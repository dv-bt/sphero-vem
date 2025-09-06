"""
This module contains functions used for the registration of volume stacks
"""

import os
import shutil
import json
from typing import Any
from collections import deque
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Callable, Optional
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import tifffile
from sphero_vem.io import imwrite, read_tensor
from sphero_vem.metrics import LossDispatcher
from sphero_vem.preprocessing import downscale_tensor
from sphero_vem.utils import generate_manifest, detect_torch_device, infer_dataset


class TransformType(Enum):
    SIMILARITY = "similarity"
    RIGID = "rigid"
    AFFINE = "affine"


@dataclass
class RegistrationConfig:
    """Configuration class for registration"""

    data_dir: Path
    out_dir: Optional[Path] = None
    pyramid_levels: int = 4
    pyramid_factors: int | list[int] = 2
    pyramid_epochs: int | list[int] = 300
    learning_rate: float | list[float] = 1e-4
    loss_type: str = "ncc"
    loss_kwargs: dict = field(default_factory=dict)
    optimizer: str = "Adam"
    max_pairs: Optional[int] = None
    progress_steps: bool = False
    verbose: bool = True
    early_stopping: bool = True
    stop_window: int = 15
    stop_tol: float = 1e-6
    transformation: str = "similarity"
    init_std: float = 0.001

    # Derived values, initialized by post_init
    dataset: Path = field(init=False)
    data_root: Path = field(init=False)
    wandb_api_key: str = field(init=False)
    images: list[Path] = field(init=False)
    fixed_images: list[Path] = field(init=False)
    moving_images: list[Path] = field(init=False)
    num_pairs: int = field(init=False)
    device: torch.device = field(init=False)

    def __post_init__(self):
        """Load environment variables and init derived values"""
        self.wandb_api_key = os.getenv("API_KEY")
        self.data_root = Path(os.getenv("DATA_ROOT"))
        self.data_dir = self.data_root / self.data_dir
        self.out_dir = self.data_root / self.out_dir if self.out_dir else None
        self.device = detect_torch_device()
        self.dataset = infer_dataset(self.data_dir)

        # Get image lists
        self.images = sorted(self.data_dir.glob("*.tif"))
        if self.max_pairs:
            self.images = self.images[: self.max_pairs + 1]
        self.fixed_images = self.images[:-1]
        self.moving_images = self.images[1:]
        self.num_pairs = len(self.fixed_images)

        # Populate lists for downscaling factors and epochs for each pyramid level
        if isinstance(self.pyramid_factors, int):
            self.pyramid_factors = [
                self.pyramid_factors**i for i in reversed(range(self.pyramid_levels))
            ]
        self.pyramid_epochs = _expand_pyramid_list(
            self.pyramid_epochs, self.pyramid_levels
        )
        self.learning_rate = _expand_pyramid_list(
            self.learning_rate, self.pyramid_levels
        )


def register_image_pair_itk(
    fixed_image: sitk.Image,
    moving_image: sitk.Image,
    transform_type: TransformType,
    shrink_factors: Iterable[int | float] = [1],
    smoothing_sigmas: Iterable[int | float] = [0.0],
    sampling_fractions: Iterable[float] | float = [1.0],
) -> tuple[sitk.Image, sitk.Transform]:
    """Register a pair of images and returns the calculated transformation. It uses
    gradient descent optimization on a mutual information criterion.
    Optimization is done with a multi resolution framework, but this can
    be turned off by passing a single value to shrink_factors."""

    # Transform
    transform_factory = get_transform_factory(transform_type)
    initial_transform = transform_factory()
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform)

    # Metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=30)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentagePerLevel(sampling_fractions)

    # Optimizer
    registration_method.SetOptimizerAsGradientDescentLineSearch(
        learningRate=1.0, numberOfIterations=100
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution framework
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=shrink_factors)
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=smoothing_sigmas)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Execute registration
    final_transform = registration_method.Execute(fixed_image, moving_image)

    # Apply the transform to the moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkBSpline3)
    resampler.SetTransform(final_transform)
    moving_image_final = resampler.Execute(moving_image)

    return moving_image_final, final_transform


def get_transform_factory(
    transform_type: TransformType,
) -> Callable[[], sitk.Transform]:
    """
    Returns a factory function for a specific transformation type.
    """
    transform_dispatcher = {
        TransformType.SIMILARITY: lambda: sitk.Similarity2DTransform(),
        TransformType.RIGID: lambda: sitk.Euler2DTransform(),
        TransformType.AFFINE: lambda: sitk.AffineTransform(2),
    }

    # Retrieve the function based on the Enum member
    return transform_dispatcher[transform_type]


def register_to_disk_itk(
    fixed_image_path: Path,
    moving_image_path: Path,
    dest_path: Path,
    **registration_kwargs,
) -> None:
    """Read image pair, register them, and save the registered moving image"""
    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
    final_image, _ = register_image_pair_itk(
        fixed_image, moving_image, **registration_kwargs
    )
    final_image = sitk.Cast(final_image, sitk.sitkUInt8)
    final_image_array = sitk.GetArrayViewFromImage(final_image)
    imwrite(dest_path, final_image_array, uncompressed=True)


def _affine_transform(q: torch.Tensor) -> torch.Tensor:
    """
    Build a batch of 2x3 affine transformation matrices from scaled parameters.
    Constructs 2D affine matrices with rotation and translation degrees of freedom.

    Parameters
    ----------
    q : torch.Tensor
        Tensor of shape (N, 3) containing parameters per example as
        [tx, ty, theta]. tx and ty are translation components, theta is the
        rotation angle. Angles are expected to be in radians.

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2, 3) where each 2x3 matrix has the form:
            [[cos(theta), -sin(theta), tx],
             [sin(theta),  cos(theta), ty]]
        The returned tensor preserves the dtype and device of the input `q`.
    """
    t = q[:, :2]
    ang = q[:, 2]
    c, s = torch.cos(ang), torch.sin(ang)
    A = torch.stack([c, -s, t[:, 0], s, c, t[:, 1]], dim=-1).view(-1, 2, 3)
    return A


def _warp_affine(
    img: torch.Tensor, transform_matrix: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Warp and image using an affine transform matrix"""
    grid = F.affine_grid(transform_matrix, size=img.shape, align_corners=False)
    return F.grid_sample(
        img, grid, mode=mode, padding_mode="zeros", align_corners=False
    )


class ImageTransform(torch.nn.Module):
    def __init__(self, device: torch.device, q_init: torch.Tensor | None = None):
        super().__init__()
        self.q = torch.nn.Parameter(q_init)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        A = _affine_transform(self.q)
        return _warp_affine(img, A)


def _early_stopping(
    loss_history: list[float], window: int = 20, tol: float = 1e-4
) -> bool:
    """Early stopping criterion"""
    if len(loss_history) == window:
        fluctuation = max(loss_history) - min(loss_history)
        if fluctuation < tol:
            return True
    return False


def register_image_pair(
    fixed_img: torch.Tensor, moving_img: torch.Tensor, config: RegistrationConfig
) -> tuple[torch.Tensor, list[float]]:
    fixed_img = fixed_img.to(config.device)
    moving_img = moving_img.to(config.device)

    # Initialize transformation parameters
    q_init = torch.randn(1, 3, device=config.device) * config.init_std
    loss_fun = LossDispatcher(config.loss_type)
    early_stop_history = deque(maxlen=config.stop_window)
    full_loss_history = []

    for level in range(config.pyramid_levels):
        with torch.no_grad():
            fixed_ds = downscale_tensor(fixed_img, config.pyramid_factors[level])
            moving_ds = downscale_tensor(moving_img, config.pyramid_factors[level])

        model = ImageTransform(device=config.device, q_init=q_init).to(config.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate[level])

        pbar = trange(
            config.pyramid_epochs[level],
            desc=f"Fitting transform, level {level}, "
            f"downscaling {config.pyramid_factors[level]}",
            leave=True,
            disable=not config.progress_steps,
        )
        for _ in pbar:
            optimizer.zero_grad()
            warped_img = model(moving_ds)
            loss: torch.Tensor = loss_fun(fixed_ds, warped_img, **config.loss_kwargs)
            loss.backward()
            optimizer.step()

            # Update progress bar
            pbar.set_postfix(loss=loss.item())
            early_stop_history.append(loss.item())
            full_loss_history.append([level, loss.item()])
            if config.early_stopping:
                if _early_stopping(
                    early_stop_history, config.stop_window, config.stop_tol
                ):
                    break

        q_init = model.q

    q_final = model.q.detach().cpu()

    return q_final, full_loss_history


def register_stack(config: RegistrationConfig) -> None:
    """Register volume stack."""

    # Set floating point precision
    torch.set_float32_matmul_precision("high")

    generate_manifest(
        config.dataset, config.out_dir, config.images, _create_processing_entry(config)
    )

    # Initialize storage lists
    log_data = []
    pairwise_matrices = []
    composed_matrices = []
    all_loss_histories = []

    for i, (fixed_path, moving_path) in enumerate(
        tqdm(
            zip(config.fixed_images, config.moving_images),
            "Registering images",
            disable=not config.verbose,
            total=config.num_pairs,
        )
    ):
        fixed_img = read_tensor(fixed_path, return_4d=True)
        moving_img = read_tensor(moving_path, return_4d=True)

        params, loss_history = register_image_pair(fixed_img, moving_img, config)
        A_pairwise = _affine_transform(params)
        A_composed = _compose_transform(A_pairwise, composed_matrices)

        # Copy first fixed image
        if i == 0:
            shutil.copy(fixed_path, config.out_dir / fixed_path.name)

        warped_img = _warp_affine(moving_img, A_composed, "bicubic")
        tifffile.imwrite(
            config.out_dir / moving_path.name,
            warped_img.squeeze(0).to(torch.uint8).numpy(),
        )

        # Update storage lists
        all_loss_histories.append(loss_history)
        pairwise_matrices.append(A_pairwise)
        composed_matrices.append(A_composed)
        log_data.append(_create_log_entry(i, fixed_path, moving_path, loss_history))

    # Save summary and transformations
    with open(config.out_dir / "registration_log.json", "w") as file:
        json.dump(log_data, file, indent=4)

    np.savez_compressed(
        config.out_dir / "registration_results.npz",
        pairwise_matrices=pairwise_matrices,
        composed_matrices=composed_matrices,
        loss_history=all_loss_histories,
    )


def _to_homog(A: torch.Tensor):
    """Convert a 1x2x3 affine matrix to homogeneous coordinates 1x3x3"""
    A_homog = torch.eye(3).unsqueeze(0)
    A_homog[:, :2, :] = A
    return A_homog


def _from_homog(A_homog: torch.Tensor):
    """Convert a 1x3x3 matrix in homogeneous coordinates to a 1x2x3 matrix"""
    return A_homog[:, :2, :]


def _compose_transform(
    A: torch.Tensor, composed_matrices: list[torch.Tensor]
) -> torch.Tensor:
    """Compose current transformtion matrix with previous transformations.
    It takes the full list to handle edge cases better"""
    try:
        A_previous = _to_homog(composed_matrices[-1])
        A_composed = _to_homog(A) @ A_previous
        return _from_homog(A_composed)
    except IndexError:
        return A


def _create_log_entry(
    i: int, fixed_path: Path, moving_path: Path, loss_history: list[float]
) -> dict[str, Any]:
    """Create entry for registration log JSON file"""
    log_entry = {
        "pair_index_start": i,
        "fixed_image_path": fixed_path.name,
        "moving_image_path": moving_path.name,
        "final_loss": loss_history[-1] if loss_history else None,
        "num_steps_taken": len(loss_history),
    }
    return log_entry


def _create_processing_entry(config: RegistrationConfig) -> list[dict]:
    """Create entry for the processing field in the manifest file"""
    entry = [
        {
            "step": "registration",
            "pyramid_levels": config.pyramid_levels,
            "pyramid_factors": config.pyramid_factors,
            "pyramid_epochs": config.pyramid_epochs,
            "learning_rate": config.learning_rate,
            "loss_type": config.loss_type,
            "transformation": config.transformation,
            "optimizer": config.optimizer,
            "stop_window": config.stop_window,
            "stop_tol": config.stop_tol,
            "init_std": config.init_std,
        }
    ]
    return entry


def _expand_pyramid_list(param: int | list, levels: int) -> list:
    if isinstance(param, int):
        param = [param for i in range(levels)]
    return param
