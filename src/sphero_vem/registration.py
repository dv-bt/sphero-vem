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
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import tifffile
from sphero_vem.io import read_tensor
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
    learning_rate: float | list[float] = field(
        default_factory=lambda: [1e-3, 1e-3, 5e-4, 1e-4]
    )
    loss_type: str = "ncc"
    loss_kwargs: dict = field(default_factory=dict)
    optimizer: str = "Adam"
    max_pairs: Optional[int] = None
    progress_steps: bool = False
    verbose: bool = True
    early_stopping: bool = True
    stop_window: int = 15
    stop_tol: float = 1e-5
    transformation: str = "similarity"
    init_std: float = 0.001
    scaling: bool = True
    shear: bool = True
    regularization_param: float = 0.5

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


def _affine_transform(q: torch.Tensor) -> torch.Tensor:
    """
    Build a batch of 2x3 affine transformation matrices from 6 parameters.
    Constructs 2D affine matrices with translation, rotation, scaling, and shear.
    The order of operations is: Scale -> Shear -> Rotate -> Translate.

    Parameters
    ----------
    q : torch.Tensor
        Tensor of shape (N, 6) containing parameters per example:
        [tx, ty, theta, sx, sy, k]
        - tx, ty: translation components
        - theta: rotation angle in radians
        - sx, sy: scaling factors
        - k: horizontal shear factor

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2, 3) representing the affine transformation.
        Each 2x3 matrix is the result of composing the individual transformations.
    """
    t = q[:, :2]
    ang = q[:, 2]
    scale = q[:, 3:5]
    shear_k = q[:, 5]

    c, s = torch.cos(ang), torch.sin(ang)
    sx, sy = scale[:, 0], scale[:, 1]

    # Construct the 2x2 linear transformation matrix (Scale -> Shear -> Rotate)
    # This combines the effects of scaling, shearing, and rotation
    # A = R @ K @ S
    # R = [[c, -s], [s, c]]
    # K = [[1, k], [0, 1]] (horizontal shear)
    # S = [[sx, 0], [0, sy]]
    A_11 = sx * c
    A_12 = sy * (shear_k * c - s)
    A_21 = sx * s
    A_22 = sy * (shear_k * s + c)
    A = torch.stack([A_11, A_12, t[:, 0], A_21, A_22, t[:, 1]], dim=-1)

    return A.view(-1, 2, 3)


def _warp_affine(
    img: torch.Tensor, transform_matrix: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Warp and image using an affine transform matrix"""
    grid = F.affine_grid(transform_matrix, size=img.shape, align_corners=False)
    return F.grid_sample(
        img, grid, mode=mode, padding_mode="zeros", align_corners=False
    )


class ImageTransform(torch.nn.Module):
    def __init__(
        self, config: RegistrationConfig, delta_q_init: torch.Tensor | None = None
    ):
        super().__init__()
        self.delta_q = torch.nn.Parameter(delta_q_init)
        self.q_identity = torch.tensor(
            [
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
            ],
            device=config.device,
        ).unsqueeze(0)

        # Mask transformation parameters that are not used
        # Rotation and translation are always considered
        scaling = 1.0 if (config.transformation == "affine" and config.scaling) else 0.0
        shear = 1.0 if (config.transformation == "affine" and config.shear) else 0.0
        self.params_mask = torch.tensor(
            [1.0, 1.0, 1.0, scaling, scaling, shear], device=config.device
        ).unsqueeze(0)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        q = self.delta_q * self.params_mask + self.q_identity
        A = _affine_transform(q)
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

    # Initialize transformation parameters.
    # The transformation is learned as a deviation from the identity transform.
    # This is done so that all parameters are zero-centered
    delta_q_init = torch.randn(1, 6, device=config.device) * config.init_std
    q_identity = torch.tensor(
        [
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
        ]
    )
    loss_fun = LossDispatcher(config.loss_type)
    early_stop_history = deque(maxlen=config.stop_window)
    full_loss_history = []

    for level in range(config.pyramid_levels):
        with torch.no_grad():
            fixed_ds = downscale_tensor(fixed_img, config.pyramid_factors[level])
            moving_ds = downscale_tensor(moving_img, config.pyramid_factors[level])

        model = ImageTransform(config=config, delta_q_init=delta_q_init).to(
            config.device
        )
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
            loss = loss_fun(
                fixed_ds, warped_img, **config.loss_kwargs
            ) + config.regularization_param * torch.sum(
                (model.delta_q[:, 3:] * model.params_mask[:, 3:]) ** 2
            )
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

        delta_q_init = model.delta_q

    q_final = model.delta_q.detach().cpu() + q_identity

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
            warped_img.squeeze(0).squeeze(0).to(torch.uint8).numpy(),
        )

        # Update storage lists
        all_loss_histories.append(_pad_loss_history(loss_history, config))
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
        "final_loss": loss_history[-1][1] if loss_history else None,
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
    if isinstance(param, int) or isinstance(param, float):
        param = [param for i in range(levels)]
    return param


def _pad_loss_history(loss_history: list, config: RegistrationConfig) -> np.ndarray:
    """Pad loss history with NaN to compensate for inhomogeneous lengths due to
    early stopping"""
    history_arr = np.array(loss_history)
    padded_history = np.full((sum(config.pyramid_epochs), 2), np.nan)
    padded_history[: len(history_arr), :] = history_arr
    return padded_history
