"""
Core registration pipeline for volume stacks.

This module contains the configuration, pairwise registration algorithm, and
the main stack registration pipeline with optional border cropping.
"""

from typing import Any
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm, trange
import zarr
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
from sphero_vem.io import write_zarr
from sphero_vem.metrics import LossDispatcher
from sphero_vem.preprocessing import downscale_tensor
from sphero_vem.utils import (
    BaseConfig,
    ProcessingStep,
    dirname_from_spacing,
    temporary_zarr,
    detect_torch_device,
)
from sphero_vem.registration.transforms import (
    _affine_transform,
    _warp_affine,
    _compose_transform,
)
from sphero_vem.registration.cropping import find_border_crop


@dataclass
class RegistrationConfig(BaseConfig):
    """Configuration for pairwise affine registration of a Zarr volume stack.

    Parameters
    ----------
    root_path : Path
        Root directory of the Zarr archive.
    src_path : str
        Path within the archive to the source image array (e.g.
        ``"images/50-10-10"``).
    dst_group : str, optional
        Group path under which registered slices are written.
        Default is ``"images/registered"``.
    pyramid_levels : int, optional
        Number of resolution levels in the multi-scale pyramid. Default is 4.
    pyramid_factors : int | list[int], optional
        Downsampling factor(s) between pyramid levels. An integer is expanded
        to ``[factor**(levels-1), ..., factor**0]``. Default is 2.
    pyramid_epochs : int | list[int], optional
        Training epochs per pyramid level. A scalar is broadcast to all levels.
        Default is 300.
    learning_rate : float | list[float], optional
        Learning rate per pyramid level. Default is
        ``[1e-3, 1e-3, 5e-4, 1e-4]``.
    loss_type : str, optional
        Name of the image similarity loss (see ``LossDispatcher``).
        Default is ``"ncc"``.
    loss_kwargs : dict, optional
        Extra keyword arguments forwarded to the loss function. Default is
        ``{}``.
    optimizer : str, optional
        Optimizer class name (currently only ``"Adam"`` is used).
        Default is ``"Adam"``.
    max_pairs : int | None, optional
        Maximum number of consecutive slice pairs to register. If None, all
        pairs in the stack are used. Default is None.
    progress_steps : bool, optional
        Show per-step tqdm progress bars. Default is False.
    verbose : bool, optional
        Print high-level progress messages. Default is True.
    early_stopping : bool, optional
        Enable early stopping when the loss plateaus. Default is True.
    stop_window : int, optional
        Rolling window size for early-stopping fluctuation check. Default 15.
    stop_tol : float, optional
        Fluctuation tolerance below which training stops. Default is 1e-5.
    transformation : str, optional
        Transformation family: ``"similarity"``, ``"rigid"``, or ``"affine"``.
        Default is ``"similarity"``.
    init_std : float, optional
        Standard deviation for random initialization of transform parameters.
        Default is 0.001.
    scaling : bool, optional
        Enable per-axis scaling when ``transformation="affine"``. Default True.
    shear : bool, optional
        Enable shear when ``transformation="affine"``. Default is True.
    regularization_param : float, optional
        Weight for L2 regularization on non-translation parameters.
        Default is 0.5.
    crop_borders : bool, optional
        Apply border cropping after registration. Default is True.
    crop_safety_margin : int, optional
        Extra pixels added to the most restrictive per-slice crop box.
        Default is 5.
    crop_stride : int, optional
        Pixel stride for the coarse crop search. Default is 20.
    crop_restarts : int, optional
        Number of random restarts for the crop refinement. Default is 10.
    n_workers : int, optional
        Number of Dask threads for writing the output array. Default is 4.
    """

    # Required input/output paths
    root_path: Path
    src_path: str
    dst_group: str = "images/registered"

    # Registration hyperparameters
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

    # Post-processing parameters
    crop_borders: bool = True
    crop_safety_margin: int = 5
    crop_stride: int = 20
    crop_restarts: int = 10
    n_workers: int = 4

    # Derived values (initialized in __post_init__)
    device: torch.device = field(init=False)
    src_zarr: zarr.Array = field(init=False)
    spacing: tuple[int, int, int] = field(init=False)
    dst_path: str = field(init=False)
    num_pairs: int = field(init=False)
    zarr_chunks: tuple[int, ...] = field(init=False)

    EXCLUDED_JSON_FIELDS = {"src_zarr", "device"}
    EXCLUDED_PROCESSING_FIELDS = {
        "root_path",
        "src_path",
        "dst_group",
        "verbose",
        "progress_steps",
        "crop_safety_margin",
        "crop_stride",
        "crop_restarts",
        "n_workers",
        "zarr_chunks",
        "src_zarr",
        "device",
        "spacing",
        "dst_path",
        "num_pairs",
    }

    def __post_init__(self):
        """Derive runtime fields from the source Zarr array.

        Opens ``src_zarr``, reads spacing, infers ``dst_path``, determines
        ``num_pairs``, and expands scalar pyramid parameters to per-level lists.
        """
        self.device = detect_torch_device()

        # Open source zarr array
        self.src_zarr = zarr.open_array(self.root_path / self.src_path, mode="r")
        self.spacing = tuple(self.src_zarr.attrs.get("spacing"))
        self.zarr_chunks = self.src_zarr.chunks

        # Construct destination path using spacing convention
        spacing_dir = dirname_from_spacing(self.spacing)
        self.dst_path = f"{self.dst_group}/{spacing_dir}"

        # Determine number of image pairs
        total_slices = self.src_zarr.shape[0]
        if self.max_pairs:
            self.num_pairs = min(self.max_pairs, total_slices - 1)
        else:
            self.num_pairs = total_slices - 1

        # Expand pyramid parameters to lists
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


class ImageTransform(torch.nn.Module):
    """Learnable 2D affine transform module for image registration.

    Parameterizes the transformation as a deviation from the identity so that
    all learned parameters are zero-centred at initialization.

    Parameters
    ----------
    config : RegistrationConfig
        Registration configuration. Controls which transform components
        (scaling, shear) are active via ``transformation``, ``scaling``, and
        ``shear`` fields, and provides the target device.
    delta_q_init : torch.Tensor | None, optional
        Initial parameter deviation tensor of shape (1, 6). If None, a
        zero tensor is used.
    """

    def __init__(
        self, config: RegistrationConfig, delta_q_init: torch.Tensor | None = None
    ):
        """Initialise parameters and build the active-parameter mask.

        Parameters
        ----------
        config : RegistrationConfig
            Registration configuration.
        delta_q_init : torch.Tensor | None, optional
            Initial ``delta_q`` values of shape (1, 6). Default is None
            (zero-initialized).
        """
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
        """Apply the current learned transform to *img*.

        Parameters
        ----------
        img : torch.Tensor
            Input image tensor of shape (N, C, H, W).

        Returns
        -------
        torch.Tensor
            Warped image tensor of the same shape as *img*.
        """
        q = self.delta_q * self.params_mask + self.q_identity
        A = _affine_transform(q)
        return _warp_affine(img, A)


def _early_stopping(
    loss_history: list[float], window: int = 20, tol: float = 1e-4
) -> bool:
    """Decide whether training should stop based on recent loss fluctuation.

    Parameters
    ----------
    loss_history : list[float]
        Rolling window of recent loss values.
    window : int, optional
        Number of steps to consider. Stopping is only triggered once
        ``len(loss_history) == window``. Default is 20.
    tol : float, optional
        Maximum allowed fluctuation (max − min) within the window.
        Default is 1e-4.

    Returns
    -------
    bool
        True if the loss has converged (fluctuation < *tol*), False otherwise.
    """
    if len(loss_history) == window:
        fluctuation = max(loss_history) - min(loss_history)
        if fluctuation < tol:
            return True
    return False


def register_image_pair(
    fixed_img: torch.Tensor, moving_img: torch.Tensor, config: RegistrationConfig
) -> tuple[torch.Tensor, list[float]]:
    """Register a moving image to a fixed image using multi-scale gradient descent.

    Optimizes a 2D affine transform at each pyramid level, passing the
    learned parameters as initialization to the next (finer) level.

    Parameters
    ----------
    fixed_img : torch.Tensor
        Fixed reference image of shape (1, 1, H, W).
    moving_img : torch.Tensor
        Moving image to be aligned, same shape as *fixed_img*.
    config : RegistrationConfig
        Registration configuration including pyramid settings, optimizer
        parameters, loss type, and device.

    Returns
    -------
    q_final : torch.Tensor
        Final affine parameter vector of shape (1, 6), as absolute parameters
        (deviation + identity).
    full_loss_history : list[list]
        Per-step loss records as ``[level, loss_value]`` pairs across all
        pyramid levels.
    """
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


def _slice_to_tensor(
    slice_data: np.ndarray,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert 2D zarr slice to 4D tensor (1x1xHxW) for registration.

    Parameters
    ----------
    slice_data : np.ndarray
        2D array from zarr slice
    device : torch.device
        Target device for tensor
    dtype : torch.dtype
        Target dtype (default: float32)

    Returns
    -------
    torch.Tensor
        4D tensor ready for registration pipeline
    """
    tensor = torch.from_numpy(slice_data).to(dtype=dtype, device=device)
    while tensor.dim() < 4:
        tensor = tensor.unsqueeze(0)
    return tensor


def _postprocess_registered(
    temp_zarr: zarr.Array,
    config: RegistrationConfig,
    transforms_metadata: dict,
) -> None:
    """Post-process registered stack with border cropping using dask.

    Steps:
    1. Find optimal crop box across all slices
    2. Apply crop using dask for parallel processing
    3. Write final output with write_zarr()

    Parameters
    ----------
    temp_zarr : zarr.Array
        Temporary zarr array containing registered (uncropped) stack
    config : RegistrationConfig
        Configuration object with cropping parameters
    transforms_metadata : dict
        Dictionary containing transformation matrices and registration log
    """

    crop_boxes = []
    for z in tqdm(
        range(temp_zarr.shape[0]), disable=not config.verbose, desc="Finding crops"
    ):
        crop_box = find_border_crop(
            temp_zarr[z],
            pix_stride=config.crop_stride,
            n_restarts=config.crop_restarts,
            jitter=10,
        )
        crop_boxes.append(crop_box)

    # Find most restrictive crop with safety margin
    crops_array = np.vstack(crop_boxes)
    min_crop = (
        int(crops_array[:, 0].max() + config.crop_safety_margin),
        int(crops_array[:, 1].min() - config.crop_safety_margin),
        int(crops_array[:, 2].max() + config.crop_safety_margin),
        int(crops_array[:, 3].min() - config.crop_safety_margin),
    )

    # Create dask array and apply crop lazily
    temp_dask = da.from_zarr(temp_zarr)
    cropped_dask = temp_dask[:, min_crop[0] : min_crop[1], min_crop[2] : min_crop[3]]

    # Write final output
    processing_steps = [
        ProcessingStep.from_config("registration", config),
        ProcessingStep.manual(
            "border_crop",
            {
                "crop_box": min_crop,
                "safety_margin": config.crop_safety_margin,
                "crop_stride": config.crop_stride,
                "crop_restarts": config.crop_restarts,
            },
        ),
    ]

    root = zarr.open_group(config.root_path, mode="a")
    with ProgressBar(), dask.config.set(num_workers=config.n_workers):
        write_zarr(
            root=root,
            array=cropped_dask,
            dst_path=config.dst_path,
            src_zarr=config.src_zarr,
            processing=processing_steps,
        )

    # Add transformation metadata
    dst_zarr = root[config.dst_path]
    for key, value in transforms_metadata.items():
        dst_zarr.attrs[key] = value
    dst_zarr.attrs["crop_box"] = min_crop


def _write_registered(
    temp_zarr: zarr.Array,
    config: RegistrationConfig,
    transforms_metadata: dict,
) -> None:
    """Write registered stack without cropping (using dask for consistency).

    Even without cropping, we use dask to avoid loading the full volume.

    Parameters
    ----------
    temp_zarr : zarr.Array
        Temporary zarr array containing registered stack
    config : RegistrationConfig
        Configuration object
    transforms_metadata : dict
        Dictionary containing transformation matrices and registration log
    """

    temp_dask = da.from_zarr(temp_zarr)
    processing = ProcessingStep.from_config("registration", config)
    root = zarr.open_group(config.root_path, mode="a")

    with ProgressBar(), dask.config.set(num_workers=config.n_workers):
        write_zarr(
            root=root,
            array=temp_dask,
            dst_path=config.dst_path,
            src_zarr=config.src_zarr,
            processing=processing,
        )

    # Add transformation metadata
    dst_zarr = root.get(config.dst_path)
    for key, value in transforms_metadata.items():
        dst_zarr.attrs[key] = value


def register_stack(config: RegistrationConfig) -> None:
    """Register volume stack from zarr archive.

    This function performs registration in two phases:

    1. Sequential affine calculation: Register consecutive image pairs and write to
       temporary zarr.
    2. Post-processing: Optional border cropping and writing final output.

    Parameters
    ----------
    config : RegistrationConfig
        Configuration object with all registration and processing parameters
    """
    # Set floating point precision
    torch.set_float32_matmul_precision("high")

    # Initialize storage lists for metadata
    registration_log = []
    pairwise_matrices = []
    composed_matrices = []

    # Phase 1: Sequential affine calculation with temporary zarr storage
    with temporary_zarr(
        shape=(config.num_pairs + 1, *config.src_zarr.shape[1:]),
        dtype=config.src_zarr.dtype,
        chunks=config.zarr_chunks,
    ) as temp_zarr:
        # Copy first slice (reference, no transformation)
        temp_zarr[0] = config.src_zarr[0]

        # Process pairs sequentially
        for i in tqdm(
            range(config.num_pairs),
            desc="Registering images",
            disable=not config.verbose,
        ):
            fixed_img = _slice_to_tensor(config.src_zarr[i], config.device)
            moving_img = _slice_to_tensor(config.src_zarr[i + 1], config.device)

            params, loss_history = register_image_pair(fixed_img, moving_img, config)
            A_pairwise = _affine_transform(params)
            A_composed = _compose_transform(A_pairwise, composed_matrices)

            warped_img = _warp_affine(moving_img, A_composed, "bicubic")
            temp_zarr[i + 1] = warped_img.squeeze(0).squeeze(0).cpu().numpy()

            # Store metadata
            pairwise_matrices.append(A_pairwise)
            composed_matrices.append(A_composed)
            registration_log.append(_create_log_entry(i, loss_history))

        # Prepare metadata dictionary
        transforms_metadata = {
            "pairwise_transforms": [
                A.squeeze(0).cpu().numpy().tolist() for A in pairwise_matrices
            ],
            "composed_transforms": [
                A.squeeze(0).cpu().numpy().tolist() for A in composed_matrices
            ],
            "registration_log": registration_log,
        }

        # Phase 2: Post-processing (cropping or direct write)
        if config.crop_borders:
            _postprocess_registered(temp_zarr, config, transforms_metadata)
        else:
            _write_registered(temp_zarr, config, transforms_metadata)

    if config.verbose:
        print(f"Registration complete: {config.dst_path}")


def _create_log_entry(pair_index: int, loss_history: list[float]) -> dict[str, Any]:
    """Create a registration log entry for a single slice pair.

    Parameters
    ----------
    pair_index : int
        Index of the fixed slice in the pair (moving slice is ``pair_index + 1``).
    loss_history : list[float]
        Per-step loss records for this pair, as ``[level, loss_value]`` pairs.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"pair_index"``, ``"slice_indices"``,
        ``"final_loss"``, and ``"num_steps_taken"``.
    """
    return {
        "pair_index": pair_index,
        "slice_indices": [pair_index, pair_index + 1],
        "final_loss": loss_history[-1][1] if loss_history else None,
        "num_steps_taken": len(loss_history),
    }


def _expand_pyramid_list(param: int | list, levels: int) -> list:
    """Broadcast a scalar pyramid parameter to a per-level list.

    Parameters
    ----------
    param : int | float | list
        Scalar value or already-expanded list.
    levels : int
        Number of pyramid levels.

    Returns
    -------
    list
        List of length *levels* with the parameter value repeated, or *param*
        unchanged if it is already a list.
    """
    if isinstance(param, int) or isinstance(param, float):
        param = [param for i in range(levels)]
    return param
