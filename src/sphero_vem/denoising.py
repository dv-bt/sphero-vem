"""
Functions for denoising images, based on CAREamics.
"""

import warnings
from pathlib import Path
from typing import Literal, ClassVar
from dataclasses import dataclass, field

from tqdm import tqdm
import numpy as np
import zarr
import dask.array as da
from sklearn.model_selection import train_test_split

import torch
from torch import nn

from careamics import CAREamist, Configuration
from careamics.config import create_n2v_configuration, UNetConfig
from careamics.models.layers import Conv_Block

from sphero_vem.io import write_zarr, _create_zarr_array, _write_zarr_metadata
from sphero_vem.utils import (
    timestamp,
    BaseConfig,
    ProcessingStep,
    temporary_zarr,
    dirname_from_spacing,
)
from sphero_vem.utils.logging import (
    ArtifactsCallback,
    HyperparamsCallback,
    setup_wanb_env,
    suppress_logging,
)


@dataclass
class DenoisingConfig(BaseConfig):
    """Configuration for Noise2Void training via CAREamics.

    Parameters
    ----------
    root_path : Path
        Path to the root Zarr archive containing the source data.
    src_path : str
        Path within the Zarr archive to the source image array.
    num_images : int, optional
        Number of 2-D slices to load for training. Default is 10.
    val_split : float, optional
        Fraction of slices to use for validation. Default is 0.2.
    random_state : int, optional
        Random seed for the train/validation split. Default is 42.
    batch_size : int, optional
        Training mini-batch size. Default is 128.
    patch_size : int, optional
        Spatial size of square patches extracted from slices. Default is 64.
    epochs : int, optional
        Number of training epochs. Default is 100.
    unet_depth : int, optional
        Depth of the U-Net encoder. Default is 2.
    unet_num_channels_init : int, optional
        Number of feature channels in the first U-Net encoder layer.
        Default is 32.
    n2v2 : bool, optional
        If True, use N2V2 blind-spot strategy instead of standard N2V.
        Default is False.
    num_workers : int, optional
        Number of data-loader worker processes. Default is 16.
    wandb_project : str, optional
        Weights & Biases project name for experiment tracking.
        Default is ``"denoising"``.
    work_root : Path, optional
        Root directory for model checkpoints and configs.
        Default is ``Path("data/models/n2v")``.
    model_name : str | None, optional
        Unique name for this training run. If None, a timestamp-based name
        is generated automatically. Default is None.
    """

    root_path: Path
    src_path: str
    num_images: int = 10
    val_split: float = 0.2
    random_state: int = 42

    # N2V hyperparameters
    batch_size: int = 128
    patch_size: int = 64
    epochs: int = 100
    unet_depth: int = 2
    unet_num_channels_init: int = 32
    n2v2: bool = False

    # Other params
    num_workers: int = 16
    wandb_project: str = "denoising"
    work_root: Path = Path("data/models/n2v")
    model_name: str | None = None

    work_dir: Path = field(init=False)
    n2v_config: Configuration = field(init=False)

    EXCLUDED_JSON_FIELDS = set(["n2v_config"])
    EXCLUDED_PROCESSING_FIELDS = set(
        [
            "root_path",
            "src_path",
            "num_workers",
            "n2v_config",
            "work_dir",
            "work_root",
            "wandb_project",
        ]
    )

    def __post_init__(self):
        if not self.model_name:
            self.model_name = f"n2v-{timestamp()}"
        self.work_dir = self.work_root / self.model_name
        self.n2v_config = create_n2v_configuration(
            experiment_name=self.model_name,
            data_type="array",
            axes="SYX",
            patch_size=[self.patch_size, self.patch_size],
            batch_size=self.batch_size,
            num_epochs=self.epochs,
            use_n2v2=self.n2v2,
            logger="wandb",
            model_params={
                "depth": self.unet_depth,
                "num_channels_init": self.unet_num_channels_init,
            },
            checkpoint_params={"save_top_k": 1},
            train_dataloader_params={"num_workers": self.num_workers},
            val_dataloader_params={"num_workers": self.num_workers},
        )

    def save_n2v_config(self, filepath: str | Path) -> None:
        """Saves the N2V config class to a JSON file."""
        with open(filepath, "w") as file:
            file.write(self.n2v_config.model_dump_json(indent=4))


def train_n2v(config: DenoisingConfig) -> None:
    """Train a Noise2Void model using the parameters in *config*.

    Loads 2-D slices from a Zarr array, splits them into training and
    validation sets, saves config files, and runs the CAREamics training loop
    with Weights & Biases logging.

    Parameters
    ----------
    config : DenoisingConfig
        Training configuration. The Zarr archive at ``config.root_path`` must
        be readable and the array at ``config.src_path`` must exist.
    """
    root = zarr.open_group(config.root_path, mode="r")
    src_array = root.get(config.src_path)

    # Load training and validation arrays into memory
    train_data, val_data = train_test_split(
        src_array[: config.num_images],
        test_size=config.val_split,
        random_state=config.random_state,
    )

    # Save config files
    config_path = config.work_dir / "config.json"
    n2v_config_path = config.work_dir / "n2v_config.json"

    config.work_dir.mkdir(exist_ok=True, parents=True)
    config.to_json(config_path)
    config.save_n2v_config(n2v_config_path)

    # Set up callbacks
    callback_params = HyperparamsCallback(config.processing_metadata())
    callback_artifacts = ArtifactsCallback([config_path, n2v_config_path])

    # Run training
    setup_wanb_env(config.wandb_project)
    torch.set_float32_matmul_precision("high")
    careamist = CAREamist(
        config.n2v_config,
        work_dir=config.work_dir,
        callbacks=[callback_params, callback_artifacts],
    )
    careamist.train(train_source=train_data, val_source=val_data)


def _patch_decoder_blocks(unet: nn.Module, unet_cfg: UNetConfig) -> None:
    """Patch UnetDecoder.decoder_blocks in-place to match CAREamics 0.0.10 architecture.

    In older versions of Careamics (at least <=0.0.10), decoder Conv_Blocks receive
    concatenated skip connections as input, producing different input channel counts.
    This function rebuilds only the decoder_blocks ModuleList with the old channel
    arithmetic, leaving the bottleneck, upsampling, and final conv untouched.
    This is intended to allow running older N2V models on the newer versions of the
    library, new models should be trained using the standard Careamics API.

    Parameters
    ----------
    unet : nn.Module
        UNet instance loaded with the current CAREamics version.
    unet_cfg : UNetConfig
        UNet configuration from the CAREamics checkpoint.
    """
    depth = unet_cfg.depth
    num_channels_init = unet_cfg.num_channels_init
    conv_dim = unet_cfg.conv_dims
    use_batch_norm = unet_cfg.use_batch_norm
    groups = unet_cfg.in_channels if unet_cfg.independent_channels else 1

    upsampling = nn.Upsample(
        scale_factor=2, mode="bilinear" if conv_dim == 2 else "trilinear"
    )

    decoder_blocks: list[nn.Module] = []
    for n in range(depth):
        decoder_blocks.append(upsampling)
        in_channels = (num_channels_init * 2 ** (depth - n)) * groups
        out_channels = in_channels // 2
        decoder_blocks.append(
            Conv_Block(
                conv_dim,
                in_channels=(in_channels + in_channels // 2 if n > 0 else in_channels),
                out_channels=out_channels,
                intermediate_channel_multiplier=2,
                dropout_perc=0.0,
                activation="ReLU",
                use_batch_norm=use_batch_norm,
                groups=groups,
            )
        )

    unet.decoder.decoder_blocks = nn.ModuleList(decoder_blocks)


def _load_old_model(model_path: Path) -> CAREamist:
    """Load N2V trained with older versions of Careamics (<=0.0.10)

    Parameters
    ----------
    model_path : Path
        Path to the checkpoint

    Returns
    -------
    CAREamist
        Loaded CAREamist model.
    """

    ckpt = torch.load(model_path, map_location="cpu")
    cfg = Configuration.model_validate(ckpt["hyper_parameters"])
    unet_cfg = cfg.algorithm_config.model

    # Patch data_type for array-based prediction
    cfg.data_config.data_type = "array"

    careamist = CAREamist(cfg)

    _patch_decoder_blocks(unet=careamist.model.model, unet_cfg=unet_cfg)
    unet_state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    careamist.model.model.load_state_dict(unet_state_dict)

    return careamist


def _load_model(
    model_name: str,
    model_root: Path = Path("data/models/n2v"),
    suppress_pbar: bool = True,
) -> CAREamist:
    """Load N2V model with the specified name.

    Expects exactly one best checkpoint. If multiple are present, returns
    the first found after sorting.

    Parameters
    ----------
    model_name : str
        Name of the model to load.
    model_root : Path
        Root directory containing trained models. Default is Path("data/models/n2v").
    suppress_pbar : bool
        If True, suppress progress bars during prediction. Default is True.

    Returns
    -------
    CAREamist
        Loaded CAREamist model.
    """
    ckpt_dir = model_root / f"{model_name}/checkpoints"
    model_path = sorted(ckpt_dir.glob(f"{model_name}*.ckpt"))[0]
    try:
        careamist = CAREamist(model_path)
    except RuntimeError:
        print(
            "Model architecture incopatible with current Careamics version.\n"
            "Attempting to load model with Careamics v0.0.10 UNet architecture"
        )
        careamist = _load_old_model(model_path)

    # Suppress logging predictions to WandB
    careamist.trainer.loggers = []
    careamist.trainer.logger = None

    if suppress_pbar:
        careamist.trainer.callbacks = []

    return careamist


@dataclass
class DenoisingStats:
    """Statistics accumulated during the denoising pass.

    Tracks global intensity range of the denoised output and the residual
    histogram (original - denoised) in the original intensity space, before
    any rescaling.

    Parameters
    ----------
    global_min : float
        Running minimum of denoised float32 values across all slices.
    global_max : float
        Running maximum of denoised float32 values across all slices.
    residual_counts : np.ndarray
        Histogram counts of shape (511,), covering residuals in [-255, 255].

    Notes
    -----
    Residuals are computed as original (uint8) - denoised (float32), rounded
    and clipped to [-255, 255]. A well-behaved residual histogram should be
    approximately zero-mean and Gaussian. A small negative bias is expected
    due to N2V's blind-spot averaging.
    """

    RESIDUAL_MIN: ClassVar[int] = -255
    RESIDUAL_MAX: ClassVar[int] = 255
    RESIDUAL_BINS: ClassVar[int] = 511
    BIN_CENTERS: ClassVar[np.ndarray] = np.arange(-255, 256)

    global_min: float = float("inf")
    global_max: float = float("-inf")
    residual_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(511, dtype=np.int64)
    )

    def update(self, original: np.ndarray, denoised: np.ndarray) -> None:
        """Update statistics with a new slice.

        Parameters
        ----------
        original : np.ndarray
            Original uint8 image.
        denoised : np.ndarray
            Denoised float32 image, before any rescaling.
        """
        self.global_min = min(self.global_min, float(denoised.min()))
        self.global_max = max(self.global_max, float(denoised.max()))
        residual = np.clip(
            np.round(original.astype(np.float32) - denoised),
            self.RESIDUAL_MIN,
            self.RESIDUAL_MAX,
        ).astype(np.int16)
        self.residual_counts += np.bincount(
            residual.ravel() - self.RESIDUAL_MIN,
            minlength=self.RESIDUAL_BINS,
        )


def denoise_image(
    image: np.ndarray,
    careamist: CAREamist,
    tile_size: tuple[int, ...],
    tile_overlap: tuple[int, ...],
    batch_size: int,
    num_workers: int,
    rescale: bool = False,
    stats: DenoisingStats | None = None,
) -> np.ndarray:
    """Denoise a single YX image.

    Parameters
    ----------
    image : np.ndarray
        2D image to denoise (YX).
    careamist : CAREamist
        Trained CAREamist model.
    tile_size : tuple[int, ...]
        Tile size for prediction.
    tile_overlap : tuple[int, ...]
        Overlap between tiles.
    batch_size : int
        Number of tiles to process in parallel.
    num_workers : int
        Number of dataloader workers.
    rescale : bool
        If True, rescale denoised output to uint8 [0, 255] using per-slice min/max.
        Default is False.
    stats : DenoisingStats | None
        If provided, updated in-place with global min/max and residual histogram.
        Stats are accumulated from the float32 denoised image before rescaling.
        Default is None.

    Returns
    -------
    np.ndarray
        Denoised image. If rescale=True, returns uint8, otherwise float32.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*predict_dataloader.*num_workers.*",
            category=UserWarning,
        )
        with suppress_logging():
            denoised = careamist.predict(
                source=image,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                axes="YX",
                data_type="array",
                batch_size=batch_size,
                dataloader_params={"num_workers": num_workers},
            )[0].squeeze()

    if stats is not None:
        stats.update(image, denoised)

    if rescale:
        img_min = denoised.min()
        img_max = denoised.max()
        denoised = (
            ((denoised - img_min) / (img_max - img_min) * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

    return denoised


def denoise_stack(
    root_path: Path,
    src_path: str,
    model_name: str,
    dst_group: str = "images/denoised",
    model_root: Path = Path("data/models/n2v"),
    tile_size: tuple[int, int] = (512, 512),
    tile_overlap: tuple[int, int] = (48, 48),
    batch_size: int = 64,
    num_workers: int = 16,
    temp_dir: Path | str | None = Path("data/tmp"),
    rescale_mode: Literal["per_slice", "global"] = "per_slice",
) -> None:
    """Denoise volume with Noise2Void and rescale to uint8.

    Performs denoising with either global or per-slice intensity rescaling.
    Both modes accumulate a residual histogram saved as an npz file alongside
    the output zarr.

    Parameters
    ----------
    root_path : Path
        Path to the zarr root archive.
    src_path : str
        Path to the source array within the zarr archive.
    model_name : str
        Name of the trained N2V model. Used to locate the model checkpoint
        and configuration file under `model_root`.
    dst_group : str
        Destination group within the zarr archive. The output array path is
        constructed as `{dst_group}/{dirname_from_spacing(spacing)}`.
        Default is "images/denoised".
    model_root : Path
        Root directory containing trained models. Each model should be in a
        subdirectory with a config.json and checkpoint file.
        Default is Path("data/models/n2v").
    tile_size : tuple[int, int]
        Tile size in pixels (Y, X) for prediction. Default is (512, 512).
    tile_overlap : tuple[int, int]
        Overlap in pixels (Y, X) between adjacent tiles to avoid boundary
        artifacts. Default is (48, 48).
    batch_size : int
        Number of tiles to process in parallel on the GPU. Default is 64.
    num_workers : int
        Number of dataloader workers for tile loading. Default is 16.
    temp_dir : Path | str | None
        Directory for intermediate float32 zarr storage. Only used when
        `rescale_mode='global'`. Should be on fast local storage (SSD).
        Default is Path("data/tmp").
    rescale_mode : Literal["per_slice", "global"]
        If `'global'`, use global min/max across all slices for rescaling
        (two-pass, requires temporary zarr storage).
        If `'per_slice'`, rescale each slice independently using per-slice
        min/max (single-pass, no temporary storage).
        Default is `'per_slice'`.

    Notes
    -----
    When `rescale_mode='global'`, the intermediate zarr is uncompressed and
    can be large (4 bytes per voxel). Ensure sufficient disk space in `temp_dir`.
    The residual histogram is saved to
    `{root_path}/images/tables/denoised-residual-hist.npz`.
    """
    torch.set_float32_matmul_precision("high")

    root = zarr.open_group(root_path, mode="a")
    src_zarr = root.get(src_path)

    config = None
    try:
        config = DenoisingConfig.from_json(model_root / f"{model_name}/config.json")
    except FileNotFoundError:
        pass
    careamist = _load_model(model_name=model_name, model_root=model_root)

    dst_path = f"{dst_group}/{dirname_from_spacing(src_zarr.attrs['spacing'])}"

    hist_path = root_path / "images/tables/denoised-residual-hist.npz"
    hist_path.parent.mkdir(exist_ok=True, parents=True)

    if rescale_mode == "global":
        _denoise_global_rescale(
            src_zarr=src_zarr,
            careamist=careamist,
            root=root,
            dst_path=dst_path,
            config=config,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size=batch_size,
            num_workers=num_workers,
            temp_dir=temp_dir,
            hist_path=hist_path,
        )
    elif rescale_mode == "per_slice":
        _denoise_per_slice_rescale(
            src_zarr=src_zarr,
            careamist=careamist,
            root=root,
            dst_path=dst_path,
            config=config,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size=batch_size,
            num_workers=num_workers,
            hist_path=hist_path,
        )
    else:
        raise ValueError(
            f"Invalid value {rescale_mode} for rescale_mode. "
            "Valid options are 'global' and 'per_slice'."
        )


def _save_residual_histogram(stats: DenoisingStats, hist_path: Path) -> None:
    """Save residual histogram counts and bin centers to an npz file.

    Parameters
    ----------
    stats : DenoisingStats
        Accumulated denoising statistics.
    hist_path : Path
        Output path for the npz file.
    """
    np.savez(
        hist_path,
        counts=stats.residual_counts,
        bin_centers=stats.BIN_CENTERS,
    )


def _denoise_global_rescale(
    src_zarr: zarr.Array,
    careamist: CAREamist,
    root: zarr.Group,
    dst_path: str,
    config: DenoisingConfig | None,
    tile_size: tuple[int, int],
    tile_overlap: tuple[int, int],
    batch_size: int,
    num_workers: int,
    temp_dir: Path | str | None,
    hist_path: Path,
) -> None:
    """Two-pass denoising with global rescaling.

    Parameters
    ----------
    src_zarr : zarr.Array
        Source zarr array.
    careamist : CAREamist
        Trained CAREamist model.
    root : zarr.Group
        Root zarr group.
    dst_path : str
        Destination path within the zarr archive.
    config : DenoisingConfig | None
        Denoising configuration. If None, denoising training steps will not be written
        in the zarr metadata.
    tile_size : tuple[int, int]
        Tile size in pixels (Y, X).
    tile_overlap : tuple[int, int]
        Overlap in pixels (Y, X) between tiles.
    batch_size : int
        Number of tiles to process in parallel.
    num_workers : int
        Number of dataloader workers.
    temp_dir : Path | str | None
        Directory for intermediate float32 zarr storage.
    hist_path : Path
        Output path for the residual histogram npz file.
    """
    stats = DenoisingStats()

    with temporary_zarr(
        shape=src_zarr.shape,
        chunks=(1, *src_zarr.shape[1:]),
        dtype=np.float32,
        dir=temp_dir,
    ) as intermediate:
        # Pass 1: Denoise, accumulate stats, store float32
        for z in tqdm(range(src_zarr.shape[0]), desc="Denoising"):
            denoised = denoise_image(
                src_zarr[z],
                careamist,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                batch_size=batch_size,
                num_workers=num_workers,
                stats=stats,
            )
            intermediate[z] = denoised

        # Pass 2: Rescale globally and cast to uint8
        intermediate_dask = da.from_zarr(intermediate)
        rescaled = (
            (
                (intermediate_dask - stats.global_min)
                / (stats.global_max - stats.global_min)
                * 255
            )
            .clip(0, 255)
            .astype(np.uint8)
        )

        processing = []
        if config is not None:
            processing.append(ProcessingStep.from_config("denoising-train", config))
        processing.append(
            ProcessingStep.manual(
                "denoising-predict",
                {
                    "tile_size": tile_size,
                    "tile_overlap": tile_overlap,
                    "global_min": stats.global_min,
                    "global_max": stats.global_max,
                    "rescale_mode": "global",
                },
            ),
        )

        write_zarr(
            root=root,
            array=rescaled,
            dst_path=dst_path,
            src_zarr=src_zarr,
            dtype=np.uint8,
            processing=processing,
        )

    _save_residual_histogram(stats, hist_path)


def _denoise_per_slice_rescale(
    src_zarr: zarr.Array,
    careamist: CAREamist,
    root: zarr.Group,
    dst_path: str,
    config: DenoisingConfig | None,
    tile_size: tuple[int, int],
    tile_overlap: tuple[int, int],
    batch_size: int,
    num_workers: int,
    hist_path: Path,
) -> None:
    """Single-pass denoising with per-slice rescaling.

    Parameters
    ----------
    src_zarr : zarr.Array
        Source zarr array.
    careamist : CAREamist
        Trained CAREamist model.
    root : zarr.Group
        Root zarr group.
    dst_path : str
        Destination path within the zarr archive.
    config : DenoisingConfig | None
        Denoising configuration. If None, denoising training steps will not be written
        in the zarr metadata.
    tile_size : tuple[int, int]
        Tile size in pixels (Y, X).
    tile_overlap : tuple[int, int]
        Overlap in pixels (Y, X) between tiles.
    batch_size : int
        Number of tiles to process in parallel.
    num_workers : int
        Number of dataloader workers.
    hist_path : Path
        Output path for the residual histogram npz file.
    """
    stats = DenoisingStats()

    processing = []
    if config is not None:
        processing.append(ProcessingStep.from_config("denoising-train", config))
    processing.append(
        ProcessingStep.manual(
            "denoising-predict",
            {
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "rescale_mode": "per_slice",
            },
        ),
    )

    dst_zarr = _create_zarr_array(
        root=root,
        dst_path=dst_path,
        shape=src_zarr.shape,
        chunks=src_zarr.chunks,
        dtype=np.uint8,
    )

    for z in tqdm(range(src_zarr.shape[0]), desc="Denoising"):
        image = src_zarr[z]
        denoised = denoise_image(
            image,
            careamist,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size=batch_size,
            num_workers=num_workers,
            stats=stats,
        )
        img_min = denoised.min()
        img_max = denoised.max()
        dst_zarr[z] = (
            ((denoised - img_min) / (img_max - img_min) * 255)
            .clip(0, 255)
            .astype(np.uint8)
        )

    _write_zarr_metadata(
        root=root,
        dst_zarr=dst_zarr,
        src_zarr=src_zarr,
        processing=processing,
    )

    _save_residual_histogram(stats, hist_path)
