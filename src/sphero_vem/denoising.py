"""
Functions for denoising images, based on Careamics.
"""

from pathlib import Path
from typing import Literal
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import zarr
import dask.array as da
from sklearn.model_selection import train_test_split
import torch
from careamics import CAREamist, Configuration
from careamics.config import create_n2v_configuration
from sphero_vem.io import write_zarr
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
    """Train Noise2Void using the parameters specifed in config."""
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


def _load_model(
    model_name: str,
    model_root: Path = Path("data/models/n2v"),
    suppress_pbar: bool = True,
) -> None:
    """Load N2V model with the specified name. This function expects to find only one
    best checkpoint. If multiple are present, it will just return the first found."""
    ckpt_dir = model_root / f"{model_name}/checkpoints"
    model_path = sorted(ckpt_dir.glob(f"{model_name}*.ckpt"))[0]
    careamist = CAREamist(model_path)

    # Suppress logging predictions to WandB
    careamist.trainer.loggers = []
    careamist.trainer.logger = None

    # Suppress progress bars
    if suppress_pbar:
        careamist.trainer.callbacks = []

    return careamist


@dataclass
class IntermediateStats:
    """Container for statistics accumulated during processing."""

    global_min: float = float("inf")
    global_max: float = float("-inf")

    def update(self, array):
        self.global_min = min(self.global_min, float(array.min()))
        self.global_max = max(self.global_max, float(array.max()))


def denoise_image(
    image: np.ndarray,
    careamist: CAREamist,
    tile_size: tuple[int, ...],
    tile_overlap: tuple[int, ...],
    batch_size: int,
    num_workers: int,
    rescale: bool = False,
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
        If True, rescale denoised output to uint8 [0, 255] using min/max.
        Default is False.

    Returns
    -------
    np.ndarray
        Denoised image. If rescale=True, returns uint8, otherwise float32.
    """
    with suppress_logging():
        denoised = careamist.predict(
            source=image,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            axes="YX",
            batch_size=batch_size,
            dataloader_params={"num_workers": num_workers},
        )[0]

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
    Global rescaling uses two-pass processing with a temporary zarr. Per-slice
    rescaling processes each slice independently without intermediate storage.

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
        Number of tiles to process in parallel on the GPU.
        Default is 64.
    num_workers : int
        Number of dataloader workers for tile loading.
        Default is 16.
    temp_dir : Path | str | None
        Directory for intermediate float32 zarr storage. Only used when
        `rescale_mode='global'`. Should be on fast local storage (SSD).
        Default is Path("data/tmp").
    rescale_mode : Literal["per_slice", "global"]
        If `'global'`, use global min/max across all slices for rescaling (two-pass).
        If `'per_slice'`, rescale each slice independently using per-slice min/max.
        Default is True.

    Notes
    -----
    When `rescale_mode='global'`, the intermediate zarr is uncompressed and
    can be large (4 bytes per voxel). Ensure sufficient disk space in `temp_dir`.
    """
    torch.set_float32_matmul_precision("high")

    root = zarr.open_group(root_path, mode="a")
    src_zarr = root.get(src_path)

    config = DenoisingConfig.from_json(model_root / f"{model_name}/config.json")
    careamist = _load_model(model_name=model_name, model_root=model_root)

    dst_path = f"{dst_group}/{dirname_from_spacing(src_zarr.attrs['spacing'])}"

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
        )
    else:
        raise ValueError(
            f"Invalid value {rescale_mode} for rescale_mode. "
            "Valid options are 'global' and 'per_slice'."
        )


def _denoise_global_rescale(
    src_zarr,
    careamist,
    root,
    dst_path: str,
    config,
    tile_size: tuple[int, int],
    tile_overlap: tuple[int, int],
    batch_size: int,
    num_workers: int,
    temp_dir: Path | str | None,
) -> None:
    """Two-pass denoising with global rescaling."""
    stats = IntermediateStats()

    with temporary_zarr(
        shape=src_zarr.shape,
        chunks=(1, *src_zarr.shape[1:]),
        dtype=np.float32,
        dir=temp_dir,
    ) as intermediate:
        # Pass 1: Denoise and accumulate stats
        for z in tqdm(range(src_zarr.shape[0]), desc="Denoising"):
            denoised = denoise_image(
                src_zarr[z],
                careamist,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            stats.update(denoised)
            intermediate[z] = denoised

        # Pass 2: Rescale and cast to uint8
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

        processing = [
            ProcessingStep.from_config("denosing-train", config),
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
        ]

        write_zarr(
            root=root,
            array=rescaled,
            dst_path=dst_path,
            src_zarr=src_zarr,
            dtype=np.uint8,
            processing=processing,
        )


def _denoise_per_slice_rescale(
    src_zarr,
    careamist,
    root,
    dst_path: str,
    config,
    tile_size: tuple[int, int],
    tile_overlap: tuple[int, int],
    batch_size: int,
    num_workers: int,
) -> None:
    """Single-pass denoising with per-slice rescaling."""

    # Create dask array from source
    src_dask = da.from_zarr(src_zarr)

    # Apply denoising and rescaling per slice along Z axis
    rescaled = da.map_blocks(
        lambda img: denoise_image(
            img,
            careamist,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            batch_size=batch_size,
            num_workers=num_workers,
            rescale=True,
        ),
        src_dask,
        chunks=(1, *src_zarr.shape[1:]),
        dtype=np.uint8,
    )

    processing = [
        ProcessingStep.from_config("denosing-train", config),
        ProcessingStep.manual(
            "denoising-predict",
            {
                "tile_size": tile_size,
                "tile_overlap": tile_overlap,
                "rescale_mode": "per_slice",
            },
        ),
    ]

    write_zarr(
        root=root,
        array=rescaled,
        dst_path=dst_path,
        src_zarr=src_zarr,
        dtype=np.uint8,
        processing=processing,
    )
