"""
This module contains functions and classes used for segmentation
"""

import os
import re
from dataclasses import dataclass, field, asdict
import logging
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
from cellpose import models, train, io, metrics, dynamics
import torch
import numpy as np
import pandas as pd
import zarr
from sphero_vem.io import read_tensor, write_image, write_zarr
from sphero_vem.utils import (
    get_file_info,
    read_manifest,
    timestamp,
    vprint,
    dirname_from_spacing,
)
from sphero_vem.utils.accelerator import xp, ndi, ArrayLike, gpu_dispatch
from sphero_vem.postprocessing import decompose_flow, median_filter, merge_labels


@dataclass
class FinetuneConfig:
    """Configuration class for fine-tuning cellpose"""

    dir_labeled: Path | str
    downscaling: int = 1
    learning_rate: float = 5e-5
    batch_size: int = 8
    n_epochs: int = 100
    test_size: float = 0.2
    random_state: int = 42
    seg_target: str = "cells"
    save_predictions: bool = True
    use_bfloat16: bool = True
    dry_run: bool = False

    # Parameters that are initialized by post_init
    data_root: Path = field(init=False)
    model_name: str = field(init=False)
    dir_experiment: Path = field(init=False)
    dir_predictions: Path = field(init=False)
    wandb_api_key: str = field(init=False)
    downscaling_eff: int = field(init=False)
    preprocessing: list[dict] = field(init=False)

    def __post_init__(self):
        """Load environment variables and init derived values"""
        if self.seg_target == "cells":
            self.wandb_project = "cell-segmentation"
        elif self.seg_target == "nuclei":
            self.wandb_project = "nuclei-segmentation"
        self.wandb_api_key = os.getenv("API_KEY")
        self.data_root = Path("data")

        ds_text = f"-ds{self.downscaling}" if self.downscaling else ""
        self.model_name = f"cellposeSAM-{self.seg_target}{ds_text}-{timestamp()}"
        self.dir_labeled = self.dir_labeled
        self.dir_experiment = Path(f"data/models/cellpose/{self.model_name}")
        if not self.dry_run:
            self.dir_experiment.mkdir(parents=True, exist_ok=True)

        if self.save_predictions:
            self.dir_predictions = Path(
                f"data/processed/segmented/finetuning/{self.model_name}"
            )
            if not self.dry_run:
                self.dir_predictions.mkdir(parents=True, exist_ok=True)

        self.preprocessing = read_manifest(self.dir_labeled).get("processing", [])
        self.downscaling_eff = self.calculate_downscaling()

    def calculate_downscaling(self) -> int:
        """Calcualte effective donwscaling to apply to images"""
        downscaling_old = 1
        try:
            for processing in self.preprocessing:
                if processing.get("step") == "downscaling":
                    downscaling_old = processing["factor"]
        # Account for manifest not conforming to standard representation
        except AttributeError:
            pass
        if self.downscaling % downscaling_old == 0:
            return self.downscaling // downscaling_old
        else:
            raise ValueError(
                f"Supplied global downscaling {self.downscaling} is incompatible with "
                f"labeled dataset already downscaled by factor {downscaling_old}"
            )


def _generate_training_manifest(
    config: FinetuneConfig,
    train_files: list[Path],
    test_files: list[Path],
):
    """Generate training manifest"""
    training_manifest = {
        "experiment_id": config.model_name,
        "timestamp": datetime.now().isoformat(),
        "segmentation_target": config.seg_target,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "preprocessing_training": [],
        "preprocessing_labels": config.preprocessing,
        "train_files": [],
        "test_files": [],
    }

    for filepath in tqdm(train_files, "Generating train data hashes"):
        file_info = get_file_info(filepath, config.data_root)
        training_manifest["train_files"].append(file_info)

    for filepath in tqdm(test_files, "Generating test data hashes"):
        file_info = get_file_info(filepath, config.data_root)
        training_manifest["test_files"].append(file_info)

    # For now keep it manual, consider automating step recognition.
    if config.downscaling:
        preprocessing = [
            {
                "step": "downscaling",
                "factor": config.downscaling,
                "factor_eff": config.downscaling_eff,
                "normalization": None,
            }
        ]

        for preprocessing_step in preprocessing:
            training_manifest["preprocessing_training"].append(preprocessing_step)

    # Save manifest locally and send to WandB
    manifest_path = config.dir_experiment / "training_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(training_manifest, f, indent=4)
    wandb.save(manifest_path)


class _CellposeLogHandler(logging.Handler):
    """Class that captures cellpose logger and sends info to WandB"""

    def __init__(self):
        super().__init__()

    def emit(self, record):
        message = record.getMessage()
        pattern = r"(\d+), train_loss=([\d\.]+), test_loss=([\d\.]+), LR=([\d\.e\-\+]+)"
        match = re.search(pattern, message)

        if match:
            epoch = int(match.group(1))
            train_loss = float(match.group(2))
            test_loss = float(match.group(3))
            learning_rate = float(match.group(4))

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "test_loss": test_loss,
                    "learning_rate": learning_rate,
                }
            )


class CellposeLogger:
    def __init__(self, config: FinetuneConfig) -> None:
        # Activate Cellpose logging
        io.logger_setup()
        self._init_wandb(config)

        # Add WandB handler to the cellpose logger
        self.wandb_handler = _CellposeLogHandler()
        self.cellpose_logger = logging.getLogger("cellpose.train")
        self.cellpose_logger.addHandler(self.wandb_handler)

    def _init_wandb(self, config: FinetuneConfig) -> None:
        """Initialize WandB logging"""
        wandb.login(key=config.wandb_api_key)

        wandb.init(
            project=config.wandb_project,
            name=config.model_name,
            dir=config.dir_experiment,
        )
        wandb.config.update(
            {
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "downscaling": config.downscaling,
                "n_epochs": config.n_epochs,
                "use_bfloat16": config.use_bfloat16,
            }
        )

    def stop(self) -> None:
        """Stop logging and cleanup"""
        self.cellpose_logger.removeHandler(self.wandb_handler)
        wandb.finish()

    def save_losses(self, train_losses: list[float], test_losses: list[float]) -> None:
        """Log detailed epoch-by-epoch data"""
        for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss_epoch": train_loss,
                    "test_loss_epoch": test_loss if test_loss > 0 else np.nan,
                }
            )


def split_dataset(config: FinetuneConfig) -> tuple[list[Path], list[Path]]:
    """Split segmentation data into train and test datasets. This function only
    considers images that also have labels

    Labels are expected to be in a 'labels' subdirectory and have the naming
    '{image_name}-{config.seg_target}'.
    Example: with config.seg_target='cells', a valid image/labels pair is:
    - image.tif
    - labels/image-cells.tif

    """
    # Ensure an even split between different imaging planes, if present
    train_files = []
    test_files = []
    for axis in ["x", "y", "z"]:
        image_list = [
            path
            for path in config.dir_labeled.glob(f"*-{axis}_*.tif")
            if _labels_path(config, path).exists()
        ]
        if image_list != []:
            train_slices, test_slices = train_test_split(
                image_list, test_size=config.test_size, random_state=config.random_state
            )
            train_files += train_slices
            test_files += test_slices
    return train_files, test_files


def load_data(
    config: FinetuneConfig, image_files: list[Path]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load images for training/testing as a list of arrays with corresponing labels.

    Labels are expected to be in a 'labels' subdirectory and have the naming
    '{image_name}-{config.seg_target}'.
    Example: with config.seg_target='cells', a valid image/labels pair is:
    - image.tif
    - labels/image-cells.tif
    NOTE: that the downscaling factor defined for the model refers to the images in their
    original size at acquisition. When loading data, prior downscaling done on the
    train and test dataset is taken into account, and an effective downscaling is
    applied to achieve the correct global downscaling factor. Particular care must be
    used therefore when an already downscaled dataset is used, since not all factors
    will give correct images.

    Parameters
    ----------
    config : CellposeConfig
        Cellpose configuration object.
    image_files : list[Path]
        List of paths to the image files to load.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        A tuple containing two lists:
        - First list: loaded and downscaled images as numpy arrays
        - Second list: loaded and downscaled label masks as numpy arrays

    """
    data = [
        read_tensor(path, None, config.downscaling_eff).numpy() for path in image_files
    ]
    labels_files = [_labels_path(config, path) for path in image_files]
    labels = [
        read_tensor(
            path, torch.uint8, config.downscaling_eff, resample_mode="nearest"
        ).numpy()
        for path in labels_files
    ]
    return data, labels


def _labels_path(config: FinetuneConfig, image_path: Path) -> Path:
    """Generate expected label path for a given image"""
    return config.dir_labeled / f"labels/{image_path.stem}-{config.seg_target}.tif"


def finetune_cellpose(config: FinetuneConfig):
    """
    Finetune a Cellpose model using the parameters in the configuration.

    This function handles the complete fine-tuning process for a Cellpose model,
    including data splitting and logging.

    Parameters
    ----------
    config : CellposeConfig
        Configuration object containing all necessary parameters for fine-tuning.
    """

    logger = CellposeLogger(config)

    train_files, test_files = split_dataset(config)
    _generate_training_manifest(config, train_files, test_files)

    cellpose_model = models.CellposeModel(gpu=True, use_bfloat16=config.use_bfloat16)

    train_data, train_labels = load_data(config, train_files)
    test_data, test_labels = load_data(config, test_files)

    _, train_losses, test_losses = train.train_seg(
        net=cellpose_model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        model_name=config.model_name,
        save_path=config.dir_experiment,
    )

    logger.save_losses(train_losses, test_losses)
    logger.stop()

    # Save test predictions
    if config.save_predictions:
        for i, image in enumerate(test_data):
            masks = cellpose_model.eval(image)
            write_image(
                config.dir_predictions
                / f"{test_files[i].stem}-{config.seg_target}.tif",
                masks[0],
                compressed=True,
            )


def calculate_ap(
    ground_truth: np.ndarray, predictions: np.ndarray, threshold_step: float = 0.05
) -> pd.DataFrame:
    """Calculate average precision and related metrics at different IoU thresholds.

    This function evaluates segmentation predictions against ground truth by computing
    the average precision, as well as counting the number of true positives, false
    positives, and false negatives at different Jaccard index (IoU) thresholds.

    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth segmentation mask with instance labels.
        Each object should have a unique integer ID > 0.
    predictions : np.ndarray
        Predicted segmentation mask with instance labels.
        Each predicted object should have a unique integer ID > 0.
    threshold_step : float, optional
        Step size for IoU thresholds, defaults to 0.05.
        Thresholds will be generated as [threshold_step, 2*threshold_step, ..., < 1.0]

    Returns
    -------
    pd.DataFrame
        DataFrame containing evaluation metrics with columns:
        - 'iou_thresholds': IoU threshold values
        - 'average_precision': AP score at each threshold
        - 'true_positives': Number of true positive detections
        - 'false_positives': Number of false positive detections
        - 'false_negatives': Number of false negative detections
    """
    thresholds = np.arange(threshold_step, 1.0, threshold_step).round(2)
    metric = metrics.average_precision(ground_truth, predictions, threshold=thresholds)
    results_df = pd.DataFrame((thresholds, *metric)).T
    results_df.columns = [
        "iou_thresholds",
        "average_precision",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]
    return results_df


def extract_seg_target(manifest: dict) -> str | None:
    """Extract segmentation target from the training manifest"""
    # Try direct key access
    try:
        return manifest["segmentation_target"]
    # Fallback extraction from name
    except KeyError:
        match = re.search(r"cellposeSAM-(\w+)-", manifest["experiment_id"])
        if match:
            return match.group(1)

    return None


def match_predictions(ground_truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Return prediction masks with matched label indices as ground truth"""
    _, matched = metrics.mask_ious(ground_truth, predictions)
    full_range = np.unique(predictions)[1:]
    missing = np.setdiff1d(full_range, matched).tolist()
    predictions_matched = predictions.copy()
    for val in missing:
        predictions_matched[predictions_matched == val] = 2 * predictions.max() + val

    for i, val in enumerate(matched):
        predictions_matched[predictions_matched == val] = predictions.max() + i + 1

    predictions_matched[predictions_matched > 0] -= predictions.max()

    return predictions_matched


@dataclass
class SegmentationConfig:
    """Segment a volume stack using cellpose"""

    root_path: Path
    model: str
    spacing_dir: str
    out_path: Path | None = None
    verbose: bool = True
    zarr_chunks: tuple[int] | None = None
    batch_size: int = 64
    flow3D_smooth: int = 2
    augment: bool = False
    tile_overlap: float = 0.3
    median_filter_cellprob: int | None = 3
    decompose_flows: bool = False
    decompose_flows_pad_fraction: float = 0.3

    seg_target: str = field(init=False)
    model_dir: Path = field(init=False)
    spacing: list[int | float] = field(init=False)
    src_zarr: zarr.Array = field(init=False)

    def __post_init__(self):
        self.seg_target = re.search(r"cellposeSAM-(\w+)-", self.model).group(1)
        self.model_dir = Path(f"data/models/cellpose/{self.model}/models/{self.model}")
        self.src_zarr = zarr.open_array(
            self.root_path / f"images/{self.spacing_dir}", mode="r"
        )
        self.spacing = self.src_zarr.attrs.get("spacing")

        # If out_dir is not specified, save under the Zarr path of the images
        if not self.out_path:
            self.out_path = self.root_path
        else:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.zarr_chunks:
            self.zarr_chunks = self.src_zarr.chunks


def calculate_flows(config: SegmentationConfig) -> None:
    """Segment volume stack using config as input. This function calculates flows and
    cell probability, masks have to be calculated in a following step."""

    processing = [
        {
            "step": "segmentation",
            "model": config.model,
            "seg_target": config.seg_target,
            "batch_size": config.batch_size,
            "flow3D_smooth": config.flow3D_smooth,
            "augment": config.augment,
            "tile_overlap": config.tile_overlap,
            "median_filter_cellprob": config.median_filter_cellprob,
            "decompose_flows": config.decompose_flows,
            "decompose_flows_pad_fraction": config.decompose_flows_pad_fraction,
        }
    ]

    image: np.ndarray = config.src_zarr[:]
    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=config.model_dir)

    time_start = datetime.now()
    vprint(f"Starting segmentation at {time_start}", config.verbose)

    # Shape and channel settings to handle 2D and 3D images correctly
    settings_ndim = {
        2: {"z_axis": None, "channel_axis": None, "do_3D": False},
        3: {
            "z_axis": 0,
            "channel_axis": None,
            "do_3D": True,
        },
    }

    with torch.inference_mode():
        _, flows, _ = cellpose_model.eval(
            image,
            **settings_ndim[image.ndim],
            batch_size=config.batch_size,
            tile_overlap=config.tile_overlap,
            flow3D_smooth=config.flow3D_smooth,
            augment=config.augment,
            compute_masks=False,
        )

    # Ensure GPU memory is garbage collected
    cellpose_model.net.to("cpu")
    del cellpose_model
    torch.cuda.empty_cache()

    time_finish = datetime.now()
    vprint(f"Completed segmentation at {time_finish}", config.verbose)
    vprint(f"Elapsed time: {time_finish - time_start}", config.verbose)

    # Post process flows
    vprint("Starting flow postprocessing", config.verbose)

    dP = np.ascontiguousarray(flows[1]).astype(np.float16, copy=False)
    cellprob = np.ascontiguousarray(flows[2]).astype(np.float16, copy=False)

    if config.decompose_flows:
        dP = decompose_flow(
            dP, config.decompose_flows_pad_fraction, torch.device("cuda")
        )
    if config.median_filter_cellprob:
        cellprob = median_filter(cellprob, config.median_filter_cellprob)

    # Saving flows
    vprint("Saving flows", config.verbose)
    save_root = zarr.open_group(config.out_path, mode="a")

    write_zarr(
        save_root,
        cellprob,
        f"labels/{config.seg_target}/flows/cellprob/{config.spacing_dir}",
        src_zarr=config.src_zarr,
        processing=processing,
        zarr_chunks=config.zarr_chunks,
        dtype="f2",
    )

    write_zarr(
        save_root,
        dP,
        f"labels/{config.seg_target}/flows/dP/{config.spacing_dir}",
        src_zarr=config.src_zarr,
        processing=processing,
        zarr_chunks=(3, *config.zarr_chunks),
        dtype="f2",
        multichannel=True,
    )


@dataclass
class SegmentationMaskParams:
    """Parameters used to calculate cellpose masks"""

    root_path: Path
    seg_target: str
    spacing_dir: str = "100-100-100"
    niter: int = 200
    cellprob_threshold: float = -0.5
    flow_threshold: float = 0.4
    min_diam: float = 3
    merge_masks: bool = True
    gaussian_edge_sigma: float = 2.0
    merge_weight_threshold: float = 0.2
    merge_contact_threshold: float = 0.2
    device: str = "cuda"
    zarr_chunks: tuple[int] | None = None

    min_size: int = field(init=False)
    spacing: list[int | float] = field(init=False)

    def __post_init__(self):
        # Celculate min_size in pixel from min_diam in micrometers
        src_zarr = zarr.open_array(
            self.root_path / f"images/{self.spacing_dir}", mode="r"
        )
        self.spacing = src_zarr.attrs.get("spacing")

        # Determine whether min_size should be area or volume
        if len(self.spacing) == 2:
            pixel_um = np.prod(self.spacing) * 1e-6
            min_area_um = np.pi * (self.min_diam / 2) ** 2
            self.min_size = int(min_area_um / pixel_um)
        elif len(self.spacing) == 3:
            voxel_um = np.prod(self.spacing) * 1e-9
            min_vol_um = 4 / 3 * np.pi * (self.min_diam / 2) ** 3
            self.min_size = int(min_vol_um / voxel_um)

        if not self.zarr_chunks:
            self.zarr_chunks = src_zarr.chunks


def calculate_masks(config: SegmentationMaskParams):
    """Calculate segmentation masks using cellpose flows"""

    device = torch.device(config.device)

    root = zarr.open_group(config.root_path, mode="a")
    cellprob_zarr = root.get(
        f"labels/{config.seg_target}/flows/cellprob/{config.spacing_dir}"
    )
    dp_zarr = root.get(f"labels/{config.seg_target}/flows/dP/{config.spacing_dir}")

    cellprob: np.ndarray = cellprob_zarr[:]
    dP: np.ndarray = dp_zarr[:]

    do_3d = True if cellprob.ndim == 3 else False

    masks = dynamics.compute_masks(
        dP=dP,
        cellprob=cellprob,
        niter=config.niter,
        cellprob_threshold=config.cellprob_threshold,
        flow_threshold=config.flow_threshold,
        do_3D=do_3d,
        min_size=config.min_size,
        device=device,
    )

    # Post-process labels
    image_arr = root.get(f"images/{config.spacing_dir}")
    if config.merge_masks:
        image: np.ndarray = image_arr[:]
        masks, _ = merge_labels(
            masks,
            cellprob=cellprob,
            image=image,
            rel_contact_thresh=config.merge_contact_threshold,
            weight_thresh=config.merge_weight_threshold,
            sigma=config.gaussian_edge_sigma,
        )

    processing_dict = asdict(config)
    excluded_keys = ["root_path", "device", "zarr_chunks"]
    processing = cellprob_zarr.attrs.get("processing") + [
        {
            "step": "segmentation mask generation",
            **{
                key: val
                for key, val in processing_dict.items()
                if key not in excluded_keys
            },
        }
    ]

    write_zarr(
        root,
        masks,
        f"labels/{config.seg_target}/masks/{config.spacing_dir}",
        src_zarr=cellprob_zarr,
        dtype=np.uint8 if masks.max() <= 255 else np.uint16,
        inputs=[image_arr.path, cellprob_zarr.path, dp_zarr.path],
        processing=processing,
    )


@gpu_dispatch(return_to_host=True)
def _calc_seeds(
    labels_lr: ArrayLike, erosion_iterations: int, zoom_factors: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Generate high res seeds. Returns also low-res mask foreground."""

    foreground_lr = labels_lr > 0
    eroded_fg = foreground_lr
    for _ in range(erosion_iterations):
        eroded_fg = ndi.binary_erosion(eroded_fg)
    seeds_lr = labels_lr * eroded_fg
    seeds_hr = ndi.zoom(seeds_lr, zoom_factors, order=0)
    return seeds_hr


@gpu_dispatch(return_to_host=True)
def _calc_foreground(
    cellprob_hr: ArrayLike,
    cellprob_threshold: float,
) -> np.ndarray:
    """Calculate high res mask"""
    foreground_hr = cellprob_hr > cellprob_threshold
    return foreground_hr


@gpu_dispatch(return_to_host=True)
def _region_fill(
    seeds_hr: ArrayLike, foreground_hr: ArrayLike, max_expansion_steps: int = 10
) -> np.ndarray:
    """Region fill aglorithm. Runs for at most max_expansion_steps"""
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)

    # Create dilation buffer to save memory
    dilation_buffer = xp.empty_like(seeds_hr)

    for i in range(max_expansion_steps):
        ndi.grey_dilation(seeds_hr, footprint=structure, output=dilation_buffer)
        should_fill = (seeds_hr == 0) & foreground_hr & (dilation_buffer > 0)
        if not xp.any(should_fill):
            break
        seeds_hr[should_fill] = dilation_buffer[should_fill]

    # Trim any accidental overflows
    seeds_hr *= foreground_hr
    return seeds_hr


def upsample_region_fill(
    labels_lr: np.ndarray,
    cellprob_hr: np.ndarray,
    erosion_iterations: int = 2,
    cellprob_threshold: float = 0.0,
    max_expansion_steps: int = 10,
) -> np.ndarray:
    """
    Upsample cellpose labels with a region fill algorithm using thresholded cellprob
    logits.
    """

    zoom_factors = np.array(cellprob_hr.shape) / np.array(labels_lr.shape)

    seeds_hr = _calc_seeds(labels_lr, erosion_iterations, zoom_factors)
    foregroung_hr = _calc_foreground(cellprob_hr, cellprob_threshold)
    labels_hr = _region_fill(
        seeds_hr, foregroung_hr, max_expansion_steps=max_expansion_steps
    )

    return labels_hr.astype(labels_lr.dtype)


def upsample_masks(
    root_path: Path,
    seg_target: str,
    target_spacing: tuple[int, float],
    src_spacing: tuple[int, int, int] = (100, 100, 100),
    erosion_iterations: int = 2,
    cellprob_threshold: float = 0.0,
    store_chunks: tuple[int] | None = None,
) -> None:
    """Upsample cellpose labels"""

    root = zarr.open_group(root_path, mode="a")
    labels_lr_zarr: zarr.Array = root.get(
        f"labels/{seg_target}/masks/{dirname_from_spacing(src_spacing)}"
    )
    cellprob_hr_zarr: zarr.Array = root.get(
        f"labels/{seg_target}/flows/cellprob/{dirname_from_spacing(target_spacing)}"
    )

    labels_lr: np.ndarray = labels_lr_zarr[:]
    cellprob_hr: np.ndarray = cellprob_hr_zarr[:]
    labels_hr = upsample_region_fill(
        labels_lr,
        cellprob_hr,
        erosion_iterations=erosion_iterations,
        cellprob_threshold=cellprob_threshold,
    )

    processing = labels_lr_zarr.attrs.get("processing") + [
        {
            "step": "upsample masks",
            "erosion_iterations": erosion_iterations,
            "cellprob_threshold": cellprob_threshold,
        }
    ]

    write_zarr(
        root,
        labels_hr,
        f"labels/{seg_target}/masks/{dirname_from_spacing(target_spacing)}",
        src_zarr=labels_lr_zarr,
        spacing=target_spacing,
        processing=processing,
        zarr_chunks=store_chunks if store_chunks else labels_lr_zarr.chunks,
        inputs=[labels_lr_zarr.path, cellprob_hr_zarr.path],
    )
