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
from cellpose import models, train, io, metrics
import torch
import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from sphero_vem.io import read_tensor, write_image, read_stack
from sphero_vem.utils import (
    get_file_info,
    read_manifest,
    timestamp,
    generate_manifest,
    vprint,
)


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
class SegmentationParams:
    """Parameters passed to the segmentation function"""

    batch_size: int = 64
    flow3D_smooth: int = 0
    cellprob_threshold: float = 0.0
    min_size: int = 100
    tile_overlap: float = 0.1
    niter: int = 200
    flow_threshold: float = 0.4
    augment: bool = False
    anisotropy: float | None = None


@dataclass
class SegmentationConfig:
    """Segment a volume stack using cellpose"""

    data_dir: Path
    model: str
    seg_params: SegmentationParams
    verbose: bool = True
    compute_stats: bool = False
    out_dir: Path | None = None
    mode: str = "inference"
    save_flows: bool = True
    zarr_chunks: tuple[int, int, int] = (128, 512, 512)
    spacing: tuple[int, int, int] = (100, 100, 100)

    dataset: str = field(init=False)
    seg_target: str = field(init=False)
    model_dir: Path = field(init=False)
    out_path: Path = field(init=False)
    flows_path: Path = field(init=False)

    def __post_init__(self):
        self.dataset = re.search(r"(Au_\d+-vol_\d+)", str(self.data_dir)).group(1)
        self.seg_target = re.search(r"cellposeSAM-(\w+)-", self.model).group(1)
        self.model_dir = Path(f"data/models/cellpose/{self.model}/models/{self.model}")
        if not self.out_dir:
            if self.mode == "inference":
                self.out_dir = Path(
                    f"data/processed/segmented/{self.dataset}/{self.seg_target}"
                )
            elif self.mode == "param_optim":
                self.out_dir = Path(
                    "data/processed/segmented/optimization_3d/"
                    f"{self.seg_target}-run{timestamp()}"
                )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.out_path = self.out_dir / f"{self.dataset}-{self.seg_target}.tif"
        self.flows_path = self.out_dir / f"{self.dataset}-{self.seg_target}-flows.zarr"


def segment_stack(config: SegmentationConfig) -> None:
    """Segment volume stack using config as input"""
    seg_params = asdict(config.seg_params)

    processing = [
        {
            "step": "segmentation",
            "model": config.model,
            "seg_target": config.seg_target,
            **seg_params,
        }
    ]

    volume_stack = read_stack(config.data_dir, verbose=config.verbose)
    cellpose_model = models.CellposeModel(gpu=True, pretrained_model=config.model_dir)

    time_start = datetime.now()
    vprint(f"Starting segmentation at {time_start}", config.verbose)

    with torch.inference_mode():
        masks, flows, _ = cellpose_model.eval(
            volume_stack,
            do_3D=True,
            channel_axis=1,
            z_axis=0,
            **seg_params,
        )

    # Compute masks statistics
    if config.compute_stats:
        ninst = int(masks.max())
        volumes = np.bincount(masks.ravel())[1:] if ninst else np.array([np.nan])
        # Equivalent diameter in micrometers, assuming 100 nm pixel size
        diams = 2 * (volumes * 3 / (4 * np.pi)) ** (1 / 3) * 0.1
        np.savez(config.out_dir / "masks_stats.npz", volume=volumes, diameter=diams)

    # Ensure GPU memory is garbage collected
    cellpose_model.net.to("cpu")
    del cellpose_model
    torch.cuda.empty_cache()

    time_finish = datetime.now()
    vprint(f"Completed segmentation at {time_finish}", config.verbose)
    vprint(f"Elapsed time: {time_finish - time_start}", config.verbose)

    vprint("Saving segmentation mask", config.verbose)
    write_image(config.out_path, masks, compressed=True)

    if config.save_flows:
        vprint("Saving flows", config.verbose)
        dP = np.ascontiguousarray(flows[1]).astype(np.float16, copy=False)
        cellprob = np.ascontiguousarray(flows[2]).astype(np.float16, copy=False)

        compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
        zarr_root = zarr.open(config.flows_path, mode="w")
        cellprob_arr = zarr_root.create_array(
            "cellprob",
            shape=(1, *cellprob.shape),
            chunks=(1, *config.zarr_chunks),
            dtype="f2",
            compressors=compressor,
        )
        dp_arr = zarr_root.create_array(
            "dP",
            shape=dP.shape,
            chunks=(3, *config.zarr_chunks),
            dtype="f2",
            compressors=compressor,
        )

        cellprob_arr[...] = cellprob[None, ...]
        dp_arr[...] = dP

        # Update zarr metadata
        manifest = read_manifest(config.data_dir)
        for arr in zarr_root.arrays():
            arr.attrs["spacing"] = config.spacing
            arr.attrs["processing"] = manifest["processing"] + processing
            arr.attrs["inputs"] = sorted(config.data_dir.glob("*.tif"))

    generate_manifest(
        config.dataset,
        config.out_dir,
        sorted(config.data_dir.glob("*.tif")),
        processing,
    )
