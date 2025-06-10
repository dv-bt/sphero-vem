"""
This module contains functions and classes used for segmentation
"""

import os
import re
from dataclasses import dataclass
import logging
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
from cellpose import models, train, io
from tifffile import imwrite
import numpy as np
from sphero_vem.io import imread_downscaled, imread_labels_downscaled
from sphero_vem.utils import get_file_info


@dataclass
class CellposeConfig:
    """Configuration class for fine-tuning cellpose"""

    dir_labeled: Path | str
    downscaling: int = 16
    learning_rate: float = 5e-5
    batch_size: int = 128
    n_epochs: int = 100
    test_size: float = 0.2
    random_state: int = 42
    seg_target: str = "cells"
    save_predictions: bool = True

    # Parameters that are initialized by post_init
    model_name: str = ""
    data_root: Path = Path()
    dir_experiment: Path = Path()
    dir_predictions: Path = Path()
    wandb_api_key: str = ""

    def __post_init__(self):
        """Load environment variables and init derived values"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.seg_target == "cells":
            self.wandb_project = "cell-segmentation"
        elif self.seg_target == "nuclei":
            self.wandb_project = "nuclei-segmentation"
        self.wandb_api_key = os.getenv("API_KEY")
        self.data_root = Path(os.getenv("DATA_ROOT"))

        self.model_name = (
            f"cellposeSAM-{self.seg_target}-ds{self.downscaling}-{timestamp}"
        )
        self.dir_labeled = self.data_root / self.dir_labeled
        self.dir_experiment = self.data_root / "models/cellpose" / self.model_name
        self.dir_experiment.mkdir(parents=True, exist_ok=True)

        if self.save_predictions:
            self.dir_predictions = (
                self.data_root / "processed/segmented/finetuning" / self.model_name
            )
            self.dir_predictions.mkdir(parents=True, exist_ok=True)


def _generate_manifest(
    config: CellposeConfig,
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
        "preprocessing_steps": [],
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
    preprocessing = [
        {
            "step": "downscaling",
            "factor": config.downscaling,
            "normalization": None,
        }
    ]

    for preprocessing_step in preprocessing:
        training_manifest["preprocessing_steps"].append(preprocessing_step)

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
    def __init__(self, config: CellposeConfig) -> None:
        # Activate Cellpose logging
        io.logger_setup()
        self._init_wandb(config)

        # Add WandB handler to the cellpose logger
        self.wandb_handler = _CellposeLogHandler()
        self.cellpose_logger = logging.getLogger("cellpose.train")
        self.cellpose_logger.addHandler(self.wandb_handler)

    def _init_wandb(self, config: CellposeConfig) -> None:
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


def split_dataset(config: CellposeConfig) -> tuple[list[Path], list[Path]]:
    """Split segmentation data into train and test datasets. This function only
    considers images that also have labels

    Labels are expected to be in a 'labels' subdirectory and have the naming
    '{image_name}-{config.seg_target}'.
    Example: with config.seg_target='cells', a valid image/labels pair is:
    - image.tif
    - labels/image-cells.tif

    """
    image_list = [
        path
        for path in config.dir_labeled.glob("*.tif")
        if _labels_path(config, path).exists()
    ]
    train_files, test_files = train_test_split(
        image_list, test_size=config.test_size, random_state=config.random_state
    )
    return train_files, test_files


def load_data(
    config: CellposeConfig, image_files: list[Path]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load images for training/testing as a list of arrays with corresponing labels.

    Labels are expected to be in a 'labels' subdirectory and have the naming
    '{image_name}-{config.seg_target}'.
    Example: with config.seg_target='cells', a valid image/labels pair is:
    - image.tif
    - labels/image-cells.tif

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
    data = [imread_downscaled(path, config.downscaling) for path in image_files]
    labels_files = [_labels_path(config, path) for path in image_files]
    labels = [
        imread_labels_downscaled(path, config.downscaling) for path in labels_files
    ]
    return data, labels


def _labels_path(config: CellposeConfig, image_path: Path) -> Path:
    """Generate expected label path for a given image"""
    return config.dir_labeled / f"labels/{image_path.stem}-{config.seg_target}.tif"


def finetune_cellpose(config: CellposeConfig):
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
    _generate_manifest(config, train_files, test_files)

    cellpose_model = models.CellposeModel(gpu=True)

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
            imwrite(
                config.dir_predictions
                / f"{test_files[i].stem}-{config.seg_target}.tif",
                masks[0],
                compression="deflate",
                compressionargs={"level": 6},
                predictor=2,
                tile=(256, 256),
            )
