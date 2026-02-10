"""
This module contains functions used to finetune cellpose models
"""

import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
import re
import json
import logging
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from cellpose import models, train, io
from tifffile import imread
from sphero_vem.utils import timestamp
from sphero_vem.io import write_image
from sphero_vem.utils import BaseConfig


@dataclass
class CellposeFinetuneConfig(BaseConfig):
    """Configuration class for fine-tuning cellpose"""

    dir_labeled: Path | str
    learning_rate: float = 5e-5
    batch_size: int = 8
    n_epochs: int = 100
    test_size: float = 0.2
    random_state: int = 42
    seg_target: str = "cells"
    save_predictions: bool = False
    use_bfloat16: bool = True

    # Parameters that are initialized by post_init
    model_name: str = field(init=False)
    dir_experiment: Path = field(init=False)
    dir_predictions: Path = field(init=False)
    spacing: list = field(init=False)

    def __post_init__(self):
        """Load environment variables and init derived values"""
        if self.seg_target == "cells":
            self.wandb_project = "cell-segmentation"
        elif self.seg_target == "nuclei":
            self.wandb_project = "nuclei-segmentation"

        self.model_name = f"cellposeSAM-{self.seg_target}-{timestamp()}"
        self.dir_experiment = Path(f"data/models/cellpose/{self.model_name}")
        if not self.dry_run:
            self.dir_experiment.mkdir(parents=True, exist_ok=True)

        if self.save_predictions:
            self.dir_predictions = Path(
                f"data/processed/segmented/finetuning/{self.model_name}"
            )

        # Load processing and spacing from manifest
        with open(self.dir_labeled / "manifest.json") as file:
            manifest = json.load(file)
            self.spacing = manifest.get("spacing")


def _generate_training_manifest(
    config: CellposeFinetuneConfig,
    train_files: list[Path],
    test_files: list[Path],
):
    """Generate training manifest"""

    # Load processing
    with open(config.dir_labeled / "manifest.json") as file:
        manifest = json.load(file)
        processing = manifest.get("processing", [])

    training_manifest = {
        "experiment_id": config.model_name,
        "timestamp": timestamp(),
        "segmentation_target": config.seg_target,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "n_epochs": config.n_epochs,
        "processing": processing,
        "spacing": config.spacing,
        "train_files": [str(path) for path in train_files],
        "test_files": [str(path) for path in test_files],
    }

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
    def __init__(self, config: CellposeFinetuneConfig) -> None:
        # Activate Cellpose logging
        io.logger_setup()
        self._init_wandb(config)

        # Add WandB handler to the cellpose logger
        self.wandb_handler = _CellposeLogHandler()
        self.cellpose_logger = logging.getLogger("cellpose.train")
        self.cellpose_logger.addHandler(self.wandb_handler)

    def _init_wandb(self, config: CellposeFinetuneConfig) -> None:
        """Initialize WandB logging"""
        wandb_api_key = os.getenv("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)

        wandb.init(
            project=config.wandb_project,
            name=config.model_name,
            dir=config.dir_experiment,
        )
        wandb.config.update(asdict(config))

        # Save config to dir and upload to wandb
        config_path = config.dir_experiment / "config.json"
        config.to_json(config_path)
        wandb.save(config_path)

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


def _split_dataset(config: CellposeFinetuneConfig) -> tuple[list[Path], list[Path]]:
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


def _load_data(
    config: CellposeFinetuneConfig, image_files: list[Path]
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
        - First list: loaded images as numpy arrays
        - Second list: loaded label masks as numpy arrays

    """
    data = [imread(path) for path in image_files]
    labels_files = [_labels_path(config, path) for path in image_files]
    labels = [imread(path) for path in labels_files]
    return data, labels


def _labels_path(config: CellposeFinetuneConfig, image_path: Path) -> Path:
    """Generate expected label path for a given image"""
    return config.dir_labeled / f"labels/{image_path.stem}-{config.seg_target}.tif"


def finetune_cellpose(config: CellposeFinetuneConfig):
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

    train_files, test_files = _split_dataset(config)
    _generate_training_manifest(config, train_files, test_files)

    cellpose_model = models.CellposeModel(gpu=True, use_bfloat16=config.use_bfloat16)

    train_data, train_labels = _load_data(config, train_files)
    test_data, test_labels = _load_data(config, test_files)

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
