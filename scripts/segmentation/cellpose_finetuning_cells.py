"""
Script to finetune cell segmentation using CellposeSAM on our dataset
"""

import os
import hashlib
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
import wandb
import torch
from sphero_vem.preprocessing import imread_downscaled, imread_labels_downscaled
from dotenv import load_dotenv
from cellpose.models import CellposeModel
from cellpose.train import train_seg

load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))
DIR_LABELED = DATA_ROOT / "processed/labeled/Au_01-vol_01/labeled-01"

DOWNSCALING = 20
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_NAME = f"cellposeSAM-ds{DOWNSCALING}-{timestamp}"
DIR_EXPERIMENT = DATA_ROOT / f"models/cellpose/{MODEL_NAME}"
DIR_EXPERIMENT.mkdir(parents=True, exist_ok=False)

# Training parameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 128

# Other variables
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_EPOCHS = 100


def get_file_info(filepath: Path) -> dict:
    """Get file metadata without uploading the file"""
    stat = os.stat(filepath)

    # Calculate hash for file integrity (optional but recommended)
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return {
        "path": str(filepath.relative_to(DATA_ROOT)),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "md5_hash": hash_md5.hexdigest(),
    }


def generate_manifest(
    train_files: list[Path], test_files: list[Path], preprocessing: list[dict]
):
    training_manifest = {
        "experiment_id": f"{MODEL_NAME}",
        "timestamp": datetime.now().isoformat(),
        "preprocessing_steps": [],
        "train_files": [],
        "test_files": [],
    }

    for filepath in train_files:
        file_info = get_file_info(filepath)
        training_manifest["train_files"].append(file_info)

    for filepath in test_files:
        file_info = get_file_info(filepath)
        training_manifest["test_files"].append(file_info)

    for preprocessing_step in preprocessing:
        training_manifest["preprocessing_steps"].append(preprocessing_step)

    # Save manifest locally
    manifest_path = DIR_EXPERIMENT / f"training_manifest_{MODEL_NAME}.json"
    with open(manifest_path, "w") as f:
        json.dump(training_manifest, f, indent=4)

    return manifest_path


class Logger:
    """WandB logger class"""

    def __init__(self, log_frequency=10):
        self.step_count = 0
        self.log_frequency = log_frequency
        self.hooks = []

    def register_hooks(self, model):
        """Register forward hooks on the model's PyTorch nn.Module"""

        def forward_hook(module, input, output):
            if module.training and self.step_count % self.log_frequency == 0:
                # Try to extract loss from output
                loss = None
                if hasattr(output, "loss"):
                    loss = output.loss
                elif isinstance(output, dict) and "loss" in output:
                    loss = output["loss"]
                elif (
                    isinstance(output, torch.Tensor) and output.dim() == 0
                ):  # scalar tensor
                    loss = output
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    # Sometimes loss is first element in tuple/list
                    first = output[0]
                    if isinstance(first, torch.Tensor) and first.dim() == 0:
                        loss = first

                if loss is not None and hasattr(loss, "item"):
                    wandb.log({"batch_loss": loss.item(), "step": self.step_count})

            self.step_count += 1

        # Register hook on the PyTorch module
        hook = model.register_forward_hook(forward_hook)
        self.hooks.append(hook)
        return hook

    def cleanup(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()


def main():
    image_list = [
        path
        for path in DIR_LABELED.glob("*.tif")
        if (DIR_LABELED / f"labels/{path.stem}-cells.tif").exists()
    ]
    train_files, test_files = train_test_split(
        image_list, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    preprocessing = [
        {
            "step": "downscaling",
            "factor": DOWNSCALING,
            "normalization": None,
        }
    ]
    manifest_path = generate_manifest(train_files, test_files, preprocessing)

    wandb_api_key = os.getenv("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    wandb.init(project="cell-segmentation", name=MODEL_NAME, dir=DIR_EXPERIMENT)
    wandb.config.update({"learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE})
    wandb.save(manifest_path)

    cellpose_model = CellposeModel(gpu=True)
    logger = Logger(log_frequency=10)
    logger.register_hooks(cellpose_model.net)

    train_data = [imread_downscaled(path, DOWNSCALING) for path in train_files]
    path_labels_train = [
        (DIR_LABELED / f"labels/{path.stem}-cells.tif") for path in train_files
    ]
    train_labels = [
        imread_labels_downscaled(path, DOWNSCALING) for path in path_labels_train
    ]

    test_data = [imread_downscaled(path, DOWNSCALING) for path in test_files]
    path_labels_test = [
        (DIR_LABELED / f"labels/{path.stem}-cells.tif") for path in test_files
    ]
    test_labels = [
        imread_labels_downscaled(path, DOWNSCALING) for path in path_labels_test
    ]

    checkpoint_path, train_losses, test_losses = train_seg(
        net=cellpose_model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=test_data,
        test_labels=test_labels,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        model_name=MODEL_NAME,
        save_path=DIR_EXPERIMENT,
    )

    # Log epoch-level metrics
    for epoch, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
        wandb.log(
            {
                "epoch": epoch,
                "epoch_train_loss": train_loss,
                "epoch_test_loss": test_loss,
            }
        )

    # Log the final checkpoint as an artifact
    wandb.save(checkpoint_path)

    # Log final metrics
    wandb.log(
        {
            "final_train_loss": train_losses[-1],
            "final_test_loss": test_losses[-1],
            "total_epochs": len(train_losses),
        }
    )

    logger.cleanup()
    wandb.finish()


if __name__ == "__main__":
    main()
