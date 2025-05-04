from pathlib import Path
import shutil
from tqdm import tqdm
import torch
from careamics import CAREamist
from careamics.config import create_n2v_configuration


# Script variables
# Dataset
DATA_PATH = Path("data/raw/Au_01-vol_01")
NUM_IMAGES = 10
TRAIN_DIR_NAME = "n2v_training"
VAL_SPLIT = 0.2

# Hyperparameters
BATCH_SIZE = 128
PATCH_SIZE = 128
EPOCHS = 100
UNET_DEPTH = 3
N2V2 = True

# Dataloader parameters
NUM_WORKERS = 8

# Experiment
EXP_NAME = f"n2v-depth{UNET_DEPTH}-patch{PATCH_SIZE}-nimages{NUM_IMAGES}{'-n2v2' if N2V2 else ''}"
WORK_DIR = Path("data/models/n2v") / EXP_NAME


def configure_n2v(train_path) -> CAREamist:
    """Create careamist object"""

    WORK_DIR.mkdir(exist_ok=True, parents=True)

    config = create_n2v_configuration(
        experiment_name=EXP_NAME,
        data_type="tiff",
        axes="YX",
        patch_size=[PATCH_SIZE, PATCH_SIZE],
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        use_n2v2=N2V2,
        logger="wandb",
        model_params={
            "depth": UNET_DEPTH,
        },
        train_dataloader_params={"num_workers": NUM_WORKERS},
        val_dataloader_params={"num_workers": NUM_WORKERS},
    )

    careamist = CAREamist(
        config,
        work_dir=WORK_DIR,
    )

    return careamist


def prepare_dataset(data_path: Path, num_images: int, train_dir_name: str) -> Path:
    """
    Move first N images to a separate folder for training.
    This removes the folder if it already exists.
    It seems that there's no way to select a subset of images when loading
    data, and that I cannot pass a custom dataloader.
    """
    train_path = data_path / train_dir_name
    if train_path.exists():
        shutil.rmtree(train_path)
    train_path.mkdir()

    file_list = sorted(list(data_path.glob("*.tif")))
    for file in tqdm(file_list[:num_images], "Moving images to training directory..."):
        shutil.copy(file, train_path / file.name)

    return train_path


def train_n2v():
    train_path = prepare_dataset(DATA_PATH, NUM_IMAGES, TRAIN_DIR_NAME)
    careamist = configure_n2v(train_path)
    careamist.train(
        train_source=train_path, use_in_memory=False, val_percentage=VAL_SPLIT
    )


if __name__ == "__main__":
    # Set lower matmul precision to use tensor cores
    torch.set_float32_matmul_precision("high")
    train_n2v()
