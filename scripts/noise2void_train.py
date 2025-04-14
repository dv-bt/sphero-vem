from pathlib import Path
import shutil
from tqdm import tqdm
import torch
from careamics import CAREamist
from careamics.config import create_n2v_configuration

# Set lower matmul precision to use tensor cores
torch.set_float32_matmul_precision("high")

work_dir = Path("../data/models/n2v")
work_dir.mkdir(exist_ok=True, parents=True)

config = create_n2v_configuration(
    experiment_name="n2v_2D",
    data_type="tiff",
    axes="YX",
    patch_size=[128, 128],
    batch_size=32,
    num_epochs=20,
    logger="tensorboard",
)

careamist = CAREamist(config, work_dir=work_dir)

# Move 100 images to a separate folder for training.
# It seems that there's no way to select a subset of images when loading
# data, and that I cannot pass a custom dataloader.
data_path = Path("../data/raw/Au_01-vol_01")
train_path = data_path / "n2v_training"
train_path.mkdir(exist_ok=True)

file_list = sorted(list(data_path.glob("*.tif")))
for file in tqdm(file_list[:50], "Moving images to training directory..."):
    file_dst = train_path / file.name
    if not file_dst.exists():
        shutil.copy(file, train_path / file.name)

careamist.train(train_source=train_path, use_in_memory=False, val_minimum_split=5)
