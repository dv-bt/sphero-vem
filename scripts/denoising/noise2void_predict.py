"""
Predict denoised images using Noise2Void
"""

import os
from tqdm import tqdm
from pathlib import Path
import tifffile
import numpy as np
from careamics import CAREamist
import torch
from dotenv import load_dotenv

# Common variables
load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))
MODEL_NAME = "n2v-depth3-patch128-nimages10"
DATASET = "Au_01-vol_01"

# Set lower matmul precision to use tensor cores
torch.set_float32_matmul_precision("high")


# Load model and remove loggers, not useful for inference
def load_model(model_name: str) -> None:
    """Load N2V model with the specified name"""
    model_path = DATA_ROOT / f"models/n2v/{model_name}/checkpoints/{model_name}-v2.ckpt"
    careamist = CAREamist(model_path)
    trainer = careamist.trainer
    trainer.loggers = []
    trainer.logger = None
    return careamist


def predict_denoised(careamist: CAREamist, image_raw: np.ndarray) -> np.ndarray:
    """Predict denoised image, returned as rescaled unsigned integer"""
    image_raw = image_raw.astype("float")
    image_denoised: np.ndarray = careamist.predict(
        source=image_raw,
        data_type="array",
        tile_size=[128, 128],
        tile_overlap=[64, 64],
        batch_size=128,
        dataloader_params={"num_workers": 16},
    )[0].squeeze()
    # Rescale image output and convert to uint8
    image_denoised = (
        (image_denoised - image_denoised.min())
        / (image_denoised.max() - image_denoised.min())
        * 255
    )
    image_denoised = image_denoised.astype(np.uint8)
    return image_denoised


# Inference
def main() -> None:
    """Denoise all images in the specified dataset using the specified model"""
    careamist = load_model(MODEL_NAME)
    list_path_images = sorted(list((DATA_ROOT / f"raw/{DATASET}").glob("*.tif")))
    for path_image_raw in tqdm(
        list_path_images, desc="Denoising images", position=1, leave=True
    ):
        image_raw = tifffile.imread(path_image_raw)
        image_denoised = predict_denoised(careamist, image_raw)
        # Save images
        dir_denoised = DATA_ROOT / f"processed/denoised/{DATASET}/{MODEL_NAME}"
        dir_denoised.mkdir(parents=True, exist_ok=True)
        path_denoised = dir_denoised / path_image_raw.name
        tifffile.imwrite(path_denoised, image_denoised)


if __name__ == "__main__":
    main()
