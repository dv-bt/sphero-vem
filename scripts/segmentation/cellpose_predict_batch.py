"""
Predict segmentation masks in batch on test data, used for checking finetuning performance.
NOTE: prediction is done at the same downscale level as training.
"""

import json
from pathlib import Path
import os
from dotenv import load_dotenv
from cellpose.models import CellposeModel
from sphero_vem.preprocessing import imread_downscaled
from tifffile import imwrite
from tqdm import tqdm

load_dotenv(".env")
DATA_ROOT = Path(os.getenv("DATA_ROOT"))
SEG_TARGET = "cells"
DIR_SEGMENTATION = DATA_ROOT / "processed/segmented/finetuning"


def main():
    """Main execution loop"""
    model_dirs = [
        dir
        for dir in (DATA_ROOT / "models/cellpose").glob("*")
        if SEG_TARGET in dir.name
    ]
    for model_dir in tqdm(
        model_dirs, f"Evaluating models for {SEG_TARGET} segmentation"
    ):
        model_is_loaded = False
        model_path = model_dir / "models" / model_dir.name
        with open(model_dir / "training_manifest.json", "r") as f:
            json_dict = json.load(f)
            test_files = [DATA_ROOT / file["path"] for file in json_dict["test_files"]]
        downscale_factor = json_dict["preprocessing_steps"][0]["factor"]
        dir_save = DIR_SEGMENTATION / model_dir.name
        dir_save.mkdir(parents=True, exist_ok=True)
        for image_path in tqdm(test_files, "Predicting masks", leave=False):
            masks_path = dir_save / f"{image_path.stem}-{SEG_TARGET}.tif"
            if not masks_path.exists():
                if not model_is_loaded:
                    cellpose_model = CellposeModel(
                        pretrained_model=model_path, gpu=True
                    )
                    model_is_loaded = True
                image = imread_downscaled(image_path, downscale_factor)
                output = cellpose_model.eval(image)
                imwrite(
                    masks_path,
                    output[0],
                    compression="deflate",
                    compressionargs={"level": 6},
                    predictor=2,
                    tile=(256, 256),
                )


if __name__ == "__main__":
    main()
