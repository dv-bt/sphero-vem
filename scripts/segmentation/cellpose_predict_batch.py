"""
Predict segmentation masks in batch on test data, used for checking finetuning performance.
The main prediction is done at the same downscale level as training.
The model is also run at original size and downscale factor 5 to check how robust it is
against ROI size.
"""

import json
from pathlib import Path
import os
import shutil
from dotenv import load_dotenv
from cellpose.models import CellposeModel
from sphero_vem.io import imread_downscaled, imwrite_labels
from tqdm import tqdm

load_dotenv(".env")
DATA_ROOT = Path(os.getenv("DATA_ROOT"))
SEG_TARGET = "cells"
DIR_SEGMENTATION = DATA_ROOT / "processed/segmented/finetuning"
DOWNSCALING_FACTORS = [1, 5, 10]


def compute_targets(model_dir: Path) -> list[tuple[int, Path, Path]]:
    """
    Calculate target values for the prediction loop.

    This function processes a training manifest JSON file to identify test files
    and their corresponding downscale factors. It creates directories for saving
    predicted masks at various resolutions and generates a list of targets for
    the prediction pipeline. It also copies the training manifest in the new
    predicted mask directory.

    Parameters
    ----------
    model_dir : Path
        Directory containing the model and training manifest.json file

    Returns
    -------
    list[tuple[int, Path, Path]]
        A list of target tuples where each tuple contains:
        - The downscaling factor (int)
        - The input image path (Path)
        - The output mask path (Path)
    """
    manifest_path = model_dir / "training_manifest.json"
    with open(manifest_path, "r") as f:
        json_dict = json.load(f)
    test_files: list[Path] = [
        DATA_ROOT / file["path"] for file in json_dict["test_files"]
    ]
    downscale_factor_orig: int = json_dict["preprocessing_steps"][0]["factor"]

    # Directory for saving masks at the same resolution of training images
    dir_save = DIR_SEGMENTATION / model_dir.name
    dir_save.mkdir(parents=True, exist_ok=True)
    shutil.copy(manifest_path, dir_save / "training_manifest.json")

    # Directories for saving masks predicted from images at different
    # resolutions (higher)
    dirs_res: list[Path] = [
        dir_save / "other-resolutions" / f"downscaled-{factor}"
        for factor in DOWNSCALING_FACTORS
    ]
    for dir in dirs_res:
        dir.mkdir(exist_ok=True, parents=True)

    # Prepare final file and downscaling list
    dirs = [dir_save] + dirs_res
    factors = [downscale_factor_orig] + DOWNSCALING_FACTORS
    targets = [
        (factor, file, dir / f"{file.stem}-{SEG_TARGET}.tif")
        for factor, dir in zip(factors, dirs)
        for file in test_files
    ]
    return targets


def main():
    """Main execution loop"""
    model_dirs = [
        dir
        for dir in (DATA_ROOT / "models/cellpose").glob("*")
        if SEG_TARGET in dir.name
    ]

    for model_dir in tqdm(model_dirs, "Evaluating models"):
        model_path = model_dir / "models" / model_dir.name
        cellpose_model = CellposeModel(True, model_path)
        targets = compute_targets(model_dir)

        for ds_factor, image_path, masks_path in tqdm(
            targets, "Predicting masks", leave=False
        ):
            if not masks_path.exists():
                image = imread_downscaled(image_path, ds_factor)
                output = cellpose_model.eval(image)
                imwrite_labels(masks_path, output[0])


if __name__ == "__main__":
    main()
