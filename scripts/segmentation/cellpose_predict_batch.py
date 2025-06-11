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
from sphero_vem.segmentation import calculate_ap, extract_seg_target
from sphero_vem.io import imread_downscaled, imwrite_labels, imread_labels_downscaled
from tqdm import tqdm


load_dotenv(".env")
DATA_ROOT = Path(os.getenv("DATA_ROOT"))
DIR_SEGMENTATION = DATA_ROOT / "processed/segmented/finetuning"
DOWNSCALING_FACTORS = [1, 5, 10]


def compute_targets(model_dir: Path) -> list[tuple[int, Path, Path, Path]] | None:
    """
    Calculate target values for the prediction loop.

    This function processes a training manifest JSON file to identify test files
    and their corresponding downscale factors. It creates directories for saving
    predicted masks at various resolutions and generates a list of targets for
    the prediction pipeline. It also copies the training manifest in the new
    predicted mask directory. Returns None if no segmentation target was

    Parameters
    ----------
    model_dir : Path
        Directory containing the model and training manifest.json file

    Returns
    -------
    list[tuple[int, Path, Path, Path]]
        A list of target tuples where each tuple contains:
        - The downscaling factor (int)
        - The input image path (Path)
        - The ground truth mask path (Path)
        - The output mask path (Path)
    """
    manifest_path = model_dir / "training_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    test_files: list[Path] = [
        DATA_ROOT / file["path"] for file in manifest["test_files"]
    ]
    seg_target = extract_seg_target(manifest)
    if not seg_target:
        print(
            f"Segmentation target for {manifest['experiment_id']} could not be inferred"
        )
        return
    gt_files: list[Path] = [
        file.parent / "labels" / f"{file.stem}-{seg_target}.tif" for file in test_files
    ]
    downscale_factor_orig: int = manifest["preprocessing_steps"][0]["factor"]

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
        (factor, file, gt_file, dir / f"{file.stem}-{seg_target}.tif")
        for factor, dir in zip(factors, dirs)
        for file in test_files
        for gt_file in gt_files
    ]
    return targets


def main():
    """Main execution loop"""
    model_dirs = [dir for dir in (DATA_ROOT / "models/cellpose").glob("*")]

    for model_dir in tqdm(model_dirs, "Evaluating models"):
        model_path = model_dir / "models" / model_dir.name
        targets = compute_targets(model_dir)
        if not targets:
            print("Evaluation skipped")
            continue
        cellpose_model = CellposeModel(True, model_path)

        for ds_factor, image_path, gt_path, masks_path in tqdm(
            targets, "Predicting masks", leave=False
        ):
            image = imread_downscaled(image_path, ds_factor)
            ground_truth = imread_labels_downscaled(gt_path, ds_factor)
            if not masks_path.exists():
                output = cellpose_model.eval(image)
                imwrite_labels(masks_path, output[0])
            results = calculate_ap(ground_truth, output, 0.01)
            results_path = masks_path.parent / f"{masks_path.stem}-ap.csv"
            results.to_csv(results_path, index=False)


if __name__ == "__main__":
    main()
