"""
Script to split a denoised dataset into a subset for manual labeling. The dataset can
then be used for training segmentation algorithms
"""

import os
import shutil
from pathlib import Path
import numpy as np
import argparse
import yaml
import datetime
from dotenv import load_dotenv

# Common variables
load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))


def get_args() -> argparse.Namespace:
    """
    Parse command line arguments for dataset splitting.

    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Split a dataset into a subset for manual labeling."
    )

    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the input dataset",
    )

    parser.add_argument(
        "-o",
        "--output-name",
        type=str,
        required=True,
        help="Name of the output dataset",
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Number of samples to select for the labeling subset",
    )

    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )

    return parser.parse_args()


def main() -> None:
    # Parse the command line arguments
    args = get_args()

    # Set the random seed for reproducibility
    rng = np.random.default_rng(seed=args.random_seed)

    # Get the path to the input directory
    input_dir = Path(args.input_dir)
    dataset = input_dir.parts[-2]
    denoising_model = input_dir.parts[-1]
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    # Get all images in the input directory
    images = list(input_dir.glob("*"))

    # Check if there are enough images
    if len(images) < args.subset_size:
        print(
            f"Warning: Input directory contains only {len(images)} files, "
            f"but {args.subset_size} were requested. Using all available files."
        )
        subset_size = len(images)
    else:
        subset_size = args.subset_size

    # Randomly select images for the subset
    subset_indices = rng.choice(len(images), size=subset_size, replace=False)
    selected_images = [images[i] for i in subset_indices]
    selected_images = sorted(selected_images)

    # Define output directory
    out_dir = DATA_ROOT / f"processed/labeled/{dataset}/{args.output_name}"
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Error: Output dataset already exists. Please use a different name")
        return

    # Generate manifest file
    generate_manifest(out_dir, selected_images, denoising_model)

    # Copy selected images to the output directory
    for image in selected_images:
        shutil.copy(image, out_dir / image.name)


def generate_manifest(out_dir: Path, images: list[Path], denoising_model: str) -> None:
    """Generate manifest.yaml file with processing steps"""

    manifest = {
        "dataset": str(out_dir.relative_to(DATA_ROOT / "processed/labeled")),
        "generated_on": datetime.datetime.now().isoformat(),
        "processing": {
            "step": "denoise",
            "algorithm": denoising_model,
        },
        "inputs": [str(p.relative_to(DATA_ROOT)) for p in images],
        "outputs": [str(p.name) for p in images],
    }

    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


if __name__ == "__main__":
    main()
