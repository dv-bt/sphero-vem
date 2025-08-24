"""
Downscale aligned tiles
"""

import os
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from sphero_vem.io import imread_downscaled, imwrite
from sphero_vem.utils import generate_manifest


load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Downscale aligned images")
    parser.add_argument(
        "-f",
        "--factor",
        type=int,
        help="Dowscaling factor",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    factor = args.factor
    for data_dir in (DATA_ROOT / "processed/aligned").glob("*/"):
        out_dir = data_dir / f"downscaled/downscaled-{factor}"
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(list(data_dir.glob("*.tif")))
        generate_manifest(
            data_dir.name,
            out_dir,
            images,
            processing=[
                {
                    "step": "downscaling",
                    "factor": factor,
                }
            ],
        )
        for image_path in tqdm(images, desc="Downscaling images"):
            image = imread_downscaled(image_path, factor)
            imwrite(out_dir / image_path.name, image, uncompressed=True)


if __name__ == "__main__":
    main()
