"""
Downscale aligned tiles
"""

import os
import argparse
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from sphero_vem.io import imread_downscaled, imwrite
from sphero_vem.utils import generate_manifest, read_section_errors


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
    extra_fields = {}
    for data_dir in (DATA_ROOT / "processed/aligned").glob("*/"):
        out_dir = data_dir / f"downscaled/downscaled-{factor}"
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(data_dir.glob("*.tif"))

        # If factor 10, use every other image to have isotropic voxels (100 nm side)
        if args.factor == 10:
            section_errors = read_section_errors(data_dir)
            all_images = sorted([image.name for image in images] + section_errors)
            selected = all_images[::2]
            discarded = all_images[1::2]
            images = [
                data_dir / name for name in selected if name not in section_errors
            ]
            extra_fields["discarded"] = discarded

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
            **extra_fields,
        )
        for image_path in tqdm(images, desc="Downscaling images"):
            image = imread_downscaled(image_path, factor)
            imwrite(out_dir / image_path.name, image, uncompressed=True)


if __name__ == "__main__":
    main()
