"""
Downscale denoised tiles, keeping information on sectioning errors to preserve the
correct Z-coordinate in the volume
"""

import os
import argparse
import yaml
import datetime
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from sphero_vem.io import imread_downscaled, imwrite
from sphero_vem.utils import read_section_errors


load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Downscale denoised images")
    parser.add_argument(
        "--factor",
        type=int,
        default=10,
        help="Dowscaling factor",
    )
    return parser.parse_args()


def generate_manifest(
    data_dir: Path,
    out_dir: Path,
    images: list[Path],
    factor: int,
    error_tiles: list[Path] | None,
) -> None:
    """Generate manifest.yaml file with processing steps"""

    manifest = {
        "dataset": data_dir.parent.name,
        "generated_on": datetime.datetime.now().isoformat(),
        "processing": [
            {
                "step": "denoise",
                "algorithm": data_dir.name,
            },
            {
                "step": "downscaling",
                "factor": factor,
            },
        ],
        "inputs": [str(p.relative_to(DATA_ROOT)) for p in images],
        "outputs": [str(p.name) for p in images],
    }

    if error_tiles:
        manifest["folded_section_tiles"] = error_tiles

    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def main():
    args = parse_args()
    factor = args.factor
    for data_dir in (DATA_ROOT / "processed/denoised").glob("*/*/"):
        out_dir = data_dir / f"downscaled/downscaled-{factor}"
        out_dir.mkdir(parents=True, exist_ok=True)
        error_tiles = read_section_errors(data_dir)
        images = sorted(list(data_dir.glob("*.tif")))
        generate_manifest(data_dir, out_dir, images, factor, error_tiles)
        for image_path in tqdm(images, desc="Downscaling images"):
            if error_tiles and (image_path.name in error_tiles):
                continue
            image = imread_downscaled(image_path, factor)
            imwrite(out_dir / image_path.name, image)


if __name__ == "__main__":
    main()
