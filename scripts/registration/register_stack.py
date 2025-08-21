"""
Register denoised volume stacks
"""

import os
import datetime
import argparse
import yaml
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from sphero_vem.registration import register_to_disk, TransformType
from sphero_vem.utils import read_section_errors


load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Register a volume stack")
    parser.add_argument(
        "-t",
        "--transform",
        type=str,
        choices=[i.value for i in TransformType],
        help="Type of transformation used for image registration",
    )
    parser.add_argument(
        "-sf",
        "--shrink_factors",
        type=int,
        nargs="+",
        help="shrink factors used during multiresolution registration. The number"
        "of parameters should match those of smoothing sigmas and sampling fractions",
    )
    parser.add_argument(
        "-sm",
        "--smoothing_sigmas",
        type=float,
        nargs="+",
        help="Smoothing sigmas used during multiresolution registration. The number"
        "of parameters should match those of shrink factors and sampling fractions",
    )
    parser.add_argument(
        "-sa",
        "--sampling_fractions",
        type=float,
        nargs="+",
        help="Sampling fractions used during multiresolution registration. The number"
        "of parameters should match those of shrink factors and smoothing sigmas",
    )

    args = parser.parse_args()
    args.transform = TransformType(args.transform)

    return args


def generate_manifest(
    data_dir: Path,
    out_dir: Path,
    images: list[Path],
    args: argparse.Namespace,
    error_tiles: list[Path] | None,
) -> None:
    """Generate manifest.yaml file with processing steps"""

    manifest = {
        "dataset": data_dir.parent.name,
        "generated_on": datetime.datetime.now().isoformat(),
        "processing": [
            {
                "step": "denoising",
                "algorithm": data_dir.name,
            },
            {
                "step": "registration",
                "transform": args.transform,
                "shriking factors": args.shrink_factors,
                "smoothing sigmas": args.smoothing_sigmas,
                "metric sampling fractions": args.sampling_fractions,
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
    """Execute script"""

    args = parse_args()
    for data_dir in (DATA_ROOT / "processed/denoised").glob("*/*/"):
        out_dir = DATA_ROOT / f"processed/aligned/{data_dir.parent.name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        error_tiles = read_section_errors(data_dir)
        images = sorted(list(data_dir.glob("*.tif")))
        if error_tiles:
            images = [path for path in images if path.name not in error_tiles]
        generate_manifest(data_dir, out_dir, images, args, error_tiles)

        # Temporarily stop after a few images for testing
        images = images[:10]

        for i in tqdm(range(len(images) - 1), "Aligning images"):
            register_to_disk(
                out_dir / images[i].name,
                images[i + 1],
                out_dir / images[i + 1].name,
            )


if __name__ == "__main__":
    main()
