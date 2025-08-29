"""
Merge images into a volume stack
"""

import argparse
import os
import shutil
from pathlib import Path
from dotenv import load_dotenv
from sphero_vem.io import write_stack

load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Merge volume stack")
    parser.add_argument(
        "-s",
        "--source",
        type=Path,
        help="Source directory",
    )
    args = parser.parse_args()
    args.source = DATA_ROOT / args.source
    return args


def infer_dataset(data_dir: Path) -> str:
    """Infer dataset name from the name of images in the data directory"""

    image_name = list(data_dir.glob("*.tif"))[0].name
    return image_name[: image_name.rfind("-z_")]


def main():
    """Main execution"""

    args = parse_args()
    dataset = infer_dataset(args.source)
    out_dir = args.source / "stacked"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{dataset}-stacked.tif"
    shutil.copy(args.source / "manifest.yaml", out_dir / "manifest.yaml")
    write_stack(args.source, out_file)


if __name__ == "__main__":
    main()
