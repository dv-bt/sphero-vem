"""
This script renames tiles in the raw data folder to a consistent format.
"""

import os
from pathlib import Path
import re
from dotenv import load_dotenv
import argparse

# Define the path to the raw data folder
load_dotenv("../.env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))
DATA_PATH = DATA_ROOT / "raw"

# Set up command line arguments
parser = argparse.ArgumentParser(
    description="Rename tile files to a consistent format."
)
parser.add_argument(
    "--dry-run",
    action="store_true",
    dest="dry_run",
    help="Only print what would be renamed without actually renaming files",
)
parser.add_argument(
    "--verbose", action="store_true", dest="verbose", help="Enable verbose output"
)
args = parser.parse_args()

DRY_RUN = args.dry_run
VERBOSE = args.verbose


def rename_tile(old_path: Path, verbose: bool = False, dry_run: bool = True) -> Path:
    """Rename the tile and return the new path. If dry_run is True, only print
    the new path. For safety purposes, this is the default behavior."""

    # Extract the tile number from the old path
    try:
        tile_number = re.search(r"\.s(\d+)_", old_path.name).group(1)
        new_name = f"{old_path.parent.name}-z_{tile_number.zfill(4)}.tif"
        new_path = old_path.parent / new_name

        if verbose:
            print(f"Renaming {old_path.name} to {new_name}")

        if not dry_run:
            old_path.rename(new_path)
    except AttributeError:
        if verbose:
            print(f"{old_path.name} already formatted correctly.")
        new_path = old_path

    return new_path


def main():
    for old_path in DATA_PATH.rglob("*.tif"):
        _ = rename_tile(old_path, verbose=VERBOSE, dry_run=DRY_RUN)


if __name__ == "__main__":
    main()
