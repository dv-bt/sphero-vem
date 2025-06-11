"""
Cleanup directories for failed runs, i.e. runs that crashed or were interrupted
before a model checkpoint could be saved. The user will be prompted for admin
password if elevation is required.
"""

import os
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import argparse

load_dotenv(".env")
DATA_ROOT = Path(os.getenv("DATA_ROOT"))


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Clean up model directories for failed cellpose finetuning runs."
    )
    parser.add_argument(
        "--dry-run", "-dr", action="store_true", help="Execute a dry run"
    )
    return parser.parse_args()


def remove_directory(directory_path: Path) -> None:
    """Remove directory with with admin privileges"""
    try:
        shutil.rmtree(directory_path)
    except PermissionError:
        remove_with_elevation(directory_path)
    except Exception as e:
        print(f"Error removing {directory_path}: {e}")


def remove_with_elevation(directory_path: Path) -> None:
    """Remove directory using system commands with elevation"""
    try:
        result = subprocess.run(
            ["sudo", "rm", "-rf", str(directory_path)], capture_output=True, text=True
        )

        if result.returncode != 0:
            print(f"Failed to remove {directory_path}: {result.stderr}")
    except Exception as e:
        print(f"Elevation removal failed: {e}")


def main():
    args = parse_args()
    dry_run_text = " (dry run)" if args.dry_run else ""
    for model_dir in (DATA_ROOT / "models/cellpose").glob("cellpose*/"):
        model_path = model_dir / "models" / model_dir.name
        if not model_path.exists():
            if not args.dry_run:
                remove_directory(model_dir)
            print(f"Removed directory {model_dir.name}{dry_run_text}")


if __name__ == "__main__":
    main()
