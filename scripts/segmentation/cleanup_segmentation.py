"""
Cleanup directories for failed runs, i.e. runs that crashed or were interrupted
before a model checkpoint could be saved, as well as incomplete predictions.
The user will be prompted for admin password if elevation is required.
"""

import shutil
import subprocess
from pathlib import Path
import argparse
from sphero_vem.utils import get_seg_params


def parse_args():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Clean up model directories for failed cellpose finetuning runs."
    )
    parser.add_argument(
        "--dry-run", "-dr", action="store_true", help="Execute a dry run"
    )
    return parser.parse_args()


def remove_directory(directory_path: Path, dry_run: bool = False) -> None:
    """Remove directory with with admin privileges"""
    if dry_run:
        return
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

    # Removed failed finetuning runs
    print("Clean up models")
    for model_dir in Path("data/models/cellpose").glob("cellpose*/"):
        model_path = model_dir / "models" / model_dir.name
        if not model_path.exists():
            remove_directory(model_dir, args.dry_run)
            print(f"Removed directory {model_dir.name}{dry_run_text}")

    # Remove failed prediction runs
    print("Clean up predictions")
    for seg_dir in Path("data/processed/segmented").glob("*/*/"):
        # Skip finetuning results directories
        if seg_dir.parent.name == "pretrained":
            continue

        image_list = list(seg_dir.glob("*.tif"))
        if len(image_list) == 0:
            remove_directory(seg_dir, args.dry_run)
            print(f"Removed empty directory {seg_dir.name}{dry_run_text}")
        elif seg_dir.parent.name != "finetuning":
            seg_params = get_seg_params(seg_dir)
            if not seg_params.get("seg_target"):
                remove_directory(seg_dir, args.dry_run)
                print(
                    f"Removed directory with malformed manifest {seg_dir.name}{dry_run_text}"
                )


if __name__ == "__main__":
    main()
