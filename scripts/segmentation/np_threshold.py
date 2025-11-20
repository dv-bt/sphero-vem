"""
Theshold np posterior
"""

from pathlib import Path
from sphero_vem.segmentation_np import threshold_posterior


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    threshold = 0.95
    spacing_dir = "50-50-50"
    threshold_posterior(root_path, threshold=threshold, spacing_dir=spacing_dir)


if __name__ == "__main__":
    main()
