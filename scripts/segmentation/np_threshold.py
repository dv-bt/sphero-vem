"""
Theshold np posterior
"""

from pathlib import Path
from sphero_vem.segmentation_np import threshold_posterior, downsample_posterior


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    threshold = 0.95
    sigma = (0.5, 1.5, 1.5)
    spacing = (50, 50, 50)
    downsample_posterior(root_path, sigma, dst_spacing=spacing)
    threshold_posterior(root_path, threshold=threshold, spacing_dir=spacing)


if __name__ == "__main__":
    main()
