"""
Resample zarr array
"""

from pathlib import Path
from sphero_vem.preprocessing import resample_array


def main():
    """Resample zarr array"""
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    target_spacing = (100, 100, 100)
    array_path = "images/50-50-50"
    resample_array(root_path, array_path, target_spacing, n_workers=8)


if __name__ == "__main__":
    main()
