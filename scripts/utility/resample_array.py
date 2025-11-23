"""
Resample zarr array
"""

from pathlib import Path
from sphero_vem.preprocessing import resample_array


def main():
    """Resample zarr array"""
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    target_spacing = (50, 25, 25)
    array_path = "labels/nps/posterior/50-10-10"
    resample_array(root_path, array_path, target_spacing)


if __name__ == "__main__":
    main()
