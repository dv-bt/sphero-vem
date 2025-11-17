"""
Resample zarr array
"""

from pathlib import Path
from sphero_vem.preprocessing import resample_array


def main():
    """Resample zarr array"""
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    target_spacing = (50, 50, 50)
    array_path = "labels/nuclei/flows/cellprob/100-100-100"
    resample_array(root_path, array_path, target_spacing, num_workers=4)


if __name__ == "__main__":
    main()
