"""
Calculate distance from nucleus for each cell
"""

from pathlib import Path
from sphero_vem.measure import nuclei_distance


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    nuclei_distance(root_path)


if __name__ == "__main__":
    main()
