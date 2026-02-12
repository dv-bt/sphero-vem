"""
Repair OME multiscales in target zarr store
"""

from pathlib import Path
from sphero_vem.utils import repair_multiscales


def main():
    """Repair multiscales"""
    root = Path("data/processed/segmented/Au_01-vol_01.zarr")
    repair_multiscales(root)


if __name__ == "__main__":
    main()
