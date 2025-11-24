"""
Theshold np posterior
"""

from pathlib import Path
from sphero_vem.segmentation_np import label_nanoparticles


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    spacing = (50, 50, 50)

    threshold = 0.95
    radius = 1
    connectivity = 2
    min_size = 10

    label_nanoparticles(
        root_path,
        spacing=spacing,
        threshold=threshold,
        radius=radius,
        connectivity=connectivity,
        min_size=min_size,
    )


if __name__ == "__main__":
    main()
