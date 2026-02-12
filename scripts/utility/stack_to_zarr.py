"""
Convert an image stack to a zarr archive
"""

from pathlib import Path
import zarr
from sphero_vem.io import stack_to_zarr, write_zarr
from sphero_vem.preprocessing import crop_to_valid


def main():
    root_tiff = Path("data/raw/FaDu-2D/tiff-stacks")
    root_dst = Path("data/raw/FaDu-2D")

    for dir in root_tiff.glob("*/"):
        zarr_path = root_dst / f"{dir.name}.zarr"
        stack_to_zarr(
            stack_dir=dir, root_path=zarr_path, spacing=(50, 50, 50), verbose=True
        )

        array = zarr.open_array(zarr_path / "images/50-50-50")
        cropped = crop_to_valid(array[:])
        write_zarr(
            root=zarr_path, array=cropped, dst_path="cropped/50-50-50", src_zarr=array
        )

    return


if __name__ == "__main__":
    main()
