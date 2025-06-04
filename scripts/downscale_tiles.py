"""
Downscale images by a specified integer factor.
"""

from pathlib import Path
import tifffile
from tqdm import tqdm
from sphero_vem.io import imread_downscaled


DATA_DIR = Path("../data/raw")
TARGET_DIR = Path("../data/processed")
DOWNSCALE_FACTOR = 5


def read_excluded_list():
    """Read the list of excluded tiles from the raw data folder."""
    excluded_list = []
    for excluded_file in DATA_DIR.rglob("excluded_tiles.txt"):
        excluded_list.extend(excluded_file.read_text().splitlines())
    return excluded_list


if __name__ == "__main__":
    file_list = list(Path(DATA_DIR).rglob("*/*.tif"))
    excluded_list = read_excluded_list()
    file_list = [path for path in file_list if path.name not in excluded_list]
    for image_path in tqdm(file_list, desc="Downscaling images"):
        target_path = (
            TARGET_DIR / image_path.parent.name
        ) / f"downscaled_{DOWNSCALE_FACTOR}"
        target_path.mkdir(parents=True, exist_ok=True)
        image_ds = imread_downscaled(image_path, DOWNSCALE_FACTOR)
        tifffile.imwrite(
            target_path / f"{image_path.stem}-ds_{DOWNSCALE_FACTOR}.tif", image_ds
        )
