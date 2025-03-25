"""
Downscale images by a specified integer factor.
"""

from pathlib import Path
import skimage as ski
from tqdm import tqdm


DATA_DIR = Path("../data/raw")
TARGET_DIR = Path("../data/processed")
DOWNSCALE_FACTOR = 2


def downscale_image(image_path, factor):
    """Downscale images by a specified integer factor, and save as uint8."""
    image = ski.io.imread(image_path)
    image = ski.util.img_as_float(image)
    image_ds = ski.transform.downscale_local_mean(image, (factor, factor))
    image_ds = ski.util.img_as_ubyte(image_ds)
    return image_ds


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
        target_path = TARGET_DIR / image_path.parent.name
        target_path.mkdir(parents=True, exist_ok=True)
        image_ds = downscale_image(image_path, DOWNSCALE_FACTOR)
        ski.io.imsave(target_path / f"{image_path.stem}-ds_2.tif", image_ds)
