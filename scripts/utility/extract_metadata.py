"""
Extract metadata from raw images
"""

import os
import json
from tqdm import tqdm
from pathlib import Path
from imageio.v3 import immeta
from dotenv import load_dotenv

load_dotenv(".env")
DATA_ROOT = Path(os.getenv("DATA_ROOT"))


def write_metadata(image_path: Path) -> None:
    """Extract image metadata and write them to a json file in a metadata subfolder"""
    metadata = immeta(image_path)
    dir_metadata = image_path.parent / "metadata"
    dir_metadata.mkdir(exist_ok=True)
    path_metadata = dir_metadata / f"{image_path.stem}-metadata.json"

    with open(path_metadata, "w") as file:
        json.dump(metadata, file, indent=4)


def main() -> None:
    """Loop through all the raw files and extract metadata"""
    path_raw = DATA_ROOT / "raw"
    image_list = list(path_raw.rglob("*.tif"))
    for image_path in tqdm(image_list):
        write_metadata(image_path)


if __name__ == "__main__":
    main()
