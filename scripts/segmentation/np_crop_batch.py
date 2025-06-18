"""
Batch crop nanoparticle ground truth ROIs using bounding box info stored in csv files
"""

import os
from pathlib import Path
import pandas as pd
import tifffile
from dotenv import load_dotenv

# Common variables
load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))
DIR_CROPS = DATA_ROOT / "processed/labeled/Au_01-vol_01/labeled-01/np-crops"


def main():
    """Load bounding box info from csv files, crop and save images"""
    for file_path in DIR_CROPS.glob("*-np-crops.csv"):
        image_path = DIR_CROPS.parent / file_path.name.replace("-np-crops.csv", ".tif")
        image = tifffile.imread(image_path)
        bboxes = pd.read_csv(file_path)
        bboxes.columns.values[0] = "crop_ix"
        for bbox in bboxes.itertuples():
            image_crop = image[
                bbox.BY : bbox.BY + bbox.Height,
                bbox.BX : bbox.BX + bbox.Width,
            ]
            tifffile.imwrite(
                DIR_CROPS / f"{file_path.stem}-{bbox.crop_ix:02d}.tif", image_crop
            )


if __name__ == "__main__":
    main()
