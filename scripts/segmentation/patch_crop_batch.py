"""
Batch crop NP and background patches using bounding box info stored in csv files
"""

import re
import os
from pathlib import Path
import pandas as pd
import tifffile
import numpy as np
from dotenv import load_dotenv
from sphero_vem.io import imwrite

# Common variables
load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))
DIR_LABELED = DATA_ROOT / "processed/labeled/Au_01-vol_01/labeled-01/"


def main():
    """Load bounding box info from csv files, crop and save images"""
    for file_path in DIR_LABELED.rglob("*-crops.csv"):
        patch_type = re.search(r"-(\w+)-crops.csv", file_path.name).group(1)
        image_path = DIR_LABELED / file_path.name.replace(
            f"-{patch_type}-crops.csv", ".tif"
        )
        image = tifffile.imread(image_path)
        bboxes = pd.read_csv(file_path)
        bboxes.columns.values[0] = "crop_ix"
        for bbox in bboxes.itertuples():
            image_crop = image[
                bbox.BY : bbox.BY + bbox.Height,
                bbox.BX : bbox.BX + bbox.Width,
            ]
            np.ascontiguousarray(image_crop, dtype=np.uint8)
            imwrite(
                file_path.parent / f"{file_path.stem}-{bbox.crop_ix:02d}.tif",
                image_crop,
            )


if __name__ == "__main__":
    main()
