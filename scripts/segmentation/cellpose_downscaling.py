"""
Trying out inference with pretrained CellposeSAM as a function
of image downscaling
"""

import os
import time
from pathlib import Path
from sphero_vem.preprocessing import imread_downscaled
from dotenv import load_dotenv
from cellpose.models import CellposeModel
import numpy as np
import tifffile
import pandas as pd

# Common variables
load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))
DIR_LABELED = DATA_ROOT / "processed/labeled/Au_01-vol_01/labeled-01"
DIR_OUTPUT = DATA_ROOT / "processed/segmented/Au_01-vol_01/downscaling_tests"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
DOWNSCALE_FACTORS = [1, 2, 4, 5, 8, 10, 16, 20, 32]
SEG_TARGET = "cells"


# Function definitions
def imwrite(fname: Path, image: np.ndarray) -> None:
    """Wrapper function to save images with common default options"""
    tifffile.imwrite(
        fname,
        image,
        compression="deflate",
        compressionargs={"level": 6},
        predictor=2,
        tile=(256, 256),
    )


def main() -> None:
    timing_results = []

    for image_path in DIR_LABELED.glob("*.tif"):
        mask_path = DIR_LABELED / f"labels/{image_path.stem}_{SEG_TARGET}.tif"
        if not mask_path.exists():
            continue

        for dowscale_factor in DOWNSCALE_FACTORS:
            image = imread_downscaled(image_path, dowscale_factor)

            cellpose_model = CellposeModel(gpu=True)

            start_time = time.time()
            output = cellpose_model.eval(image)
            elapsed_time = time.time() - start_time

            # Store downsample factor and corresponding time
            timing_results.append((dowscale_factor, elapsed_time))
            print(f"Downscale factor {dowscale_factor}: {elapsed_time:.2f} seconds")

            imwrite(
                DIR_OUTPUT
                / f"{image_path.stem}-cellposeSAM-pretrained-{SEG_TARGET}-ds{dowscale_factor}.tif",
                output[0],
            )

    timing_df = pd.DataFrame(timing_results, columns=["downscale_factor", "time"])
    timing_df.to_csv(
        DIR_OUTPUT / f"cellposeSAM-{SEG_TARGET}-downscaled-time.csv", index=False
    )


if __name__ == "__main__":
    main()
