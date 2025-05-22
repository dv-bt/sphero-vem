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
dir_labeled = DATA_ROOT / "processed/labeled/Au_01-vol_01/labeled-01"
dir_output = DATA_ROOT / "processed/segmented/Au_01-vol_01/downscaling_tests"
dir_output.mkdir(parents=True, exist_ok=True)


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
    image_path = dir_labeled / "Au_01-vol_01-z_0425.tif"
    factors = [1, 2, 4, 5, 8, 10, 16, 20, 32]
    timing_results = []

    for dowscale_factor in factors:
        image = imread_downscaled(image_path, dowscale_factor)

        cellpose_model = CellposeModel(gpu=True)

        start_time = time.time()
        output = cellpose_model.eval(image)
        elapsed_time = time.time() - start_time

        # Store downsample factor and corresponding time
        timing_results.append((dowscale_factor, elapsed_time))
        print(f"Downscale factor {dowscale_factor}: {elapsed_time:.2f} seconds")

        imwrite(dir_output / f"{image_path.stem}-ds{dowscale_factor}.tif", image)
        imwrite(
            dir_output / f"cellposeSAM-pretrained-mask-ds{dowscale_factor}.tif",
            output[0],
        )

    timing_df = pd.DataFrame(timing_results, columns=["downscale_factor", "time"])
    timing_df.to_csv(dir_output / "cellposeSAM-downscaled-time.csv", index=False)


if __name__ == "__main__":
    main()
