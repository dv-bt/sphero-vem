"""
Trying out inference with pretrained CellposeSAM as a function
of image downscaling
"""

import os
import time
from pathlib import Path
from sphero_vem.io import imread_downscaled, imread_labels_downscaled, write_image
from sphero_vem.segmentation import calculate_ap
from dotenv import load_dotenv
from cellpose.models import CellposeModel
import pandas as pd
from tqdm import tqdm

# Common variables
load_dotenv(".env")
DATA_ROOT = Path(os.environ.get("DATA_ROOT"))
DIR_LABELED = DATA_ROOT / "processed/labeled/Au_01-vol_01/labeled-01"
DIR_OUTPUT = DATA_ROOT / "processed/segmented/pretrained/"
DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
DOWNSCALE_FACTORS = [1, 2, 4, 5, 8, 10, 16, 20, 32]
SEG_TARGETS = ["cells", "nuclei"]


def main() -> None:
    timing_results = []

    image_list = [
        path
        for path in DIR_LABELED.glob("*.tif")
        if (DIR_LABELED / f"labels/{path.stem}-cells.tif").exists()
    ]

    for image_path in tqdm(image_list, desc="Analyzing images", position=0):
        for ds_factor in tqdm(
            DOWNSCALE_FACTORS,
            desc="Analyzing different scales",
            position=1,
            leave=False,
        ):
            image = imread_downscaled(image_path, ds_factor)

            cellpose_model = CellposeModel(gpu=True)

            start_time = time.time()
            predictions = cellpose_model.eval(image)[0]
            elapsed_time = time.time() - start_time

            results_dir = DIR_OUTPUT / f"other-resolutions/downscaled-{ds_factor}"
            results_dir.mkdir(parents=True, exist_ok=True)

            for seg_tartet in SEG_TARGETS:
                gt_path = (
                    image_path.parent / "labels" / f"{image_path.stem}-{seg_tartet}.tif"
                )
                ground_truth = imread_labels_downscaled(gt_path, ds_factor)
                results = calculate_ap(ground_truth, predictions, 0.01)

                pred_path = results_dir / gt_path.name
                write_image(pred_path, predictions, compressed=True)

                results_path = results_dir / f"{gt_path.stem}-ap.csv"
                results.to_csv(results_path, index=False)

            # Store downsample factor and corresponding time
            timing_results.append((ds_factor, elapsed_time))

    timing_df = pd.DataFrame(timing_results, columns=["downscale_factor", "time"])
    timing_df.to_csv(DIR_OUTPUT / "cellposeSAM-downscaled-time.csv", index=False)


if __name__ == "__main__":
    main()
