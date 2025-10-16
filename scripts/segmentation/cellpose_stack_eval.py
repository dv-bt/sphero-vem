"""
Calculate metrics for evaluating stack segmentation performance
"""

from pathlib import Path
import warnings
import re
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd
from tifffile import imread
from cellpose import metrics
from sphero_vem.utils import get_seg_params


def eval_stack(stack_dir: Path, labels_dir: Path) -> pd.DataFrame:
    seg_target = get_seg_params(stack_dir)["seg_target"]
    image = imread(stack_dir / f"Au_01-vol_01-{seg_target}.tif")

    vertical_slices = sorted(labels_dir.glob(f"*virt-{seg_target}.tif"))

    gts = []
    preds = []
    for mask_path in vertical_slices:
        mask = imread(mask_path).astype(np.uint8)
        gts.append(
            cv2.resize(
                mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
            )
        )
        axis, idx = re.search(r"(x|y)_(\d+)", mask_path.name).groups()
        if axis == "x":
            pred = image[:, :, int(idx) // 2]
        elif axis == "y":
            pred = image[:, int(idx) // 2, :]
        preds.append(pred)

    results = []
    thresholds = [0.5]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            metric = metrics.average_precision(gt, pred, threshold=thresholds)
            entry = pd.DataFrame((thresholds, *metric)).T
            entry.columns = [
                "iou_thresholds",
                "average_precision",
                "true_positives",
                "false_positives",
                "false_negatives",
            ]
            entry["ground_truth"] = vertical_slices[i].name
            results.append(entry)

        results_df = pd.concat(results).reset_index(drop=True)
        results_df["aggregated_jaccard_index"] = metrics.aggregated_jaccard_index(
            gts, preds
        )
    return results_df


def main() -> None:
    stack_root = Path("data/processed/segmented/Au_01-vol_01")
    labels_dir = Path("data/processed/labeled/Au_01-vol_01/labeled-03/labels")
    stacks = list(stack_root.glob("*/"))
    for stack_dir in tqdm(stacks, "Evaluating predictions"):
        try:
            results = eval_stack(stack_dir, labels_dir)
            results.to_csv(stack_dir / "segmentation_results.csv", index=False)
            merged_dir = stack_dir / "merged-labels"
            # If separate merged labels, perform evaluation also on those
            if merged_dir.exists():
                results = eval_stack(merged_dir, labels_dir)
                results.to_csv(merged_dir / "segmentation_results.csv", index=False)
        except (FileNotFoundError, TypeError):
            continue


if __name__ == "__main__":
    main()
