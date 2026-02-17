"""
Evaluate prediction accuracy metrics from a segmented stack
"""

import warnings
from pathlib import Path
import re
import pandas as pd
from tifffile import imread
import zarr
from cellpose.metrics import aggregated_jaccard_index
from sphero_vem.segmentation.cellpose.utils import calculate_ap


def slice_indexer(path: Path) -> tuple:
    """Get tuple for indexing volume stack slice corresponding to ground truth path"""

    axis_map = {"x": 2, "y": 1, "z": 0}

    matches = re.search(r"-([xyz])_(\d+)", path.name)
    axis = axis_map[matches.group(1)]
    idx = int(matches.group(2))
    indexer = tuple(idx if i == axis else slice(None) for i in range(3))

    return indexer


def get_seg_target(array: zarr.Array) -> str:
    """Get segmentation target from array metadata"""
    seg_step = next(
        (
            d
            for d in array.attrs.get("processing")
            if "segmentation" in d.get("step", d.get("step_name", ""))
        ),
        None,
    )
    seg_target = seg_step.get("parameters", {}).get("seg_target") or seg_step.get(
        "seg_target"
    )
    return seg_target


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    gt_root = Path("data/processed/labeled/Au_01-vol_01/labeled-05")
    array_path = "labels/cells/masks/50-50-50"

    root = zarr.open(root_path, mode="r")
    masks = root.get(array_path)
    seg_target = get_seg_target(masks)
    scale_dir = Path(array_path).name

    gt_paths = sorted(gt_root.glob(f"{scale_dir}/labels/*-{seg_target}.tif"))
    gts = []
    preds = []
    for path in gt_paths:
        gt = imread(path)
        pred = masks[slice_indexer(path)]
        assert gt.shape == pred.shape

        gts.append(gt)
        preds.append(pred)

    results = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, (gt, pred) in enumerate(zip(gts, preds)):
            entry = calculate_ap(gt, pred, threshold_step=0.05)
            entry["ground_truth"] = gt_paths[i].name
            results.append(entry)

        results_df = pd.concat(results).reset_index(drop=True)

        results_aji = pd.DataFrame(
            {
                "ground_truth": [path.name for path in gt_paths],
                "aggregated_jaccard_index": aggregated_jaccard_index(gts, preds),
            }
        )
        results_df: pd.DataFrame = results_df.merge(results_aji)
        results_df["spacing"] = scale_dir

    save_path = root_path / f"labels/{seg_target}/segmentation-eval.parquet"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_parquet(save_path, index=False)


if __name__ == "__main__":
    main()
