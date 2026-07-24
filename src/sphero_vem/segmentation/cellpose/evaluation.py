"""
Evaluation of Cellpose segmentation accuracy
"""

import warnings
from typing import Literal
from pathlib import Path
import re
import numpy as np
import pandas as pd
from tifffile import imread
import zarr
from cellpose.metrics import aggregated_jaccard_index, average_precision
from sphero_vem.utils import dirname_from_spacing


def calculate_ap(
    ground_truth: np.ndarray, predictions: np.ndarray, threshold_step: float = 0.05
) -> pd.DataFrame:
    """Calculate average precision and related metrics at different IoU thresholds.

    This function evaluates segmentation predictions against ground truth by computing
    the average precision, as well as counting the number of true positives, false
    positives, and false negatives at different Jaccard index (IoU) thresholds.

    Parameters
    ----------
    ground_truth : np.ndarray
        Ground truth segmentation mask with instance labels.
        Each object should have a unique integer ID > 0.
    predictions : np.ndarray
        Predicted segmentation mask with instance labels.
        Each predicted object should have a unique integer ID > 0.
    threshold_step : float, optional
        Step size for IoU thresholds, defaults to 0.05.
        Thresholds will be generated as [threshold_step, 2*threshold_step, ..., < 1.0]

    Returns
    -------
    pd.DataFrame
        DataFrame containing evaluation metrics with columns:
        - 'iou_thresholds': IoU threshold values
        - 'average_precision': AP score at each threshold
        - 'true_positives': Number of true positive detections
        - 'false_positives': Number of false positive detections
        - 'false_negatives': Number of false negative detections
    """
    thresholds = np.arange(threshold_step, 1.0, threshold_step).round(2)
    metric = average_precision(ground_truth, predictions, threshold=thresholds)
    results_df = pd.DataFrame((thresholds, *metric)).T
    results_df.columns = [
        "iou_thresholds",
        "average_precision",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]
    return results_df


def _to_index(vals: list[str]) -> int | slice:
    """Return an int for a single value, a slice for a start/stop pair."""
    return int(vals[0]) if len(vals) == 1 else slice(int(vals[0]), int(vals[1]))


def _slice_indexer(path: Path, ndim: int = 3) -> tuple[int | slice, ...]:
    """Build a numpy indexer tuple for the raw-image region matching a label file.

    Label TIFFs are crops or slices of a larger array stored as zarr, and encode
    their position in the filename as components of the form ``-{axis}_{index}``
    or ``-{axis}_{start}_{stop}``, e.g. ``sample-x_2600_3600-y_200_400-z_120.tif``.
    An axis given a single value becomes an ``int`` and is dropped from the
    indexed result; an axis given a start/stop pair becomes a ``slice`` and is
    retained. Axes absent from the filename become ``slice(None)``, so a label
    covering the full raw image needs no coordinates at all.

    Parameters
    ----------
    path : Path
        Path to a label TIFF whose name optionally encodes one or more axes.
    ndim : int, default 3
        Dimensionality of the array to be indexed, e.g. ``image.ndim``. Axis names
        are taken from the trailing ``ndim`` characters of ``"zyx"``, i.e.
        ``("y", "x")`` in 2D.

    Returns
    -------
    tuple of int or slice
        Index tuple of length ``ndim``, compatible with numpy/zarr basic indexing.
        All-``slice(None)`` if the filename encodes no coordinates.
    """
    axes = "zyx"[-ndim:]
    axis_re = re.compile(r"-([xyz])((?:_\d+)+)")
    coords = {
        ax: _to_index(vals.split("_")[1:]) for ax, vals in axis_re.findall(path.name)
    }
    return tuple(coords.get(ax, slice(None)) for ax in axes)


def _get_seg_target(array: zarr.Array) -> str:
    """Extract the segmentation target name from a Zarr array's processing metadata.

    Parameters
    ----------
    array : zarr.Array
        Zarr array whose ``"processing"`` attribute contains a segmentation step.

    Returns
    -------
    str
        Segmentation target string (e.g. ``"cells"`` or ``"nuclei"``).
    """
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


def evaluate_segmentation(
    root_path: Path,
    gt_root_path: Path,
    array_path: str,
    out_dir: Path | None = None,
    seg_target: Literal["cells", "nuclei"] | None = None,
) -> pd.DataFrame:
    """Calculate accuracy for a volume segmentation with several metrics and saves them.

    Parameters
    ----------
    root_path : Path
        The path the the zarr root store.
    gt_root_path : Path
        The path the root of the labeled dataset. It should have subdirectories with
        structure `gt_root_path/spacing_dir/labels`. `spacing_dir` should is the last
        element of `array_path`.
    array_path : str
        Path to the mask array to analyse relative to `root_path`.
    out_dir : Path | None, optional
        Optional destination path for the calculated metrics. If specified, metrics
        will be saved as `out_dir/segmentation-eval.parquet`, otherwise they will
        be saved as `tables/segmentation-eval.parquet` within the parent group of the
        mask array.
        Default is None.
    seg_target : str | None
        If this is specified, accuracy metrics will be calculated against ground truths
        for this segmentation target, regardless of the actual target they were
        calculated on. If None, segmentation target will be read from the mask array.
        This is useful when comparing cross-class detections, or when evaluating
        pretrained models, which were not trained on a specific custom target.
        Default is None.

    Returns
    -------
    pd.DataFrame
        A dataframe with the calculated metrics.
    """

    # Process inputs
    label_folder = Path(array_path).parents[1]

    # Load data and get parameters
    root = zarr.open(root_path, mode="r")
    masks = root.get(array_path)
    scale_dir = Path(array_path).name

    if seg_target is None:
        seg_target = _get_seg_target(masks)
    if out_dir is None:
        out_dir = root_path / f"{label_folder.parent}/{seg_target}/tables"

    gt_paths = sorted(gt_root_path.glob(f"{scale_dir}/labels/*-{seg_target}.tif"))
    gts = []
    preds = []
    for path in gt_paths:
        gt = imread(path)
        pred = masks[_slice_indexer(path=path, ndim=gt.ndim)]
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
        results_df["spacing"] = dirname_from_spacing(masks.attrs.get("spacing"))

    out_dir.mkdir(exist_ok=True, parents=True)
    save_path = out_dir / "segmentation-eval.parquet"
    results_df.to_parquet(save_path, index=False)

    return results_df
