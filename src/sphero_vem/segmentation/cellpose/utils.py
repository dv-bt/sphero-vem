"""
Various utility functions for cellposes
"""

from pathlib import Path
import numpy as np
import pandas as pd
from cellpose import metrics
import zarr
from skimage import graph
from sphero_vem.io import write_zarr
from sphero_vem.utils import dirname_from_spacing
from sphero_vem.utils.accelerator import xp, ndi, ArrayLike, gpu_dispatch, to_host


@gpu_dispatch(return_to_host=True)
def _calc_seeds(
    labels_lr: ArrayLike, erosion_iterations: int, zoom_factors: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Generate high res seeds. Returns also low-res mask foreground."""

    foreground_lr = labels_lr > 0
    eroded_fg = foreground_lr
    for _ in range(erosion_iterations):
        eroded_fg = ndi.binary_erosion(eroded_fg)
    seeds_lr = labels_lr * eroded_fg
    seeds_hr = ndi.zoom(seeds_lr, zoom_factors, order=0)
    return seeds_hr


@gpu_dispatch(return_to_host=True)
def _calc_foreground(
    cellprob_hr: ArrayLike,
    cellprob_threshold: float,
) -> np.ndarray:
    """Calculate high res mask"""
    foreground_hr = cellprob_hr > cellprob_threshold
    return foreground_hr


@gpu_dispatch(return_to_host=True)
def _region_fill(
    seeds_hr: ArrayLike, foreground_hr: ArrayLike, max_expansion_steps: int = 10
) -> np.ndarray:
    """Region fill aglorithm. Runs for at most max_expansion_steps"""
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)

    # Create dilation buffer to save memory
    dilation_buffer = xp.empty_like(seeds_hr)

    for i in range(max_expansion_steps):
        ndi.grey_dilation(seeds_hr, footprint=structure, output=dilation_buffer)
        should_fill = (seeds_hr == 0) & foreground_hr & (dilation_buffer > 0)
        if not xp.any(should_fill):
            break
        seeds_hr[should_fill] = dilation_buffer[should_fill]

    # Trim any accidental overflows
    seeds_hr *= foreground_hr
    return seeds_hr


def _upsample_region_fill(
    labels_lr: np.ndarray,
    cellprob_hr: np.ndarray,
    erosion_iterations: int = 2,
    cellprob_threshold: float = 0.0,
    max_expansion_steps: int = 10,
) -> np.ndarray:
    """
    Upsample cellpose labels with a region fill algorithm using thresholded cellprob
    logits.
    """

    zoom_factors = np.array(cellprob_hr.shape) / np.array(labels_lr.shape)

    seeds_hr = _calc_seeds(labels_lr, erosion_iterations, zoom_factors)
    foregroung_hr = _calc_foreground(cellprob_hr, cellprob_threshold)
    labels_hr = _region_fill(
        seeds_hr, foregroung_hr, max_expansion_steps=max_expansion_steps
    )

    return labels_hr.astype(labels_lr.dtype)


def upsample_masks(
    root_path: Path,
    seg_target: str,
    target_spacing: tuple[int, float],
    src_spacing: tuple[int, int, int] = (100, 100, 100),
    erosion_iterations: int = 2,
    cellprob_threshold: float = 0.0,
    store_chunks: tuple[int] | None = None,
) -> None:
    """Upsample cellpose labels"""

    root = zarr.open_group(root_path, mode="a")
    labels_lr_zarr: zarr.Array = root.get(
        f"labels/{seg_target}/masks/{dirname_from_spacing(src_spacing)}"
    )
    cellprob_hr_zarr: zarr.Array = root.get(
        f"labels/{seg_target}/flows/cellprob/{dirname_from_spacing(target_spacing)}"
    )

    labels_lr: np.ndarray = labels_lr_zarr[:]
    cellprob_hr: np.ndarray = cellprob_hr_zarr[:]
    labels_hr = _upsample_region_fill(
        labels_lr,
        cellprob_hr,
        erosion_iterations=erosion_iterations,
        cellprob_threshold=cellprob_threshold,
    )

    processing = labels_lr_zarr.attrs.get("processing") + [
        {
            "step": "upsample masks",
            "erosion_iterations": erosion_iterations,
            "cellprob_threshold": cellprob_threshold,
        }
    ]

    write_zarr(
        root,
        labels_hr,
        f"labels/{seg_target}/masks/{dirname_from_spacing(target_spacing)}",
        src_zarr=labels_lr_zarr,
        spacing=target_spacing,
        processing=processing,
        zarr_chunks=store_chunks if store_chunks else labels_lr_zarr.chunks,
        inputs=[labels_lr_zarr.path, cellprob_hr_zarr.path],
    )


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
    metric = metrics.average_precision(ground_truth, predictions, threshold=thresholds)
    results_df = pd.DataFrame((thresholds, *metric)).T
    results_df.columns = [
        "iou_thresholds",
        "average_precision",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]
    return results_df


def match_predictions(ground_truth: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    """Return prediction masks with matched label indices as ground truth"""
    _, matched = metrics.mask_ious(ground_truth, predictions)
    full_range = np.unique(predictions)[1:]
    missing = np.setdiff1d(full_range, matched).tolist()
    predictions_matched = predictions.copy()
    for val in missing:
        predictions_matched[predictions_matched == val] = 2 * predictions.max() + val

    for i, val in enumerate(matched):
        predictions_matched[predictions_matched == val] = predictions.max() + i + 1

    predictions_matched[predictions_matched > 0] -= predictions.max()

    return predictions_matched


@gpu_dispatch(return_to_host=True)
def _get_edges_and_nodes(
    labels: ArrayLike, cellprob: ArrayLike, edge_map: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds adjacencies and samples cellprob and edgemap at the boundaries.
    """

    # Build 6-connectivity element
    structure = ndi.generate_binary_structure(rank=3, connectivity=1)

    # Find edges
    dilated_labels = ndi.grey_dilation(labels, footprint=structure)
    boundary_mask = labels != dilated_labels

    labels_a = to_host(labels[boundary_mask])
    labels_b = to_host(dilated_labels[boundary_mask])

    probs = to_host(cellprob[boundary_mask])
    edges = to_host(edge_map[boundary_mask])

    # Find nodes
    all_labels, pixel_counts = xp.unique(labels, return_counts=True)
    nodes = to_host(all_labels)
    counts = to_host(pixel_counts)

    return labels_a, labels_b, probs, edges, nodes, counts


def build_rag(
    labels: np.ndarray, cellprob: np.ndarray, edge_map: np.ndarray
) -> graph.RAG:
    """
    Builds a RAG of cellpose labels using GPU acceleration when possible.

    Parameters
    ----------
    labels : np.ndarray
        Integer array containing the predicted labels
    cellprob : np.ndarray
        Cellprob array containing the cell probability logits
    edge_map : np.ndarray
        Edge map normalized to [0, 1]

    Returns
    -------
    graph.RAG
        Region adjacency graph with the edge parameters
        - 'prob_weight': mean cell probability
        - 'edge_weight': mean edge probability
        - 'count': number of boundary voxels
        - 'weight': 1 - prob_weight + edge_weight
    """

    # 1. Get all node and edge data
    (labels_a, labels_b, probs, edges, node_ids, node_counts) = _get_edges_and_nodes(
        labels, cellprob, edge_map
    )

    # 2. Format Node Data
    node_attr = [
        (label, {"labels": [label], "pixel_count": count, "total_surface_area": 0})
        for label, count in zip(node_ids, node_counts)
    ]

    # 3. Format Edge Data
    df = pd.DataFrame({"prob": probs, "edge": edges})
    pair = np.sort(np.stack([labels_a, labels_b], axis=1), axis=1)
    df["label_1"] = pair[:, 0]
    df["label_2"] = pair[:, 1]

    # Aggregate probability, edge, and count
    edge_stats = (
        df.groupby(["label_1", "label_2"])
        .agg(
            prob_weight=("prob", "mean"),
            edge_weight=("edge", "mean"),
            count=("prob", "size"),
        )
        .reset_index()
    )
    edge_stats["weight"] = 1 - (edge_stats["prob_weight"] - edge_stats["edge_weight"])

    # Create the edge list for the graph constructor
    edge_list = []
    for _, row in edge_stats.iterrows():
        # Correct for 1-sided detection
        true_count = int(row["count"]) * 2
        edge_list.append(
            (
                row["label_1"],
                row["label_2"],
                {
                    "weight": row["weight"],
                    "prob_weight": row["prob_weight"],
                    "edge_weight": row["edge_weight"],
                    "count": true_count,
                },
            )
        )

    rag = graph.RAG(label_image=None, data=edge_list)
    rag.add_nodes_from(node_attr)
    return rag


def calc_surface_rag(rag: graph.RAG) -> dict[int, float]:
    """Calculate an approximation of label area from a region adjacency graph (RAG).

    Parameters
    ----------
    rag : skimage.graph.RAG
        A region adjaceny graph. The RAG should include background nodes and use only
        face connectivity (connectivity=1). The function will use the "count" edge
        parameter to calculate the total surface of each label.

    Returns
    -------
    dict[int, float]
        A dictionary in the form {label_num: total_surface}
    """
    total_surface = {n: 0 for n in rag.nodes}
    for u, v, d in rag.edges(data=True):
        c = int(d.get("count", 0))
        total_surface[u] += c
        total_surface[v] += c
    return total_surface


@gpu_dispatch(return_to_host=True)
def gaussian_edge_map(image: ArrayLike, sigma: float | int) -> ArrayLike:
    """Calculate edge map of an image using Gaussian-smoothed gradient magnitude

    The edge map is clipped to 1st and 99th percentile and normalized.
    The function automatically uses GPU acceleration when available.

    Parameters
    ----------
    image : ArrayLike
        The image to be analyzed.
    sigma : float | int
        The standard deviation of the Gaussian filter applied before gradient
        calculation.

    Returns
    -------
    ArrayLike
        The edge map of the image, normalized to [0, 1].
    """
    edge_map = ndi.gaussian_gradient_magnitude(image, sigma, np.float32)
    p1, p99 = xp.percentile(edge_map, (1, 99))
    edge_map = xp.clip((edge_map - p1) / (p99 - p1), 0, 1)
    return edge_map
