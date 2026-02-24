"""
Various utility functions for cellposes
"""

from pathlib import Path
import numpy as np
from scipy.special import expit
import pandas as pd
from cellpose import metrics
import zarr
from skimage import graph
from sphero_vem.io import write_zarr
from sphero_vem.preprocessing import resample_array
from sphero_vem.utils import dirname_from_spacing
from sphero_vem.utils.accelerator import (
    xp,
    ndi,
    ArrayLike,
    gpu_dispatch,
    to_host,
)


@gpu_dispatch(return_to_host=True)
def _upsample_seeds(
    labels_lr: ArrayLike, erosion_iterations: int, zoom_factors: ArrayLike
) -> np.ndarray:
    """Zoom low-res labels to high-res seeds, with erosion to prevent bleed on zoom.

    Parameters
    ----------
    labels_lr : ArrayLike
        Low-resolution integer label array.
    erosion_iterations : int
        Number of binary erosion steps applied before zooming.
    zoom_factors : ArrayLike
        Per-axis zoom factors.

    Returns
    -------
    np.ndarray
        High-resolution seed array.
    """
    eroded = labels_lr > 0
    for _ in range(erosion_iterations):
        eroded = ndi.binary_erosion(eroded)
    seeds_lr = labels_lr * eroded
    return ndi.zoom(seeds_lr, zoom_factors, order=0)


@gpu_dispatch(return_to_host=True)
def _calc_foreground(
    cellprob_hr: ArrayLike,
    cellprob_threshold: float,
) -> np.ndarray:
    """Calculate high res mask"""
    foreground_hr = cellprob_hr > cellprob_threshold
    return foreground_hr


@gpu_dispatch(return_to_host=True)
def region_fill(
    seeds: ArrayLike, foreground: ArrayLike, max_expansion_steps: int = 10
) -> np.ndarray:
    """Expand integer seeds by iterative grey dilation, constrained to a foreground mask.

    Each iteration expands labels by 1 voxel (6-connectivity). Background regions
    act as hard barriers since expansion is strictly limited to the foreground mask.

    Parameters
    ----------
    seeds : ArrayLike
        Integer label array (0 = background).
    foreground : ArrayLike
        Boolean mask. Expansion is strictly limited to True voxels.
    max_expansion_steps : int, optional
        Maximum dilation iterations. Default 50.

    Returns
    -------
    np.ndarray
        Expanded label array, zero outside foreground.
    """
    seeds = xp.copy(seeds)
    structure = ndi.generate_binary_structure(rank=seeds.ndim, connectivity=1)
    dilation_buffer = xp.empty_like(seeds)

    for _ in range(max_expansion_steps):
        ndi.grey_dilation(seeds, footprint=structure, output=dilation_buffer)
        should_fill = (seeds == 0) & foreground & (dilation_buffer > 0)
        if not xp.any(should_fill):
            break
        seeds[should_fill] = dilation_buffer[should_fill]

    seeds *= foreground
    return seeds


def _upsample_masks_region_fill(
    labels_lr: np.ndarray,
    cellprob_hr: np.ndarray,
    erosion_iterations: int = 2,
    cellprob_threshold: float = 0.0,
    max_expansion_steps: int = 50,
) -> np.ndarray:
    """Upsample Cellpose labels using region fill constrained by cellprob logits.

    Parameters
    ----------
    labels_lr : np.ndarray
        Low-resolution integer label volume.
    cellprob_hr : np.ndarray
        High-resolution cellprob logit volume.
    erosion_iterations : int, optional
        Erosion steps before zooming seeds. Default 2.
    cellprob_threshold : float, optional
        Logit threshold for foreground mask. Default 0.0.
    max_expansion_steps : int, optional
        Maximum dilation iterations. Default 50.

    Returns
    -------
    np.ndarray
        Upsampled label volume, same shape as `cellprob_hr`.
    """
    zoom_factors = np.array(cellprob_hr.shape) / np.array(labels_lr.shape)
    seeds = _upsample_seeds(labels_lr, erosion_iterations, zoom_factors)
    foreground = cellprob_hr > cellprob_threshold
    return region_fill(seeds, foreground, max_expansion_steps).astype(labels_lr.dtype)


def upsample_masks(
    root_path: Path,
    seg_target: str,
    target_spacing: tuple[int, float],
    src_spacing: tuple[int, int, int] = (100, 100, 100),
    erosion_iterations: int = 2,
    cellprob_threshold: float = 0.0,
    store_chunks: tuple[int] | None = None,
    label_root: str | None = None,
    n_workers: int = 4,
) -> None:
    """Upsample cellpose labels"""

    root = zarr.open_group(root_path, mode="a")
    label_path = (
        f"{label_root}/labels/{seg_target}"
        if label_root is not None
        else f"labels/{seg_target}"
    )
    labels_lr_zarr: zarr.Array = root.get(
        f"{label_path}/masks/{dirname_from_spacing(src_spacing)}"
    )

    # Try to load high resolution cellprob, and calculate it if it doesn't exit
    cellprob_hr_path = (
        f"{label_path}/flows/cellprob/{dirname_from_spacing(target_spacing)}"
    )
    cellprob_hr_zarr: zarr.Array = root.get(cellprob_hr_path)
    if cellprob_hr_zarr is None:
        resample_array(
            zarr_path=root_path,
            array_path=f"{label_path}/flows/cellprob/{dirname_from_spacing(src_spacing)}",
            target_spacing=target_spacing,
            n_workers=n_workers,
        )
        cellprob_hr_zarr: zarr.Array = root.get(cellprob_hr_path)

    labels_lr: np.ndarray = labels_lr_zarr[:]
    cellprob_hr: np.ndarray = cellprob_hr_zarr[:]
    labels_hr = _upsample_masks_region_fill(
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
        f"{label_path}/masks/{dirname_from_spacing(target_spacing)}",
        src_zarr=labels_lr_zarr,
        spacing=target_spacing,
        processing=processing,
        zarr_chunks=store_chunks if store_chunks else labels_lr_zarr.chunks,
        inputs=[labels_lr_zarr.path, cellprob_hr_zarr.path],
    )


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
    Finds adjacencies and samples cellprob (logits) and edgemap at the boundaries.
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
        - 'prob_weight': mean cell probability (probability of mean logit value)
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
    edge_stats["prob_weight"] = expit(edge_stats["prob_weight"])
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


def rag_to_df(rag: graph.RAG) -> pd.DataFrame:
    """Convert a RAG to a tidy DataFrame for inspection and debugging.

    Flattens the edge data of a RAG (as produced by `build_rag`) into a
    DataFrame where each row corresponds to one edge between two adjacent
    label regions.

    Parameters
    ----------
    rag : graph.RAG
        Region adjacency graph, as returned by `build_rag`.

    Returns
    -------
    pd.DataFrame
        One row per edge with columns:

        - ``u``, ``v`` : int — the two adjacent label IDs.
        - ``weight`` : float — merge cost (1 - prob_weight + edge_weight).
        - ``prob_weight`` : float — mean cell probability across boundary voxels
          (sigmoid of mean logit).
        - ``edge_weight`` : float — mean edge map value across boundary voxels.
        - ``count`` : int — number of boundary voxels between the two regions.

    See Also
    --------
    build_rag : Constructs the RAG from Cellpose labels, cellprob logits, and an edge map.
    """
    return pd.DataFrame([{"u": u, "v": v, **d} for u, v, d in rag.edges(data=True)])
