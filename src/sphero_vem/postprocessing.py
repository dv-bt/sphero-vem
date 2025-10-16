"""
Postprocessing functions
"""

import numpy as np
import scipy.ndimage as ndi
from skimage import graph, segmentation


def merge_labels(
    labels: np.ndarray,
    image: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
    rel_contact_thresh: float = 0.1,
    edge_thresh: float = 0.15,
    sphericity_thresh: float | None = None,
    sigma: int = 1,
    remove_background: bool = True,
) -> tuple[np.ndarray, graph.RAG]:
    """
    Merge adjacent labeled regions based on boundary strength and relative contact area.
    This function constructs a region adjacency graph (RAG) from an integer-labeled
    segmentation and an edge strength map, annotates edges with the relative contact
    area between regions, preventing merging for small contacts, and then
    performs a threshold-based region merging. If no precomputed edge map is
    provided, an edge map is computed from a supplied grayscale image.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer-labeled segmentation image. Labels should be non-negative integers
        where each connected component with the same integer value represents a
        region
    image : numpy.ndarray, optional
        Grayscale image used to compute an edge map when `edge_map` is not provided.
        If given, the function computes the Gaussian gradient magnitude of this
        image and rescales it using the 1st and 99th percentiles to yield values in
        [0, 1]. Required if `edge_map` is None.
    edge_map : numpy.ndarray, optional
        Precomputed normalized edge strength map (float array, expected in range [0, 1])
        used for building the RAG. If provided, `image` is ignored.
    rel_contact_thresh : float, optional
        Minimum relative contact area (fraction of region perimeter in contact with
        a neighbor) required for an edge to be considered mergeable. Edges with
        maximal relative contact < this threshold will have their 'weight' set to
        1.0 to discourage merging. Default is 0.1.
    edge_thresh : float, optional
        Threshold applied to the RAG by `graph.cut_threshold` to perform merging.
        Edges with weight (or other computed metric) below this threshold will be
        merged. Default is 0.15.
    sphericity_thresh: float | None, optional
        Minimum node sphericity to consider an edge a candidate for merging. Values
        should be in the range [0, 1]. Edges where both nodes have a sphericity larger
        than this value will have their weights set to 1. If set to None, no sphericity
        check is done. Default is None.
    sigma : int, optional
        Standard deviation for the Gaussian kernel used when computing the image
        gradient magnitude if `edge_map` is not provided. Default is 1.
    connectivity : int, optional
        Pixel connectivity used when constructing the RAG (e.g. 1 for 4-connectivity,
        2 for 8-connectivity in 2D). Forwarded to the underlying RAG constructor.
        Default is 2.
    remove_background : bool, optional
        Remove background label (0). Default is True.

    Returns
    -------
    merged : numpy.ndarray
        Label image after region merging. Same shape as `labels`. Label values may
        be relabeled by the merging procedure.
    rag : graph.RAG
        The region adjacency graph built from the original inputs. Edge attributes
        are updated with:
          - 'count': original boundary pixel count (if present from rag construction)
          - 'rel_contact_max': maximum of the two relative contact areas between the
            pair of adjacent regions (float in [0, 1])
          - 'weight': original edge weights.

    Raises
    ------
    ValueError
        If neither `edge_map` nor `image` is provided (an edge map is required to
        construct the RAG).
    """

    if edge_map is None:
        if image is None:
            raise ValueError(
                "When not supplying a precomputed edge map, "
                "a valid image must be specified"
            )
        edge_map = ndi.gaussian_gradient_magnitude(image, sigma, np.float32)
        p1, p99 = np.percentile(edge_map, (1, 99))
        edge_map = np.clip((edge_map - p1) / (p99 - p1), 0, 1)

    # Relabel labels sequentially
    labels, fwd, inv = segmentation.relabel_sequential(labels)
    rag = graph.rag_boundary(labels, edge_map, connectivity=1)

    # Calculate total surface per node
    total_surface = calc_surface(rag)
    label_volume = calc_volume(labels)

    if remove_background and 0 in rag:
        rag.remove_node(0)

    # Calculate edge contact area relative to total label area and minimum sphericity
    # per label
    for u, v, d in rag.edges(data=True):
        iface = int(d.get("count", 0))
        su = total_surface.get(u, 0) or 1
        sv = total_surface.get(v, 0) or 1
        rel_u = iface / su
        rel_v = iface / sv
        d["rel_contact_max"] = float(max(rel_u, rel_v))

        vu = label_volume.get(u)
        vv = label_volume.get(v)
        sph_u = calc_sphericity(su, vu)
        sph_v = calc_sphericity(sv, vv)
        d["min_sphericity"] = min(sph_u, sph_v)

    # Make edge unmergeable if the contact area is below the threshold, then optionally
    # consider only edges that connect to at least a node with a sphericity lower than
    # the set threshold
    rag_th = rag.copy()
    for _, _, d in rag_th.edges(data=True):
        if d["rel_contact_max"] < rel_contact_thresh:
            d["weight"] = 1.0
        elif sphericity_thresh and (d["min_sphericity"] > sphericity_thresh):
            d["weight"] = 1.0

    merged = graph.cut_threshold(labels.copy(), rag_th, edge_thresh, in_place=False)

    return merged, rag


def calc_surface(rag: graph.RAG) -> dict[int, float]:
    """Calculate an approximation of label area from a region adjacency graph (RAG).
    The RAG should include background nodes and use only face connectivity
    (connectivity=1).
    """
    total_surface = {n: 0 for n in rag.nodes}
    for u, v, d in rag.edges(data=True):
        c = int(d.get("count", 0))
        total_surface[u] += c
        total_surface[v] += c
    return total_surface


def calc_volume(labels: np.ndarray) -> dict[int, float]:
    """Calculate label volume from a labeled array"""
    counts = np.bincount(labels.ravel())
    bins = np.unique(labels)
    return {bin: count for bin, count in zip(bins, counts)}


def calc_sphericity(area: float, volume: float) -> float:
    """Calculate sphericity from object area and volume"""
    return (np.pi ** (1 / 3)) * ((6 * volume) ** (2 / 3)) / area
