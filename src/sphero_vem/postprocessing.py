"""
Postprocessing functions
"""

import numpy as np
import scipy.ndimage as ndi
from skimage import graph


def merge_labels(
    labels: np.ndarray,
    image: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
    rel_contact_thresh: float = 0.1,
    edge_thresh: float = 0.15,
    sigma: int = 1,
    connectivity: int = 2,
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
          - 'weight': may be set to 1.0 for edges that should not be merged due to
            small contact area; other weight values are those produced by the RAG
            construction or left for downstream thresholds.

    Raises
    ------
    ValueError
        If neither `edge_map` nor `image` is provided (an edge map is required to
        construct the RAG).
    """

    if edge_map is None:
        if image is not None:
            raise ValueError(
                "When not supplying a precomputed edge map, "
                "a valid image must be specified"
            )
        edge_map = ndi.gaussian_gradient_magnitude(image, sigma, np.float32)
        p1, p99 = np.percentile(edge_map, (1, 99))
        edge_map = np.clip((edge_map - p1) / (p99 - p1), 0, 1)

    rag = graph.rag_boundary(labels, edge_map, connectivity=connectivity)

    # Calculate total surface per node
    total_surface = {n: 0 for n in rag.nodes}
    for u, v, d in rag.edges(data=True):
        c = int(d.get("count", 0))
        total_surface[u] += c
        total_surface[v] += c

    if remove_background and 0 in rag:
        rag.remove_node(0)

    # Calculate edge contact area relative to total label area
    for u, v, d in rag.edges(data=True):
        iface = int(d.get("count", 0))
        su = total_surface.get(u, 0) or 1
        sv = total_surface.get(v, 0) or 1
        rel_u = iface / su
        rel_v = iface / sv
        rel_max = rel_u if rel_u >= rel_v else rel_v
        d["rel_contact_max"] = float(rel_max)

        # Make edge unmergeable if the contact area is below the threshold
        if rel_max < rel_contact_thresh:
            d["weight"] = 1.0

    merged = graph.cut_threshold(labels.copy(), rag, edge_thresh, in_place=False)

    return merged, rag
