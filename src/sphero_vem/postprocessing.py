"""
Postprocessing functions
"""

from pathlib import Path
import itertools
import numpy as np
import torch
import torch.fft
import torch.nn.functional as F
from skimage import graph
from skimage.morphology import ball
import zarr
import pandas as pd
from sphero_vem.utils.accelerator import (
    xp,
    ndi,
    ski,
    ArrayLike,
    gpu_dispatch,
    to_device,
    to_host,
)
from sphero_vem.io import read_tensor, write_image


@gpu_dispatch(return_to_host=True)
def _get_edges_and_nodes(
    labels: ArrayLike, cellprob: ArrayLike, edge_map: ArrayLike
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds adjacencies and samples cellprob and edgemap at the boundaries.
    """

    # Build 6-connectivity element
    structure = xp.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=xp.bool_,
    )

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
    Builds a RAG with edge parameters:
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


def merge_labels(
    labels: np.ndarray,
    cellprob: np.ndarray,
    image: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
    rel_contact_thresh: float = 0.1,
    weight_thresh: float = 0.15,
    sigma: int = 1,
) -> tuple[np.ndarray, graph.RAG]:
    """
    Merge adjacent labeled regions based on boundary strength and relative contact area.

    This function constructs a region adjacency graph (RAG) from an integer-labeled
    segmentation and an edge strength map, annotates edges with the relative contact
    area between regions, preventing merging for small contacts, and then
    performs a threshold-based region merging.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer-labeled segmentation image
    cellprob : numpy.ndarray
        Cell probability map (logits) output by cellpose.
    image : numpy.ndarray, optional
        Grayscale image used to compute an edge map when `edge_map` is not provided.
        Required if `edge_map` is None.
    edge_map : numpy.ndarray, optional
        Precomputed normalized edge strength map (float array, expected in range [0, 1])
        used for building the RAG. If provided, `image` is ignored.
    rel_contact_thresh : float, optional
        Minimum relative contact area (fraction of region perimeter in contact with
        a neighbor) required for an edge to be considered mergeable. Edges with
        maximal relative contact < this threshold will have their 'weight' set to
        1.0 to discourage merging. Default is 0.1.
    weight_thresh : float, optional
        Threshold applied to the RAG by `graph.cut_threshold` to perform merging.
        Edges with weight (or other computed metric) below this threshold will be
        merged. Weight is calculated as (1 - cellprob) + edge_map. Default is 0.15.
    sigma : int, optional
        Standard deviation for the Gaussian kernel used when computing the image
        gradient magnitude if `edge_map` is not provided. Default is 1.

    Returns
    -------
    merged : numpy.ndarray
        Label image after region merging. Same shape as `labels`. Label values may
        be relabeled by the merging procedure.
    rag : graph.RAG
        The region adjacency graph built from the original inputs

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
        edge_map = gaussian_edge_map(image, sigma=sigma)

    rag = build_rag(labels, cellprob, edge_map)
    total_surface = calc_surface(rag)

    def _rel_contact(node: int, edge_data: dict) -> float:
        """Edge contact relative to total label surface"""
        edge_contact = int(edge_data.get("count", 0))
        tot_surface = total_surface.get(node, 0) or 1
        return float(edge_contact / tot_surface)

    # Calculate edge contact area relative to total label area
    for u, v, d in rag.edges(data=True):
        rel_u = _rel_contact(u, d)
        rel_v = _rel_contact(v, d)
        d["rel_contact_max"] = max(rel_u, rel_v)

    # Make edge unmergeable if the contact area is below the threshold, then optionally
    # consider only edges that connect to at least a node with a sphericity lower than
    # the set threshold.
    # Edges with the background node are also set to unmergeable.
    rag_th = rag.copy()
    for u, v, d in rag_th.edges(data=True):
        if (u == 0) or (v == 0):
            d["weight"] = 1.0
        elif d["rel_contact_max"] < rel_contact_thresh:
            d["weight"] = 1.0

    merged = graph.cut_threshold(labels.copy(), rag_th, weight_thresh, in_place=False)

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


@gpu_dispatch()
def expand_labels(labels: ArrayLike, grow_dist: int = 1) -> ArrayLike:
    """Expand labels by grow_dist (in voxels) while preserving instance labels"""
    out = labels.copy()
    struct = xp.asarray(ndi.generate_binary_structure(3, 1))
    for _ in range(grow_dist):
        foreground = out > 0
        dilated_foreground = ndi.binary_dilation(foreground, structure=struct)
        new_voxels = dilated_foreground & (~foreground)
        if not new_voxels.any():
            break

        neigh_labels = []
        for axis, shift in itertools.product((0, 1, 2), (-1, 1)):
            shifted = xp.roll(out, shift=shift, axis=axis)
            neigh_labels.append(shifted)
        neigh_stack = xp.stack(neigh_labels, axis=0)
        propagated = neigh_stack.max(axis=0)
        out = xp.where(new_voxels, propagated, out)
    return out


@gpu_dispatch(return_to_host=True)
def fill_internal_seams(
    labels: ArrayLike, close_radius: int = 2, grow_dist: int = 3
) -> ArrayLike:
    """Fill internal seams between labels

    This function should be used with caution when there are separate instances close
    together, as it might lead to incorrect instance edges.
    The function automatically uses GPU acceleration when available.

    Parameters
    ----------
    labels : ArrayLike
        An array of integer labels
    close_radius : int
        The radius of a ball element used for morphological closing of the image.
        Default is 2.
    grow_dist : int
        The distance in voxels use to expand the labels for assigning the newly closed
        regions to the correct instance. Default is 3.

    Returns
    -------
    ArrayLike
        The labels with closed internal seams.
    """
    binary = labels > 0

    selem = to_device(ball(close_radius))
    binary_closed = ndi.binary_closing(binary, structure=selem)
    binary_closed = xp.logical_or(binary_closed, binary)

    # Expand instance labels to new voxels and keep only true expanded
    grown_gpu = expand_labels(labels, grow_dist=grow_dist)
    return xp.where(binary_closed, grown_gpu, xp.zeros_like(grown_gpu))


def upscale_labels(label_path: Path, dest_path: Path, shape: tuple) -> None:
    """Takes a torch tensor as input and returns a numpy array ready for saving"""
    labels = read_tensor(label_path)
    while labels.dim() < 5:
        labels = labels.unsqueeze(0)
    upscaled_tensor = F.interpolate(labels, size=shape, mode="nearest")
    upscaled_array = upscaled_tensor.squeeze().numpy()
    # Infer data type
    if upscaled_array.max() <= 255:
        dtype = np.uint8
    else:
        dtype = np.uint16
    write_image(dest_path, upscaled_array.astype(dtype), compressed=True)


def seg_params_zarr(arr: zarr.Array) -> dict:
    """Get segmentation parameters from a Zarr array. Returns empty dict if not found"""
    for step in arr.attrs.get("processing", {}):
        if step.get("step") == "segmentation":
            return step
    return {}


def _get_curl_free_component(
    input: torch.Tensor,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    padding: tuple[int, int, int] = (0, 0, 0),
) -> torch.Tensor:
    """
    Computes the curl-free (irrotational) component of a 3D vector field unsing a
    FFT-based approach.

    Parameters
    ----------
    input_vec : torch.Tensor
        The input 3D vector field (3, Z, Y, X)
    spacing : tuple[float, float, float], optional
        Grid spacing (dz, dy, dx). Default is (1.0, 1.0, 1.0)
    padding : tuple[int, int, int], optional)
        Zero-padding for (Z, Y, X) to add to each side.

    Returns
    --------
    torch.Tensor
        The curl-free (irrotational) component (3, Z, Y, X).
    """

    if input.dim() != 4 or input.shape[0] != 3:
        raise ValueError("Input tensor must have shape (3, Z, Y, X)")

    device = input.device
    dz, dy, dx = spacing
    pad_z, pad_y, pad_x = padding

    # Padding
    if any(p > 0 for p in padding):
        pad_dims = (pad_x, pad_x, pad_y, pad_y, pad_z, pad_z)
        vec_padded = F.pad(input, pad_dims, mode="constant", value=0.0)
    else:
        vec_padded = input.clone()

    z_pad, y_pad, x_pad = vec_padded.shape[1:]

    # Create k-vectors and k-squared
    k_z_freqs = torch.fft.fftfreq(z_pad, d=dz, device=device)
    k_y_freqs = torch.fft.fftfreq(y_pad, d=dy, device=device)
    k_x_freqs = torch.fft.fftfreq(x_pad, d=dx, device=device)

    Kz, Ky, Kx = torch.meshgrid(k_z_freqs, k_y_freqs, k_x_freqs, indexing="ij")

    # Stack k-vectors into a (3, Z, Y, X) tensor
    k_vec = torch.stack([Kz, Ky, Kx], dim=0)
    del Kz, Ky, Kx, k_z_freqs, k_y_freqs, k_x_freqs
    k_sq = torch.sum(k_vec**2, dim=0)
    # Avoid division by zero at k=0
    k_sq[0, 0, 0] = 1.0

    # Forward FFT
    vec_k_padded = torch.fft.fftn(vec_padded, dim=(1, 2, 3))
    del vec_padded  # Free padded real-space tensor

    # Projection in Fourier space
    # (k · V_k)
    k_dot_Vk = torch.sum(k_vec * vec_k_padded, dim=0)
    del vec_k_padded
    projection_scalar = k_dot_Vk / k_sq
    del k_dot_Vk, k_sq

    # Calculate the curl-free component in k-space
    # cf_k = projection_scalar * k_vec
    cf_k = projection_scalar.unsqueeze(0) * k_vec
    del projection_scalar, k_vec

    # Inverse FFT
    cf_padded = torch.fft.ifftn(cf_k, dim=(1, 2, 3))
    del cf_k
    cf_padded_real = cf_padded.real

    if any(p > 0 for p in padding):
        cf_component = cf_padded_real[
            :, pad_z : z_pad - pad_z, pad_y : y_pad - pad_y, pad_x : x_pad - pad_x
        ].clone()
    else:
        cf_component = cf_padded_real.clone()

    return cf_component


def decompose_flow(
    dP: np.ndarray,
    z_pad_fraction: float = 0.3,
    device: torch.DeviceObjType = torch.device("cpu"),
) -> np.ndarray[np.float32]:
    """Decompose dP output from cellpose into its curl-free component.

    This is useful to remove banding artifacts that appear when nuclei shapes
    are very complex and have "holes" or large convexities. It is detrimental when
    working with more regular-shaped cells with many contact points.

    Parameters
    ----------
    dP : np.ndarray
        A numpy array of shape (3, Z, Y, X) containing cellpose flows
    z_pad_fraction : float
        The fraction of the total Z to be padded on both sides. This is important
        to avoid ghosting artifacts. Default is 0.3
    device : torch.device
        Torch device where the computation should be executed. Default is
        torch.device("cpu")

    Returns
    -------
    np.ndarray
        The curl-free component of the flows. This is returned in np.float32.
    """
    dP = torch.from_numpy(dP).to(device=device, dtype=torch.float32)
    pad_z_amount = int(dP.shape[1] * z_pad_fraction)
    padding_tuple = (pad_z_amount, 0, 0)

    cf_component = _get_curl_free_component(
        dP,
        padding=padding_tuple,
    )
    return cf_component.cpu().numpy()


@gpu_dispatch(return_to_host=True)
def median_filter(cellprob_array: ArrayLike, size: int = 3) -> np.ndarray:
    """
    Applies a 3D median filter, using GPU acceleration if possible.
    """
    cellprob_smoothed = ndi.median_filter(cellprob_array, size=size, mode="nearest")
    return cellprob_smoothed


@gpu_dispatch(return_to_host=True)
def filter_and_relabel(labels: ArrayLike, min_size: int) -> np.ndarray:
    """
    Remove components smaller than `min_size` voxels and relabel remaining components
    to contiguous IDs (0=background, 1..K)

    Parameters
    ----------
    labels : np.ndarray
        Integer label volume, background must be 0.
    min_size : int
        Minimum number of voxels for a component to be kept.

    Returns
    -------
    labels_out : np.ndarray
        New label volume with small components removed and contiguous IDs.
    """
    # Flatten array
    flat = labels.ravel()

    max_label = int(flat.max())
    counts = xp.bincount(flat, minlength=max_label + 1)

    # Keep labels with at least min_size voxels, except background
    keep = counts >= min_size
    keep[0] = False
    keep_voxel = keep[flat]

    filtered = xp.where(keep_voxel, flat, 0).reshape(labels.shape)
    del keep_voxel, labels

    relabeled = ski.segmentation.relabel_sequential(filtered)
    return relabeled[0]


@gpu_dispatch(return_to_host=True)
def binary_closing(binary_image: ArrayLike, radius: int = 1) -> np.ndarray:
    """Mrophological opening with stucturing element with given connectivity"""
    struct = ski.morphology.ball(radius=radius)
    return ndi.binary_closing(binary_image, structure=struct, border_value=0)
