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
from sphero_vem.utils.accelerator import xp, ndi, ArrayLike, gpu_dispatch, to_device
from sphero_vem.io import read_tensor, write_image


def merge_labels(
    labels: np.ndarray,
    image: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
    rel_contact_thresh: float = 0.1,
    edge_thresh: float = 0.15,
    sphericity_thresh: float | None = None,
    sigma: int = 1,
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
        edge_map = gaussian_edge_map(image, sigma=sigma)

    rag = graph.rag_boundary(labels, edge_map, connectivity=1)

    # Calculate total surface per node
    total_surface = calc_surface(rag)
    label_volume = calc_volume(labels)

    def _rel_contact(node: int, edge_data: dict) -> float:
        """Edge contact relative to total label surface"""
        edge_contact = int(edge_data.get("count", 0))
        tot_surface = total_surface.get(node, 0) or 1
        return float(edge_contact / tot_surface)

    def _sphericity(node: int) -> float:
        """Label sphericity. Assign sphericity np.nan to background"""
        surface = total_surface.get(node, 0) or 1
        volume = label_volume.get(node)
        return calc_sphericity(surface, volume) if node != 0 else np.nan

    # Calculate edge contact area relative to total label area and minimum sphericity
    # per label
    for u, v, d in rag.edges(data=True):
        rel_u = _rel_contact(u, d)
        rel_v = _rel_contact(v, d)
        d["rel_contact_max"] = max(rel_u, rel_v)

        sph_u = _sphericity(u)
        sph_v = _sphericity(v)
        d["min_sphericity"] = np.nanmin([sph_u, sph_v])

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
    cellprob_smoothed = ndi.median_filter(cellprob_array, size=size)
    return cellprob_smoothed
