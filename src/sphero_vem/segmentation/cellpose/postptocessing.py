"""
Various functions used for postprocessing cellpose flows and masks
"""

import numpy as np
from skimage import graph
import torch
import torch.nn.functional as F
from sphero_vem.segmentation.cellpose.utils import (
    gaussian_edge_map,
    build_rag,
    calc_surface_rag,
)


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
    total_surface = calc_surface_rag(rag)

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
