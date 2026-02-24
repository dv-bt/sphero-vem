"""
Various functions used for postprocessing cellpose flows and masks
"""

import networkx as nx
import numpy as np
from skimage import graph
import torch
import torch.nn.functional as F
from sphero_vem.utils import vprint
from sphero_vem.segmentation.cellpose.utils import (
    gaussian_edge_map,
    build_rag,
    calc_surface_rag,
)


def _merge_by_threshold(
    labels: np.ndarray,
    rag: graph.RAG,
    weight_thresh: float,
    rel_contact_thresh: float,
) -> np.ndarray:
    """Merge RAG nodes connected by edges that pass both threshold checks.

    An edge is kept (allows merging) when ALL conditions are met:
    - Neither node is background (label 0)
    - Relative contact area >= rel_contact_thresh
    - Edge weight <= weight_thresh
    """
    rag_cut = rag.copy()
    to_remove = [
        (u, v)
        for u, v, d in rag_cut.edges(data=True)
        if u == 0
        or v == 0
        or d["rel_contact_max"] < rel_contact_thresh
        or d["weight"] > weight_thresh
    ]
    rag_cut.remove_edges_from(to_remove)

    map_array = np.arange(labels.max() + 1, dtype=labels.dtype)
    for i, nodes in enumerate(nx.connected_components(rag_cut)):
        for node in nodes:
            for label in rag_cut.nodes[node]["labels"]:
                map_array[label] = i

    return map_array[labels]


def merge_labels(
    labels: np.ndarray,
    cellprob: np.ndarray,
    image: np.ndarray | None = None,
    edge_map: np.ndarray | None = None,
    rel_contact_thresh: float = 0.1,
    weight_thresh: float = 0.15,
    sigma: int = 1,
    verbose: bool = False,
) -> tuple[np.ndarray, graph.RAG]:
    """
    Merge adjacent labeled regions based on boundary strength and relative contact area.

    This function constructs a region adjacency graph (RAG) from an integer-labeled
    segmentation and an edge strength map, annotates edges with the relative contact
    area between regions, and performs threshold-based region merging. Merging is
    applied iteratively: after each round the RAG is rebuilt from the updated labels
    so that newly eligible merges (e.g. due to increased contact area) are captured.
    The process stops when no further merges occur.

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
        a neighbor) required for an edge to be considered mergeable. Edges whose
        maximal relative contact is below this value are excluded from merging.
        Default is 0.1.
    weight_thresh : float, optional
        Maximum edge weight for merging. Edges with weight at or below this
        threshold will be merged. Weight is calculated as
        (1 - cellprob) + edge_map. Default is 0.15.
    sigma : int, optional
        Standard deviation for the Gaussian kernel used when computing the image
        gradient magnitude if `edge_map` is not provided. Default is 1.
    verbose : bool, optional
        Toogles verbose output, useful for debugging unexpected behavior.
        Default is False.

    Returns
    -------
    merged : numpy.ndarray
        Label image after region merging. Same shape as `labels`. Label values may
        be relabeled by the merging procedure.
    rag : graph.RAG
        The region adjacency graph from the final iteration (stable state).

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

    current = labels
    rag_orig = build_rag(current, cellprob, edge_map)
    i = 0
    vprint(f"Unique labels: {np.unique(current).size}", verbose)
    while True:
        rag = rag_orig if i == 0 else build_rag(current, cellprob, edge_map)
        i += 1
        total_surface = calc_surface_rag(rag)

        for u, v, d in rag.edges(data=True):
            edge_contact = int(d.get("count", 0))
            rel_u = edge_contact / (total_surface.get(u, 0) or 1)
            rel_v = edge_contact / (total_surface.get(v, 0) or 1)
            d["rel_contact_max"] = max(rel_u, rel_v)

        merged = _merge_by_threshold(current, rag, weight_thresh, rel_contact_thresh)
        vprint(
            f"Iteration {i}: Unique labels after merging: {np.unique(merged).size}",
            verbose,
        )

        if np.unique(merged).size == np.unique(current).size:
            vprint(f"No improvement, stop merging at iteration {i}", verbose)
            break
        current = merged

    return merged, rag


def _get_curl_free_component(
    input: torch.Tensor,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_padding: int = 0,
) -> torch.Tensor:
    """
    Compute the curl-free (irrotational) component of a 3D vector field.

    Uses FFT-based Helmholtz-Hodge decomposition with finite-difference
    wavenumbers, following Bhatia et al. (2013) and Hinkle et al. (2009).

    Parameters
    ----------
    input : torch.Tensor
        Input 3D vector field with shape (3, Z, Y, X).
    spacing : tuple[float, float, float], optional
        Grid spacing (dz, dy, dx).
    z_padding : int, optional
        Zero-padding to add to each side of Z dimension.

    Returns
    -------
    torch.Tensor
        Curl-free (irrotational) component with shape (3, Z, Y, X).

    References
    ----------
    .. [1] Bhatia et al., "The Helmholtz-Hodge Decomposition—A Survey",
           IEEE TVCG, 2013. DOI:10.1109/TVCG.2012.316
    .. [2] Hinkle et al., "4D MAP Image Reconstruction Incorporating
           Organ Motion", IPMI, 2009. DOI:10.1007/978-3-642-02498-6_56
    """
    if input.dim() != 4 or input.shape[0] != 3:
        raise ValueError("Input must have shape (3, Z, Y, X)")

    device = input.device
    dz, dy, dx = spacing

    if z_padding > 0:
        input = F.pad(
            input,
            (0, 0, 0, 0, z_padding, z_padding),
            mode="constant",
            value=0.0,
        )

    _, Nz, Ny, Nx = input.shape

    W_vec, W_sq = _build_wavenumbers(Nz, Ny, Nx, dz, dy, dx, device)
    V_cf_hat = _project_curl_free(input, W_vec, W_sq)
    V_cf = torch.fft.ifftn(V_cf_hat, dim=(1, 2, 3)).real

    if z_padding > 0:
        V_cf = V_cf[:, z_padding:-z_padding, :, :].clone()

    return V_cf


def _build_wavenumbers(
    Nz: int,
    Ny: int,
    Nx: int,
    dz: float,
    dy: float,
    dx: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build finite-difference wavenumber arrays."""

    freq_z = torch.fft.fftfreq(Nz, d=1.0, device=device)
    freq_y = torch.fft.fftfreq(Ny, d=1.0, device=device)
    freq_x = torch.fft.fftfreq(Nx, d=1.0, device=device)

    Fz, Fy, Fx = torch.meshgrid(freq_z, freq_y, freq_x, indexing="ij")

    Wz = torch.sin(2 * torch.pi * Fz) / dz
    Wy = torch.sin(2 * torch.pi * Fy) / dy
    Wx = torch.sin(2 * torch.pi * Fx) / dx

    W_vec = torch.stack([Wz, Wy, Wx], dim=0)
    W_sq = Wz**2 + Wy**2 + Wx**2

    return W_vec, W_sq


def _project_curl_free(
    input: torch.Tensor,
    W_vec: torch.Tensor,
    W_sq: torch.Tensor,
) -> torch.Tensor:
    """Project vector field onto curl-free component in Fourier space."""

    V_hat = torch.fft.fftn(input, dim=(1, 2, 3))
    W_dot_V = torch.sum(W_vec * V_hat, dim=0)

    # Handle DC component
    W_sq[0, 0, 0] = 1.0
    projection = W_dot_V / W_sq
    projection[0, 0, 0] = 0.0

    return projection.unsqueeze(0) * W_vec


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
    dP_tensor = torch.from_numpy(dP).to(device=device, dtype=torch.float32)
    z_padding = int(dP_tensor.shape[1] * z_pad_fraction)

    cf_component = _get_curl_free_component(
        input=dP_tensor,
        z_padding=z_padding,
    )

    # Move result to CPU and convert to numpy
    result = cf_component.cpu().numpy()

    # Explicitly free GPU memory
    del dP_tensor, cf_component
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result
