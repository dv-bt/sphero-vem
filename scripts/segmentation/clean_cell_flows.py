"""
Smoothen flows and probability output from cellpose segmentation.

TODO: these functions should be incorporated in the main API, perhaps even as a final
call during segmentation.
"""

from pathlib import Path
from urllib.parse import urlparse
import numpy as np
from tqdm import tqdm
import zarr
import torch
import torch.fft
import torch.nn.functional as F
from zarr.codecs import BloscCodec, BloscShuffle
from sphero_vem.utils.accelerator import ArrayLike, ndi, gpu_dispatch


def seg_params_zarr(arr: zarr.Array) -> dict:
    """Get segmentation parameters from a Zarr array. Returns empty dict if not found"""
    for step in arr.attrs.get("processing", {}):
        if step.get("step") == "segmentation":
            return step
    return {}


def get_curl_free_component(
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
    dP: np.ndarray, z_pad_fraction: float = 0.3
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

    Returns
    -------
    np.ndarray
        The curl-free component of the flows. This is returned in np.float32.
    """
    dP = torch.from_numpy(dP).to(device="cuda", dtype=torch.float32)
    pad_z_amount = int(dP.shape[1] * z_pad_fraction)
    padding_tuple = (pad_z_amount, 0, 0)

    cf_component = get_curl_free_component(
        dP,
        padding=padding_tuple,
    )
    return cf_component.cpu().numpy()


def smoothen_flows(zarr_group: zarr.Group, z_pad_fraction=0.3) -> None:
    """Smoothen flow gradients"""
    dP_zarr = zarr_group["dP"]
    dP_numpy = dP_zarr[:]
    dP_smooth = decompose_flow(dP_numpy, z_pad_fraction=z_pad_fraction)

    processing = [{"step": "flow decomposition", "z_pad_fraction": z_pad_fraction}]

    # Save array
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
    dP_smooth_zarr = zarr_group.create_array(
        "dP-smooth",
        shape=dP_smooth.shape,
        chunks=dP_zarr.chunks,
        dtype="f2",
        compressors=compressor,
    )
    dP_smooth_zarr[...] = dP_smooth
    dP_smooth_zarr.attrs["spacing"] = dP_zarr.attrs["spacing"]
    dP_smooth_zarr.attrs["processing"] = dP_zarr.attrs["processing"] + processing
    dP_smooth_zarr.attrs["inputs"] = urlparse(str(dP_zarr.store_path)).path


@gpu_dispatch(return_to_host=True)
def median_filter(cellprob_array: ArrayLike, size: int = 3) -> np.ndarray:
    """
    Applies a 3D median filter, using GPU acceleration if possible.
    """
    cellprob_smoothed = ndi.median_filter(cellprob_array, size=size)
    return cellprob_smoothed


def smoothen_cellprob(zarr_group: zarr.Group, size: int = 3) -> None:
    """Smoothen cellprob by applying a median filter of given size"""
    cellprob_zarr = zarr_group["cellprob"]
    # Cellprob is stored as (1, Z, Y, X) but we need (Z, Y, X)
    cellprob_numpy = cellprob_zarr[0]
    cellprob_smooth = median_filter(cellprob_numpy, size=size)

    processing = [{"step": "median filter", "size": size}]

    # Save array
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
    cellprob_smooth_zarr = zarr_group.create_array(
        "cellprob-smooth",
        shape=(1, *cellprob_smooth.shape),
        chunks=cellprob_zarr.chunks,
        dtype="f2",
        compressors=compressor,
    )
    cellprob_smooth_zarr[0, ...] = cellprob_smooth
    cellprob_smooth_zarr.attrs["spacing"] = cellprob_zarr.attrs["spacing"]
    cellprob_smooth_zarr.attrs["processing"] = (
        cellprob_zarr.attrs["processing"] + processing
    )
    cellprob_smooth_zarr.attrs["inputs"] = urlparse(str(cellprob_zarr.store_path)).path


def main():
    for seg_target in tqdm(["cells", "nuclei"]):
        root = Path(f"data/processed/segmented/Au_01-vol_01/{seg_target}")
        flows_path = root / f"Au_01-vol_01-{seg_target}-flows.zarr"
        flows = zarr.open_group(flows_path, mode="r+")

        # Process flows
        if seg_target == "nuclei":
            smoothen_flows(flows)

        # Process cell probability
        smoothen_cellprob(flows)


if __name__ == "__main__":
    main()
