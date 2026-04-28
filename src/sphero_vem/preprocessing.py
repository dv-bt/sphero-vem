"""
Functions for preprocessing images.
"""

import warnings
from pathlib import Path
from typing import Literal
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import resize
import zarr
import dask
import dask.array as da
import dask_image
import dask_image.ndinterp
from dask.diagnostics import ProgressBar
from sphero_vem.utils import dirname_from_spacing, create_ome_multiscales


def create_pyramid(
    image: torch.Tensor, num_levels: int, factor: int
) -> list[torch.tensor]:
    """Build a multi-resolution image pyramid.

    Parameters
    ----------
    image : torch.Tensor
        Input image tensor at full resolution.
    num_levels : int
        Total number of pyramid levels, including the full-resolution image.
    factor : int
        Downsampling factor between consecutive levels.

    Returns
    -------
    list[torch.Tensor]
        List of image tensors ordered from coarsest to finest resolution.
    """
    pyramid = [image]
    for _ in range(num_levels - 1):
        image = resize(image, image.shape[-1] // factor)
        pyramid.append(image)
    return list(reversed(pyramid))


def downscale_tensor(
    image: torch.Tensor, factor: int, mode: str = "bilinear"
) -> torch.tensor:
    """Downscale a tensor or batch of tensors by an integer factor.

    Parameters
    ----------
    image : torch.Tensor
        Input tensor of shape ``(..., H, W)``. Unsqueezed to 4-D internally
        if necessary before interpolation.
    factor : int
        Integer downsampling factor. Output spatial dimensions are
        ``H // factor`` × ``W // factor``.
    mode : str, optional
        Interpolation mode passed to ``torch.nn.functional.interpolate``.
        Default is ``"bilinear"``. Use ``"nearest"`` for label maps.

    Returns
    -------
    torch.Tensor
        Downscaled tensor with the same number of dimensions as the input.
    """
    n_dim = image.dim()
    while image.dim() < 4:
        image = image.unsqueeze(0)
    if mode == "nearest":
        image_ds: torch.Tensor = F.interpolate(
            image,
            scale_factor=1 / factor,
            mode=mode,
        )
    else:
        image_ds: torch.Tensor = F.interpolate(
            image,
            scale_factor=1 / factor,
            mode=mode,
            align_corners=False,
            antialias=True,
        )
    # Make sure that output has the same number of dimensions of input
    while image_ds.dim() > n_dim:
        image_ds = image_ds.squeeze(0)
    return image_ds


def resample_array(
    zarr_path: Path,
    array_path: str,
    target_spacing: tuple[int, int, int],
    order: int = 1,
    zarr_chunks: tuple[int, int, int] = (1, 1024, 1024),
    n_workers: int = 4,
) -> None:
    """Resample an array in a Zarr archive to the target voxel spacing.

    Uses a lazy Gaussian pre-blur followed by affine transform via dask_image,
    keeping memory usage bounded to chunk size throughout. Anti-aliasing is
    applied only along downsampled axes, mirroring skimage.transform.resize.
    Integer label data (integer dtype + order=0) is resampled without
    anti-aliasing. float16 arrays are promoted to float32 for processing
    and cast back on output, as scipy.ndimage does not support float16.

    Parameters
    ----------
    zarr_path : Path
        Path to the Zarr archive.
    array_path : str
        Path to the source array within the archive.
    target_spacing : tuple[int, int, int]
        Target voxel spacing (Z, Y, X) in nanometers.
    order : int
        Spline interpolation order. 0 = nearest neighbour (labels),
        1 = linear (images). Default 1.
    zarr_chunks : tuple[int, int, int]
        Chunk shape for the output Zarr array.
    n_workers : int
        Number of threads for dask's threaded scheduler. Default 4.
    """

    Z_SCALE_WARN_THRESHOLD = 2.5

    root = zarr.open_group(zarr_path)
    src_array: zarr.Array = root.get(array_path)
    if src_array is None:
        raise FileNotFoundError(
            f"Source array {array_path} not found under {zarr_path}"
        )

    src_dtype = np.dtype(src_array.dtype)
    original_shape = np.array(src_array.shape)
    src_spacing = np.asarray(src_array.attrs["spacing"])

    spacing_dir = dirname_from_spacing(target_spacing)
    ref_array: zarr.Array = root.get(f"images/{spacing_dir}")
    if ref_array is not None:
        target_shape = list(ref_array.shape)
    else:
        ratio = src_spacing / np.asarray(target_spacing)
        target_shape = (ratio * original_shape).astype(int).tolist()

    scale_factors = original_shape / np.array(target_shape)

    z_magnitude = max(scale_factors[0], 1.0 / scale_factors[0])
    if z_magnitude > Z_SCALE_WARN_THRESHOLD:
        direction = "downsampling" if scale_factors[0] > 1 else "upsampling"
        warnings.warn(
            f"Large Z {direction} factor {z_magnitude:.2f}x "
            f"(src spacing: {src_spacing[0]}, target: {target_spacing[0]}). "
            f"Gaussian pre-blur operates per-chunk; boundary artifacts may "
            f"appear at chunk edges for large kernels.",
            UserWarning,
            stacklevel=2,
        )

    is_label = np.issubdtype(src_dtype, np.integer) and order == 0
    working_dtype = (
        np.dtype("float32") if src_dtype == np.dtype("float16") else src_dtype
    )

    temp_chunks = (8, 1024, 1024)
    src_dask: da.Array = da.from_zarr(src_array).rechunk(temp_chunks)

    if working_dtype != src_dtype:
        src_dask = src_dask.astype(working_dtype)

    aa_sigmas = np.zeros(3)
    if not is_label:
        aa_sigmas = np.maximum(0.0, (scale_factors - 1.0) / 2.0)
        if np.any(aa_sigmas > 0):
            src_dask = dask_image.ndfilters.gaussian_filter(
                src_dask, sigma=aa_sigmas.tolist()
            )

    resampled_dask: da.Array = dask_image.ndinterp.affine_transform(
        src_dask,
        matrix=np.diag(scale_factors),
        output_shape=target_shape,
        output_chunks=temp_chunks,
        order=order,
        mode="nearest",
    )

    if working_dtype != src_dtype:
        resampled_dask = resampled_dask.astype(src_dtype)

    parent_group: zarr.Group = root.get(str(Path(src_array.path).parent))
    dst_zarr_path = f"{parent_group.path}/{spacing_dir}"
    dst_zarr = root.require_array(
        name=dst_zarr_path,
        shape=target_shape,
        chunks=zarr_chunks,
        dtype=src_dtype,
        compressors=src_array.compressors,
        overwrite=True,
    )

    with ProgressBar(), dask.config.set(num_workers=n_workers):
        resampled_dask.to_zarr(dst_zarr, overwrite=True)

    dst_zarr.attrs["spacing"] = target_spacing
    dst_zarr.attrs["processing"] = src_array.attrs.get("processing", []) + [
        {
            "step": "resample",
            "order": order,
            "scale_factors": scale_factors.tolist(),
            "anti_aliasing": not is_label,
            "anti_aliasing_sigma": aa_sigmas.tolist(),
        }
    ]
    dst_zarr.attrs["inputs"] = src_array.path
    create_ome_multiscales(parent_group)


def rechunk_array(
    root: zarr.Group,
    src_array_path: str,
    dst_array_path: str,
    dst_chunks: tuple[int, int, int] = (1, 1024, 1024),
    copy_attributes: bool = True,
    delete_src: bool = False,
    verbose: bool = True,
) -> zarr.Array:
    """Copy a Zarr array to a new path with a different chunk layout.

    Parameters
    ----------
    root : zarr.Group
        Root Zarr group containing the source array.
    src_array_path : str
        Path to the source array within *root*.
    dst_array_path : str
        Path for the destination array within *root*. Created or overwritten.
    dst_chunks : tuple[int, int, int], optional
        Chunk shape for the output array. Default is ``(1, 1024, 1024)``.
    copy_attributes : bool, optional
        If True, copy all Zarr attributes from source to destination.
        Default is True.
    delete_src : bool, optional
        If True, delete the source array after copying. Default is False.
    verbose : bool, optional
        If True, show a tqdm progress bar. Default is True.

    Returns
    -------
    zarr.Array
        The newly created destination array.

    Raises
    ------
    FileNotFoundError
        If *src_array_path* does not exist within *root*.
    """

    src_zarr: zarr.Array = root.get(src_array_path)
    if src_zarr is None:
        raise FileNotFoundError(f"Temp array {src_array_path} not found")

    compressor = src_zarr.compressors
    dst_zarr = root.require_array(
        name=dst_array_path,
        shape=src_zarr.shape,
        chunks=dst_chunks,
        dtype=src_zarr.dtype,
        compressors=compressor,
        overwrite=True,
    )

    for z in tqdm(range(src_zarr.shape[0]), disable=not verbose):
        dst_zarr[z] = src_zarr[z]

    if copy_attributes:
        dst_zarr.attrs.clear()
        dst_zarr.attrs.update(dict(src_zarr.attrs))

    if delete_src:
        del root[src_array_path]
    return dst_zarr


def crop_to_valid(
    data: np.ndarray, mode: Literal["nonzero", "notnan"] = "nonzero"
) -> np.ndarray:
    """
    Crop a 3D array to the bounding box of valid data.

    Parameters
    ----------
    data : np.ndarray
        The 3D input array.
    mode : Literal["nonzero", "notnan"], optional
        The validity criteria: "nonzero" (default) or "notnan".

    Returns
    -------
    np.ndarray
        The cropped array.

    Raises
    ------
    ValueError
        If mode is not a valid value.
    """
    if mode == "notnan":
        mask = ~np.isnan(data)
    elif mode == "nonzero":
        mask = data != 0
    else:
        raise ValueError(
            f"Mode {mode} not recognized. Valid options are 'nonzero' and 'notnan'"
        )

    coords = np.argwhere(mask)

    if coords.size == 0:
        return data

    start = coords.min(axis=0)
    stop = coords.max(axis=0) + 1
    slicer = tuple(slice(s, e) for s, e in zip(start, stop))

    return data[slicer]
