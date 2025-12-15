"""Module containing input/output functions"""

from typing import Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import tifffile
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
import dask.array as da
from dask.diagnostics import ProgressBar
from sphero_vem.preprocessing import downscale_image, downscale_labels, downscale_tensor
from sphero_vem.utils import (
    read_manifest,
    create_ome_multiscales,
    dirname_from_spacing,
)


def write_image(
    fname: Path, image: np.ndarray, compressed: bool = False, **kwargs
) -> None:
    """Save TIFF images with default zip compression. Compression can be enabled
    with the compressed argument. This also disables any other optional keyword
    argument passed"""
    default_compression = {
        "compression": "zlib",
        "compressionargs": {"level": 6},
        "predictor": 2,
        "tile": (256, 256),
    }
    options = {**default_compression, **kwargs} if compressed else {}
    return tifffile.imwrite(fname, image, **options)


def imread_downscaled(
    image_path: Path,
    factor: int,
    norm_method: str | None = None,
    scale_range: bool = True,
) -> np.ndarray:
    """Read and downscale an image. This is a convenience function that
    returns a downscaled image as unsigned 8-bit integer"""
    image = tifffile.imread(image_path)
    image_resized = downscale_image(
        image, factor, norm_method, out_type=torch.uint8, scale_range=scale_range
    )
    return image_resized.squeeze().numpy()


def imread_labels_downscaled(
    labels_path: Path,
    factor: int,
) -> np.ndarray:
    """Read and downscale a segmentation mask. The mask will be returned
    of the same type as the original one, which is typically int32"""
    labels = tifffile.imread(labels_path)
    labels_resized = downscale_labels(labels, factor)
    return labels_resized.squeeze().numpy()


def write_stack(
    data_dir: Path, out_file: Path, channel_axis: bool = False, compressed: bool = False
) -> None:
    """Merge images in a folder into a single ZYX tif. If channel_axis option is True,
    save also a channel of size 1 so that every image is CYZ"""

    volume_stack = read_stack(data_dir, channel_axis=channel_axis)
    write_image(out_file, volume_stack, compressed=compressed)


def read_stack(
    data_dir: Path, channel_axis: bool = False, verbose: bool = True
) -> np.ndarray:
    """Sequentially read images in directory and merge them into a 3D stack with shape
    ZYX. If the channel_axis option is on, a channel dimension of size 1 is added to have
    shape ZCYX"""

    image_list = sorted(list(data_dir.glob("*.tif")))
    first_image = tifffile.imread(image_list[0])
    image_shape = (1, *first_image.shape) if channel_axis else first_image.shape
    volume_stack = np.empty((len(image_list), *image_shape), first_image.dtype)
    for i, image_path in enumerate(
        tqdm(image_list, "Reading slices", disable=not verbose)
    ):
        image = tifffile.imread(image_path).reshape(*image_shape)
        volume_stack[i] = image
    return volume_stack


def read_tensor(
    image_path: Path,
    dtype: torch.dtype | None = torch.float32,
    ds_factor: int = 1,
    resample_mode: str = "bilinear",
    return_4d: bool = False,
) -> torch.Tensor:
    """Read a tiff image as a pytorch tensor. Returns a tensor of the same shape as
    the image.
    If dtype is None, use the original dtype of the image.
    If ds_factor > 1, applies downscaling by that factor to the image.
    If return_4d is true, returns a tensor of size 1 x 1 x H x W"""
    image = tifffile.imread(image_path)
    image_torch = torch.tensor(image, dtype=dtype)
    if return_4d:
        while image_torch.dim() < 4:
            image_torch = image_torch.unsqueeze(0)
    if ds_factor > 1:
        image_torch = downscale_tensor(image_torch, ds_factor, resample_mode)
    return image_torch


def stack_to_zarr(
    stack_dir: Path,
    root_path: Path,
    spacing: tuple[int, int, int] | None,
    chunk_size: tuple[int, int, int] = (1, 1024, 1024),
    verbose: bool = True,
) -> None:
    """Convert a tiff stack to a ZYX zarr archive.

    The stack will be saved under root/images/spacing_dir

    Parameters
    ----------
    stack_path : Path
        Path to the tiff stack. This should be a directory with single tif slices that
        will be concatened along the Z axis.
    dest_path : Path
        Path the root of the destination zarr store.
    spacing : tuple[int, int, int] | None
        ZYX spacing of the dataset in nanometers. If None, attempt to read the spacing
        from metadata (Currently not implemented).
    chunk_size : tuple[int, int, int] | None
        ZYX chunk size of the zarr array. If None, use (1, H, W).
    verbose : bool
        Enable verbose output.


    Raises
    ------
    NotImplementedError
        When passing None to spacing.
    """
    if not spacing:
        raise NotImplementedError("Automatic spacing determination not yet implemented")

    image_paths = sorted(stack_dir.glob("*.tif"))
    with tifffile.TiffFile(image_paths[0]) as tif:
        image_shape = tif.pages[0].shape
        image_dtype = tif.pages[0].dtype

    stack_shape = (len(image_paths), *image_shape)

    compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
    zarr_root = zarr.open(root_path, mode="a")
    image_group = zarr_root.require_group("images")
    zarr_arr = image_group.create_array(
        dirname_from_spacing(spacing),
        shape=stack_shape,
        chunks=chunk_size,
        dtype=image_dtype,
        compressors=compressor,
    )

    for i, image_path in tqdm(
        enumerate(image_paths),
        "Reading images",
        disable=not verbose,
        total=len(image_paths),
    ):
        zarr_arr[i] = tifffile.imread(image_path)

    # Update zarr metadata
    manifest = read_manifest(stack_dir)
    zarr_arr.attrs["spacing"] = spacing
    zarr_arr.attrs["processing"] = manifest["processing"]
    zarr_arr.attrs["inputs"] = [str(path) for path in image_paths]
    create_ome_multiscales(image_group)


def write_zarr(
    root: zarr.group | Path | str,
    array: np.ndarray | da.Array,
    dst_path: str,
    src_zarr: zarr.Array | None = None,
    spacing: tuple[int | float] | None = None,
    dtype: Any | None = None,
    shape: tuple[int] | None = None,
    processing: list[dict] | dict | None = None,
    inputs: list[str] | None = None,
    zarr_chunks: tuple[int] | None = None,
    multichannel: bool = False,
) -> None:
    """Write numpy or dask array to zarr.

    Parameters
    ----------
    root: zarr.Group | pathlib.Path | str
        Root zarr group used to save the image, or path to the root group.
    array: np.ndarray | da.Array
        Numpy or Dask array containing the to save. Axis order should be (CZ)YX.
    dst_path : str
        Path under root where to save the file.
    src_zarr: zarr.Array | None
        Source zarr array. Used to determine previous processing, as well as chunk size
        and spacing unless they are specified. Default is None.
    spacing: tuple[int | float] | None
        Spacing of the array. If None, uses the spacing of src_zarr if provided.
        Spacing is saved under the "spacing" key in the destination array attributes.
        NOTE: this might lead to incorrect spacing if resampling was involved!
        Default is None.
    dtype: Any | None
        Data type of the saved zarr array. This casts the array to specified dtype
        while saving it. If None, uses the array dtype. Default is None.
    shape: tuple[int] | None
        Zarr array shape. If None, uses the input array shape. Default is None.
    processing: list[dict] | dict | None
        Sequential processing steps done on the array. Each step is a dictionary
        whose first key should be "step":"step name". The function tries to read
        the previous processing of the source array by reading
        src_zarr.attrs["processing] and appends processing to this list. If None,
        processing will be an empty list. Default is None.
    inputs: list[str] | None
        Paths to the input array(s). If None, uses the path to src_array if supplied.
        Default is None.
    zarr_chunks: tuple[int] | None
        Chunks of the saved zarr array. It should have the same length as the array
        dimensions. If None, uses src_array chunks. Default is None.
    multichannel: bool = False
        Whether the image is multichannel or not. Image channel should always be the
        first axis of array. Default is False.

    Raises
    ------
    ValueError
        If spacing is None and src_zarr is None: spacing should be specified, or a
        valid src_zarr should be passed.
    ValueError
        If src_zarr doesn't have a spacing attribute.
    ValueError
        If zarr_chunks is None and src_zarr is None: zarr_chunks should be specified,
        or a valid src_zarr should be passed.
    """

    if not isinstance(root, zarr.Group):
        root = zarr.open_group(root, mode="a")

    group_path = str(Path(dst_path).parent)

    if not spacing:
        if src_zarr:
            spacing = src_zarr.attrs.get("spacing")
            if not spacing:
                raise ValueError(
                    "Source array has no spacing attribute. "
                    "Destination spacing must be specified."
                )
        else:
            raise ValueError("Spacing must be specified if src_zarr is None.")

    if not zarr_chunks:
        if src_zarr:
            zarr_chunks = src_zarr.chunks
        else:
            raise ValueError("Zarr chunks must be specified if src_zarr is None.")

    compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
    dst_zarr = root.require_array(
        dst_path,
        shape=shape if shape else array.shape,
        chunks=zarr_chunks,
        compressors=compressor,
        dtype=dtype if dtype else array.dtype,
        overwrite=True,
    )

    if isinstance(array, np.ndarray):
        dst_zarr[...] = array
    elif isinstance(array, da.Array):
        with ProgressBar():
            array.to_zarr(dst_zarr)
    else:
        raise TypeError(f"Unsuppored type {type(array)} for input array")

    # Update default processing and inputs
    if src_zarr:
        src_processing = src_zarr.attrs.get("processing", [])
    else:
        src_processing = []

    if not processing:
        processing = []
    elif isinstance(processing, dict):
        processing = [processing]

    if not inputs and src_zarr:
        inputs = [src_zarr.path]
    else:
        inputs = []

    dst_zarr.attrs["spacing"] = spacing
    dst_zarr.attrs["processing"] = src_processing + processing
    dst_zarr.attrs["inputs"] = inputs

    # Get parameters for multiscales
    if array.ndim < 3:
        spatial_dims = 2
    elif array.ndim == 3:
        spatial_dims = 2 if multichannel else 3
    else:
        spatial_dims = 3

    create_ome_multiscales(
        root.get(group_path),
        spatial_dims=spatial_dims,
        multichannel=multichannel,
    )
