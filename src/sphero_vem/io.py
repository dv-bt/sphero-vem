"""Module containing input/output functions"""

from typing import Any
from pathlib import Path
from tqdm import tqdm
import numpy as np
import tifffile
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
import dask.array as da
from dask.diagnostics import ProgressBar
from sphero_vem.utils import (
    read_manifest,
    create_ome_multiscales,
    dirname_from_spacing,
    ProcessingStep,
)


def write_image(
    fname: Path, image: np.ndarray, compressed: bool = False, **kwargs
) -> None:
    """Save a NumPy array as a TIFF file, optionally with zlib compression.

    Parameters
    ----------
    fname : Path
        Destination file path.
    image : numpy.ndarray
        Image array to save.
    compressed : bool, optional
        If True, apply zlib compression (level 6) with tiling. Extra keyword
        arguments are ignored when compression is enabled. Default is False.
    **kwargs
        Additional keyword arguments forwarded to ``tifffile.imwrite`` when
        *compressed* is False.
    """
    default_compression = {
        "compression": "zlib",
        "compressionargs": {"level": 6},
        "predictor": 2,
        "tile": (256, 256),
    }
    options = {**default_compression, **kwargs} if compressed else {}
    return tifffile.imwrite(fname, image, **options)


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
    processing: list = manifest.get("processing", [])

    zarr_arr.attrs["spacing"] = spacing
    zarr_arr.attrs["processing"] = processing
    zarr_arr.attrs["inputs"] = [str(path) for path in image_paths]
    create_ome_multiscales(image_group)


def _create_zarr_array(
    root: zarr.Group,
    dst_path: str,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
) -> zarr.Array:
    """Create a compressed zarr array at the given path.

    Parameters
    ----------
    root : zarr.Group
        Root zarr group.
    dst_path : str
        Path under root where to create the array.
    shape : tuple[int, ...]
        Array shape.
    chunks : tuple[int, ...]
        Chunk shape.
    dtype : Any
        Array dtype.

    Returns
    -------
    zarr.Array
        Created zarr array, compressed with zstd and bitshuffle.
    """
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle)
    return root.create_array(
        dst_path,
        shape=shape,
        chunks=chunks,
        compressors=compressor,
        dtype=dtype,
        overwrite=True,
    )


def _write_zarr_data(dst_zarr: zarr.Array, array: np.ndarray | da.Array) -> None:
    """Write a numpy or dask array into an existing zarr array.

    Parameters
    ----------
    dst_zarr : zarr.Array
        Destination zarr array, must have compatible shape and dtype.
    array : np.ndarray | da.Array
        Data to write.

    Raises
    ------
    TypeError
        If array is neither a numpy nor a dask array.
    """
    if isinstance(array, np.ndarray):
        dst_zarr[...] = array
    elif isinstance(array, da.Array):
        with ProgressBar():
            array.to_zarr(dst_zarr)
    else:
        raise TypeError(f"Unsupported type {type(array)} for input array")


def _write_zarr_metadata(
    root: zarr.Group,
    dst_zarr: zarr.Array,
    src_zarr: zarr.Array | None = None,
    spacing: tuple[int | float, ...] | None = None,
    processing: ProcessingStep | list[ProcessingStep] | list[dict] | dict | None = None,
    inputs: list[str] | None = None,
) -> None:
    """Write metadata to a zarr array and create OME multiscales.

    Parameters
    ----------
    root : zarr.Group
        Root zarr group.
    dst_zarr : zarr.Array
        Destination zarr array to attach metadata to.
    src_zarr : zarr.Array | None
        Source zarr array. Used to read previous processing steps, spacing,
        and inputs if not explicitly provided. Default is None.
    spacing : tuple[int | float, ...] | None
        Spacing of the array. If None, reads from src_zarr.
        Default is None.
    processing : ProcessingStep | list[ProcessingStep] | list[dict] | dict | None
        Processing steps to append to the existing processing history from
        src_zarr. Default is None.
    inputs : list[str] | None
        Paths to input arrays. If None, uses src_zarr path. Default is None.

    Raises
    ------
    ValueError
        If spacing is None and src_zarr is None or has no spacing attribute.
    """
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

    src_processing = src_zarr.attrs.get("processing", []) if src_zarr else []

    if not processing:
        processing = []
    elif not isinstance(processing, list):
        processing = [processing]
    processing = [
        step.to_dict() if isinstance(step, ProcessingStep) else step
        for step in processing
    ]

    inputs = [src_zarr.path] if (not inputs and src_zarr) else (inputs or [])

    dst_zarr.attrs["spacing"] = spacing
    dst_zarr.attrs["processing"] = src_processing + processing
    dst_zarr.attrs["inputs"] = inputs

    group_path = str(Path(dst_zarr.path).parent)
    create_ome_multiscales(root.get(group_path))


def write_zarr(
    root: zarr.Group | Path | str,
    array: np.ndarray | da.Array,
    dst_path: str,
    src_zarr: zarr.Array | None = None,
    spacing: tuple[int | float, ...] | None = None,
    dtype: Any | None = None,
    shape: tuple[int, ...] | None = None,
    processing: ProcessingStep | list[ProcessingStep] | list[dict] | dict | None = None,
    inputs: list[str] | None = None,
    zarr_chunks: tuple[int, ...] | None = None,
) -> None:
    """Write a numpy or dask array to zarr with metadata.

    Parameters
    ----------
    root : zarr.Group | Path | str
        Root zarr group, or path to one.
    array : np.ndarray | da.Array
        Array to save. Axis order should be (CZ)YX.
    dst_path : str
        Path under root where to save the array.
    src_zarr : zarr.Array | None
        Source zarr array. Used to read spacing, chunks, previous processing,
        and inputs if not explicitly provided. Default is None.
    spacing : tuple[int | float, ...] | None
        Spacing of the array. If None, reads from src_zarr. Default is None.
    dtype : Any | None
        dtype to cast to when saving. If None, uses array dtype. Default is None.
    shape : tuple[int, ...] | None
        Zarr array shape. If None, uses array shape. Default is None.
    processing : ProcessingStep | list[ProcessingStep] | list[dict] | dict | None
        Processing steps to append to src_zarr processing history.
        Default is None.
    inputs : list[str] | None
        Paths to input arrays. If None, uses src_zarr path. Default is None.
    zarr_chunks : tuple[int, ...] | None
        Chunk shape. If None, reads from src_zarr. Default is None.

    Raises
    ------
    ValueError
        If zarr_chunks is None and src_zarr is None.
    ValueError
        If spacing is None and src_zarr is None or has no spacing attribute.
    TypeError
        If array is neither a numpy nor a dask array.
    """
    if not isinstance(root, zarr.Group):
        root = zarr.open_group(root, mode="a")

    if not zarr_chunks:
        if src_zarr:
            zarr_chunks = src_zarr.chunks
        else:
            raise ValueError("zarr_chunks must be specified if src_zarr is None.")

    dst_zarr = _create_zarr_array(
        root=root,
        dst_path=dst_path,
        shape=shape if shape else array.shape,
        chunks=zarr_chunks,
        dtype=dtype if dtype else array.dtype,
    )
    _write_zarr_data(dst_zarr, array)
    _write_zarr_metadata(
        root=root,
        dst_zarr=dst_zarr,
        src_zarr=src_zarr,
        spacing=spacing,
        processing=processing,
        inputs=inputs,
    )
