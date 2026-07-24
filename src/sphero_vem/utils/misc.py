"""
Utility functions
"""

import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from collections.abc import Sequence
import torch
import zarr
import numpy as np
import pandas as pd


def vprint(text: str, verbose: bool) -> None:
    """Helper function for cleanly handling print statements with a verbose option"""
    if verbose:
        print(text)


def timestamp() -> str:
    """Returns a timestamp for the current time up to seconds, ISO-formatted and
    widely filesystem compatible"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def detect_torch_device() -> torch.device:
    """Select the best torch device available on the current machine.

    The preference order is CUDA, then Apple Metal Performance Shaders (MPS),
    then CPU. This is the single source of truth for device selection across
    the package: config dataclasses default their ``device`` field to this
    function via ``field(default_factory=detect_torch_device)``.

    Returns
    -------
    torch.device
        ``torch.device("cuda")`` if a CUDA device is available,
        ``torch.device("mps")`` on Apple silicon with a working Metal
        backend, otherwise ``torch.device("cpu")``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dirname_from_spacing(spacing: tuple[int, int, int]) -> str:
    """Convenience function to create a directory name from spacing in the format
    '{spacing_z}-{spacing_y}-{spacing_x}'"""
    return "-".join([str(i) for i in spacing])


@contextmanager
def temporary_zarr(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype=np.float32,
    prefix: str = "intermediate_",
    dir: Path | str | None = None,
):
    """Context manager for temporary zarr array.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the array.
    chunks : tuple[int, ...]
        Chunk size for the array.
    dtype : np.dtype
        Data type of the array. Default is np.float32.
    prefix : str
        Prefix for the temporary directory name. Default is ``"intermediate_"``.
    dir : Path | str | None
        Parent directory for the temporary zarr. If None, uses system temp.

    Yields
    ------
    zarr.Array
        Temporary zarr array, deleted on context exit.
    """

    # Ensure parent directory exists
    if dir is not None:
        Path(dir).mkdir(parents=True, exist_ok=True)

    tmp_dir = tempfile.mkdtemp(prefix=prefix, dir=dir)
    tmp_path = Path(tmp_dir) / "data.zarr"

    try:
        # No compression for speed
        tmp_zarr = zarr.open_array(
            tmp_path,
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
        )
        yield tmp_zarr
    finally:
        # Clean up immediately on exit
        shutil.rmtree(tmp_dir, ignore_errors=True)


def bbox_expand(bbox: tuple[int], margin: int, im_shape: tuple[int]) -> tuple[int]:
    """Expand bounding box by margin without indexing out of image bounds.

    Parameters
    ----------
    bbox : tuple[int]
        Bounding box coordinates in the form (x0_min, x1_min, ..., x0_max, x1_max, ...).
        The order of the coordinates x_i should be the same as numpy axis.
    margin : int
        Constant margin for bounding box expansion. The bounding box will be expanded
        by this value in all directions.
    im_shape : tuple[int]
        Shape of the image array in the same axis order as *bbox*. Used to
        clip the expanded bounding box so it does not exceed array bounds.

    Returns
    -------
    bbox_exp : tuple[int]
        Expanded bounding box, in the form (x0_min, x1_min, ..., x0_max, x1_max, ...).
    """
    n_dim = len(bbox) // 2
    bbox_arr = np.array(bbox)
    offsets = np.array([[-margin] * n_dim + [margin] * n_dim])
    expanded = np.clip(bbox_arr + offsets, 0, im_shape * 2)
    return tuple(*expanded.tolist())


def slice_from_bbox(bbox: tuple) -> tuple[slice]:
    """Get slice from a bounding box for easy image cropping.

    Parameters
    ----------
    bbox : tuple[int]
        Bounding box coordinates in the form (x0_min, x1_min, ..., x0_max, x1_max, ...).
        The order of the coordinates x_i should be the same as numpy axis.

    Returns
    -------
    tuple[slice]
        Tuple of slices for indexing.
    """
    n_dim = len(bbox) // 2
    return tuple(slice(bbox[i], bbox[i + n_dim]) for i in range(n_dim))


def check_isotropic(spacing: Sequence[float], raise_error: bool = False) -> bool:
    """Check if spacing is isotropic, and optionally raise an error if it's not.

    Parameters
    ----------
    spacing : Sequence[float]
        A sequence containing the voxel spacing to check.
    raise_error : bool
        Flag that controls whether to raise an error is the check fails.
        Default is False.

    Returns
    -------
    bool
        True is the spacing is isotropic.

    Raises
    ------
    ValueError
        If the spacing is not isotropic and raise_error is True.
    """
    check = True
    if len(set(spacing)) > 1:
        check = False
        if raise_error:
            raise ValueError(f"Spacing must be isotropic. Received {spacing}")
    return check


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    """Calculate the weighted standard deviation of the data.

    Parameters
    ----------
    values : np.ndarray
        Array containing the data.
    weights : np.ndarray
        Array containing the weights. It must have the same shape as values.

    Returns
    -------
    float
        The weighted standardn deviation.
    """
    mean = np.average(values, weights=weights)
    var = np.average((values - mean) ** 2, weights=weights)
    return np.sqrt(var)


def flatten_for_save(
    df: pd.DataFrame,
    sep: str = "__",
) -> pd.DataFrame:
    """
    Unpack tuple/list columns into indexed scalar columns for storage.

    Tuple columns are expanded into separate columns with names
    ``{original_name}{sep}0``, ``{original_name}{sep}1``, etc.
    The original tuple column is dropped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with possible tuple or list valued columns.
    sep : str, optional
        Separator between column name and index. Must be passed
        identically to `reconstruct_tuples` for round-tripping.
        Default is ``"__"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with all tuple columns replaced by scalar columns.

    Raises
    ------
    ValueError
        If any column name already contains `sep`, which would
        create ambiguity on reconstruction.

    See Also
    --------
    reconstruct_tuples : Inverse operation.
    """
    ambiguous = [c for c in df.columns if sep in str(c)]
    if ambiguous:
        raise ValueError(
            f"Column names already contain '{sep}', which would "
            f"create ambiguity on reconstruction: {ambiguous}"
        )

    df_out = df.copy()
    for col in df.columns:
        first = df[col].iloc[0]
        if isinstance(first, (tuple, list)):
            n = len(first)
            for i in range(n):
                df_out[f"{col}{sep}{i}"] = df[col].apply(lambda x, i=i: x[i])
            df_out = df_out.drop(columns=[col])

    return df_out


def reconstruct_tuples(
    df: pd.DataFrame,
    sep: str = "__",
) -> pd.DataFrame:
    """
    Pack indexed scalar columns back into tuple columns.

    Columns matching the pattern ``{name}{sep}0``, ``{name}{sep}1``, ...
    are merged into a single tuple column ``{name}``. The indexed
    columns are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame as loaded from parquet, with flattened tuple columns.
    sep : str, optional
        Separator used by `flatten_for_save`. Default is ``"__"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with indexed columns replaced by tuple columns.

    Raises
    ------
    ValueError
        If indexed columns for a group are not contiguous starting
        from 0 (e.g., ``bbox__0``, ``bbox__2`` without ``bbox__1``).

    See Also
    --------
    flatten_for_save : Inverse operation.
    """
    groups: dict[str, list[tuple[int, str]]] = {}
    passthrough: list[str] = []

    for col in df.columns:
        if sep in col:
            base, _, suffix = col.rpartition(sep)
            if suffix.isdigit():
                groups.setdefault(base, []).append((int(suffix), col))
            else:
                passthrough.append(col)
        else:
            passthrough.append(col)

    df_out = df[passthrough].copy()

    for base, idx_cols in groups.items():
        idx_cols.sort()
        indices = [i for i, _ in idx_cols]
        if indices != list(range(len(indices))):
            raise ValueError(
                f"Non-contiguous indices for '{base}': found {indices}, "
                f"expected {list(range(len(indices)))}"
            )
        col_names = [c for _, c in idx_cols]
        df_out[base] = list(zip(*[df[c] for c in col_names]))

    return df_out
