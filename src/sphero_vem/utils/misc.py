"""
Utility functions
"""

import tempfile
import warnings
import shutil
from pathlib import Path
from contextlib import contextmanager
import yaml
import json
from datetime import datetime
from collections.abc import Sequence
import torch
import zarr
import numpy as np
import pandas as pd


def read_manifest(data_dir: Path) -> dict:
    """Read manifest in directory"""
    try:
        with open(data_dir / "manifest.yaml", "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}


def vprint(text: str, verbose: bool) -> None:
    """Helper function for cleanly handling print statements with a verbose option"""
    if verbose:
        print(text)


def timestamp() -> str:
    """Returns a timestamp for the current time up to seconds, ISO-formatted and
    widely filesystem compatible"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def detect_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CustomJSONEncoder(json.JSONEncoder):
    """A custom JSONEncoder to handle non base data types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def create_ome_multiscales(group: zarr.Group | Path) -> None:
    """Create multiscales specifications compliant with OME-NGFF format v0.5.

    Automatically infers multichannel and spatial dimensions from existing arrays.

    Parameters
    ----------
    group : zarr.Group | Path
        Zarr group that contains the multiscale arrays, or path to it.

    Notes
    -----
    - Spatial dimensions inferred from 'spacing' attribute length
    - Channel dimension assumed if array.ndim > len(spacing)
    - Axis order is always C(Z)YX
    - Does nothing if no scale arrays found
    """
    if isinstance(group, Path):
        group = zarr.open_group(group, mode="a")

    scales = get_multiscales(group)

    # Early return if no scales present
    if not scales:
        return

    # Infer from first array
    first_array = group[scales[0]["path"]]
    spatial_dims = len(scales[0]["scale"])  # spacing length
    multichannel = first_array.ndim > spatial_dims

    # Build spatial axes
    spatial_axes = [
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    if spatial_dims == 3:
        spatial_axes = [
            {"name": "z", "type": "space", "unit": "nanometer"}
        ] + spatial_axes

    # Handle multichannel
    channel_axis = [{"name": "c", "type": "channel"}] if multichannel else []
    channel_scale = [1] if multichannel else []

    group.attrs["multiscales"] = [
        {
            "version": "0.5",
            "name": "images",
            "axes": channel_axis + spatial_axes,
            "datasets": [
                {
                    "path": s["path"],
                    "coordinateTransformations": [
                        {
                            "type": "scale",
                            "scale": channel_scale + list(s["scale"]),
                        }
                    ],
                }
                for s in scales
            ],
        }
    ]


def dirname_from_spacing(spacing: tuple[int, int, int]) -> str:
    """Convenience function to create a directory name from spacing in the format
    '{spacing_z}-{spacing_y}-{spacing_x}'"""
    return "-".join([str(i) for i in spacing])


def get_multiscales(group: zarr.Group) -> list[dict]:
    """Get array scales as a list of dicts.

    The function looks for "spacing" in the array attributes as a source of ground
    truth. If not found, the array is ignored.

    Parameters
    ----------
    group : zarr.Group
        Zarr group containing the multiscale arrays.

    Returns
    -------
    list[dict]
        A list containing the multiscale information as a dictionary. Scales
        are sorted for ascending pixel area/voxel volume.
        Example return:
            [
                {"path": "0", "scale": [50, 50, 50]},
                {"path": "1", "scale": [100, 100, 100]}
            ]
    """

    def _get_spacing(arr: zarr.Array) -> tuple[int | float] | None:
        """Access spacing and returns None if not found"""
        return arr.attrs.get("spacing", None)

    multiscales = [
        {"path": key, "scale": _get_spacing(arr)}
        for key, arr in group.arrays()
        if _get_spacing(arr)
    ]
    return sorted(multiscales, key=lambda x: np.prod(x["scale"]))


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
        Prefix for the temporary directory name. Default is "intermediate_".
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


def repair_multiscales(root: Path, start_path: str = "") -> None:
    """Recursively repair multiscales metadata for all groups in hierarchy.

    Parameters
    ----------
    root : Path
        Path to the Zarr store containing the hierarchy
    start_path : str, default=""
        Path to start repair from (empty string for root).
    """

    # Ignores warnings of non-standard zarr hierarchy components, such as tables.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Object at .* is not recognized as a component of a Zarr hierarchy",
            category=zarr.errors.ZarrUserWarning,
        )

        root = zarr.open(root, mode="a")
        group = root.get(start_path) if start_path else root

        if group is not None:
            _repair_group_recursive(group)


def _repair_group_recursive(group: zarr.Group) -> None:
    """Recursively repair a group and its children."""
    # Repair this group if it has multiscales
    if "multiscales" in group.attrs:
        create_ome_multiscales(group)

    # Recurse into all subgroups
    for key in group.group_keys():
        subgroup = group.get(key)
        if subgroup is not None and isinstance(subgroup, zarr.Group):
            _repair_group_recursive(subgroup)
