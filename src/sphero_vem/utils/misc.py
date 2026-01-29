"""
Utility functions
"""

import tempfile
import shutil
from pathlib import Path
from contextlib import contextmanager
import os
import yaml
import json
from datetime import datetime
import torch
import zarr
import numpy as np


def get_file_info(filepath: Path, data_root: Path) -> dict:
    """Get file info and generate hashes. Used for manifest generation"""
    stat = os.stat(filepath)

    return {
        "path": str(filepath.relative_to(data_root)),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def read_section_errors(data_dir: Path) -> list[str] | None:
    """Read tiles with sectioning errors, returns None if there is no file specifying
    them"""
    try:
        with open(data_dir / "folded_section_tiles.txt", "r") as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        try:
            manifest = read_manifest(data_dir)
            return manifest.get("folded_section_tiles")
        except FileNotFoundError:
            return


def generate_manifest(
    dataset: str,
    out_dir: Path,
    images: list[Path],
    processing: list[dict],
    **extra_fields,
) -> dict:
    """Generate manifest.yaml file with processing steps and saves it.

    For correct results, this should be called after processing is complete. Output
    image list is read from the directory and sorted, assuming the ordering is
    maintained to establish a correspondence.

    Parameters
    ----------
    dataset : str
        The dataset name.
    out_dir : Path
        The output directory containing the output images and where the manifest will
        be saved.
    images : list[Path]
        A list containing the input images.
    processing : list[dict]
        A list of dictionaries containing the processing steps in the order they were
        executed. Each dictionary contains the relevant parameters for that processing
        step, and should contain the key 'step' where the processing step is specifyied
        with a string.

    Returns
    -------
    dict
        The generate manifest
    """

    data_dir = images[0].parent
    data_root = _find_data_root(data_dir)
    old = read_manifest(data_dir)

    manifest = {
        "dataset": dataset,
        "generated_on": datetime.now().isoformat(),
        "processing": old.get("processing", []) + processing,
        "inputs": [str(p.relative_to(data_root)) for p in sorted(images)],
        "outputs": [str(p.name) for p in sorted(out_dir.glob("*.tif"))],
        **extra_fields,
    }

    error_tiles = read_section_errors(data_dir)
    manifest.update({"folded_section_tiles": error_tiles} if error_tiles else {})
    manifest.update({"discarded": old["discarded"]} if old.get("discarded") else {})

    with open(out_dir / "manifest.yaml", "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return manifest


def read_manifest(data_dir: Path) -> dict:
    """Read manifest in directory"""
    try:
        with open(data_dir / "manifest.yaml", "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {}


def _find_data_root(dir: Path) -> Path:
    """Find absolute path of the data root"""
    for parent in dir.parents:
        if parent.name == "data":
            return parent


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


def infer_dataset(data_dir: Path) -> str:
    """Infer dataset name from the name of images in the data directory"""
    image_name = list(data_dir.glob("*.tif"))[0].name
    return image_name[: image_name.rfind("-z_")]


def get_seg_params(dir: Path) -> dict:
    """Extract segmentation parameters from the manifest"""
    manifest = read_manifest(dir)
    seg_params = {}
    for step in manifest.get("processing", {}):
        if step.get("step") == "segmentation":
            seg_params = step
    return seg_params


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


def create_ome_multiscales(
    group: zarr.Group,
    multichannel: bool = False,
    spatial_dims: int = 3,
) -> None:
    """Create multiscales specifications compliant with OME-NGFF format v0.5.

    Parameters
    ----------
    group : zarr.Group
        Zarr group that contains the multiscale.
    multichannel : bool
        Whether a channel axis should be included. This is expected to be in C(Z)YX
    spatial_dims : int
        Number of spatial dimensions. Dimension order should be (Z)YX. Accepted values
        are 2 or 3.

    Raises
    ------
    ValueError
        If the number of spatial dimensions is not 2 or 3.

    """

    spatial_axes = [
        {"name": "y", "type": "space", "unit": "nanometer"},
        {"name": "x", "type": "space", "unit": "nanometer"},
    ]
    if spatial_dims == 3:
        spatial_axes = [
            {
                "name": "z",
                "type": "space",
                "unit": "nanometer",
            }
        ] + spatial_axes
    elif spatial_dims != 2:
        raise ValueError(
            f"Unsupported number of spatial dimensions {spatial_dims}. "
            "Supported values are 2 (YX) or 3 (ZYX)."
        )

    scales = get_multiscales(group)

    # Hanlde multichannel
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


def spacing_from_dirname(dirname) -> tuple[int, int, int]:
    """Convenience function to extract spacing from directory name in the format
    '{spacing_z}-{spacing_y}-{spacing_x}'"""
    return tuple(int(i) for i in dirname.split("-"))


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
        Contant margin for bounding box expansion. The bounding box will be expanded
        by this value in all directions.
    im_shape : tuple[int]
        Image shape.

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
