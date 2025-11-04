"""
Utility functions
"""

import os
import yaml
import json
from datetime import datetime
import xxhash
from pathlib import Path
from typing import Iterable
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import Image
from sphero_vem.preprocessing import Normalize
import tifffile
import numpy as np


class TiffDataset(Dataset):
    """
    Custom for loading TIFF images
    """

    def __init__(
        self,
        root_dir: Path,
        transform: transforms.Transform | None = None,
        slices_idx: Iterable[int] | None = None,
    ) -> None:
        """
        Initialize a dataset.

        Parameters
        ----------
        root_dir : Path
            Directory containing the dataset files.
        transform : transforms.Transform or None, optional
            Transformation to apply to the images. If None, a default transform
            will be applied. The default transformation will
        slices_idx : Iterable[int] or None, optional
            Indices of slices to include. If None, all slices will be included.

        Notes
        -----
        The dataset will load all .tif files in the root directory.
        """

        self.root_dir = root_dir
        self.transform = transform if transform else self._get_default_transform()
        file_list = sorted(list(root_dir.glob("*.tif")))
        self.file_list = [file_list[i] for i in slices_idx] if slices_idx else file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Image:
        image_path = self.file_list[idx]
        image = tifffile.imread(image_path)

        image_tv: Image = self.transform(image)

        return image_tv

    def _get_default_transform(self) -> transforms.Transform:
        """A default transformation that convert an image to a float tensor and
        standardizes it"""

        default_transform = transforms.Compose(
            [
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                Normalize("zscore"),
            ]
        )

        return default_transform


def get_file_info(filepath: Path, data_root: Path) -> dict:
    """Get file info and generate hashes. Used for manifest generation"""
    stat = os.stat(filepath)

    # Calculate hash for file integrity (optional but recommended)
    hash_value = xxhash.xxh64(open(filepath, "rb").read()).hexdigest()

    return {
        "path": str(filepath.relative_to(data_root)),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "xxh64_hash": hash_value,
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
