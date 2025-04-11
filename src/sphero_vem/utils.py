"""
Utility functions
"""

from pathlib import Path
from typing import Iterable
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from torchvision.tv_tensors import Image
from sphero_vem.preprocessing import Normalize
import tifffile


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
