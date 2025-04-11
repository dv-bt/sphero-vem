"""
Tests for the functions and classes in the utils module
"""

import pytest
from pathlib import Path
import torch
import numpy as np
import tempfile
import tifffile
from sphero_vem.utils import TiffDataset
from torchvision.tv_tensors import Image
import torchvision.transforms.v2 as transforms


@pytest.fixture
def temp_tiff_dir():
    """Create a temporary directory with test TIFF files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create 5 simple TIFF images
        for i in range(5):
            # Create a simple numpy array
            img_data = np.ones((10, 10), dtype=np.uint16) * i
            tifffile.imwrite(tmpdir_path / f"img_{i:02d}.tif", img_data)

        yield tmpdir_path


def test_tiffdataset_init(temp_tiff_dir):
    """Test TiffDataset initialization."""
    dataset = TiffDataset(temp_tiff_dir)
    assert len(dataset.file_list) == 5
    assert all(f.suffix == ".tif" for f in dataset.file_list)
    assert dataset.root_dir == temp_tiff_dir


def test_tiffdataset_len(temp_tiff_dir):
    """Test TiffDataset __len__."""
    dataset = TiffDataset(temp_tiff_dir)
    assert len(dataset) == 5

    # Test with slices_idx
    dataset_subset = TiffDataset(temp_tiff_dir, slices_idx=[0, 2, 4])
    assert len(dataset_subset) == 3


def test_tiffdataset_getitem(temp_tiff_dir):
    """Test TiffDataset __getitem__."""
    dataset = TiffDataset(temp_tiff_dir)

    # Get first item
    img = dataset[0]
    assert isinstance(img, Image)
    assert img.shape[-2:] == (10, 10)  # Assuming 2D image

    # Check if transform is applied correctly
    # Default transform standardizes the data
    assert img.dtype == torch.float32


def test_tiffdataset_custom_transform(temp_tiff_dir):
    """Test TiffDataset with custom transform."""
    custom_transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )

    dataset = TiffDataset(temp_tiff_dir, transform=custom_transform)
    img = dataset[0]

    assert isinstance(img, Image)
    assert img.dtype == torch.float32


def test_tiffdataset_slices_idx(temp_tiff_dir):
    """Test TiffDataset with slices_idx parameter."""
    dataset = TiffDataset(temp_tiff_dir, slices_idx=[1, 3])

    assert len(dataset) == 2
    # Check that we get the correct files
    assert dataset.file_list[0].name == "img_01.tif"
    assert dataset.file_list[1].name == "img_03.tif"
