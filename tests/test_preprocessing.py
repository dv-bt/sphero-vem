import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import tifffile
from sphero_vem.preprocessing import Normalize, downscale_image, imread_downscaled
from torchvision import tv_tensors


class TestNormalize:
    def test_minmax_normalization(self):
        # Create a test image
        image = tv_tensors.Image(torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 2, 2))

        # Apply normalization
        normalizer = Normalize("minmax")
        result = normalizer(image)

        # Check results: (x - min) / (max - min)
        expected = tv_tensors.Image(
            torch.tensor([0.0, 1 / 3, 2 / 3, 1.0]).view(1, 1, 2, 2)
        )
        assert torch.allclose(result, expected)

    def test_zscore_normalization(self):
        # Create a test image
        image = tv_tensors.Image(torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 2, 2))

        # Apply normalization
        normalizer = Normalize("zscore")
        result = normalizer(image)

        # Calculate mean and std manually
        mean = 2.5
        std = 1.2909944
        expected = tv_tensors.Image(
            torch.tensor(
                [(1 - mean) / std, (2 - mean) / std, (3 - mean) / std, (4 - mean) / std]
            ).view(1, 1, 2, 2)
        )
        assert torch.allclose(result, expected)

    def test_none_normalization(self):
        # Create a test image
        image = tv_tensors.Image(torch.tensor([1.0, 2.0, 3.0, 4.0]).view(1, 1, 2, 2))

        # Apply normalization
        normalizer = Normalize(None)
        result = normalizer(image)

        # Should return the original image
        assert torch.equal(result, image)

    def test_invalid_normalization(self):
        with pytest.raises(ValueError):
            Normalize("invalid_method")


class TestDownscaleImage:
    def test_downscale_numpy_array(self):
        # Create a test image
        image = np.ones((16, 16), dtype=np.float32)

        # Downscale by factor of 2
        result = downscale_image(image, factor=2)
        print(result.shape)

        # Check shape
        assert result.shape == (1, 8, 8)
        assert isinstance(result, tv_tensors.Image)

    def test_downscale_torch_tensor(self):
        # Create a test image
        image = torch.ones((1, 16, 16), dtype=torch.float32)

        # Downscale by factor of 4
        result = downscale_image(image, factor=4)

        # Check shape
        assert result.shape == (1, 4, 4)
        assert isinstance(result, tv_tensors.Image)

    def test_downscale_with_normalization(self):
        # Create a test image with values 0-255
        image = torch.linspace(0, 255, 16 * 16).reshape(1, 16, 16)

        # Downscale with minmax normalization
        result = downscale_image(image, factor=2, norm_method="minmax")

        # Values should be in [0, 1]
        assert result.min() >= 0 and result.max() <= 1

    def test_output_dtype(self):
        # Create a test image
        image = torch.ones((1, 16, 16), dtype=torch.float32)

        # Downscale with uint8 output
        result = downscale_image(image, factor=2, out_type=torch.uint8)

        # Check dtype
        assert result.dtype == torch.uint8


class TestImreadDownscaled:
    def test_imread_downscaled(self):
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix=".tiff") as tmp:
            # Create a test image and save it
            test_image = np.ones((16, 16), dtype=np.uint8) * 255
            tifffile.imwrite(tmp.name, test_image)

            # Read and downscale
            result = imread_downscaled(Path(tmp.name), factor=2, norm_method="minmax")

            # Check shape and type
            assert result.shape == (8, 8)
            assert result.dtype == np.uint8
