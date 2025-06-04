"""Module containing input/output functions"""

from pathlib import Path
import numpy as np
import torch
import tifffile
from sphero_vem.preprocessing import downscale_image, downscale_labels


def imwrite_labels(fname: Path, image: np.ndarray, **kwargs) -> None:
    """Save TIFF images with default compression for labels."""
    defaults = {
        "compression": "deflate",
        "compressionargs": {"level": 6},
        "predictor": 2,
        "tile": (256, 256),
    }
    options = {**defaults, **kwargs}
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
