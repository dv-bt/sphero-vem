"""Module containing input/output functions"""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import tifffile
from sphero_vem.preprocessing import downscale_image, downscale_labels, downscale_tensor


def imwrite(
    fname: Path, image: np.ndarray, uncompressed: bool = True, **kwargs
) -> None:
    """Save TIFF images with default zip compression. Compression can be disabled
    with the uncompressed argument. This also disables any other optional keyword
    argument passed"""
    defaults = {
        "compression": "zlib",
        "compressionargs": {"level": 6},
        "predictor": 2,
        "tile": (256, 256),
    }
    options = {**defaults, **kwargs} if not uncompressed else {}
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


def write_stack(data_dir: Path, out_file: Path, channel_axis: bool = False) -> None:
    """Merge images in a folder into a single ZYX tif. If channel_axis option is True,
    save also a channel of size 1 so that every image is CYZ"""

    volume_stack = read_stack(data_dir, channel_axis=channel_axis)
    tifffile.imwrite(out_file, volume_stack)


def read_stack(data_dir: Path, channel_axis: bool = False) -> np.ndarray:
    """Sequentially read images in directory and merge them into a 3D stack with shape
    ZYX. If the channel_axis option is on, a channel dimension of size 1 is added to have
    shape ZCYX"""

    image_list = sorted(list(data_dir.glob("*.tif")))
    first_image = tifffile.imread(image_list[0])
    image_shape = (1, *first_image.shape) if channel_axis else first_image.shape
    volume_stack = np.empty((len(image_list), *image_shape), first_image.dtype)
    for i, image_path in enumerate(tqdm(image_list, "Reading slices")):
        image = tifffile.imread(image_path).reshape(*image_shape)
        volume_stack[i] = image
    return volume_stack


def read_tensor(
    image_path: Path,
    dtype: torch.dtype = torch.float32,
    ds_factor: int = 1,
    return_4d: bool = True,
) -> torch.Tensor:
    """Read a tiff image as a pytorch tensor. Returns a tensor of shape 1 x H x W.
    If ds_factor > 1, applies downscaling by that factor to the image.
    If return_4d is true, returns a tensor of size 1 x 1 x H x W"""
    image = tifffile.imread(image_path)
    image_torch = torch.tensor(image, dtype=dtype)
    if return_4d:
        while image_torch.dim() < 4:
            image_torch = image_torch.unsqueeze(0)
    if ds_factor > 1:
        image_torch = downscale_tensor(image_torch, ds_factor)
    return image_torch
