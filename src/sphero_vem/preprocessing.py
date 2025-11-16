"""
Functions for preprocessing images.
"""

from pathlib import Path
from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
from torchvision.transforms.functional import resize
import zarr
import dask.array as da
import dask_image.ndinterp
import dask_image
from sphero_vem.utils import dirname_from_spacing, create_ome_multiscales


class Normalize(torch.nn.Module):
    """Class that normalizes an image with the give method. This is typically
    intended for use in torchvision transform pipelines"""

    def __init__(self, norm_method: Literal["minmax", "zscore"] | None):
        super().__init__()
        if norm_method == "minmax":
            self.norm_fun = self._minmax
        elif norm_method == "zscore":
            self.norm_fun = self._zscore
        elif norm_method is None:
            self.norm_fun = lambda image: image
        else:
            raise ValueError("Invalid argument for norm_method")

    def forward(self, image: tv_tensors.Image):
        image_norm = tv_tensors.wrap(self.norm_fun(image), like=image)
        return image_norm

    def _minmax(self, image: tv_tensors.Image):
        """Normalizes an image using min max normalization"""
        image_norm = (image - image.min()) / (image.max() - image.min())
        image_norm = image_norm.clamp(0, 1)
        return image_norm

    def _zscore(self, image: tv_tensors.Image):
        """Normalizes an image using z-score normalization (standardization)"""
        image_norm = (image - image.mean()) / image.std()
        return image_norm


def downscale_image(
    image: np.ndarray | torch.Tensor,
    factor: int,
    norm_method: str | None = None,
    out_type: torch.dtype = torch.float32,
    scale_range: bool = True,
) -> tv_tensors.Image:
    """Downscale an image by a specified integer factor.
    The image can be a numpy array or a torch tensor.
    Transforms are run on the CPU. The tensor should be sent to device later
    """
    if (not scale_range) and norm_method and (out_type != torch.float32):
        print(
            "WARNING: image range scaling is necessary during normalization if "
            "data type is not torch.float32\n"
            "scale_range set to True"
        )
        scale_range = True

    # Enforce correct type for factor
    factor = int(factor)

    resize = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=scale_range),
            transforms.Resize(image.shape[-1] // factor, antialias=True),
            Normalize(norm_method),
            transforms.ToDtype(out_type, scale=scale_range),
        ]
    )
    image_resized: tv_tensors.Image = resize(image)
    return image_resized


def downscale_labels(
    image: np.ndarray | torch.Tensor,
    factor: int,
) -> tv_tensors.Image:
    """Downscale a labels by a specified integer factor.
    This is the equivalent of downscale_image but specific for segmentation labels
    and masks
    """

    # Enforce correct type for factor
    factor = int(factor)

    resize = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.Resize(
                image.shape[-1] // factor,
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
        ]
    )
    image_resized: tv_tensors.Image = resize(image)
    return image_resized


def create_pyramid(
    image: torch.Tensor, num_levels: int, factor: int
) -> list[torch.tensor]:
    """Creates a multi-resolution pyramid for an image tensor."""
    pyramid = [image]
    for _ in range(num_levels - 1):
        image = resize(image, image.shape[-1] // factor)
        pyramid.append(image)
    return list(reversed(pyramid))


def downscale_tensor(
    image: torch.Tensor, factor: int, mode: str = "bilinear"
) -> torch.tensor:
    """Dowscales a tensor or a batch of tensors using bilinar interpolation"""
    n_dim = image.dim()
    while image.dim() < 4:
        image = image.unsqueeze(0)
    if mode == "nearest":
        image_ds: torch.Tensor = F.interpolate(
            image,
            scale_factor=1 / factor,
            mode=mode,
        )
    else:
        image_ds: torch.Tensor = F.interpolate(
            image,
            scale_factor=1 / factor,
            mode=mode,
            align_corners=False,
            antialias=True,
        )
    # Make sure that output has the same number of dimensions of input
    while image_ds.dim() > n_dim:
        image_ds = image_ds.squeeze(0)
    return image_ds


def resample_array(
    zarr_path: Path,
    array_path: str,
    target_spacing: tuple[int, int, int],
    order: int = 1,
) -> None:
    """Resample an array in a Zarr archive to have the target spacing."""

    root = zarr.open_group(zarr_path)
    src_array: zarr.Array = root.get(array_path)
    if not src_array:
        raise FileExistsError(
            f"The source array {array_path} was not found under {zarr_path}"
        )

    array_dask = da.from_zarr(zarr_path)
    original_shape = array_dask.shape

    spacing_dir = dirname_from_spacing(target_spacing)
    reference_array: zarr.Array = root.get(f"images/{spacing_dir}")
    if not reference_array:
        raise NotImplementedError(
            "Arbitrary spacing not yet supported. "
            "Please specify a spacing that is already under images/"
        )
    target_shape = reference_array.shape
    scale_factors = np.array(original_shape) / np.array(target_shape)
    scaling_matrix = np.diag(scale_factors)

    resampled_array = dask_image.ndinterp.affine_transform(
        array_dask,
        matrix=scaling_matrix,
        output_shape=target_shape,
        order=order,
    )

    parent_group: zarr.Group = root.get(str(Path(src_array.path).parent))
    resampled_array.to_zarr(
        parent_group,
        component=spacing_dir,
        overwrite=True,
        compute=True,
        dtype=src_array.dtype,
        compressor="zstd",
    )

    # Array metadata
    arr: zarr.Array = parent_group[spacing_dir]
    arr.attrs["spacing"] = target_spacing
    arr.attrs["processing"] = src_array.attrs["processing"] + [
        {
            "step": "resample",
            "order": order,
            "scale_factors": [float(i) for i in scale_factors],
        }
    ]
    arr.attrs["inputs"] = src_array.path
    create_ome_multiscales(parent_group)
