"""
Functions for preprocessing images.
"""

from typing import Literal
import numpy as np
import torch
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms


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
