"""
This module contains losses and metrics used throughout the library
"""

from functools import partial
import torch
import torch.nn.functional as F
from kornia.losses import ssim_loss


def ncc_loss(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized cross correlation loss"""
    img1_c = img1 - img1.mean(dim=[2, 3], keepdim=True)
    img2_c = img2 - img2.mean(dim=[2, 3], keepdim=True)
    num = (img1_c * img2_c).mean(dim=[2, 3])
    den = (
        (img1_c.pow(2).mean(dim=[2, 3]) * img2_c.pow(2).mean(dim=[2, 3]))
        .clamp_min(0)
        .sqrt()
    )
    return 1.0 - (num / (den + eps)).mean()


class LossDispatcher:
    _losses = {
        "mse": F.mse_loss,
        "mae": partial(F.l1_loss, reduction="mean"),
        "ncc": ncc_loss,
        "ssim": ssim_loss,
    }

    def __init__(self, loss_name: str):
        try:
            self._fun = self._losses[loss_name]
        except KeyError:
            raise ValueError(
                f"Invalid loss: '{loss_name}'. "
                f"Available losses are: {list(self._losses.keys())}"
            )

    def __call__(self, *args, **kwargs):
        return self._fun(*args, **kwargs)
