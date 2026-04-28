"""
This module contains losses and metrics used throughout the library
"""

from functools import partial
import torch
import torch.nn.functional as F
from kornia.losses import ssim_loss


def ncc_loss(img1: torch.Tensor, img2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalized cross-correlation loss between two image tensors.

    Parameters
    ----------
    img1 : torch.Tensor
        First input image tensor of shape (N, C, H, W).
    img2 : torch.Tensor
        Second input image tensor. Must have the same shape as *img1*.
    eps : float, optional
        Small constant added to the denominator for numerical stability.
        Default is 1e-6.

    Returns
    -------
    torch.Tensor
        Scalar NCC loss in [0, 2]; 0 for perfectly correlated images.
    """
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
    """Factory that resolves a loss function by name and forwards calls to it.

    Available loss names: ``"mse"``, ``"mae"``, ``"ncc"``, ``"ssim"``.

    Parameters
    ----------
    loss_name : str
        Name of the loss function to use. Case-sensitive.

    Raises
    ------
    ValueError
        If *loss_name* is not in the registry.
    """

    _losses = {
        "mse": F.mse_loss,
        "mae": partial(F.l1_loss, reduction="mean"),
        "ncc": ncc_loss,
        "ssim": ssim_loss,
    }

    def __init__(self, loss_name: str):
        """Resolve *loss_name* to a callable loss function.

        Parameters
        ----------
        loss_name : str
            Name of the loss function. Must be one of the keys in ``_losses``.

        Raises
        ------
        ValueError
            If *loss_name* is not a registered loss name.
        """
        try:
            self._fun = self._losses[loss_name]
        except KeyError:
            raise ValueError(
                f"Invalid loss: '{loss_name}'. "
                f"Available losses are: {list(self._losses.keys())}"
            )

    def __call__(self, *args, **kwargs):
        """Compute the loss by forwarding all arguments to the resolved function.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the loss function.
        **kwargs
            Keyword arguments forwarded to the loss function.

        Returns
        -------
        torch.Tensor
            Loss value returned by the underlying loss function.
        """
        return self._fun(*args, **kwargs)
