"""
Affine transform math for 2D image registration.

This module contains pure transform operations: building affine matrices from
parameters, warping images, and composing sequential transforms. It has no I/O
or zarr dependencies — only torch.
"""

from enum import Enum
import torch
import torch.nn.functional as F


class TransformType(Enum):
    SIMILARITY = "similarity"
    RIGID = "rigid"
    AFFINE = "affine"


def _affine_transform(q: torch.Tensor) -> torch.Tensor:
    """
    Build a batch of 2x3 affine transformation matrices from 6 parameters.
    Constructs 2D affine matrices with translation, rotation, scaling, and shear.
    The order of operations is: Scale -> Shear -> Rotate -> Translate.

    Parameters
    ----------
    q : torch.Tensor
        Tensor of shape (N, 6) containing parameters per example:
        [tx, ty, theta, sx, sy, k]
        - tx, ty: translation components
        - theta: rotation angle in radians
        - sx, sy: scaling factors
        - k: horizontal shear factor

    Returns
    -------
    torch.Tensor
        Tensor of shape (N, 2, 3) representing the affine transformation.
        Each 2x3 matrix is the result of composing the individual transformations.
    """
    t = q[:, :2]
    ang = q[:, 2]
    scale = q[:, 3:5]
    shear_k = q[:, 5]

    c, s = torch.cos(ang), torch.sin(ang)
    sx, sy = scale[:, 0], scale[:, 1]

    # Construct the 2x2 linear transformation matrix (Scale -> Shear -> Rotate)
    # This combines the effects of scaling, shearing, and rotation
    # A = R @ K @ S
    # R = [[c, -s], [s, c]]
    # K = [[1, k], [0, 1]] (horizontal shear)
    # S = [[sx, 0], [0, sy]]
    A_11 = sx * c
    A_12 = sy * (shear_k * c - s)
    A_21 = sx * s
    A_22 = sy * (shear_k * s + c)
    A = torch.stack([A_11, A_12, t[:, 0], A_21, A_22, t[:, 1]], dim=-1)

    return A.view(-1, 2, 3)


def _warp_affine(
    img: torch.Tensor, transform_matrix: torch.Tensor, mode: str = "bilinear"
) -> torch.Tensor:
    """Warp an image using an affine transform matrix.

    Parameters
    ----------
    img : torch.Tensor
        Input image tensor of shape (N, C, H, W).
    transform_matrix : torch.Tensor
        Affine transformation matrix of shape (N, 2, 3).
    mode : str, optional
        Interpolation mode for ``F.grid_sample``. Default is ``"bilinear"``.

    Returns
    -------
    torch.Tensor
        Warped image tensor of shape (N, C, H, W). Out-of-bounds pixels are
        filled with zeros.
    """
    grid = F.affine_grid(transform_matrix, size=img.shape, align_corners=False)
    return F.grid_sample(
        img, grid, mode=mode, padding_mode="zeros", align_corners=False
    )


def _to_homog(A: torch.Tensor):
    """Convert a (1, 2, 3) affine matrix to homogeneous (1, 3, 3) coordinates.

    Parameters
    ----------
    A : torch.Tensor
        Affine matrix of shape (1, 2, 3).

    Returns
    -------
    torch.Tensor
        Homogeneous matrix of shape (1, 3, 3), with the last row set to
        [0, 0, 1].
    """
    A_homog = torch.eye(3).unsqueeze(0)
    A_homog[:, :2, :] = A
    return A_homog


def _from_homog(A_homog: torch.Tensor):
    """Convert a (1, 3, 3) homogeneous matrix back to (1, 2, 3) affine form.

    Parameters
    ----------
    A_homog : torch.Tensor
        Homogeneous matrix of shape (1, 3, 3).

    Returns
    -------
    torch.Tensor
        Affine matrix of shape (1, 2, 3), obtained by dropping the last row.
    """
    return A_homog[:, :2, :]


def _compose_transform(
    A: torch.Tensor, composed_matrices: list[torch.Tensor]
) -> torch.Tensor:
    """Compose the current affine matrix with the accumulated transform chain.

    Parameters
    ----------
    A : torch.Tensor
        Current affine matrix of shape (1, 2, 3).
    composed_matrices : list[torch.Tensor]
        List of previously composed affine matrices. The last element is
        used as the preceding transform.

    Returns
    -------
    torch.Tensor
        Composed affine matrix of shape (1, 2, 3). If *composed_matrices* is
        empty, returns *A* unchanged.
    """
    try:
        A_previous = _to_homog(composed_matrices[-1])
        A_composed = _to_homog(A) @ A_previous
        return _from_homog(A_composed)
    except IndexError:
        return A
