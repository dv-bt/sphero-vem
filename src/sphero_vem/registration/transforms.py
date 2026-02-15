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
    """Warp and image using an affine transform matrix"""
    grid = F.affine_grid(transform_matrix, size=img.shape, align_corners=False)
    return F.grid_sample(
        img, grid, mode=mode, padding_mode="zeros", align_corners=False
    )


def _to_homog(A: torch.Tensor):
    """Convert a 1x2x3 affine matrix to homogeneous coordinates 1x3x3"""
    A_homog = torch.eye(3).unsqueeze(0)
    A_homog[:, :2, :] = A
    return A_homog


def _from_homog(A_homog: torch.Tensor):
    """Convert a 1x3x3 matrix in homogeneous coordinates to a 1x2x3 matrix"""
    return A_homog[:, :2, :]


def _compose_transform(
    A: torch.Tensor, composed_matrices: list[torch.Tensor]
) -> torch.Tensor:
    """Compose current transformtion matrix with previous transformations.
    It takes the full list to handle edge cases better"""
    try:
        A_previous = _to_homog(composed_matrices[-1])
        A_composed = _to_homog(A) @ A_previous
        return _from_homog(A_composed)
    except IndexError:
        return A
