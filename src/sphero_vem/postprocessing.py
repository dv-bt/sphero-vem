"""
Postprocessing functions
"""

import numpy as np
from sphero_vem.utils.accelerator import (
    xp,
    ndi,
    ski,
    ArrayLike,
    gpu_dispatch,
)


@gpu_dispatch(return_to_host=True)
def filter_and_relabel(labels: ArrayLike, min_size: int) -> np.ndarray:
    """
    Remove components smaller than `min_size` voxels and relabel remaining components
    to contiguous IDs (0=background, 1..K)

    Parameters
    ----------
    labels : np.ndarray
        Integer label volume, background must be 0.
    min_size : int
        Minimum number of voxels for a component to be kept.

    Returns
    -------
    labels_out : np.ndarray
        New label volume with small components removed and contiguous IDs.
    """
    # Flatten array
    flat = labels.ravel()

    max_label = int(flat.max())
    counts = xp.bincount(flat, minlength=max_label + 1)

    # Keep labels with at least min_size voxels, except background
    keep = counts >= min_size
    keep[0] = False
    keep_voxel = keep[flat]

    filtered = xp.where(keep_voxel, flat, 0).reshape(labels.shape)
    del keep_voxel, labels

    relabeled = ski.segmentation.relabel_sequential(filtered)
    return relabeled[0]


@gpu_dispatch(return_to_host=True)
def binary_closing(binary_image: ArrayLike, radius: int = 1) -> np.ndarray:
    """Apply 3D morphological closing with a spherical structuring element.

    Parameters
    ----------
    binary_image : ArrayLike
        Binary 3D input array.
    radius : int, optional
        Radius of the ball-shaped structuring element. Default is 1.

    Returns
    -------
    numpy.ndarray
        Binary 3D array after morphological closing.
    """
    struct = ski.morphology.ball(radius=radius)
    return ndi.binary_closing(binary_image, structure=struct, border_value=0)


@gpu_dispatch(return_to_host=True)
def median_filter(array: ArrayLike, size: int = 3) -> np.ndarray:
    """Apply a 3D median filter using GPU acceleration if available.

    Parameters
    ----------
    array : ArrayLike
        Input 3D array.
    size : int, optional
        Size of the median filter kernel in each dimension. Default is 3.

    Returns
    -------
    numpy.ndarray
        Filtered array of the same shape as *array*.
    """
    cellprob_smoothed = ndi.median_filter(array, size=size, mode="nearest")
    return cellprob_smoothed


@gpu_dispatch(return_to_host=True)
def guided_filter(
    array: ArrayLike,
    guide: ArrayLike,
    radius: int,
    eps: float,
) -> np.ndarray:
    """Apply a guided filter using a guide image to smooth an input image.

    Parameters
    ----------
    array : np.ndarray
        Input image to be filtered. It should have the same shape as guide.
    guide : np.ndarray
        Guide image. Edges in this image are preserved in the output.
        Should be normalised to [0, 1].
    radius : int
        Half-size of the local window. Window size is (2*radius+1)^2.
    eps : float
        Regularisation parameter. Larger values give more smoothing, smaller values
        make the filter more edge-preserving.

    Returns
    -------
    np.ndarray
        Filtered image of same shape as inp.

    References
    ----------
    .. 1. He, K., Sun, J. & Tang, X. Guided Image Filtering.
       IEEE Transactions on Pattern Analysis and Machine Intelligence 35, 1397-1409 (2013).
       https://doi.org/10.1109/TPAMI.2012.213.

    """

    def box(x: ArrayLike) -> ArrayLike:
        return ndi.uniform_filter(x.astype(xp.float32), size=2 * radius + 1)

    mean_G = box(guide)
    mean_P = box(array)
    mean_GP = box(guide * array)
    mean_GG = box(guide * guide)

    cov_GP = mean_GP - mean_G * mean_P
    del mean_GP

    var_G = mean_GG - mean_G * mean_G
    del mean_GG

    a = cov_GP / (var_G + eps)
    del cov_GP, var_G

    b = mean_P - a * mean_G
    del mean_P, mean_G

    return box(a) * guide + box(b)
