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
    """Mrophological opening with stucturing element with given connectivity"""
    struct = ski.morphology.ball(radius=radius)
    return ndi.binary_closing(binary_image, structure=struct, border_value=0)


@gpu_dispatch(return_to_host=True)
def median_filter(array: ArrayLike, size: int = 3) -> np.ndarray:
    """
    Applies a 3D median filter, using GPU acceleration if possible.
    """
    cellprob_smoothed = ndi.median_filter(array, size=size, mode="nearest")
    return cellprob_smoothed
