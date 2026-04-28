"""
Utility functions used for nanoparticle segmentation
"""

from pathlib import Path
import zarr
import numpy as np
import dask.array as da
import dask
from dask_image.ndfilters import gaussian_filter
from sphero_vem.io import write_zarr
from sphero_vem.utils import dirname_from_spacing
from sphero_vem.utils.accelerator import (
    xp,
    gpu_dispatch,
    ArrayLike,
)


@gpu_dispatch(return_to_host=True)
def bincount_ubyte(image: ArrayLike) -> np.ndarray:
    """Compute a 256-bin intensity histogram for a uint8 image.

    This function uses GPU acceleration when available.

    Parameters
    ----------
    image : ArrayLike
        Input uint8 image array of any shape.

    Returns
    -------
    numpy.ndarray
        Integer histogram of shape (256,) with bin counts per intensity level.
    """
    return xp.bincount(image.ravel(), minlength=256).astype(xp.int64)


def downsample_posterior(
    root_path: Path,
    sigma: tuple[float, float, float],
    dst_spacing: tuple[int, int, int],
    src_spacing: tuple[int, int, int] = (50, 10, 10),
    n_workers: int = 4,
) -> None:
    """Downsample posterior probabilities via logit-space Gaussian smoothing and average pooling.

    Parameters
    ----------
    root_path : Path
        Path to the Zarr archive.
    sigma : tuple[float, float, float]
        Gaussian smoothing sigma (Z, Y, X) applied in logit space.
    dst_spacing : tuple[int, int, int]
        Target voxel spacing (Z, Y, X) in nanometers.
    src_spacing : tuple[int, int, int]
        Source voxel spacing (Z, Y, X) in nanometers. Default (50, 10, 10).
    n_workers : int
        Number of threads for dask threaded scheduler. Default 4.
    """
    root = zarr.open_group(root_path, mode="a")
    spacing_dir = dirname_from_spacing(src_spacing)
    src_zarr: zarr.Array = root.get(f"labels/nps/posterior/{spacing_dir}")

    ds_ratio = np.asarray(dst_spacing) / np.asarray(src_spacing)
    final_shape = (np.array(src_zarr.shape) / ds_ratio).astype(int).tolist()

    tmp_chunks = (64, 1000, 1000)
    prob: da.Array = da.from_zarr(src_zarr).rechunk(tmp_chunks).astype(np.float32)

    eps = np.float32(1e-5)
    prob = da.clip(prob, eps, 1 - eps)
    logit = da.log(prob / (1 - prob))
    logit_smooth = gaussian_filter(logit, sigma=list(sigma))
    prob_smooth = 1 / (1 + da.exp(-logit_smooth))

    axes_dict = {i: int(ds_ratio[i]) for i in range(3)}
    downsampled = da.coarsen(np.mean, prob_smooth, axes_dict, trim_excess=True)

    with dask.config.set(num_workers=n_workers):
        write_zarr(
            root,
            downsampled,
            dst_path=f"labels/nps/posterior/{dirname_from_spacing(dst_spacing)}",
            src_zarr=src_zarr,
            shape=final_shape,
            dtype="f2",
            processing={
                "step": "downscaling",
                "sigma": sigma,
                "target spacing": dst_spacing,
            },
        )
