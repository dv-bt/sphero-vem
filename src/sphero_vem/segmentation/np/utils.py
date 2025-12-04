"""
Utility functions used for nanoparticle segmentation
"""

from pathlib import Path
import zarr
import numpy as np
import dask.array as da
from dask_image.ndfilters import gaussian_filter
from sphero_vem.io import write_zarr
from sphero_vem.utils import dirname_from_spacing
from sphero_vem.utils.accelerator import (
    xp,
    gpu_dispatch,
    ArrayLike,
    da_to_device,
    da_to_host,
)


@gpu_dispatch(return_to_host=True)
def bincount_ubyte(image: ArrayLike) -> np.ndarray:
    """Calculates image histogram with GPU acceleration"""
    return xp.bincount(image.ravel(), minlength=256).astype(xp.int64)


def downsample_posterior(
    root_path: Path,
    sigma: tuple[float, float, float],
    dst_spacing: tuple[int, int, int],
    src_spacing: tuple[int, int, int] = (50, 10, 10),
) -> None:
    """Downsample posterior using average pooling after Gaussian antialiasing"""
    root = zarr.open_group(root_path, mode="a")
    spacing_dir = dirname_from_spacing(src_spacing)
    src_zarr: zarr.Array = root.get(f"labels/nps/posterior/{spacing_dir}")

    # Calculate downsampling parameters
    ds_ratio = np.asarray(dst_spacing) / np.asarray(src_spacing)
    final_shape = (src_zarr.shape // ds_ratio).astype(int).tolist()

    ## Dask pipeline
    tmp_chunks = (64, 1000, 1000)
    prob = da.from_zarr(src_zarr).rechunk(tmp_chunks)
    prob = da_to_device(prob)
    prob = prob.astype(xp.float32)

    # Convert to logit and smooth
    eps = 1e-5
    prob = da.clip(prob, eps, 1 - eps)
    logit = da.log(prob / (1 - prob))
    logit_smooth = gaussian_filter(logit, sigma=sigma)
    prob_smooth = 1 / (1 + da.exp(-logit_smooth))

    # Average pooling
    axes_dict = {i: int(ds_ratio[i]) for i in range(3)}
    downsampled = da.coarsen(
        xp.mean,
        prob_smooth,
        axes_dict,
        trim_excess=True,
    )
    downsampled = da_to_host(downsampled)

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
