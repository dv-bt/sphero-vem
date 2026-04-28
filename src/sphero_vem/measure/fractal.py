"""Fractal dimension estimation via Minkowski-Bouligand tube scaling."""

import numpy as np
from scipy.stats import linregress
from sphero_vem.utils import check_isotropic
from sphero_vem.utils.accelerator import gpu_dispatch, xp, ArrayLike, to_host
from sphero_vem.measure.sdf import _calc_sdf


@gpu_dispatch()
def props_fractal(
    label_idx: int,
    labels: ArrayLike,
    spacing: tuple[float],
    sigma_frac: float = 0.7,
    n_steps: int = 30,
) -> dict:
    """
    Compute fractal dimension via Minkowski-Bouligand tube volume scaling.

    Computes a lightly smoothed signed distance field from the label mask,
    then measures how the volume of an espilon-tube around the zero level set
    scales with epsilon. For a smooth surface, D ≈ 2.0.

    Parameters
    ----------
    label_idx : int
        Index of the label to analyze.
    labels : ArrayLike
        3D integer label array (typically a cropped region containing `label_idx`).
    spacing : tuple[float]
        Isotropic voxel spacing in physical units.
    sigma_frac : float, optional
        Light Gaussian smoothing sigma in voxels for the SDF. Should be
        small (0.5-1.0) to remove EDT quantization artifacts while
        preserving meso-scale roughness. Default is 0.7.
    n_steps : int, optional
        Number of epsilon values sampled in log-space. Default is 30.

    Returns
    -------
    dict
        fractal_dim : float
            Estimated fractal dimension (D = 3 - slope).
        fractal_r2 : float
            R^2 of the log-log linear fit.
        fractal_eps_min : float
            Lower epsilon bound used for fitting.
        fractal_eps_max : float
            Upper epsilon bound (after auto-trimming).
        fractal_n_points : int
            Number of points used in the fit.

    Notes
    -----
    Uses a light smoothing (sigma = 0.5-1.0 voxels) rather than the raw SDF
    to suppress EDT quantization artifacts near small ε, but avoid the heavy
    smoothing (sigma > 1.5) used for curvature.

    The lower epsilon bound is calcualtes as the maximum between `1.5 * voxel_size` and
    `3 * sigma_frac * voxel_size`.

    The upper epsilon bound is auto-trimmed to exclude the finite-size
    saturation regime where tube volume growth slows. A minimum of 5 epsilon values
    are used for the log-log regression.

    Raises
    ------
    ValueError
        If the spacing is not isotropic
    """

    check_isotropic(spacing=spacing, raise_error=True)
    voxel_size = spacing[0]
    sdf = _calc_sdf(label_idx, labels, spacing=spacing, sigma=sigma_frac)

    object_radius = float(
        ((3 / (4 * xp.pi)) * xp.sum(sdf < 0) * voxel_size**3) ** (1 / 3)
    )
    eps_min = max(3 * sigma_frac * voxel_size, 1.5 * voxel_size)
    eps_max = object_radius / 2

    epsilons = np.geomspace(eps_min, eps_max, n_steps)
    abs_sdf = xp.abs(sdf)

    volumes = xp.array(
        [xp.sum(abs_sdf < float(eps)) * voxel_size**3 for eps in epsilons]
    )

    # Move to CPU for log and regression
    volumes_cpu = to_host(volumes)

    # Filter zero volumes to avoid numerical artifacts.
    valid = volumes_cpu > 0
    if np.sum(valid) < 5:
        return {
            "fractal_dim": np.nan,
            "fractal_r2": np.nan,
            "fractal_eps_min": eps_min,
            "fractal_eps_max": eps_max,
            "fractal_n_points": 0,
        }

    log_eps = np.log(epsilons[valid])
    log_vol = np.log(volumes_cpu[valid])

    # Auto-trim saturation regime from upper end
    best_r2 = 0
    best_idx = len(log_eps)
    for i in range(len(log_eps), 5, -1):
        slope, _, r_value, _, _ = linregress(log_eps[:i], log_vol[:i])
        if r_value**2 > best_r2:
            best_r2 = r_value**2
            best_idx = i

    slope, _, r_value, _, _ = linregress(log_eps[:best_idx], log_vol[:best_idx])

    return {
        "fractal_dim": 3 - slope,
        "fractal_r2": r_value**2,
        "fractal_eps_min": eps_min,
        "fractal_eps_max": float(np.exp(log_eps[best_idx - 1])),
        "fractal_n_points": best_idx,
    }
