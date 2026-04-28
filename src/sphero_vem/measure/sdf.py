"""SDF computation, surface area estimation, and mesh extraction."""

import numpy as np
from sphero_vem.utils import check_isotropic
from sphero_vem.utils.accelerator import ndi, gpu_dispatch, xp, ArrayLike, to_host


@gpu_dispatch()
def _calc_sdf(
    label_idx: int,
    labels: ArrayLike,
    spacing: tuple[float, float, float],
    sigma: float = 3,
) -> tuple[ArrayLike, ArrayLike]:
    """Calculate smooethed signed distance function (SDF) for the given label.

    The function uses the sign convention SDF < 0 on the inside of the object. When
    CUDA is available, this function uses GPU acceleration.

    Parameters
    ----------
    label_idx : int
        Index of the label.
    labels : ArrayLike
        Array with the labeled image. Consider passing the image cropped to the region
        of interest of the label for efficiency.
    spacing : tuple[float, float, float]
        Physical spacing of the voxel grid. The SDF will be returned in these units.
    sigma : float
        Standard deviation (in voxels) of the Gaussian kernel used for smoothing the SDF.
        If 0, smoothing is skipped and the function returns the un-smoothed SDF.
        Default is 3.

    Returns
    -------
    ArrayLike
        Signed distance function array with the same shape as *labels*.
        Values are negative inside the object and positive outside,
        expressed in the physical units defined by *spacing*.
    """
    mask = labels == label_idx
    sdf = ndi.distance_transform_edt(
        ~mask, sampling=spacing
    ) - ndi.distance_transform_edt(mask, sampling=spacing)
    if sigma > 0:
        sdf = ndi.gaussian_filter(sdf, sigma=sigma)
    return sdf


@gpu_dispatch()
def _calc_surface(
    sdf: ArrayLike, spacing: ArrayLike, epsilon_voxels: float = 1.5
) -> float:
    """
    Calculate surface are from the 0-level set of the SDF.

    Surface is calculated using a smeared-out heavyside function, following the approach
    described in Osher, S. & Fedkiw, R. (2003).
    The function currently only supports calcultion on isotropic voxels.

    Parameters
    ----------
    sdf : ArrayLike
        Array contining the smoothed SDF.
    spacing : ArrayLike
        Physical spacing of the voxel grid. It should have the same units as the SDF
        values. Spacing must be isotropic.
    epsilon_voxels : float
        Bandwith of the smeared heavyside function used for calculating the surface,
        in voxels.
        Default is 1.5

    Returns
    -------
    float
        The surface area of the object.

    Raises
    ------
    ValueError
        If the spacing is not isotropic.


    References
    ----------
    .. 1. Osher, S. & Fedkiw, R. Level Set Methods and Dynamic Implicit Surfaces.
          vol. 153 (Springer New York, New York, NY, 2003).

    """
    check_isotropic(to_host(spacing), raise_error=True)

    # Convert epsilon to physical units.
    epsilon = epsilon_voxels * spacing[0]

    # Smoothed delta
    mask = xp.abs(sdf) <= epsilon
    delta = xp.zeros_like(sdf)
    delta[mask] = (1 + xp.cos(np.pi * sdf[mask] / epsilon)) / (2 * epsilon)

    # Gradient magnitude
    grad = xp.gradient(sdf, *spacing)
    grad_mag = xp.sqrt(sum(g**2 for g in grad))

    voxel_volume = xp.prod(spacing)

    return float(xp.sum(delta * grad_mag) * voxel_volume)


@gpu_dispatch()
def props_sdf(
    label_idx: int,
    labels: ArrayLike,
    spacing: tuple,
    sigma: float = 3,
    eps_voxels: float = 1.5,
) -> tuple[dict, ArrayLike]:
    """
    Calculate label properties directly from the implicit SDF representation.

    Parameters
    ----------
    label_idx : int
        Index of the label.
    labels : ArrayLike
        Array with the labeled image. Consider passing the image cropped to the region
        of interest of the label for efficiency.
    spacing : tuple[float, float, float]
        Physical spacing of the voxel grid. The SDF will be returned in these units.
    sigma : float
        Standard deviation of the Gaussian kernel used for smoothing the SDF in voxels.
        Default is 3.
    epsilon_voxels : float
        Bandwith of the smeared Heavyside function used for calculating the surface,
        in voxels.
        Default is 1.5.

    Returns
    -------
    dict
        Dictionary of properties for the given label:
        - volume: volume of the object.
        - surface_area_real: surface area of the boundary.
        - sphericity: ratio of the surface area of a sphere of equivalent volume over
          the measured surface area.
        - surface_area_boundary: surface are of the object that is cropped by the
          image boundary.
        - truncation_fraction: surface_area_boundary / surface_area_total.
        - diam_equiv: diameter of a sphere with equivalent volume.
    ArrayLike
        An array containing the signed distance function (SDF < 0 on the inside).


    """
    spacing_arr = xp.asarray(spacing)
    sdf = _calc_sdf(
        label_idx=label_idx,
        labels=labels,
        spacing=spacing,
        sigma=sigma,
        _to_host=False,
    )
    volume = float(xp.sum(sdf < 0) * xp.prod(spacing_arr))
    surface_area = _calc_surface(
        sdf, spacing=spacing_arr, epsilon_voxels=eps_voxels, _to_host=False
    )
    sphericity = xp.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / surface_area
    crop_area = _get_surface_boundary(sdf, spacing=spacing)

    results = {
        "volume": volume,
        "surface_area_real": surface_area,
        "sphericity": sphericity,
        "surface_area_boundary": crop_area,
        "truncation_fraction": crop_area / (surface_area + crop_area),
        "diam_equiv": (3 / (4 * xp.pi) * volume) ** (1 / 3) * 2,
    }
    return results, sdf


@gpu_dispatch()
def _get_surface_boundary(sdf: ArrayLike, spacing: tuple) -> float:
    """
    Get surface of portions of objects cropped by image bounds

    Parameters
    ----------
    sdf : ArrayLike
        Array contining the smoothed SDF.
    spacing : ArrayLike
        Physical spacing of the voxel grid. It should have the same units as the SDF
        values.

    Returns
    -------
    float
        The surface of the cropped areas.
    """

    boundary_mask = np.zeros(sdf.shape, dtype=bool)
    for dim in range(sdf.ndim):
        slc = [slice(None)] * sdf.ndim

        slc[dim] = 0
        boundary_mask[tuple(slc)] = True

        slc[dim] = -1
        boundary_mask[tuple(slc)] = True

    return float(xp.sum(sdf[boundary_mask] < 0)) * spacing[0] ** 2


@gpu_dispatch()
def _sample(
    sdf: ArrayLike,
    voxel_coords: ArrayLike,
    offset: tuple[float, float, float] = (0, 0, 0),
) -> ArrayLike:
    """
    Sample SDF values at coordinates with optional offset.

    Parameters
    ----------
    sdf : ArrayLike
        Signed distance function array.
    voxel_coords : ArrayLike
        Sampling coordinates in voxel units, shape (N, 3).
    offset : tuple[float, float, float], optional
        Offset to apply to coordinates in voxel units, by default (0, 0, 0).

    Returns
    -------
    ArrayLike
        Interpolated SDF values at the offset coordinates, shape (N,).
    """

    coords = (voxel_coords + xp.asarray(offset)).T
    return ndi.map_coordinates(sdf, coords, order=1)


@gpu_dispatch()
def _first_deriv(
    sdf: ArrayLike,
    voxel_coords: ArrayLike,
    axis: int,
    spacing: tuple[float, float, float],
    h: float = 0.5,
) -> ArrayLike:
    """
    Compute first partial derivative using central difference approximation.

    Parameters
    ----------
    sdf : ArrayLike
        Signed distance function array.
    voxel_coords : ArrayLike
        Sampling coordinates in voxel units, shape (N, 3).
    axis : int
        Axis along which to compute the derivative (0=z, 1=y, 2=x).
    spacing : tuple[float, float, float]
        Voxel spacing in physical units (z, y, x).
    h : float, optional
        Step size in voxel units for finite difference, by default 0.5.

    Returns
    -------
    ArrayLike
        First derivative values in physical units, shape (N,).
    """

    o = xp.zeros(3)
    o[axis] = h
    return (_sample(sdf, voxel_coords, o) - _sample(sdf, voxel_coords, -o)) / (
        2 * h * spacing[axis]
    )


@gpu_dispatch()
def _second_deriv(
    sdf: ArrayLike,
    voxel_coords: ArrayLike,
    axis1: int,
    axis2: int,
    spacing: tuple[float, float, float],
    h: float = 0.5,
    f0: ArrayLike | None = None,
) -> ArrayLike:
    """
    Compute second partial derivative using central difference approximation.

    For pure second derivatives (axis1 == axis2), uses the standard three-point
    stencil. For mixed derivatives, uses the four-point crossed stencil.

    Parameters
    ----------
    sdf : ArrayLike
        Signed distance function array.
    voxel_coords : ArrayLike
        Sampling coordinates in voxel units, shape (N, 3).
    axis1 : int
        First axis for differentiation (0=z, 1=y, 2=x).
    axis2 : int
        Second axis for differentiation (0=z, 1=y, 2=x).
    spacing : tuple[float, float, float]
        Voxel spacing in physical units (z, y, x).
    h : float, optional
        Step size in voxel units for finite difference, by default 0.5.
    f0 : ArrayLike | None, optional
        Pre-computed SDF values at voxel_coords. If None and axis1 == axis2,
        will be computed internally. Ignored for mixed derivatives.

    Returns
    -------
    ArrayLike
        Second derivative values in physical units, shape (N,).
    """

    o1 = np.zeros(3)
    o2 = np.zeros(3)
    o1[axis1] = h
    o2[axis2] = h

    if axis1 == axis2:
        # Pure second derivative: (f(+) - 2f(0) + f(-)) / h²
        if f0 is None:
            f0 = _sample(sdf, voxel_coords)
        return (
            _sample(sdf, voxel_coords, o1) - 2 * f0 + _sample(sdf, voxel_coords, -o1)
        ) / (h * spacing[axis1]) ** 2
    else:
        # Mixed derivative: (f(+,+) - f(+,-) - f(-,+) + f(-,-)) / 4h²
        return (
            _sample(sdf, voxel_coords, o1 + o2)
            - _sample(sdf, voxel_coords, o1 - o2)
            - _sample(sdf, voxel_coords, -o1 + o2)
            + _sample(sdf, voxel_coords, -o1 - o2)
        ) / (4 * h**2 * spacing[axis1] * spacing[axis2])
