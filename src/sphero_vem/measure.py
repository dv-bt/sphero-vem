"""
Module containing functions for shape analysis and geometric transforms.
"""

from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from skimage.measure import marching_cubes
from sphero_vem.utils import bbox_expand, slice_from_bbox
from sphero_vem.utils.accelerator import ndi, gpu_dispatch, xp, ArrayLike, ski, to_host


@gpu_dispatch(return_to_host=True)
def calc_regionprops(labels: ArrayLike, bbox_margin: int = 15) -> list[dict]:
    """Calculate region propertis using GPU acceleration"""

    props = ski.measure.regionprops(labels)
    bounds = (0, 0, 0, *labels.shape)

    results = [
        {
            "label": i.label,
            "bbox": i.bbox,
            "bbox_exp": bbox_expand(i.bbox, margin=bbox_margin, im_shape=labels.shape),
            "cut": any(x == y for x, y in zip(bounds, i.bbox)),
            "inertia_tensor": i.inertia_tensor,
            "inertia_tensor_eigvals": i.inertia_tensor_eigvals,
            "centroid": i.centroid,
        }
        for i in props
    ]

    return results


@gpu_dispatch(return_to_host=True)
def calc_sdf(
    label_idx: int,
    labels_crop: ArrayLike,
    spacing: tuple,
    sigma: float = 1.5,
) -> ArrayLike:
    """Calculate SDF for the given label with GPU acceleration"""
    mask = labels_crop == label_idx
    sdt = ndi.distance_transform_edt(
        ~mask, sampling=spacing
    ) - ndi.distance_transform_edt(mask, sampling=spacing)
    sdt_smooth = ndi.gaussian_filter(sdt, sigma=sigma)
    return sdt_smooth


@gpu_dispatch()
def compute_surface_area(
    sdf: ArrayLike, spacing: ArrayLike, epsilon_voxels: float = 1.5
) -> float:
    # Convert epsilon to physical units (same as SDF).
    # Caution: this assumes isotropic voxels!
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


@gpu_dispatch(return_to_host=True)
def calc_descriptors(
    label_idx: int,
    labels: ArrayLike,
    spacing: tuple,
    sigma: float = 1.5,
    eps_voxels: float = 1.5,
) -> dict:
    spacing_arr = xp.asarray(spacing)
    sdf = calc_sdf(
        label_idx=label_idx,
        labels_crop=labels,
        spacing=spacing,
        sigma=sigma,
        _to_host=False,
    )
    volume = float(xp.sum(sdf < 0) * xp.prod(spacing_arr))
    surface_area = compute_surface_area(
        sdf, spacing=spacing_arr, epsilon_voxels=eps_voxels, _to_host=False
    )
    sphericity = xp.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / surface_area
    crop_area = get_crop_surface(sdf, spacing=spacing)

    results = {
        "volume": volume,
        "surface_area": surface_area,
        "sphericity": sphericity,
        "crop_area": crop_area,
    }
    return results


@gpu_dispatch()
def get_crop_surface(sdf: ArrayLike, spacing: tuple) -> float:
    """Get surface of portions of objects cropped by image bounds"""

    boundary_mask = np.zeros(sdf.shape, dtype=bool)
    for dim in range(sdf.ndim):
        slc = [slice(None)] * sdf.ndim

        slc[dim] = 0
        boundary_mask[tuple(slc)] = True

        slc[dim] = -1
        boundary_mask[tuple(slc)] = True

    return float(xp.sum(sdf[boundary_mask] < 0)) * spacing[0] ** 2


def analyze_labels(
    labels: np.ndarray,
    spacing: tuple[float],
    bbox_margin: int = 15,
    sigma: float = 1.5,
    eps_voxels: int = 1.5,
) -> pd.DataFrame:
    """Analyze labels"""

    if len(set(spacing)) > 1:
        raise ValueError(f"Spacing must be isotropic. Received {spacing}")

    results = calc_regionprops(labels, bbox_margin=bbox_margin)
    for entry in tqdm(results):
        sel_slice = slice_from_bbox(entry["bbox_exp"])
        labels_crop = labels[sel_slice]
        props = calc_descriptors(
            label_idx=entry["label"],
            labels=labels_crop,
            spacing=spacing,
            sigma=sigma,
            eps_voxels=eps_voxels,
        )
        entry |= props

    return pd.DataFrame(results)


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


@gpu_dispatch()
def compute_derivatives(
    sdf: ArrayLike,
    voxel_coords: ArrayLike,
    spacing: tuple[float, float, float],
    h: float = 1.0,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute gradient and Hessian at surface points.

    Returns
    -------
    grad : np.ndarray
        Shape (3, N) - gradient vector at each point.
    hess : np.ndarray
        Shape (3, 3, N) - symmetric Hessian matrix at each point.
    """
    n_points = len(voxel_coords)
    grad = xp.zeros((3, n_points))
    hess = xp.zeros((3, 3, n_points))

    f0 = _sample(sdf, voxel_coords)

    for i in range(3):
        grad[i] = _first_deriv(sdf, voxel_coords, i, spacing, h)
        hess[i, i] = _second_deriv(sdf, voxel_coords, i, i, spacing, h, f0=f0)
        for j in range(i + 1, 3):
            hess[i, j] = hess[j, i] = _second_deriv(sdf, voxel_coords, i, j, spacing, h)

    return grad, hess


@gpu_dispatch()
def compute_curvatures(grad: np.ndarray, hess: np.ndarray) -> tuple[ArrayLike]:
    """
    Compute curvatures from gradient and Hessian of an implicit surface, evalated at
    the supplied points.

    It uses the equations described in Goldman (2005), but with opposite sign convetion
    for the SDF (negative on the inside).

    Parameters
    ----------
    grad : np.ndarray
        Shape (3, N).
    hess : np.ndarray
        Shape (3, 3, N).

    Returns
    -------
    curv_mean : ArrayLike
        Mean curvature evaluated at N points.
    curv_gauss : ArrayLike
        Gaussian curvature evaluated at N points.
    kappa1 : ArrayLike
        First principal curvature evaluated at N points.
    kappa2 : ArrayLike
        Second principal curvature evaluated at N points.

    References
    ----------
    .. 1. Goldman, R. Curvature formulas for implicit curves and surfaces.
          Computer Aided Geometric Design 22, 632-658 (2005).
          https://doi.org/10.1016/j.cagd.2005.06.005

    """
    grad_mag_sq = xp.sum(grad**2, axis=0)
    grad_mag = xp.sqrt(grad_mag_sq)

    # Mean curvature: H = (grad^T @ hess @ grad - grad_mag^2 * trace(hess)) / (2 * grad_mag^3)
    # Flipped sign because SDF has opposite sign convetion than in reference.
    curv_mean = -(
        xp.einsum("in,ijn,jn->n", grad, hess, grad) - xp.trace(hess) * grad_mag_sq
    ) / (2 * grad_mag_sq * grad_mag + 1e-10)

    # Gaussian curvature: K = grad^T @ adj(hess) @ grad / grad_mag^4
    # Adjugate = cofactor for symmetric matrix
    adj = xp.array(
        [
            xp.cross(hess[1], hess[2], axis=0),
            xp.cross(hess[2], hess[0], axis=0),
            xp.cross(hess[0], hess[1], axis=0),
        ]
    )

    curv_gauss = xp.einsum("in,ijn,jn->n", grad, adj, grad) / (grad_mag_sq**2 + 1e-10)

    disc = xp.sqrt(xp.maximum(curv_mean**2 - curv_gauss, 0))

    return curv_mean, curv_gauss, curv_mean + disc, curv_mean - disc


def _get_vertex_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area associated with each vertex (1/3 of adjacent face areas)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    vertex_area = np.zeros(len(verts))
    for i in range(3):
        np.add.at(vertex_area, faces[:, i], face_areas / 3)

    return vertex_area


def calc_mesh(
    sdf: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mesh from 0-level set of the SDF using the marching cubes algorithm.

    Parameters
    ----------
    sdf : np.ndarray
        Signed distance function, negative on the inside.
    spacing : tuple[float, float, float]
        Physical spacing of the SDF voxel grid. It should have the same units as SDF.

    Returns
    -------
    verts : np.ndarray
        Vertices of the mesh
    faces : np.ndarray
        Faces of the mesh
    vertex_area s: np.ndarray
        Area associated to each vertex, equal to 1/3 of the adjacent face areas.
    """
    verts, faces, _, _ = marching_cubes(to_host(sdf), level=0.0, spacing=spacing)
    vertex_areas = _get_vertex_areas(verts, faces)
    return verts, faces, vertex_areas


@gpu_dispatch(return_to_host=True)
def calc_curvature(
    sdf: ArrayLike,
    verts: ArrayLike,
    spacing: tuple[float, float, float],
    h: float = 1.5,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Calculate curvatures at the mesh vertices using the implicit SDF representation.

    The function uses first and second order central finite differences to approximate
    the gradient and the Hessian matrix of the SDF, sampled at the vertices of the
    mesh.

    Parameters
    ----------
    sdf : ArrayLike
        Signed distance function, negative on the inside.
    verts : np.ndarray
        Vertices of the mesh, used to sample the SDF first and second order derivatives.
    spacing : tuple[float, float, float]
        Physical spacing of the SDF voxel grid. It should have the same units as SDF.
    h : float, Optional
        Step for the finite differences, in voxels.
        Default is 1.5

    Returns
    -------
    curv_mean : ArrayLike
        Mean curvature evaluated at N points.
    curv_gauss : ArrayLike
        Gaussian curvature evaluated at N points.
    kappa1 : ArrayLike
        First principal curvature evaluated at N points.
    kappa2 : ArrayLike
        Second principal curvature evaluated at N points.

    """
    voxel_coords = verts / xp.asarray(spacing)
    grad, hess = compute_derivatives(sdf, voxel_coords, spacing, h=h)
    curv_mean, curv_gauss, kappa1, kappa2 = compute_curvatures(grad, hess)
    return curv_mean, curv_gauss, kappa1, kappa2
