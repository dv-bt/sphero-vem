"""
Module containing functions for shape analysis and geometric transforms.
"""

from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import pandas as pd
import zarr
from skimage.measure import marching_cubes
from sphero_vem.utils import bbox_expand, slice_from_bbox, check_isotropic, weighted_std
from sphero_vem.utils.accelerator import ndi, gpu_dispatch, xp, ArrayLike, ski, to_host
from sphero_vem.utils.config import BaseConfig


@dataclass
class LabelAnalysisConfig(BaseConfig):
    """
    Configuration for 3D label morphology analysis pipeline.

    Defines paths, SDF parameters, and mesh extraction settings for computing
    shape descriptors from segmented volumetric data stored in zarr format.

    Parameters
    ----------
    root_path : Path
        Root path to the zarr store containing segmentation data.
    seg_target : str
        Name of the segmentation target (e.g., 'cells', 'nuclei').
        Used to construct paths: `labels/{seg_target}/tables/`.
    scale_dir : str
        Scale directory name within the masks folder (e.g., '50-50-50', 's1').
    bbox_margin : int, optional
        Margin in voxels to expand bounding boxes for label cropping.
        Default is 15.
    sigma : float, optional
        Gaussian smoothing sigma in voxels for SDF computation.
        Controls surface smoothness for curvature estimation. Default is 3.
    eps_voxels : float, optional
        Epsilon in voxels for Heaviside volume/area integration.
        Default is 1.5.
    mesh_downsample_factor : int, optional
        Factor by which to downsample SDF before marching cubes.
        Reduces vertex count and computation time. Default is 2.
    h : float, optional
        Step size in voxels for finite difference curvature estimation.
        Default is 1.5.
    voxel_only : bool, optional
        If True, compute only voxel-based properties (skip SDF and mesh).
        Default is False.

    Attributes
    ----------
    array_path : Path
        Computed path to the label zarr array.
    save_root : Path
        Computed path for saving output tables and meshes. Evaluates to
        `labels/{seg_target}/tables/`.
    spacing : tuple[float]
        Voxel spacing in micrometers, read from zarr attributes.
    """

    root_path: Path
    seg_target: str
    scale_dir: str
    bbox_margin: int = 15
    sigma: float = 3
    eps_voxels: int = 1.5
    mesh_downsample_factor: int = 2
    h: float = 1.5
    voxel_only: bool = False

    array_path: Path = field(init=False)
    save_root: Path = field(init=False)
    spacing: tuple[float] = field(init=False)

    def __post_init__(self):
        self.array_path = (
            self.root_path / f"labels/{self.seg_target}/masks/{self.scale_dir}"
        )
        self.save_root = self.root_path / f"labels/{self.seg_target}/tables"
        src_zarr = zarr.open_array(self.array_path)

        # Spacing is assumed to be in nm and converted to µm.
        self.spacing = tuple(i / 1000 for i in src_zarr.attrs.get("spacing"))


@gpu_dispatch()
def props_voxel(
    labels: ArrayLike, bbox_margin: int = 15, calc_volume: bool = False
) -> list[dict]:
    """Calculate voxel-based properties using GPU acceleration.

    This function uses skimage.measure.regionprops (or its CuCIM equivalent if CUDA is
    available) calculate per-label properties from a voxel image.

    Parameters
    ----------
    labels : ArrayLike
        Array with the labeled image.
    bbox_margin : int
        Constant margin for bounding box expansion. The returned bounding box will be
        expanded by this value in all directions.
    calc_volume : bool
        Flag that controls whether to calculate volume and related quantities directly
        from the voxel map. This is useful when SDF and mesh-based approaches are
        not feasible.
        Default is False.

    Returns
    -------
    list[dict]
        List of dictionaries containing the calculated properties for each label:
        - label: label ID
        - bbox: bounding box of the object
        - bbox_exp: bounding box expanded by bbox_margin
        - eigvals: eigenvalues of the gyration tensor (sorted ascending)
        - centroid: coordinates of the image centroid
        - volume: label volume (if calc_volume=True)
        - diam_equiv: equivalent diameter (if calc_volume=True)

    """

    props = ski.measure.regionprops(labels)

    results = [
        {
            "label": prop.label,
            "bbox": prop.bbox,
            "bbox_exp": bbox_expand(
                prop.bbox, margin=bbox_margin, im_shape=labels.shape
            ),
            "eigvals": tuple(sorted(prop.inertia_tensor_eigvals)),
            "centroid": prop.centroid,
        }
        for prop in props
    ]

    if calc_volume:
        for i, prop in enumerate(props):
            results[i]["volume"] = prop.area
            results[i]["diam_equiv"] = prop.equivalent_diameter_area

    return results


@gpu_dispatch()
def _calc_sdf(
    label_idx: int,
    labels: ArrayLike,
    spacing: tuple[float, float, float],
    sigma: float = 3,
) -> ArrayLike:
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
        Standard deviation of the Gaussian kernel used for smoothing the SDF in voxels.
        Default is 3.

    Returns
    -------
    ArrayLike
        The smoothed SDF.
    """
    mask = labels == label_idx
    sdt = ndi.distance_transform_edt(
        ~mask, sampling=spacing
    ) - ndi.distance_transform_edt(mask, sampling=spacing)
    sdt_smooth = ndi.gaussian_filter(sdt, sigma=sigma)
    return sdt_smooth


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


@gpu_dispatch()
def _compute_derivatives(
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
def _compute_curvatures(grad: np.ndarray, hess: np.ndarray) -> tuple[ArrayLike]:
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

    return curv_mean, curv_gauss, curv_mean - disc, curv_mean + disc


def _get_vertex_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute area associated with each vertex (1/3 of adjacent face areas)."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)

    vertex_area = np.zeros(len(verts))
    for i in range(3):
        np.add.at(vertex_area, faces[:, i], face_areas / 3)

    return vertex_area


def get_mesh(
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
def _calc_curvature(
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
    grad, hess = _compute_derivatives(sdf, voxel_coords, spacing, h=h)
    curv_mean, curv_gauss, kappa1, kappa2 = _compute_curvatures(grad, hess)
    return curv_mean, curv_gauss, kappa1, kappa2


@gpu_dispatch(return_to_host=True)
def props_mesh(
    sdf: ArrayLike,
    spacing: tuple[float],
    mesh_downsample_factor: int = 2,
    h: float = 1.5,
    mesh_save_path: Path | None = None,
) -> tuple[ArrayLike]:
    """Compute curvature-based shape descriptors from SDF via mesh sampling.

    Extracts a surface mesh from the downsampled SDF using marching cubes,
    then samples curvature values from the full-resolution SDF at mesh vertex
    locations. Returns area-weighted aggregate statistics of local curvature
    descriptors.

    Parameters
    ----------
    sdf : ArrayLike
        Smoothed signed distance function (negative inside).
    spacing : tuple[float]
        Voxel spacing in physical units (z, y, x).
    mesh_downsample_factor : int, optional
        Factor by which to downsample SDF before mesh extraction.
        Reduces vertex count and computation time. Default is 2.
    h : float, optional
        Step size in voxels for finite difference curvature estimation.
        Default is 1.5.
    mesh_save_path : Path | None, optional
        If provided, saves mesh geometry and per-vertex curvature values
        to this path as .npz file.

    Returns
    -------
    results : dict
        Area-weighted curvature statistics:
        - curv_mean_avg, curv_mean_std : Mean curvature statistics
        - curv_gauss_avg, curv_gauss_std : Gaussian curvature statistics
        - curvedness_avg, curvedness_std : Curvedness statistics
        - shape_index_avg, shape_index_std : Shape index statistics

    Notes
    -----
    Curvature sign convention: positive mean curvature indicates convex
    regions (outward curving), negative indicates concave regions.

    Shape index follows Koenderink (1992) convention with kappa_1 <= kappa_2:
    - S = -1: spherical cup (concave)
    - S = 0: saddle
    - S = +1: spherical cap (convex)

    References
    ----------
    .. 1. Koenderink, J. J. & van Doorn, A. J. Surface shape and curvature scales.
          Image and Vision Computing 10, 557-564 (1992).

    """

    # Calculate mesh on downsampled SDF to reduce the number of vertices.
    verts, faces, vertex_areas = get_mesh(
        sdf=ndi.zoom(sdf, 1 / mesh_downsample_factor, order=1), spacing=spacing
    )

    curv_mean, curv_gauss, kappa1, kappa2 = _calc_curvature(
        sdf=sdf, verts=verts, spacing=spacing, h=h
    )

    if mesh_save_path is not None:
        np.savez(
            mesh_save_path,
            verts=verts,
            faces=faces,
            vertex_areas=vertex_areas,
            curv_mean=curv_mean,
            curv_gauss=curv_gauss,
            kappa1=kappa1,
            kappa2=kappa2,
        )

    curvedness = np.sqrt((kappa1**2 + kappa2**2) / 2)
    shape_index = (2 / np.pi) * np.arctan2(kappa1 + kappa2, kappa2 - kappa1)

    results = {
        "curv_mean_avg": np.average(curv_mean, weights=vertex_areas),
        "curv_mean_std": weighted_std(curv_mean, weights=vertex_areas),
        "curv_gauss_avg": np.average(curv_gauss, weights=vertex_areas),
        "curv_gauss_std": weighted_std(curv_gauss, weights=vertex_areas),
        "curvedness_avg": np.average(curvedness, weights=vertex_areas),
        "curvedness_std": weighted_std(curvedness, weights=vertex_areas),
        "shape_index_avg": np.average(shape_index, weights=vertex_areas),
        "shape_index_std": weighted_std(shape_index, weights=vertex_areas),
    }

    return results


def label_properties(
    labels: np.ndarray,
    spacing: tuple[float],
    bbox_margin: int = 15,
    sigma: float = 3,
    eps_voxels: int = 1.5,
    mesh_downsample_factor: int = 2,
    h: float = 1.5,
    mesh_save_root: Path | None = None,
    voxel_only: bool = False,
) -> pd.DataFrame:
    """
    Compute morphological properties for all labels in a 3D volume.

    Extracts voxel-based, SDF-based, and mesh-based shape descriptors for
    each labeled region. Requires isotropic voxel spacing.

    Parameters
    ----------
    labels : np.ndarray
        3D integer array of labeled regions. Background should be 0.
    spacing : tuple[float]
        Isotropic voxel spacing in physical units (z, y, x).
    bbox_margin : int, optional
        Margin in voxels to expand bounding boxes when cropping labels.
        Default is 15.
    sigma : float, optional
        Gaussian smoothing sigma in voxels for SDF computation.
        Default is 3.
    eps_voxels : float, optional
        Epsilon in voxels for Heaviside volume/area integration.
        Default is 1.5.
    mesh_downsample_factor : int, optional
        Downsampling factor for SDF before mesh extraction. Default is 2.
    h : float, optional
        Step size in voxels for finite difference curvature estimation.
        Default is 1.5.
    mesh_save_root : Path | None, optional
        If provided, per-label meshes and curvature data are saved as .npz
        files under `{mesh_save_root}/meshes/`. Default is None.
    voxel_only : bool, optional
        If True, skip SDF and mesh computations; return only voxel-based
        properties. Default is False.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per label containing:
        - Voxel-based: label, bbox, centroid, inertia eigenvalues, etc.
        - SDF-based: volume, surface_area, sphericity (if not voxel_only)
        - Mesh-based: curvature statistics (if not voxel_only)

    Raises
    ------
    ValueError
        If spacing is not isotropic.

    See Also
    --------
    `props_voxel` : Voxel-based property extraction.
    `props_sdf` : SDF-based volume and surface area computation.
    `props_mesh` : Mesh-based curvature computation.
    """

    check_isotropic(spacing, raise_error=True)

    results = props_voxel(labels, bbox_margin=bbox_margin)
    if not voxel_only:
        for entry in tqdm(results):
            sel_slice = slice_from_bbox(entry["bbox_exp"])
            labels_crop = labels[sel_slice]
            props, sdf = props_sdf(
                label_idx=entry["label"],
                labels=labels_crop,
                spacing=spacing,
                sigma=sigma,
                eps_voxels=eps_voxels,
            )
            entry |= props

            mesh_save_path = (
                mesh_save_root / f"meshes/mesh-{entry['label']}.npz"
                if mesh_save_root is not None
                else None
            )
            if mesh_save_path:
                mesh_save_path.parent.mkdir(exist_ok=True, parents=True)

            props = props_mesh(
                sdf=sdf,
                spacing=spacing,
                mesh_downsample_factor=mesh_downsample_factor,
                h=h,
                mesh_save_path=mesh_save_path,
            )
            entry |= props

    return pd.DataFrame(results)


def analyze_labels(config: LabelAnalysisConfig) -> None:
    """
    Run label morphology analysis pipeline from configuration.

    Loads labels from zarr, computes shape descriptors via `label_properties`,
    and saves results to parquet along with the configuration.

    Parameters
    ----------
    config : LabelAnalysisConfig
        Configuration object specifying paths and analysis parameters.

    Raises
    ------
    ValueError
        If spacing is not isotropic.

    Notes
    -----
    Outputs are saved to `{config.save_root}/`:
    - `regionprops.parquet`: DataFrame with all computed properties
    - `analysis-config.json`: Serialized configuration for reproducibility
    - `meshes/mesh-{label}.npz`: Per-label mesh data (if not voxel_only)
    """
    root = zarr.open_group(config.root_path, mode="a")
    label_array = root.get(config.array_path)
    results = label_properties(
        labels=label_array[:],
        spacing=config.spacing,
        bbox_margin=config.bbox_margin,
        sigma=config.sigma,
        eps_voxels=config.eps_voxels,
        mesh_downsample_factor=config.mesh_downsample_factor,
        h=config.h,
        mesh_save_root=config.save_root,
    )
    results = flatten_for_save(results)
    results.to_parquet(config.save_root / "regionprops.parquet")
    config.to_json(config.save_root / "analysis-config.json")


def flatten_for_save(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten tuple columns to indexed columns for parquet storage."""
    df_out = df.copy()

    for col in df.columns:
        if isinstance(df[col].iloc[0], tuple):
            for i in range(len(df[col].iloc[0])):
                df_out[f"{col}_{i}"] = df[col].apply(lambda x, i=i: x[i])
            df_out = df_out.drop(columns=[col])

    return df_out
