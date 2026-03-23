"""
Module containing functions for shape analysis and geometric transforms.
"""

from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import pandas as pd
import zarr
from scipy.stats import linregress
from skimage.measure import marching_cubes
from sphero_vem.io import write_zarr
from sphero_vem.utils import (
    bbox_expand,
    slice_from_bbox,
    check_isotropic,
    weighted_std,
    flatten_for_save,
    reconstruct_tuples,
)
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
    sigma_frac : float, optional
        Gaussian smoothing sigma in voxels for SDF computation used during fractal
        dimension calculation. This should be in the 0.5-1.0 range.
        Default is 0.7.
    n_steps_frac : int, optional
        Number of epsilon values sampled in log-space during fractal dimension
        calculation.
        Default is 30.
    sep : str
        Separator used for unpacking tuple columns when saving the region properties
        dataframe to parquet using `save_regionprops`. This should be used by
        `read_regionprops` to reconstruct the tuple columns, e.g. bbox, centroid...
        Default is `"__"`

    Attributes
    ----------
    array_path : Path
        Computed path to the label zarr array.
    save_root : Path
        Computed path for saving output tables and meshes. Evaluates to
        `labels/{seg_target}/tables/`.
    spacing : tuple[float]
        Voxel spacing in micrometers, read from zarr attributes.
        NOTE: this assumes that the spacing stored in the zarr attributes is in nm.
    cell_array_path : Path
        Path to the array containing the cell labels with the same spacing as the
        analyzed label array. This is used when analyzing targets other than cells to
        assign their parent cell.
    """

    root_path: Path
    seg_target: str
    scale_dir: str
    bbox_margin: int = 15
    sigma: float = 3.0
    eps_voxels: float = 1.5
    mesh_downsample_factor: int = 2
    h: float = 1.5
    voxel_only: bool = False
    sigma_frac: float = 0.7
    n_steps_frac: int = 30
    sep: str = "__"

    array_path: Path = field(init=False)
    save_root: Path = field(init=False)
    spacing: tuple[float, float, float] = field(init=False)
    cell_array_path: Path = field(init=False)

    def __post_init__(self):
        self.array_path = (
            self.root_path / f"labels/{self.seg_target}/masks/{self.scale_dir}"
        )
        self.save_root = self.root_path / f"labels/{self.seg_target}/tables"
        self.cell_array_path = self.root_path / f"labels/cells/masks/{self.scale_dir}"

        # Spacing is assumed to be in nm and converted to µm.
        src_zarr = zarr.open_array(self.array_path)
        self.spacing = tuple(i / 1000 for i in src_zarr.attrs.get("spacing"))


@gpu_dispatch()
def props_voxel(
    labels: ArrayLike,
    spacing: tuple[float, float, float],
    bbox_margin: int = 15,
    calc_volume: bool = False,
) -> list[dict]:
    """Calculate voxel-based properties using GPU acceleration.

    This function uses skimage.measure.regionprops (or its CuCIM equivalent if CUDA is
    available) calculate per-label properties from a voxel image.

    Parameters
    ----------
    labels : ArrayLike
        Array with the labeled image.
    spacing : tuple[float, float, float]
        Physical spacing of the voxel grid. This will only be used when calc_volume
        is True.
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
        - volume: label volume in µm^3 (if calc_volume=True)
        - diam_equiv: equivalent diameter in µm^2 (if calc_volume=True)

    """

    props = ski.measure.regionprops(labels)

    results = [
        {
            "label": prop.label,
            "bbox": prop.bbox,
            "bbox_exp": bbox_expand(
                prop.bbox, margin=bbox_margin, im_shape=labels.shape
            ),
            "eigvals": tuple(float(i) for i in sorted(prop.inertia_tensor_eigvals)),
            "centroid": prop.centroid,
        }
        for prop in props
    ]

    if calc_volume:
        props_spacing = ski.measure.regionprops(labels, spacing=spacing)
        for i, prop in enumerate(props_spacing):
            results[i]["volume"] = float(prop.area)
            results[i]["diam_equiv"] = float(prop.equivalent_diameter_area)

    return results


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
        The calculated SDF.
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


def _remove_boundary_caps(
    verts: np.ndarray,
    faces: np.ndarray,
    sdf_shape: tuple[int, ...],
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Remove faces where all vertices lie on the same array boundary plane.

    Parameters
    ----------
    verts : np.ndarray
        Mesh vertices in physical coordinates, shape (V, 3).
    faces : np.ndarray
        Triangle face indices, shape (F, 3).
    sdf_shape : tuple[int, ...]
        Shape of the SDF array passed to marching cubes.
    spacing : tuple[float, float, float]
        Physical voxel spacing used by marching cubes (z, y, x).

    Returns
    -------
    np.ndarray
        Filtered face indices with boundary caps removed.
    """
    cap_mask = np.zeros(len(faces), dtype=bool)
    v = [verts[faces[:, i]] for i in range(3)]
    for axis in range(3):
        for boundary in [0.0, (sdf_shape[axis] - 1) * spacing[axis]]:
            on_plane = np.stack(
                [np.isclose(v[i][:, axis], boundary) for i in range(3)], axis=1
            ).all(axis=1)
            cap_mask |= on_plane
    return faces[~cap_mask]


def get_mesh(
    sdf: np.ndarray, spacing: tuple[float, float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate mesh from 0-level set of the SDF using the marching cubes algorithm.

    Parameters
    ----------
    sdf : np.ndarray
        Signed distance function, negative on the inside.
    spacing : tuple[float, float, float]
        Physical spacing of the SDF voxel grid, same units as SDF values.

    Returns
    -------
    verts : np.ndarray
        Vertices of the mesh.
    faces : np.ndarray
        Faces of the mesh, with boundary cap faces removed.
    vertex_areas : np.ndarray
        Area associated to each vertex, equal to 1/3 of the adjacent face areas.
    """
    sdf_host = to_host(sdf)
    verts, faces, _, _ = marching_cubes(sdf_host, level=0.0, spacing=spacing)
    faces = _remove_boundary_caps(
        verts, faces, sdf_shape=sdf_host.shape, spacing=spacing
    )
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
    # Use effective spacing for the downsampled grid so that marching cubes returns
    # vertices in the correct physical coordinates (matching the full-res SDF grid).
    downsampled_spacing = tuple(s * mesh_downsample_factor for s in spacing)
    verts, faces, vertex_areas = get_mesh(
        sdf=ndi.zoom(sdf, 1 / mesh_downsample_factor, order=1),
        spacing=downsampled_spacing,
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
    sigma_frac: float = 0.7,
    n_steps_frac: int = 30,
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

    results = props_voxel(
        labels, spacing=spacing, bbox_margin=bbox_margin, calc_volume=voxel_only
    )
    if not voxel_only:
        for entry in tqdm(results, "Analyzing labels"):
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

            props = props_fractal(
                label_idx=entry["label"],
                labels=labels_crop,
                spacing=spacing,
                sigma_frac=sigma_frac,
                n_steps=n_steps_frac,
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
    label_array = zarr.open_array(config.array_path)
    props = label_properties(
        labels=label_array[:],
        spacing=config.spacing,
        bbox_margin=config.bbox_margin,
        sigma=config.sigma,
        eps_voxels=config.eps_voxels,
        mesh_downsample_factor=config.mesh_downsample_factor,
        h=config.h,
        mesh_save_root=config.save_root,
        voxel_only=config.voxel_only,
        sigma_frac=config.sigma_frac,
        n_steps_frac=config.n_steps_frac,
    )

    if config.seg_target != "cells":
        cell_array = zarr.open_array(config.cell_array_path)
        props = assign_cell(props=props, cells=cell_array[:])

    save_regionprops(
        props, dst_path=config.save_root / "regionprops.parquet", sep=config.sep
    )
    config.to_json(config.save_root / "analysis-config.json")


def save_regionprops(
    props: pd.DataFrame,
    dst_path: Path,
    sep: str = "__",
) -> None:
    """
    Save region properties to parquet with tuple columns flattened.

    Tuple and list columns are unpacked into indexed scalar columns
    (e.g., ``centroid`` → ``centroid__0``, ``centroid__1``, ...) for
    parquet compatibility. The index is not saved; all information
    should be encoded in the columns.

    Parameters
    ----------
    props : pd.DataFrame
        DataFrame of region properties, potentially containing tuple
        or list valued columns.
    dst_path : Path
        Destination path for the parquet file.
    sep : str, optional
        Separator for flattened column names. Must match the `sep`
        passed to `read_regionprops` for round-tripping. Default is
        ``"__"``.

    See Also
    --------
    read_regionprops : Inverse operation.
    flatten_for_save : Underlying flattening logic.
    """
    props = flatten_for_save(props, sep=sep)
    props.to_parquet(dst_path, index=False)


def read_regionprops(
    src_path: Path,
    sep: str = "__",
) -> pd.DataFrame:
    """
    Read region properties from parquet and reconstruct tuple columns.

    Indexed scalar columns (e.g., ``centroid__0``, ``centroid__1``, ...)
    are packed back into tuple columns (``centroid``).

    Parameters
    ----------
    src_path : Path
        Path to the parquet file saved by `save_regionprops`.
    sep : str, optional
        Separator used when the file was saved. Default is ``"__"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with tuple columns reconstructed.

    See Also
    --------
    save_regionprops : Inverse operation.
    reconstruct_tuples : Underlying reconstruction logic.
    """
    props = pd.read_parquet(src_path)
    props = reconstruct_tuples(props, sep=sep)
    return props


def assign_cell(props: pd.DataFrame, cells: np.ndarray) -> pd.DataFrame:
    """Assign parent cell and return dataframe by looking up centroid

    Parameters
    ----------
    props : pd.DataFrame
        The dataframe containing the region properties. It should have a `"centroid"`
        column containing the tuple indexing the object centroid.
    cells : np.ndarray
        An array containing the cells masks labeled by instance.

    Returns
    -------
    pd.DataFrame
        The updated region properties dataframe with a new column `"parent_cell"`.
    """
    indices = np.array(props["centroid"].tolist()).astype(int)
    props["parent_cell"] = cells[tuple(indices.T)]
    return props


@gpu_dispatch(return_to_host=True)
def _distance_map_cell(
    nuclei_region: ArrayLike,
    cells_region: ArrayLike,
    cell_label: int,
    nuclei_in_cell: ArrayLike,
    spacing: tuple[float, float, float],
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute Euclidean distance from nuclei within a single cell's bounding box.

    Computes the EDT from all nuclei belonging to the given cell. Returns the distance
    map and cell mask for the bounding box region, which are used by the caller to
    assign values into the full-volume output.

    Parameters
    ----------
    nuclei_region : ArrayLike
        Cropped nuclei label array (bounding box of the cell).
    cells_region : ArrayLike
        Cropped cell label array (same bounding box).
    cell_label : int
        Label index of the cell being processed.
    nuclei_in_cell : ArrayLike
        Array of nuclei label indices that belong to this cell.
    spacing : tuple[float, float, float]
        Voxel spacing in physical units (z, y, x).

    Returns
    -------
    nuclei_edt_region : ArrayLike
        Euclidean distance transform from the nuclei surfaces within the bounding box,
        in the same units as `spacing`.
    cells_mask_region : ArrayLike
        Boolean mask of voxels belonging to `cell_label` within the bounding box.
    """

    cells_mask_region = cells_region == cell_label
    nuclei_region_inv = ~xp.isin(nuclei_region, nuclei_in_cell)

    nuclei_edt_region = ndi.distance_transform_edt(nuclei_region_inv, sampling=spacing)
    return nuclei_edt_region, cells_mask_region


def _calc_nuclei_distance(
    nuclei: np.ndarray,
    cells: np.ndarray,
    props_nuclei: pd.DataFrame,
    props_cells: pd.DataFrame,
    spacing: tuple[float, float, float],
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute per-voxel Euclidean distance to the nearest nucleus for each cell.

    Iterates over cells that contain at least one nucleus, computing the distance
    transform from the nuclear surfaces within each cell's bounding box. Voxels outside
    any cell or in cells without a detected nucleus are set to NaN.

    Parameters
    ----------
    nuclei : np.ndarray
        Full-volume 3D nuclei label array.
    cells : np.ndarray
        Full-volume 3D cell label array (same shape as `nuclei`).
    props_nuclei : pd.DataFrame
        Nuclei properties table, indexed by nucleus label. Must contain a
        ``parent_cell`` column mapping each nucleus to its enclosing cell label.
    props_cells : pd.DataFrame
        Cell properties table, indexed by cell label. Must contain a ``bbox`` column
        with bounding box coordinates.
    spacing : tuple[float, float, float]
        Voxel spacing in physical units (z, y, x).
    verbose : bool, optional
        If True, show a progress bar. Default is True.

    Returns
    -------
    np.ndarray
        Float32 array (same shape as input) with Euclidean distance to the nearest
        nucleus in each cell. NaN for background voxels and cells without a detected
        nucleus.
    """

    nuclei_edt = np.full(nuclei.shape, np.nan, dtype=np.float32)
    cells_with_nuclei = props_nuclei["parent_cell"].unique()

    for label in tqdm(cells_with_nuclei, "Analyzing cells", disable=not verbose):
        bbox_slice = slice_from_bbox(props_cells.at[label, "bbox"])
        nuclei_in_cell = props_nuclei.loc[
            props_nuclei["parent_cell"] == label
        ].index.to_numpy()

        nuclei_edt_region, cells_mask_region = _distance_map_cell(
            nuclei_region=nuclei[bbox_slice],
            cells_region=cells[bbox_slice],
            cell_label=label,
            nuclei_in_cell=nuclei_in_cell,
            spacing=spacing,
        )
        nuclei_edt[cells == label] = nuclei_edt_region[cells_mask_region]

    return nuclei_edt


def nuclei_distance(root_path: Path, verbose: bool = True) -> None:
    """
    Compute and save the nucleus distance map for all cells in a dataset.

    For each cell containing at least one nucleus, computes the Euclidean distance from
    every voxel to the nearest nuclear surface. The result is saved as a zarr array
    under ``labels/nuclei/distance/``.

    Requires that label analysis (via `analyze_labels`) has been run for both ``cells``
    and ``nuclei`` segmentation targets at the same spacing, and that nuclei properties
    contain a ``parent_cell`` column.

    Parameters
    ----------
    root_path : Path
        Root path to the zarr store.
    verbose : bool, optional
        If True, show a progress bar. Default is True.

    Raises
    ------
    ValueError
        If cells and nuclei were analyzed at different spacings.

    Notes
    -----
    Output is written to ``labels/nuclei/distance/{scale_dir}`` in the zarr store, with
    units in micrometers. Voxels outside cells or in cells without a detected nucleus
    are set to NaN.
    """

    root = zarr.open(root_path, mode="a")
    data = {
        "cells": {},
        "nuclei": {},
    }

    for seg_target in data.keys():
        tables_path = root_path / f"labels/{seg_target}/tables/"
        config = LabelAnalysisConfig.from_json(tables_path / "analysis-config.json")
        data[seg_target]["spacing"] = config.spacing
        data[seg_target]["array"] = zarr.open_array(config.array_path)
        props = read_regionprops(tables_path / "regionprops.parquet")
        data[seg_target]["props"] = props.set_index("label")

    if data["cells"]["spacing"] != data["nuclei"]["spacing"]:
        raise ValueError(
            "Cells and nuclei properties were not calculated at the same spacing"
        )

    nuclei_edt = _calc_nuclei_distance(
        nuclei=data["nuclei"]["array"][:],
        cells=data["cells"]["array"][:],
        props_nuclei=data["nuclei"]["props"],
        props_cells=data["cells"]["props"],
        spacing=data["nuclei"]["spacing"],
        verbose=verbose,
    )

    write_zarr(
        root,
        nuclei_edt,
        dst_path=f"labels/nuclei/distance/{config.scale_dir}",
        src_zarr=data["nuclei"]["array"],
        processing={
            "step": "euclidean distance transform",
            "units": "micrometers",
        },
        inputs=[data["nuclei"]["array"].path, data["cells"]["array"].path],
    )
