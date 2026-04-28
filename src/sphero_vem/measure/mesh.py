"""Mesh calculation and curvature computation from SDF at interpolated points."""

from pathlib import Path
import numpy as np
from skimage.measure import marching_cubes
from sphero_vem.utils import weighted_std
from sphero_vem.utils.accelerator import ndi, gpu_dispatch, xp, ArrayLike, to_host
from sphero_vem.measure.sdf import _sample, _first_deriv, _second_deriv


def _get_vertex_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute the area associated with each mesh vertex.

    Each vertex is assigned one-third of the area of each adjacent face
    (barycentric area weighting).

    Parameters
    ----------
    verts : numpy.ndarray
        Vertex coordinates, shape (V, 3).
    faces : numpy.ndarray
        Triangle face indices into *verts*, shape (F, 3).

    Returns
    -------
    numpy.ndarray
        Per-vertex area, shape (V,).
    """
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


@gpu_dispatch()
def _compute_derivatives(
    sdf: ArrayLike,
    voxel_coords: ArrayLike,
    spacing: tuple[float, float, float],
    h: float = 1.0,
) -> tuple[ArrayLike, ArrayLike]:
    """
    Compute gradient and Hessian of the SDF at surface points.

    Parameters
    ----------
    sdf : ArrayLike
        3-D signed distance function array.
    voxel_coords : ArrayLike
        Integer voxel coordinates of surface points, shape (N, 3).
    spacing : tuple[float, float, float]
        Physical voxel spacing in ZYX order, used to scale finite differences.
    h : float, optional
        Step size (in voxels) for finite-difference stencils. Default is 1.0.

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
