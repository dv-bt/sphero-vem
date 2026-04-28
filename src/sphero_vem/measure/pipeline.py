"""Pipeline orchestration, I/O, and configuration for label morphology analysis."""

from pathlib import Path
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import pandas as pd
import zarr
from sphero_vem.utils import (
    check_isotropic,
    slice_from_bbox,
    flatten_for_save,
    reconstruct_tuples,
)
from sphero_vem.utils.config import BaseConfig
from sphero_vem.measure.voxel import props_voxel, assign_cell
from sphero_vem.measure.sdf import props_sdf
from sphero_vem.measure.mesh import props_mesh
from sphero_vem.measure.fractal import props_fractal


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
