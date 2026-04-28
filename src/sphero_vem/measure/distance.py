"""Nucleus distance map computation."""

from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import zarr
from sphero_vem.io import write_zarr
from sphero_vem.utils import slice_from_bbox
from sphero_vem.utils.accelerator import ndi, gpu_dispatch, xp, ArrayLike
from sphero_vem.measure.pipeline import LabelAnalysisConfig, read_regionprops


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
