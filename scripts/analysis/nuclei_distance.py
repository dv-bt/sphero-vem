"""
Calculate distance from nucleus for each cell
"""

from pathlib import Path
import zarr
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sphero_vem.io import write_zarr
from sphero_vem.utils import dirname_from_spacing


def bbox_to_slice(data: pd.DataFrame, label: int) -> tuple[slice]:
    """Create a slice from bbox"""
    obj_slice = (
        slice(data.at[label, "bbox-0"], data.at[label, "bbox-3"]),
        slice(data.at[label, "bbox-1"], data.at[label, "bbox-4"]),
        slice(data.at[label, "bbox-2"], data.at[label, "bbox-5"]),
    )
    return obj_slice


def calc_nuclei_distance(
    nuclei: np.ndarray,
    cells: np.ndarray,
    data_nuclei: pd.DataFrame,
    data_cells: pd.DataFrame,
    spacing_um: tuple[float, float, float],
    verbose: bool = True,
) -> np.ndarray:
    """Calculate the Euclidean distance to the nucleus for each cell. Outside cells and
    in cells without a detected nucleus the array is np.nan."""

    nuclei_edt = np.full(nuclei.shape, np.nan, dtype=np.float32)
    cells_with_nuclei = data_nuclei["parent_cell"].unique()

    for label in tqdm(cells_with_nuclei, "Analyzing cells", disable=not verbose):
        bbox_slice = bbox_to_slice(data_cells, label)
        nuclei_region = nuclei[bbox_slice]
        cells_region = cells[bbox_slice]

        nuclei_in_cell = data_nuclei.loc[
            data_nuclei["parent_cell"] == label
        ].index.tolist()

        nuclei_region_inv = ~np.isin(nuclei_region, nuclei_in_cell)
        cells_mask_region = cells_region == label
        cells_mask_global = cells == label

        nuclei_edt_region = distance_transform_edt(
            nuclei_region_inv, sampling=spacing_um
        )
        nuclei_edt[cells_mask_global] = nuclei_edt_region[cells_mask_region]

    return nuclei_edt


def nuclei_distance(
    root_path: Path, spacing: tuple[int, int, int], verbose: bool = True
) -> None:
    root = zarr.open(root_path, mode="a")
    spacing_dir = dirname_from_spacing(spacing)
    # Convert spacing to µm for compatibility with calculated region props
    spacing_um = tuple(i / 1000 for i in spacing)

    nuclei_zarr = root.get(f"labels/nuclei/masks/{spacing_dir}")
    cells_zarr = root.get(f"labels/cells/masks/{spacing_dir}")

    nuclei = nuclei_zarr[:]
    cells = cells_zarr[:]
    data_nuclei = pd.read_parquet(
        root_path / "labels/nuclei/tables/regionprops.parquet"
    )
    data_nuclei = data_nuclei.set_index("label")
    data_cells = pd.read_parquet(root_path / "labels/cells/tables/regionprops.parquet")
    data_cells = data_cells.set_index("label")

    nuclei_edt = calc_nuclei_distance(
        nuclei=nuclei,
        cells=cells,
        data_nuclei=data_nuclei,
        data_cells=data_cells,
        spacing_um=spacing_um,
        verbose=verbose,
    )

    write_zarr(
        root,
        nuclei_edt,
        dst_path=f"labels/nuclei/distance/{spacing_dir}",
        src_zarr=nuclei_zarr,
        processing={
            "step": "euclidean distance transform",
            "units": "micrometers",
        },
        inputs=[nuclei_zarr.path, cells_zarr.path],
    )


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    spacing = (50, 50, 50)
    nuclei_distance(root_path, spacing)


if __name__ == "__main__":
    main()
