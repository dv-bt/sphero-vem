"""
Calculate distance from nucleus for each NP voxel and assign parent cell.

The resulting dataframe is saved under "labels/nps/tables/distance.parquet".
"""

from pathlib import Path
import zarr
import polars as pl


def calc_dist_data(
    np_mask: zarr.Array,
    dist_map: zarr.Array,
    data_nps: pl.DataFrame,
    data_cells: pl.DataFrame,
) -> pl.DataFrame:
    """Calculate distance map"""
    np_mask_array = np_mask[:]
    np_selection = np_mask_array > 0

    np_dist = dist_map[np_selection]
    np_label = np_mask_array[np_selection]

    data_dist = pl.DataFrame(
        {
            "label": np_label,
            "dist": np_dist,
        }
    )

    data_cells = data_cells.with_columns(parent_cell="label")

    data_dist = data_dist.join(
        data_nps.select(["label", "parent_cell"]),
        on="label",
    ).join(data_cells.select(["parent_cell", "truncation_fraction"]), on="parent_cell")

    return data_dist


def main() -> None:
    """Main execution"""

    path_root = Path("data/processed/segmented/Au_01-vol_01.zarr")
    scale_dir = "50-50-50"

    # Read data
    root: zarr.Group = zarr.open(path_root, mode="r")
    np_mask = root.get(f"labels/nps/masks/{scale_dir}")
    dist_map = root.get(f"labels/nuclei/distance/{scale_dir}")
    data_nps = pl.read_parquet(path_root / "labels/nps/tables/regionprops.parquet")
    data_cells = pl.read_parquet(path_root / "labels/cells/tables/regionprops.parquet")

    data_dist = calc_dist_data(
        np_mask=np_mask, dist_map=dist_map, data_nps=data_nps, data_cells=data_cells
    )

    data_dist.write_parquet(path_root / "labels/nps/tables/nuclei-distance.parquet")


if __name__ == "__main__":
    main()
