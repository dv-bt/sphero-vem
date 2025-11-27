"""
Analyze segmentation masks
"""

from pathlib import Path
import zarr
import numpy as np
import skimage as ski
import pandas as pd
from sphero_vem.utils import dirname_from_spacing


def _assign_spacing(
    data: pd.DataFrame, spacing_um: tuple[float, float, float]
) -> pd.DataFrame:
    """Assign spacing in µm to dataframe"""
    for i in range(3):
        data[f"spacing_um-{i}"] = spacing_um[i]
    return data


def _assign_cell(data: pd.DataFrame, cells: np.ndarray) -> pd.DataFrame:
    """Assign parent cell and return dataframe by looking up centroid"""
    data["parent_cell"] = 0
    for idx in data.index:
        centroid = tuple(
            round(data.loc[idx, f"centroid-{i}"] / data.loc[idx, f"spacing_um-{i}"])
            for i in range(3)
        )
        data.loc[idx, "parent_cell"] = cells[centroid]
    return data


def analyze_labels(root_path: Path, seg_target: str, spacing: tuple[int, int, int]):
    """Analyze lables and write results to a parquet file"""
    root = zarr.open(root_path, mode="r")

    # Use spacing in µm for calculations
    spacing_um = tuple(i / 1000 for i in spacing)
    image_zarr = root.get(f"labels/{seg_target}/masks/{dirname_from_spacing(spacing)}")
    image = image_zarr[:]

    properties = [
        "label",
        "area",
        "centroid",
        "equivalent_diameter_area",
        "inertia_tensor",
        "inertia_tensor_eigvals",
        "moments_central",
        "bbox",
    ]

    props = ski.measure.regionprops_table(
        image, properties=properties, spacing=spacing_um
    )

    data = pd.DataFrame(props)
    data = _assign_spacing(data, spacing_um)

    if seg_target != "cells":
        cells = root.get(f"labels/cells/masks/{dirname_from_spacing(spacing)}")[:]
        data = _assign_cell(data, cells)

    save_dir = root_path / f"labels/{seg_target}/tables"
    save_dir.mkdir(exist_ok=True, parents=True)
    data.to_parquet(save_dir / "regionprops.parquet", index=False)


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    spacing = (50, 50, 50)

    for seg_target in ["cells", "nuclei", "nps"]:
        analyze_labels(root_path, seg_target, spacing)
        print(f"Completed {seg_target} analysis")


if __name__ == "__main__":
    main()
