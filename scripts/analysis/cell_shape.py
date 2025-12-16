"""
Calculate cell geometric parameters
"""

from pathlib import Path
import zarr
from tqdm import tqdm
import pandas as pd
import numpy as np
import porespy as ps
from sphero_vem.utils.accelerator import ski, gpu_dispatch, ArrayLike


def get_bbox_slice(row: pd.Series):
    """Get slice to index bounding box"""
    ndim = 3
    bbox_slice = tuple(
        slice(int(row[f"bbox-{i}"]), int(row[f"bbox-{i + ndim}"])) for i in range(ndim)
    )
    return bbox_slice


@gpu_dispatch(return_to_host=True)
def convex_hull(binary: ArrayLike) -> np.ndarray:
    """Calculate convex hull of object"""
    return ski.morphology.convex_hull_image(binary)


def calc_convex_vol(row: pd.Series, labels: zarr.Array, voxel_vol: float) -> int:
    """Calculate volume of convex hull for a given record"""
    binary = labels[row["bbox_slice"]] == row["label"]
    hull = convex_hull(binary)
    return hull.sum() * voxel_vol


@gpu_dispatch(return_to_host=True)
def region_boundary(binary: ArrayLike) -> np.ndarray:
    """Calculate region boundary"""
    return ski.segmentation.find_boundaries(binary)


def calc_fractal_dimension(row: pd.Series, labels: zarr.Array) -> float:
    """Calculate fractal dimension"""
    binary = labels[row["bbox_slice"]] == row["label"]
    boundary = region_boundary(binary)
    max_dim = min(boundary.shape)
    box_sizes = 2 ** np.arange(2, int(np.log2(max_dim)))
    data = ps.metrics.boxcount(boundary, bins=box_sizes)
    return data.slope.mean()


def main():
    """Calculate cell parameters"""
    data_root = Path("data/processed/segmented/Au_01-vol_01.zarr")

    paths = {
        seg_target: {
            "labels_path": data_root / f"labels/{seg_target}/masks/50-50-50",
            "data_path": data_root / f"labels/{seg_target}/tables/regionprops.parquet",
        }
        for seg_target in ["cells", "nuclei"]
    }

    for target, path in paths.items():
        labels = zarr.open_array(path["labels_path"])
        data = pd.read_parquet(path["data_path"])
        voxel_vol = np.prod(labels.attrs["spacing"]) / 1e9

        data["bbox_slice"] = data.apply(get_bbox_slice, axis=1)

        hull_vol = []
        fractal_dim = []
        for _, row in tqdm(
            data.iterrows(), total=len(data), desc=f"Analyzing {target}"
        ):
            hull_vol.append(calc_convex_vol(row, labels, voxel_vol))
            fractal_dim.append(calc_fractal_dimension(row, labels))

        data["convex_hull_volume"] = hull_vol
        data["solidity"] = data["area"] / data["convex_hull_volume"]
        data["fractal_dim"] = fractal_dim
        data = data.drop(columns="bbox_slice")

        # Overwrite regionprops with added data
        data.to_parquet(path["data_path"], index=False)


if __name__ == "__main__":
    main()
