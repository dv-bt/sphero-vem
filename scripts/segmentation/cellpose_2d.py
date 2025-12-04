"""
Segment 2D datasets with pretrained cellpose models
"""

from pathlib import Path
from tqdm import tqdm
import zarr
from sphero_vem.segmentation.cellpose import (
    CellposeFlowConfig,
    CellposeMaskConfig,
    calculate_flows,
    calculate_masks,
)
from sphero_vem.utils import get_multiscales


def segment_cells(root_path: Path, spacing_dir: str) -> None:
    """Segment cells"""
    model = "cpsam"
    config_flows = CellposeFlowConfig(
        root_path=root_path,
        model=model,
        spacing_dir=spacing_dir,
        median_filter_cellprob=None,
        decompose_flows=False,
    )

    config_masks = CellposeMaskConfig(
        root_path=root_path,
        seg_target="cells",
        merge_masks=False,
        spacing_dir=spacing_dir,
    )
    calculate_flows(config_flows)
    calculate_masks(config_masks)


# def segment_nuclei(root_path: Path, spacing_dir: str) -> None:
#     """Segment cells"""
#     model = "cpsam"
#     config_flows = CellposeFlowConfig(
#         root_path=root_path,
#         model=model,
#         spacing_dir=spacing_dir,
#         median_filter_cellprob=None,
#         decompose_flows=False,
#     )

#     config_masks = CellposeMaskConfig(
#         root_path=root_path,
#         seg_target="nuclei",
#         merge_masks=False,
#         spacing_dir=spacing_dir,
#         min_diam=1,
#     )
#     calculate_flows(config_flows)
#     calculate_masks(config_masks)


def main():
    data_root = Path("data/processed/segmented/datasets_2d/cpsam")
    for dataset in tqdm(data_root.glob("*.zarr"), "Segmenting datasets"):
        image_group = zarr.open_group(dataset / "images", mode="a")

        # Get smallest scale for predictions
        scales = get_multiscales(image_group)
        arr_path = scales[-1]["path"]

        segment_cells(root_path=dataset, spacing_dir=arr_path)
        # segment_nuclei(root_path=dataset, spacing_dir=arr_path)


if __name__ == "__main__":
    main()
