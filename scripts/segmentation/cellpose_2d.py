"""
Segment 2D datasets with pretrained cellpose models
"""

from typing import NamedTuple
from pathlib import Path
from tqdm import tqdm
import zarr
from sphero_vem.segmentation.cellpose import (
    CellposeFlowConfig,
    CellposeMaskConfig,
    calculate_flows,
    calculate_masks,
)
from sphero_vem.io import _get_multiscales


class SegParams(NamedTuple):
    """Segmentation parameters for each model"""

    seg_target: str
    model: str
    dataset_dir: str


def segment_targets(root_path: Path, spacing_dir: str, params: SegParams) -> None:
    """Segment cells"""
    config_flows = CellposeFlowConfig(
        root_path=root_path,
        model=params.model,
        spacing_dir=spacing_dir,
        median_filter_cellprob=None,
        decompose_flows=False,
    )

    config_masks = CellposeMaskConfig(
        root_path=root_path,
        seg_target=params.seg_target,
        merge_masks=False,
        spacing_dir=spacing_dir,
    )
    calculate_flows(config_flows)
    calculate_masks(config_masks)


def main():
    params = [
        SegParams(seg_target="cells", model="cpsam", dataset_dir="cpsam"),
        SegParams(
            seg_target="cells",
            model="cellposeSAM-cells-20260223_093152",
            dataset_dir="finetuned",
        ),
        SegParams(
            seg_target="nuclei",
            model="cellposeSAM-nuclei-20260223_103423",
            dataset_dir="finetuned",
        ),
    ]

    for item in params:
        data_root = Path(f"data/processed/segmented/datasets_2d/{item.dataset_dir}")
        dataset_list = list(data_root.glob("*.zarr"))

        for dataset in tqdm(dataset_list, "Segmenting datasets"):
            image_group = zarr.open_group(dataset / "images", mode="a")

            # Get smallest scale for predictions
            scales = _get_multiscales(image_group)
            arr_path = scales[-1]["path"]

            segment_targets(root_path=dataset, spacing_dir=arr_path, params=item)


if __name__ == "__main__":
    main()
