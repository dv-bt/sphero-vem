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


def segment_cells(root_path: Path, spacing_dir: str, model: str) -> None:
    """Segment cells"""
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


def main():
    params = [
        {
            "dataset_path": "cpsam",
            "model": "cpsam",
        },
        {"dataset_path": "finetuned", "model": "cellposeSAM-cells-20260223_093152"},
    ]

    for item in params:
        data_root = Path(f"data/processed/segmented/datasets_2d/{item['dataset_path']}")
        for dataset in tqdm(data_root.glob("*.zarr"), "Segmenting datasets"):
            image_group = zarr.open_group(dataset / "images", mode="a")

            # Get smallest scale for predictions
            scales = get_multiscales(image_group)
            arr_path = scales[-1]["path"]

            segment_cells(root_path=dataset, spacing_dir=arr_path, model=item["model"])


if __name__ == "__main__":
    main()
