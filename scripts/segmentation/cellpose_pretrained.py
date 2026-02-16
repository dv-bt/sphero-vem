"""
Run whole volume inference with pretrained Cellpose-SAM
"""

from pathlib import Path
from sphero_vem.segmentation.cellpose import (
    CellposeFlowConfig,
    calculate_flows,
    CellposeMaskConfig,
    calculate_masks,
)


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    out_path = root_path / "pretrained"
    spacing_dir = "100-100-100"

    flow_config = CellposeFlowConfig(
        root_path=root_path,
        model="cpsam",
        out_path=out_path,
        spacing_dir=spacing_dir,
        flow3D_smooth=3,
        augment=True,
        tile_overlap=0.3,
        batch_size=64,
        decompose_flows=False,
        median_filter_cellprob=None,
        guided_filter_cellprob=False,
        save_raw_flows=False,
    )
    calculate_flows(flow_config)

    mask_config = CellposeMaskConfig(
        root_path=root_path,
        seg_target="cells",
        spacing_dir=spacing_dir,
        label_root="pretrained",
        niter=400,
        min_diam=4.5,
        merge_masks=False,
    )
    calculate_masks(mask_config)


if __name__ == "__main__":
    main()
