"""
Run whole volume inference with pretrained Cellpose-SAM
"""

from pathlib import Path
from sphero_vem.segmentation.cellpose import CellposeFlowConfig, calculate_flows


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    out_path = root_path / "pretrained"
    spacing_dir = "100-100-100"

    config = CellposeFlowConfig(
        root_path=root_path,
        model="cpsam",
        out_path=out_path,
        spacing_dir=spacing_dir,
        flow3D_smooth=2,
        augment=True,
        tile_overlap=0.3,
        batch_size=64,
        decompose_flows=False,
        median_filter_cellprob=None,
        guided_filter_cellprob=False,
        save_raw_flows=False,
    )
    calculate_flows(config)


if __name__ == "__main__":
    main()
