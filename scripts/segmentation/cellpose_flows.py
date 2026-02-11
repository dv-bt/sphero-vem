"""
Segment a volume stack using cellpose
"""

import shutil
from pathlib import Path
import zarr
from sphero_vem.segmentation.cellpose import CellposeFlowConfig, calculate_flows


def main():
    # Ensure that images are in the right directory
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    source_path = Path("data/processed/cropped/Au_01-vol_01.zarr")
    try:
        zarr.open_group(root_path, mode="r")
    except FileNotFoundError:
        shutil.copytree(source_path, root_path)

    # Set segmentation parameters
    seg_params = {
        "cells": {
            "model": "cellposeSAM-cells-20260210_184023",
            "decompose_flows": True,
        },
        # "nuclei": {
        #     "model": "cellposeSAM-nuclei-20260210_202304",
        #     "decompose_flows": True,
        # },
    }
    spacing_dir = "100-100-100"

    # Segment stack
    for seg_target, params in seg_params.items():
        print(f"Segmenting {seg_target}")
        config = CellposeFlowConfig(
            root_path=root_path,
            model=params["model"],
            spacing_dir=spacing_dir,
            decompose_flows=params["decompose_flows"],
        )
        calculate_flows(config)


if __name__ == "__main__":
    main()
