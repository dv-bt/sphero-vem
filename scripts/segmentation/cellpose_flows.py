"""
Segment a volume stack using cellpose
"""

from pathlib import Path
import torch
from sphero_vem.segmentation.cellpose import CellposeFlowConfig, calculate_flows


def main():
    # Ensure that images are in the right directory
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")

    # Set segmentation parameters
    seg_params = {
        "cells": {
            "model": "cellposeSAM-cells-20260223_093152",
            "augment": True,
            "flow3D_smooth": 3,
            "decompose_flows": False,
            "median_filter_cellprob": False,
            "guided_filter_cellprob": True,
            "guided_filter_radius": 8,
            "guided_filter_eps": 0.01,
        },
        "nuclei": {
            "model": "cellposeSAM-nuclei-20260223_103423",
            "decompose_flows": True,
            "median_filter_cellprob": False,
            "guided_filter_cellprob": True,
            "guided_filter_eps": 0.001,
        },
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
            tile_overlap=0.3,
            batch_size=64,
        )
        calculate_flows(config)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
