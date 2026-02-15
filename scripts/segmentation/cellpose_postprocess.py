"""
Post-process raw flows. This scripts assumes that "cellprob-raw" and "dP-raw" have been
saved, and it's mostly intendend for finetuning post processing hyperparameters.

Masks are also saved using the post-processed data.
"""

from tqdm import tqdm
from pathlib import Path
import zarr
from sphero_vem.segmentation.cellpose import (
    CellposeFlowConfig,
    postprocess_flows,
    CellposeMaskConfig,
    calculate_masks,
)


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    spacing_dir = "100-100-100"

    root = zarr.open(root_path, mode="a")

    # Set segmentation parameters
    flow_params = {
        "cells": {
            "model": "cellposeSAM-cells-20260211_171241",
            "decompose_flows": False,
        },
        "nuclei": {
            "model": "cellposeSAM-nuclei-20260211_170320",
            "decompose_flows": True,
        },
    }

    mask_params = {
        "cells": {
            "merge_weight_threshold": 0.2,
            "merge_contact_threshold": 0.3,
        },
        "nuclei": {
            "merge_weight_threshold": 0.1,
            "merge_contact_threshold": 0.15,
        },
    }

    # Segment stack
    for seg_target, params in tqdm(flow_params.items(), "Post-processing flows"):
        config = CellposeFlowConfig(
            root_path=root_path,
            model=params["model"],
            spacing_dir=spacing_dir,
            decompose_flows=params["decompose_flows"],
            tile_overlap=0.3,
            batch_size=64,
            median_filter_cellprob=False,
            guided_filter_cellprob=True,
            guided_filter_eps=0.001,
            save_raw_flows=False,
            verbose=False,
        )
        dP_raw = root.get(f"labels/{seg_target}/flows/dP-raw/{spacing_dir}")
        cellprob_raw = root.get(f"labels/{seg_target}/flows/cellprob-raw/{spacing_dir}")
        postprocess_flows(
            config=config,
            dP=dP_raw[:],
            cellprob=cellprob_raw[:],
        )

    for seg_target, params in tqdm(mask_params.items(), "Calculating masks"):
        config = CellposeMaskConfig(
            root_path=root_path, seg_target=seg_target, merge_masks=False, **params
        )
        calculate_masks(config)


if __name__ == "__main__":
    main()
