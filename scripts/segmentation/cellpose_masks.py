"""
Calculate cellpose segmentation masks
"""

from pathlib import Path
from tqdm import tqdm
from sphero_vem.segmentation.cellpose import CellposeMaskConfig, calculate_masks


def main() -> None:
    """Calculate masks"""
    stack_root = Path("data/processed/segmented/Au_01-vol_01.zarr")

    # Set segmentation parameters
    seg_params = {
        "cells": {
            "merge_masks": False,
            "min_diam": 4.5,
            "niter": 400,
        },
        "nuclei": {
            "merge_weight_threshold": 0.12,
            "merge_contact_threshold": 0.15,
            "merge_masks": True,
            "niter": 400,
            "min_diam": 3.7,
        },
    }

    for seg_target, params in tqdm(seg_params.items(), "Calculating masks"):
        config = CellposeMaskConfig(
            root_path=stack_root, seg_target=seg_target, merge_masks=False, **params
        )
        calculate_masks(config)


if __name__ == "__main__":
    main()
