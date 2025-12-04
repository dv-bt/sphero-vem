"""
Calculate cellpose segmentation masks
"""

from pathlib import Path
from tqdm import tqdm
from sphero_vem.segmentation.cellpose import SegmentationMaskParams, calculate_masks


def main() -> None:
    """Calculate masks"""
    stack_root = Path("data/processed/segmented/Au_01-vol_01.zarr")

    # Set segmentation parameters
    seg_params = {
        "cells": {
            "merge_weight_threshold": 0.2,
            "merge_contact_threshold": 0.3,
        },
        "nuclei": {
            "merge_weight_threshold": 0.1,
            "merge_contact_threshold": 0.15,
        },
    }

    for seg_target, params in tqdm(seg_params.items(), "Calculating masks"):
        config = SegmentationMaskParams(
            root_path=stack_root, seg_target=seg_target, **params
        )
        calculate_masks(config)


if __name__ == "__main__":
    main()
