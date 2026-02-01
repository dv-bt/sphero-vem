"""
Analyze segmentation masks
"""

from pathlib import Path
from tqdm import tqdm
from sphero_vem.measure import analyze_labels, LabelAnalysisConfig


def main():
    root_path = Path("data/processed/segmented/Au_01-vol_01.zarr")
    scale_dir = "50-50-50"

    analysis_params = [
        {"seg_target": "cells"},
        {"seg_target": "nuclei"},
        {"seg_target": "nps", "voxel_only": True},
    ]

    pbar = tqdm(analysis_params, desc="Processing")
    for params in pbar:
        pbar.set_postfix(target=params["seg_target"])
        config = LabelAnalysisConfig(
            root_path=root_path,
            seg_target=params.get("seg_target"),
            scale_dir=scale_dir,
            voxel_only=params.get("voxel_only", False),
        )
        analyze_labels(config)


if __name__ == "__main__":
    main()
