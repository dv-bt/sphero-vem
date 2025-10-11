"""
Run a parameter sweep for cellpose stack segmentation parameters
"""

from itertools import product
from pathlib import Path
from tqdm import tqdm
from sphero_vem.segmentation import (
    SegmentationConfig,
    SegmentationParams,
    segment_stack,
)


def main():
    # Fixed inputs
    DATASET = Path("data/processed/aligned/Au_01-vol_01/downscaled/downscaled-10")
    MODEL = "cellposeSAM-nuclei-ds10-20250911_181746"

    # Sweep grids
    FLOW_SMOOTHING = [1, 2]
    CELLPROB_TH = [0.0]
    TILE_OVERLAP = [0.2]
    N_ITER = [200]
    MIN_SIZE = [14000]
    AUGMENT = [False]
    FLOW_THRESHOLD = [0.5, 0.6, 0.7]

    sweep = list(
        product(
            FLOW_SMOOTHING,
            CELLPROB_TH,
            TILE_OVERLAP,
            N_ITER,
            MIN_SIZE,
            AUGMENT,
            FLOW_THRESHOLD,
        )
    )
    for sm, th, to, ni, ms, au, ft in tqdm(sweep, "Segmentation parameter sweep"):
        seg_params = SegmentationParams(
            flow3D_smooth=sm,
            cellprob_threshold=th,
            tile_overlap=to,
            niter=ni,
            min_size=ms,
            augment=au,
            batch_size=64,
            flow_threshold=ft,
        )

        config = SegmentationConfig(
            data_dir=DATASET,
            model=MODEL,
            seg_params=seg_params,
            verbose=False,
            compute_stats=True,
        )
        segment_stack(config)


if __name__ == "__main__":
    main()
