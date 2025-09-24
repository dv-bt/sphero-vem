#!/bin/bash

# Parameters to vary
FLOW_SMOOTHING=(0 2 4 8)
STITCH_THRESHOLDS=(0.0 0.1 0.25 0.5)


# Fixed parameters
DATASET="data/processed/aligned/Au_01-vol_01/downscaled/downscaled-10"
MODEL="cellposeSAM-cells-ds10-20250911_174443"

for SMOOTHING in "${FLOW_SMOOTHING[@]}"; do
    for THRESHOLD in "${STITCH_THRESHOLDS[@]}"; do
        poetry run python scripts/segmentation/cellpose_stack.py \
            --data-dir $DATASET \
            --model $MODEL \
            --seg-params.stitch-threshold $THRESHOLD \
            --seg-params.flow3D-smooth $SMOOTHING
    done
done

echo "All segmentation experiments completed!"
