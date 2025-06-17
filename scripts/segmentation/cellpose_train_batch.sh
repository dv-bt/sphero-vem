#!/bin/bash

# Parameters to vary
DS_FACTORS=(5 8 10 20)

# Fixed parameters
EPOCHS=201
DATASET="processed/labeled/Au_01-vol_01/labeled-01"
SEG_TARGET="nuclei"

for DS_FACTOR in "${DS_FACTORS[@]}"; do
    poetry run python scripts/segmentation/cellpose_finetuning.py \
        --dataset $DATASET \
        --ds $DS_FACTOR \
        --epochs $EPOCHS \
        --seg-target $SEG_TARGET
done

echo "All finetuning experiments completed!"
