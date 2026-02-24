#!/bin/bash

poetry run python scripts/segmentation/cellpose_flows.py
poetry run python scripts/segmentation/cellpose_masks.py
poetry run python scripts/segmentation/cellpose_upsample.py
poetry run python scripts/segmentation/cellpose_eval.py
