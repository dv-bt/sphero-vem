"""
Cellpose-based cell segmentation.

This module provides tools for training and applying Cellpose models for
3D cell segmentation, including flow field computation, mask generation,
and model fine-tuning.

Main Components
---------------
CellposeFlowConfig
    Configuration for Cellpose flow field computation
CellposeMaskConfig
    Configuration for mask generation from flows
CellposeFinetuneConfig
    Configuration for fine-tuning Cellpose models
"""

from .core import (
    CellposeFlowConfig,
    CellposeMaskConfig,
    calculate_flows,
    calculate_masks,
    compute_raw_flows,
    postprocess_flows,
)
from .finetuning import CellposeFinetuneConfig, finetune_cellpose
from .utils import upsample_masks
from .evaluation import evaluate_segmentation

__all__ = [
    "CellposeFlowConfig",
    "CellposeMaskConfig",
    "calculate_flows",
    "calculate_masks",
    "compute_raw_flows",
    "postprocess_flows",
    "CellposeFinetuneConfig",
    "finetune_cellpose",
    "upsample_masks",
    "evaluate_segmentation",
]
