"""
Nanoparticle segmentation.

This module provides tools for detecting and segmenting nanoparticles in
high-resolution microscopy images, including background subtraction and
threshold-based segmentation.

Main Components
---------------
NanoparticleSegConfig
    Configuration for nanoparticle segmentation
NanoparticleSegmentation
    Main segmentation class for nanoparticle detection
label_nanoparticles
    High-level function for labeling nanoparticles in images
"""

from .core import NanoparticleSegConfig, NanoparticleSegmentation, label_nanoparticles
from .utils import downsample_posterior

__all__ = [
    "NanoparticleSegConfig",
    "NanoparticleSegmentation",
    "label_nanoparticles",
    "downsample_posterior",
]
