"""
Registration subpackage for volume stack alignment.

This subpackage provides tools for registering sequential slices in
volumetric microscopy data using learned affine transformations, with
optional border cropping of the registered output.

Main Components
---------------
RegistrationConfig
    Configuration for zarr-based registration
register_stack
    Register a volume stack from a zarr archive
register_image_pair
    Register a single pair of images
"""

from .core import (
    RegistrationConfig,
    register_image_pair,
    register_stack,
)
from .transforms import TransformType

__all__ = [
    "RegistrationConfig",
    "register_image_pair",
    "register_stack",
    "TransformType",
]
