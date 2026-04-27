"""
sphero-vem: 3D Volumetric Analysis Tools for Electron Microscopy

A scientific Python package for processing, registering, segmenting, and analyzing
3D volumetric microscopy data, with focus on spheroid cell cultures and electron
microscopy imaging.
"""

__version__ = "0.1.0"

# Core I/O operations
from .io import (
    write_image,
    stack_to_zarr,
    write_zarr,
)

# Preprocessing
from .preprocessing import (
    downscale_tensor,
)

# Registration
from .registration import (
    RegistrationConfig,
    register_image_pair,
    register_stack,
    TransformType,
)

# Denoising
from .denoising import (
    DenoisingConfig,
    train_n2v,
    denoise_stack,
)

# Measurement
from .measure import (
    LabelAnalysisConfig,
    props_voxel,
    props_sdf,
    props_mesh,
    props_fractal,
    label_properties,
    analyze_labels,
    save_regionprops,
    read_regionprops,
    get_mesh,
)

# Segmentation
from .segmentation.cellpose import (
    CellposeFlowConfig,
    calculate_flows,
    calculate_masks,
)

# Utilities
from .utils import (
    BaseConfig,
    ProcessingStep,
    timestamp,
    detect_torch_device,
)

__all__ = [
    # Version
    "__version__",
    # I/O
    "write_image",
    "stack_to_zarr",
    "write_zarr",
    # Preprocessing
    "downscale_tensor",
    # Registration
    "RegistrationConfig",
    "register_image_pair",
    "register_stack",
    "TransformType",
    # Denoising
    "DenoisingConfig",
    "train_n2v",
    "denoise_stack",
    # Measurement
    "LabelAnalysisConfig",
    "props_voxel",
    "props_sdf",
    "props_mesh",
    "props_fractal",
    "label_properties",
    "analyze_labels",
    "save_regionprops",
    "read_regionprops",
    "get_mesh",
    # Segmentation
    "CellposeFlowConfig",
    "calculate_flows",
    "calculate_masks",
    # Utilities
    "BaseConfig",
    "ProcessingStep",
    "timestamp",
    "detect_torch_device",
]
