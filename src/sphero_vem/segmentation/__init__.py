"""
Segmentation subpackage for cell and organelle segmentation.

This subpackage provides tools for segmenting different cellular structures
in volumetric microscopy data:

- **cellpose**: Cell segmentation using Cellpose models with support for
  custom training and CellposeSAM integration
- **np**: Nuclear pore segmentation and analysis

Modules
-------
cellpose
    Cell segmentation using Cellpose models
np
    Nuclear pore segmentation
"""

from . import cellpose
from . import np

__all__ = ["cellpose", "np"]
