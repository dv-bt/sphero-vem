# sphero-vem

**3D Volumetric Analysis Tools for Electron Microscopy**

A comprehensive Python library for processing, analyzing, and segmenting 3D volumetric microscopy data, with emphasis on spheroid cell cultures and electron microscopy imaging.

## Features

- **3D Image Segmentation**: Cell and nuclei detection using Cellpose integration and custom algorithms
- **Advanced Shape Analysis**: Surface curvature, fractal dimensions, sphericity, and morphological measurements
- **Image Registration**: Multi-resolution alignment of serial sections with PyTorch optimization
- **Deep Learning Denoising**: Noise2Void-based noise reduction via CAREamics
- **GPU Acceleration**: Automatic CPU/GPU abstraction with CuPy (Linux with CUDA)
- **Zarr-Native Pipeline**: Efficient handling of large datasets with compression and metadata tracking
- **Reproducible Workflows**: Configuration-driven analysis with processing history preservation

## Installation

### Using pip

```bash
pip install sphero-vem
```

### Using Poetry

```bash
poetry add sphero-vem
```

### Development Installation

```bash
git clone https://github.com/yourusername/sphero-vem.git
cd sphero-vem
poetry install
```

## Quick Start

```python
import sphero_vem as sv
from pathlib import Path

# Load and register image stack
config = sv.RegistrationConfig(
    data_dir=Path("data/slices"),
    out_dir=Path("data/registered"),
    transformation="similarity",
)
sv.register_stack(config)

# Segment cells with Cellpose
flow_config = sv.CellposeFlowConfig(
    root_path=Path("data/sample.zarr"),
    src_path="images/registered",
    diameter=30,
)
sv.calculate_flows(flow_config)
sv.segment_from_flows(flow_config)

# Analyze segmented regions
analysis_config = sv.LabelAnalysisConfig(
    root_path=Path("data/sample.zarr"),
    seg_target="cells",
    spacing=(1.0, 0.5, 0.5),
)
results = sv.analyze_labels(analysis_config)
print(results[['label', 'volume_sdf', 'sphericity_sdf']])
```

## Key Capabilities

### Image Processing

- Multi-resolution downscaling and normalization
- PyTorch-based registration (similarity, rigid, affine transforms)
- Noise2Void training and inference
- Morphological operations

### Segmentation

- Cellpose 3D cell segmentation with flow field computation
- Custom model fine-tuning and CellposeSAM support
- Nanoparticle and nuclear pore detection
- Post-processing and refinement

### Shape Analysis

- **Voxel-based**: Fast properties using scikit-image regionprops
- **SDF-based**: Precise volume/surface via signed distance functions
- **Mesh-based**: Curvature statistics from mesh extraction
- **Fractal**: Minkowski-Bouligand dimension for surface complexity

### Data Management

- Zarr storage with chunking and compression
- OME-NGFF multiscale pyramid support
- Processing metadata tracking for reproducibility

<!-- ## Documentation

Full documentation is available at: [https://yourusername.github.io/sphero-vem/](https://yourusername.github.io/sphero-vem/)

- [Installation Guide](https://yourusername.github.io/sphero-vem/getting-started/installation/)
- [Quick Start Tutorial](https://yourusername.github.io/sphero-vem/getting-started/quickstart/)
- [API Reference](https://yourusername.github.io/sphero-vem/api/) -->

## Requirements

- Python 3.11-3.13
- PyTorch 2.0+
- Optional: CUDA 12.x for GPU acceleration (Linux only)

## System Requirements

### GPU Acceleration

GPU support is automatic on Linux systems with CUDA 12.x installed. The following packages provide GPU acceleration:

- CuPy: GPU-accelerated NumPy
- Dask-CUDA: Distributed GPU computing
- CuCIM: GPU image processing

macOS and Windows installations use CPU-only versions.

## Contributing

Contributions are welcome! Please see the [Contributing Guide](https://yourusername.github.io/sphero-vem/development/contributing/) for details.

### Development Setup

```bash
git clone https://github.com/yourusername/sphero-vem.git
cd sphero-vem
poetry install
pre-commit install
pytest
```

## Citation

If you use sphero-vem in your research, please cite:

```bibtex
@software{sphero_vem,
  title = {sphero-vem: 3D Volumetric Analysis Tools for Electron Microscopy},
  author = {dv-bt},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/yourusername/sphero-vem}
}
```

## License

See [LICENSE](LICENSE) for details.

## Contact

- Author: dv-bt
- Email: d.bottone@pm.me
- GitHub: [https://github.com/yourusername/sphero-vem](https://github.com/yourusername/sphero-vem)

## Acknowledgments

This project builds on excellent open-source tools:

- [Cellpose](https://github.com/MouseLand/cellpose) for cell segmentation
- [CAREamics](https://github.com/CAREamics/careamics) for Noise2Void denoising
- [PyTorch](https://pytorch.org/) for deep learning
- [zarr](https://zarr.readthedocs.io/) for array storage
- [scikit-image](https://scikit-image.org/) for image processing
