# sphero-vem

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org)
[![CI](https://github.com/dv-bt/sphero-vem/actions/workflows/ci.yml/badge.svg)](https://github.com/dv-bt/sphero-vem/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/dv-bt/sphero-vem/graph/badge.svg?token=0VqxCn7LKp)](https://codecov.io/gh/dv-bt/sphero-vem)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Python library for quantitative analysis of volumetric electron microscopy (vEM) data.

`sphero-vem` was developed for the end-to-end analysis pipeline described in:

> Bottone et al., *3D Reconstruction of Nanoparticle Distribution in Tumor Spheroids with Volume Electron Microscopy*, [Preprint: [https://doi.org/10.64898/2026.04.17.719153](https://doi.org/10.64898/2026.04.17.719153)]

While the library was originally developed for SBF-SEM data of nanoparticle-loaded tumor spheroids, the individual components are designed to be reusable for other vEM datasets and workflows.

## Capabilities

- **Denoising**: self-supervised Noise2Void via CAREamics on large zarr volumes
- **Registration**: intensity-based pairwise slice alignment with multi-resolution PyTorch optimization
- **Segmentation**: fine-tuning and inference with Cellpose-SAM for cells and nuclei; empirical Bayes approach for nanoparticles
- **Shape analysis**: 3D morphological descriptors via signed distance functions and mesh-based curvature (mean curvature, curvedness, shape index, fractal dimension)
- **Spatial analysis**: nanoparticle-to-nucleus distance quantification per cell
- **Data management**: zarr-native I/O with OME-NGFF multiscale support and processing metadata tracking

## Installation

Clone the repository and install with [Poetry](https://python-poetry.org/):

```bash
git clone https://github.com/dv-bt/sphero-vem.git
cd sphero-vem
poetry install           # CPU and limited GPU acceleration
poetry install -E cuda   # with full CUDA 12.x GPU acceleration (Linux/Windows only)
```

## Documentation

Full documentation is available on the [official website](https://dv-bt.github.io/sphero-vem/).

## Requirements

- Python 3.12–3.13
- [PyTorch](https://pytorch.org/) ≥ 2.0
- [NumPy](https://numpy.org/) ≥ 2.0
- [zarr](https://zarr.readthedocs.io/) ≥ 3.0
- [Cellpose](https://cellpose.readthedocs.io/) ≥ 4.0
- [CAREamics](https://careamics.github.io/) ≥ 0.0.9

## GPU acceleration

PyTorch-based stages (denoising, registration, Cellpose segmentation) support GPU execution via PyTorch's native device management, including CUDA and MPS (Apple Silicon), and require no additional dependencies.

Array operation stages (nanoparticle segmentation, shape analysis, spatial analysis) additionally support CUDA acceleration via CuPy and CuCIM, installed with the `cuda` optional dependency group above. GPU acceleration is handled automatically with a custom orchestrator.

## Dataset and model weights

The annotated SBF-SEM dataset used to develop this pipeline is available at BioImage Archive (https://doi.org/10.6019/S-BIAD3263). Fine-tuned Cellpose-SAM model weights are available on Zenodo (https://doi.org/10.5281/zenodo.19616546).

## Citation

If you use this library, please cite the accompanying paper:

```bibtex
@article {Bottone2026,
	author = {Bottone, Davide and Gerken, Lukas RH and Habermann, Sebastian and Mateos, Jose Maria and Lucas, Miriam S and Riemann, Johannes and Fachet, Melanie and Resch-Genger, Ute and Kissling, Vera M and Roesslein, Matthias and Gogos, Alexander and Herrmann, Inge K},
	title = {3D Reconstruction of Nanoparticle Distribution in Tumor Spheroids with Volume Electron Microscopy},
	year = {2026},
	doi = {10.64898/2026.04.17.719153},
	eprint = {https://www.biorxiv.org/content/early/2026/04/21/2026.04.17.719153},
}
```

## License

See [LICENSE](LICENSE) for details.
