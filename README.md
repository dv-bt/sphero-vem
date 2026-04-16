# sphero-vem

Python library for quantitative analysis of volumetric electron microscopy (vEM) data.

`sphero-vem` was developed for the end-to-end analysis pipeline described in:

> Bottone et al., *3D Reconstruction of Nanoparticle Distribution in Tumor Spheroids with Volume Electron Microscopy*, [preprint DOI]

While the library was originally developed for SBF-SEM data of nanoparticle-loaded tumor spheroids, the individual components are designed to be reusable for other vEM datasets and workflows.

## Capabilities

- **Denoising**: self-supervised Noise2Void via CAREamics on large zarr volumes
- **Registration**: intensity-based pairwise slice alignment with multi-resolution PyTorch optimization
- **Segmentation**: fine-tuning and inference with Cellpose-SAM for cells and nuclei; empirical Bayes approach for nanoparticles
- **Shape analysis**: 3D morphological descriptors via signed distance functions and mesh-based curvature (mean curvature, curvedness, shape index, fractal dimension)
- **Spatial analysis**: nanoparticle-to-nucleus distance quantification per cell
- **Data management**: zarr-native I/O with OME-NGFF multiscale support and processing metadata tracking

## Installation

```bash
pip install sphero-vem
```

For development:

```bash
git clone https://github.com/yourusername/sphero-vem.git
cd sphero-vem
poetry install
```

GPU acceleration (CUDA 12.x, Linux only) is enabled automatically when CuPy and CuCIM are available.

## Requirements

- Python 3.11–3.13
- PyTorch 2.0+

## Documentation

Full documentation is available at [link].

## Dataset and model weights

The annotated SBF-SEM dataset used to develop this pipeline is available at BioImage Archive ([accession]). Fine-tuned Cellpose-SAM model weights are available at Zenodo ([DOI]).

## Citation

If you use this library, please cite the accompanying paper:

```bibtex
@article{Bottone2025,
  title   = {3D Reconstruction of Nanoparticle Distribution in Tumor Spheroids with Volume Electron Microscopy},
  author  = {Bottone, Davide and others},
  year    = {2025},
  doi     = {[preprint DOI]}
}
```

## License

See [LICENSE](LICENSE) for details.
