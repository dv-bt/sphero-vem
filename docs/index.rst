sphero-vem
==========

**sphero-vem** is a Python library for end-to-end quantitative analysis of
volume electron microscopy (vEM) data, with a focus on 3D cell and nucleus
morphology and nanoparticle (NP) localization and quantification in tumor
spheroid models.
It covers the full analysis pipeline from raw image ingestion to per-label
quantitative property extraction, with GPU acceleration throughout: automatic
via CuPy and CuCIM for array operations, and PyTorch-native for registration
and segmentation.
Cell and nucleus segmentation builds on `Cellpose-SAM
<https://cellpose.readthedocs.io/>`_, self-supervised denoising on
`CAREamics <https://careamics.github.io/>`_, and image registration on
`PyTorch <https://pytorch.org/>`_.

Originally developed for SBF-SEM imaging of gold nanoparticle (AuNP)
distribution in FaDu head-and-neck tumor spheroids, the pipeline is designed
to generalize to other vEM modalities and biological specimens.

**sphero-vem** accompanies the paper:

   Bottone et al., *3D Reconstruction of Nanoparticle Distribution in Tumor
   Spheroids with Volume Electron Microscopy*, 2026.
   `https://doi.org/10.64898/2026.04.17.719153 <https://doi.org/10.64898/2026.04.17.719153>`_

The annotated dataset is publicly available at `BioImage Archive (S-BIAD3263)
<https://doi.org/10.6019/S-BIAD3263>`_, and finetuned model weights for cell and
nucleus segmentation are available on `Zenodo
<https://doi.org/10.5281/zenodo.19616546>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   pipeline
   api/index
