Analysis Pipeline
=================

The **sphero-vem** analysis pipeline processes image stacks through
eight ordered stages, from raw TIFF acquisition to quantitative per-label
property tables.

Stage 1: Ingest
----------------

Raw TIFF stacks are converted to `zarr <https://zarr.readthedocs.io/>`_ archives
with voxel spacing metadata. The zarr format enables efficient chunked I/O
for large volumetric datasets without loading the full stack into memory.

Stage 2: Denoising
-------------------

Each slice is denoised with self-supervised
`Noise2Void <https://careamics.github.io/>`_ via the CAREamics library.
Training and inference operate directly on zarr arrays; output is uint8.

Stage 3: Registration
----------------------

Consecutive slices are aligned with intensity-based pairwise affine registration.
Optimization uses a multi-resolution PyTorch schedule; detected misalignment
borders are cropped after registration.

Stage 4: Cell and nucleus segmentation
---------------------------------------

Cellpose-SAM flows are computed at a reduced resolution and upsampled before mask
generation. The library supports both pretrained and fine-tuned Cellpose-SAM
models; fine-tuned weights for FaDu spheroid data are available on Zenodo.

Stage 5: Nanoparticle segmentation
-----------------------------------

Nanoparticle (NP) voxels are identified by decomposing the intensity histogram
via an empirical Bayes EM algorithm into background and NP distributions,
followed by posterior thresholding.

Stage 6: Shape analysis
------------------------

Per-label 3D morphological descriptors are computed:

- **Voxel-based**: bounding box, centroid, (optional) volume
- **SDF-based**: volume, surface area, sphericity via signed distance functions
- **Mesh-based curvature**: mean curvature, Gaussian curvature, shape index,
  curvedness via mesh extraction and derivative estimation
- **Fractal dimension**: Minkowski-Bouligand tube scaling

Stage 7 — Spatial analysis
---------------------------

A Euclidean distance transform to the nearest nuclear surface is computed for
each cell. Each NP is assigned to its parent cell and its distance to the
nuclear surface is recorded.

Stage 8 — Output
-----------------

Results are written as Parquet tables (per-label properties) and zarr arrays
(distance maps), suitable for downstream statistical analysis.

GPU dispatch
-------------

All computationally intensive stages support transparent GPU acceleration via
the ``utils.accelerator`` module. When CuPy and CuCIM are available, array
operations are automatically dispatched to GPU; results are returned as NumPy
arrays where required.
