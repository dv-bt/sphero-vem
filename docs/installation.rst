Installation
============

Requirements
------------

- Python 3.12–3.13
- `PyTorch <https://pytorch.org/>`_ ≥ 2.0
- `NumPy <https://numpy.org/>`_ ≥ 2.0
- `zarr <https://zarr.readthedocs.io/>`_ ≥ 3.0
- `Cellpose <https://cellpose.readthedocs.io/>`_ ≥ 4.0
- `CAREamics <https://careamics.github.io/>`_ ≥ 0.0.9

PyPI install
------------

The library is available on `PyPI <https://pypi.org/project/sphero-vem/>`_.
We recommend installing it in a dedicated virtual environment:

.. code-block:: bash

   pip install sphero-vem

For full CUDA 12.x GPU acceleration (Linux only; Windows untested):

.. code-block:: bash

   pip install "sphero-vem[cuda]"

Development install
-------------------

Clone the repository and install with `Poetry <https://python-poetry.org/>`_:

.. code-block:: bash

   git clone https://github.com/dv-bt/sphero-vem.git
   cd sphero-vem
   poetry install           # base install
   poetry install -E cuda   # with CUDA 12.x GPU acceleration (Linux only; Windows untested)

GPU acceleration
----------------

PyTorch-based stages (denoising, registration, Cellpose-based segmentation) support
GPU execution via PyTorch's native device management, including CUDA and MPS
(Apple Silicon), with no additional dependencies.

Array operation stages (nanoparticle segmentation, shape analysis, spatial analysis)
additionally support CUDA acceleration via CuPy and CuCIM, available with the
``cuda`` extra above. Automatic backend switching (NumPy ↔ CuPy,
scikit-image ↔ CuCIM) is handled at import time by ``utils.accelerator``.

Dataset and model weights
--------------------------

The annotated SBF-SEM dataset used to develop and benchmark this pipeline is
available at `BioImage Archive <https://doi.org/10.6019/S-BIAD3263>`_.
Fine-tuned Cellpose-SAM model weights are available on
`Zenodo <https://doi.org/10.5281/zenodo.19616546>`_.
