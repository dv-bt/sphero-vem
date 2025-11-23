"""
numpy/cupy switch with a single `xp` and a safe `ArrayLike` type.
- If cupy is importable, `xp` is cupy; otherwise it's numpy.
- `ArrayLike` works with Pylance/mypy without requiring cupy at runtime.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Union
from typing import TypeAlias
import functools

import numpy as np
import numpy.typing as npt
import dask.array as da


# --- Typing ---

if TYPE_CHECKING:
    import cupy

    ArrayLike: TypeAlias = Union[npt.NDArray[Any], "cupy.ndarray"]
else:
    ArrayLike: TypeAlias = npt.NDArray[Any]


# --- Backend selection ---


class GPUFallbackWarning(UserWarning):
    """Issued when cupy isn't available and the code falls back to numpy (CPU)"""

    pass


# Defaults
xp = np
cp = None
GPU_AVAILABLE = False

try:
    import cupy as _cp
    import cupyx.scipy.ndimage as _ndi
    import cucim.skimage as _ski

    cp = _cp
    xp = _cp
    GPU_AVAILABLE = True
except Exception:
    import scipy.ndimage as _ndi
    import skimage as _ski

    warnings.warn(
        "cupy is not installed/available. Falling back to numpy (CPU).",
        GPUFallbackWarning,
        stacklevel=2,
    )

# Updates ndimage
ndi = _ndi
ski = _ski


# --- Helpers ---


def to_host(arr: ArrayLike) -> npt.NDArray[Any]:
    """Return a numpy array on host memory, or no operation if already a numpy array."""
    if GPU_AVAILABLE and cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)  # type: ignore[arg-type]
    return np.asarray(arr)


def to_device(arr: ArrayLike) -> ArrayLike:
    """Return an array on the current device backend. A cp.array if cupy is available,
    else a numpy array."""
    if GPU_AVAILABLE and cp is not None:
        if isinstance(arr, cp.ndarray):
            return arr
        return cp.asarray(arr)  # type: ignore[no-any-return]
    return np.asarray(arr)  # type: ignore[no-any-return]


def _map_arrays(obj, fn):
    """Apply fn to arrays inside simple containers (tuple/list/dict). Useful to convert
    nested data structures to the correct device."""
    if isinstance(obj, (np.ndarray,)) or (
        cp is not None and isinstance(obj, cp.ndarray)
    ):  # type: ignore[name-defined]
        return fn(obj)
    if isinstance(obj, list):
        return [_map_arrays(x, fn) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_map_arrays(x, fn) for x in obj)
    if isinstance(obj, dict):
        return {k: _map_arrays(v, fn) for k, v in obj.items()}
    return obj


def gpu_dispatch(
    *,
    return_to_host: bool = False,
    host_kwarg: str = "_to_host",
):
    """
    Decorator that dispatches function inputs to GPU, if available, before the call.

    Kernels should use the global `xp` and `ndi` imported by this module instead of
    numpy or scipy.ndimage to ensure that calculations work as expected.

    Parameters
    ----------
    return_to_host : bool, optional
        Default behavior for the wrapped function:
        - False (default): return arrays on the active device (CuPy if GPU, NumPy otherwise)
        - True: always convert result back to NumPy if GPU is used
    host_kwarg : str, optional
        Name of a special keyword argument that can override `return_to_host` at
        *call time*. The kwarg is popped from kwargs and never forwarded to the
        wrapped function.
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            use_gpu = GPU_AVAILABLE and (cp is not None)

            # Call-time override, if provided
            host_flag = kwargs.pop(host_kwarg, return_to_host)

            if use_gpu:
                args = tuple(_map_arrays(a, to_device) for a in args)
                kwargs = {k: _map_arrays(v, to_device) for k, v in kwargs.items()}

            out = func(*args, **kwargs)

            if use_gpu and host_flag:
                return _map_arrays(out, to_host)
            return out

        return wrapper

    return decorate


def da_to_device(x: da.Array) -> da.Array:
    """Convert Dask array blocks to CuPy or NumPy depending on availability."""
    if GPU_AVAILABLE and cp is not None:
        meta = cp.empty((0,) * x.ndim, dtype=x.dtype)
    else:
        meta = np.empty((0,) * x.ndim, dtype=x.dtype)

    return da.map_blocks(to_device, x, meta=meta)


def da_to_host(x: da.Array) -> da.Array:
    """Return a Dask array whose blocks are NumPy ndarrays on host memory."""

    # If there's no GPU / CuPy, just ensure NumPy arrays
    if not (GPU_AVAILABLE and cp is not None):
        if isinstance(x._meta, np.ndarray):
            return x
        return x.map_blocks(
            np.asarray,
            meta=np.empty((0,) * x.ndim, dtype=x.dtype),
        )

    # GPU path: convert each block cp.ndarray -> np.ndarray
    def _to_host_block(block):
        if isinstance(block, cp.ndarray):
            return cp.asnumpy(block)
        return np.asarray(block)

    return x.map_blocks(
        _to_host_block,
        meta=np.empty((0,) * x.ndim, dtype=x.dtype),
    )
