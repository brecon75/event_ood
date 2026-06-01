"""
cuda_utils.py — CUDA/CuPy detection and array-module helper.

All corruption functions accept either numpy or cupy arrays transparently
by using `get_array_module(arr)` to pick the right backend.
"""
import numpy as np

_CUDA_AVAILABLE: bool | None = None   # cached after first check


def cuda_available() -> bool:
    """Return True if a CUDA-capable GPU and CuPy are both present."""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is None:
        try:
            import cupy as cp
            cp.zeros(1)             # force device init; raises if no GPU
            _CUDA_AVAILABLE = True
        except Exception:
            _CUDA_AVAILABLE = False
    return _CUDA_AVAILABLE


def get_array_module(arr):
    """
    Return the array module (numpy or cupy) appropriate for `arr`.

    Usage
    -----
        xp = get_array_module(arr)
        out = xp.clip(arr + 1, 0, 255)
    """
    if cuda_available():
        try:
            import cupy as cp
            return cp.get_array_module(arr)
        except Exception:
            pass
    return np


def to_gpu(arr: np.ndarray):
    """Transfer a NumPy array to the GPU. No-op if CUDA is unavailable."""
    if cuda_available():
        import cupy as cp
        return cp.asarray(arr)
    return arr


def to_cpu(arr) -> np.ndarray:
    """Transfer a GPU array back to CPU NumPy. No-op if already NumPy."""
    if cuda_available():
        try:
            import cupy as cp
            if isinstance(arr, cp.ndarray):
                return cp.asnumpy(arr)
        except Exception:
            pass
    return arr


def clear_gpu_memory():
    """Force CuPy to release all unused memory blocks."""
    if cuda_available():
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
