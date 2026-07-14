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
