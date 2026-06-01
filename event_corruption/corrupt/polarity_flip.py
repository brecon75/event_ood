"""
Polarity Flip corruption.

For a random subset of frames, swaps the ON-polarity channels (0–9)
with the OFF-polarity channels (10–19). Simulates sensor misclassification
of brightness-change direction (ON↔OFF).

GPU support: slice assignment works identically for numpy and cupy arrays.
"""
import numpy as np
from .cuda_utils import get_array_module

# Severity → fraction of frames that get ON↔OFF swapped
POLARITY_FLIP_PARAMS = {
    1: 0.05,
    2: 0.10,
    3: 0.20,
    4: 0.35,
    5: 0.50,
}


def corrupt_polarity_flip(
    histogram: np.ndarray,
    timestamps: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Swap ON and OFF channel blocks for a random fraction of frames.

    Parameters
    ----------
    histogram  : (N, 20, H, W) uint8  — numpy or cupy array
    timestamps : (N, 2) int64
    severity   : 1–5
    rng        : seeded NumPy Generator

    Returns
    -------
    Same array type as input, (N, 20, H, W) uint8
    """
    xp = get_array_module(histogram)
    flip_prob = POLARITY_FLIP_PARAMS[severity]
    N = histogram.shape[0]

    out = histogram.copy()

    # RNG decision is always on CPU
    flip_mask_cpu = rng.random(size=N) < flip_prob   # (N,) bool, NumPy

    if not flip_mask_cpu.any():
        return out

    # Transfer mask to the same device as the histogram
    if xp.__name__ == "cupy":
        import cupy as cp
        flip_mask = cp.asarray(flip_mask_cpu)
    else:
        flip_mask = flip_mask_cpu

    # Swap ON (0:10) ↔ OFF (10:20) for selected frames
    # Direct slice assignment avoids the fancy-index copy pitfall
    tmp = out[flip_mask, 0:10, :, :].copy()
    out[flip_mask, 0:10,  :, :] = out[flip_mask, 10:20, :, :]
    out[flip_mask, 10:20, :, :] = tmp

    return out   # uint8 preserved, no arithmetic needed
