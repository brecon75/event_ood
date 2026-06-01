"""
Spatial Dropout corruption.

Zeros out all channels in one or more rectangular regions across every
frame in the sequence. Simulates dead pixel clusters, sensor damage, or
persistent structured occlusion.

GPU support: slice assignment to zero works identically for numpy and cupy.
"""
import numpy as np
from .cuda_utils import get_array_module

# Severity → (num_regions, height_fraction, width_fraction)
SPATIAL_DROPOUT_PARAMS = {
    1: (1, 0.05, 0.05),
    2: (1, 0.10, 0.10),
    3: (2, 0.10, 0.10),
    4: (2, 0.15, 0.15),
    5: (3, 0.20, 0.20),
}


def corrupt_spatial_dropout(
    histogram: np.ndarray,
    timestamps: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Zero out rectangular regions across all frames and channels.

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
    xp = get_array_module(histogram)   # needed to verify same-device; op is slice=0
    n_regions, h_frac, w_frac = SPATIAL_DROPOUT_PARAMS[severity]
    _N, _C, H, W = histogram.shape

    out = histogram.copy()

    rh = max(1, int(H * h_frac))
    rw = max(1, int(W * w_frac))

    for _ in range(n_regions):
        r0 = int(rng.integers(0, max(1, H - rh + 1)))
        c0 = int(rng.integers(0, max(1, W - rw + 1)))
        out[:, :, r0:r0 + rh, c0:c0 + rw] = 0   # all frames, all channels

    return out
