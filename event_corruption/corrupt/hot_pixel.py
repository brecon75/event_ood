"""
Hot Pixel Noise corruption.

Selects a fixed set of (row, col) pixel locations and adds a constant count
to every time bin and every frame at those locations, simulating defective
pixels that fire continuously regardless of scene content.

GPU support: works with both numpy and cupy arrays via get_array_module().
"""
import numpy as np
from .cuda_utils import get_array_module

# Severity → (num_hot_pixels, count_to_add_per_bin)
HOT_PIXEL_PARAMS = {
    1: (10,  20),
    2: (30,  40),
    3: (80,  80),
    4: (150, 140),
    5: (300, 200),
}


def corrupt_hot_pixel(
    histogram: np.ndarray,    # (N, 20, H, W) uint8 — numpy or cupy
    timestamps: np.ndarray,   # (N, 2) int64  — unused, kept for uniform API
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Inject hot-pixel noise into every frame at `n_pixels` random locations.

    Parameters
    ----------
    histogram  : (N, 20, H, W) uint8  — numpy or cupy array
    timestamps : (N, 2) int64
    severity   : 1–5
    rng        : seeded NumPy Generator (coordinates sampled on CPU, applied on GPU)

    Returns
    -------
    Same array type as input, (N, 20, H, W) uint8
    """
    xp = get_array_module(histogram)
    n_pixels, add_count = HOT_PIXEL_PARAMS[severity]
    _N, _C, H, W = histogram.shape

    # Cast to int16 on the same device to prevent uint8 wrap-around
    out = histogram.astype(xp.int16)

    # Sample coordinates on CPU (RNG is always NumPy)
    rows = rng.integers(0, H, size=n_pixels)
    cols = rng.integers(0, W, size=n_pixels)

    for r, c in zip(rows, cols):
        out[:, :, int(r), int(c)] += add_count   # all frames, all channels

    return xp.clip(out, 0, 255).astype(xp.uint8)
