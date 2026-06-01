"""
Event Flood corruption.

Inserts sudden bursts of saturating events into random spatial patches
centred on random frames. Simulates electromagnetic interference or
sensor saturation affecting a localised region of the array.

GPU support: works with both numpy and cupy arrays via get_array_module().
"""
import numpy as np
from .cuda_utils import get_array_module

# Severity → (num_burst_centres, patch_fraction_of_HW, count_to_add)
EVENT_FLOOD_PARAMS = {
    1: (2,  0.05, 60),
    2: (4,  0.08, 100),
    3: (8,  0.12, 150),
    4: (15, 0.18, 200),
    5: (25, 0.25, 255),
}

BURST_RADIUS = 2   # frames on each side of the burst centre


def corrupt_event_flood(
    histogram: np.ndarray,
    timestamps: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Add localised count bursts at random frames and spatial patches.

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
    n_bursts, patch_frac, add_count = EVENT_FLOOD_PARAMS[severity]
    N, _C, H, W = histogram.shape

    out = histogram.astype(xp.int16)

    ph = max(1, int(H * patch_frac))
    pw = max(1, int(W * patch_frac))

    # All random decisions made on CPU
    burst_centres = rng.integers(0, N, size=n_bursts)
    r0s = rng.integers(0, max(1, H - ph + 1), size=n_bursts)
    c0s = rng.integers(0, max(1, W - pw + 1), size=n_bursts)

    for bc, r0, c0 in zip(burst_centres, r0s, c0s):
        f_lo = max(0, int(bc) - BURST_RADIUS)
        f_hi = min(N, int(bc) + BURST_RADIUS + 1)
        out[f_lo:f_hi, :, int(r0):int(r0) + ph, int(c0):int(c0) + pw] += add_count

    return xp.clip(out, 0, 255).astype(xp.uint8)
