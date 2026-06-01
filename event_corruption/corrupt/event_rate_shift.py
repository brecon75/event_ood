"""
Event Rate Shift corruption.

Scales all histogram bin counts by a fixed factor to simulate a domain
gap in event density. Factors < 1 simulate under-rate (low-texture /
slow-motion scene), factors > 1 simulate over-rate (high-texture /
fast-motion scene or sensor with lower threshold).

GPU support: float32 multiply + clip run entirely on-device.
This is the most GPU-friendly corruption — large speedup expected.
"""
import numpy as np
from .cuda_utils import get_array_module

# Severity → (under_scale, over_scale)
EVENT_RATE_PARAMS = {
    1: (0.80, 1.20),
    2: (0.65, 1.40),
    3: (0.50, 1.65),
    4: (0.35, 2.00),
    5: (0.20, 2.50),
}


def corrupt_event_rate_shift(
    histogram: np.ndarray,
    timestamps: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Multiply all counts by a severity-dependent scale factor.

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
    under_scale, over_scale = EVENT_RATE_PARAMS[severity]

    # Coin flip determines direction; seeded so reproducible
    scale = float(under_scale if rng.integers(0, 2) == 0 else over_scale)

    # Use float32 for scaling, but do it carefully to avoid holding extra copies
    # Clipping and rounding on the same device
    out = (histogram.astype(xp.float32) * scale).round()
    out = xp.clip(out, 0, 255).astype(xp.uint8)
    return out
