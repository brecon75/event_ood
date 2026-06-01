"""
Temporal Jitter corruption.

Rolls the ON-polarity bin channels and the OFF-polarity bin channels
independently by a random integer offset within ±max_shift. Vacated
bin positions are zeroed. This simulates clock noise / timestamp
inaccuracy that shifts which time-bin an event is counted into.

Channel layout:
    channels  0–9  : ON  polarity, time bins 0–9 (oldest → newest)
    channels 10–19 : OFF polarity, time bins 0–9

GPU support: uses xp.roll() which works with both numpy and cupy.
"""
import numpy as np
from .cuda_utils import get_array_module

# Severity → max shift in bins  (1 bin = 50 µs)
TEMPORAL_JITTER_PARAMS = {1: 1, 2: 2, 3: 3, 4: 5, 5: 8}

ON_SLICE  = slice(0, 10)
OFF_SLICE = slice(10, 20)


def _roll_bins(arr, shift: int, xp):
    """
    Roll along axis=1 (bin axis) by `shift` steps, zero-filling vacated edges.
    Positive shift moves content towards higher bin indices.
    Works with both numpy and cupy arrays.
    """
    if shift == 0:
        return arr.copy()
    result = xp.zeros_like(arr)
    if shift > 0:
        result[:, shift:, :, :] = arr[:, :-shift, :, :]
    else:  # shift < 0
        result[:, :shift, :, :] = arr[:, -shift:, :, :]
    return result


def corrupt_temporal_jitter(
    histogram: np.ndarray,
    timestamps: np.ndarray,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Roll ON and OFF bin channels independently by a random shift.

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
    max_shift = TEMPORAL_JITTER_PARAMS[severity]

    shift_on  = int(rng.integers(-max_shift, max_shift + 1))
    shift_off = int(rng.integers(-max_shift, max_shift + 1))

    out = histogram.copy()
    out[:, ON_SLICE,  :, :] = _roll_bins(histogram[:, ON_SLICE,  :, :], shift_on,  xp)
    out[:, OFF_SLICE, :, :] = _roll_bins(histogram[:, OFF_SLICE, :, :], shift_off, xp)

    return out   # dtype preserved (uint8), zeros stay in range
