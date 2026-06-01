"""
Corruption registry.

Maps corruption names to their implementation functions and provides
a single dispatch entry point `apply_corruption`.
"""
import numpy as np

from .hot_pixel        import corrupt_hot_pixel
from .event_flood      import corrupt_event_flood
from .temporal_jitter  import corrupt_temporal_jitter
from .polarity_flip    import corrupt_polarity_flip
from .event_rate_shift import corrupt_event_rate_shift
from .spatial_dropout  import corrupt_spatial_dropout

CORRUPTIONS: dict = {
    "hot_pixel":        corrupt_hot_pixel,
    "event_flood":      corrupt_event_flood,
    "temporal_jitter":  corrupt_temporal_jitter,
    "polarity_flip":    corrupt_polarity_flip,
    "event_rate_shift": corrupt_event_rate_shift,
    "spatial_dropout":  corrupt_spatial_dropout,
}

SEVERITIES: list = [1, 2, 3, 4, 5]


def apply_corruption(
    histogram: np.ndarray,
    timestamps: np.ndarray,
    name: str,
    severity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Dispatch to the named corruption function.

    Parameters
    ----------
    histogram  : (N, 20, H, W) uint8
    timestamps : (N, 2) int64
    name       : one of CORRUPTIONS.keys()
    severity   : 1–5
    rng        : seeded np.random.Generator

    Returns
    -------
    (N, 20, H, W) uint8  — corrupted copy
    """
    if name not in CORRUPTIONS:
        raise ValueError(
            f"Unknown corruption '{name}'. Valid: {list(CORRUPTIONS.keys())}"
        )
    if severity not in SEVERITIES:
        raise ValueError(
            f"Severity must be in {SEVERITIES}, got {severity}"
        )
    return CORRUPTIONS[name](histogram, timestamps, severity, rng)
