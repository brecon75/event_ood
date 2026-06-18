"""Corruption bridge (corruption_wrap.apply_corruption_to_tensor): determinism,
per-sequence seeding, dtype/shape contract. Skips a corruption if its
implementation needs an unavailable backend (e.g. CuPy)."""
import contextlib

import numpy as np
import pytest
import torch

from vmem_benchmark.corruption_wrap import apply_corruption_to_tensor


def _apply(name, sev, seed):
    base = torch.zeros((4, 20, 16, 16), dtype=torch.uint8)
    try:
        return apply_corruption_to_tensor(base, name, sev, seed=seed)
    except (ImportError, ModuleNotFoundError) as e:        # CuPy / backend missing
        pytest.skip(f"{name}: backend unavailable ({e})")


@pytest.mark.parametrize("name", ["hot_pixel", "event_flood", "event_rate_shift",
                                  "temporal_jitter", "polarity_flip", "spatial_dropout"])
def test_corruption_preserves_shape_and_dtype(name):
    out = _apply(name, 3, seed=[42, 0])
    assert out.shape == (4, 20, 16, 16)
    assert out.dtype == torch.uint8


def test_corruption_deterministic_same_seed():
    a = _apply("hot_pixel", 5, seed=[42, 0])
    b = _apply("hot_pixel", 5, seed=[42, 0])
    assert torch.equal(a, b)


def test_corruption_varies_with_seed():
    a = _apply("hot_pixel", 5, seed=[42, 0])
    b = _apply("hot_pixel", 5, seed=[42, 1])
    assert not torch.equal(a, b), "per-sequence seeds must give different realizations"


def test_additive_corruption_increases_events_on_blank_input():
    out = _apply("hot_pixel", 5, seed=[42, 0])
    assert int(out.sum()) > 0
