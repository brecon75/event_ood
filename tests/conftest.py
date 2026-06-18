"""Shared fixtures for the extraction + evaluation pipeline test suite.

No real model weights are ever loaded. The monitor is exercised by injecting
synthetic membrane tensors directly into its hook buffers; phi/detector tests
use deterministic synthetic arrays.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless backend for figure/plot tests

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# vmem_benchmark modules (extract.py, corruption_wrap.py) do a bare
# `import benchmark_config`, which only resolves with vmem_benchmark/ on the
# path — true when they run as scripts, but pytest must add it explicitly.
_VB = ROOT / "vmem_benchmark"
if str(_VB) not in sys.path:
    sys.path.insert(0, str(_VB))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def make_membrane():
    """Factory for a synthetic V_mem tensor shaped (T, B, C, H, W)."""
    def _make(T=10, B=1, C=4, H=8, W=8, mean=0.5, noise=0.1, seed=0):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(T, B, C, H, W, generator=g) * noise + mean
    return _make


@pytest.fixture
def stub_monitor():
    """A VmemMonitor with no real PLIF layers; inject _v buffers by hand.

    `v_by_layer` maps layer index -> a single (T,B,C,H,W) tensor.
    """
    from vmem_benchmark.monitor import VmemMonitor

    def _make(v_by_layer):
        m = VmemMonitor(nn.Module())
        m._v = {i: [t] for i, t in v_by_layer.items()}
        return m
    return _make


@pytest.fixture
def phi_factory():
    """Factory for clean-like phi arrays at the full TOTAL_PHI_DIM width."""
    from analysis.vmem_utils import TOTAL_PHI_DIM

    def _make(n, shift=0.0, scale=1.0, dim=None, seed=0):
        d = dim or TOTAL_PHI_DIM
        r = np.random.default_rng(seed)
        return (r.normal(0, 1, (n, d)) * scale + shift).astype(np.float32)
    return _make
