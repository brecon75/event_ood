"""phi slicing, representation extraction, fusion alignment, margin histogram."""
import numpy as np
import pytest
import torch

from analysis.vmem_utils import (
    slice_phi_stat, slice_phi_layer, LAYER_SPECS, TOTAL_PHI_DIM,
)
from analysis.representation_ablation import extract_representation
from analysis.fusion_features import align_and_concat
from analysis.extract_offline_features import extract_margin_hist


def _phi(n=10):
    rng = np.random.default_rng(0)
    return rng.normal(0, 1, (n, TOTAL_PHI_DIM)).astype(np.float32)


def test_slice_phi_stat_widths():
    phi = _phi()
    total_C = sum(s["C"] for s in LAYER_SPECS)
    for stat in ("mu", "var", "kurtosis"):
        assert slice_phi_stat(phi, stat).shape[1] == total_C
    assert slice_phi_stat(phi, "mu").shape[1] * 3 == TOTAL_PHI_DIM


def test_slice_phi_stat_picks_correct_columns():
    # construct phi where each layer block is [0..C-1 | 100.. | 200..] so we can
    # tell mu/var/kurt apart.
    phi = np.zeros((1, TOTAL_PHI_DIM), np.float32)
    for s in LAYER_SPECS:
        C = s["C"]
        phi[0, s["phi_start"]:s["phi_start"] + C] = 1.0          # mu block
        phi[0, s["phi_start"] + C:s["phi_start"] + 2 * C] = 2.0  # var block
        phi[0, s["phi_start"] + 2 * C:s["phi_end"]] = 3.0        # kurt block
    assert np.all(slice_phi_stat(phi, "mu") == 1.0)
    assert np.all(slice_phi_stat(phi, "var") == 2.0)
    assert np.all(slice_phi_stat(phi, "kurtosis") == 3.0)


def test_slice_phi_stat_rejects_unknown():
    with pytest.raises(ValueError):
        slice_phi_stat(_phi(), "skewness")


def test_slice_phi_layer_widths():
    phi = _phi()
    for i, s in enumerate(LAYER_SPECS):
        assert slice_phi_layer(phi, i).shape[1] == 3 * s["C"]


def test_extract_representation_full_and_missing():
    feats = {"phi": _phi(), "ann": {}, "spike": {}}
    assert extract_representation(feats, "full_membrane").shape == (10, TOTAL_PHI_DIM)
    assert np.array_equal(extract_representation(feats, "membrane_mean"),
                          slice_phi_stat(feats["phi"], "mu"))
    assert extract_representation(feats, "ANN") is None       # absent -> None, not crash
    assert extract_representation(feats, "nonexistent") is None


def test_align_and_concat_basic_and_truncation():
    fd = {"a": np.ones((5, 2)), "b": np.full((5, 3), 2.0)}
    out = align_and_concat(fd, ["a", "b"])
    assert out.shape == (5, 5)
    # mismatched rows -> truncate to min length (documented prefix assumption)
    fd2 = {"a": np.ones((5, 2)), "b": np.full((3, 3), 2.0)}
    out2 = align_and_concat(fd2, ["a", "b"])
    assert out2.shape == (3, 5)
    assert align_and_concat({}, ["a"]) is None
    assert align_and_concat({"x": np.ones((2, 2))}, ["missing"]) is None


def test_margin_hist_sums_to_one_per_layer():
    N, T, sumC = 6, 10, sum(s["C"] for s in LAYER_SPECS)
    tgap = torch.randn(N, T, sumC)
    mh = extract_margin_hist(tgap, bins=20)
    assert mh.shape == (N, 20 * len(LAYER_SPECS))
    # each 20-bin block is a normalized distribution
    for i in range(len(LAYER_SPECS)):
        block = mh[:, i * 20:(i + 1) * 20]
        assert np.allclose(block.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(mh).all()
