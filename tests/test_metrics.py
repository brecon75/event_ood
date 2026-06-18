"""Metric correctness: auroc_fpr95 (vmem_utils) and calc_fpr95 (representation_ablation).

Golden values are hand-computed. Tolerances are explicit; no exact float ==.
These tests also probe the documented silent-failure surfaces.
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from analysis.vmem_utils import auroc_fpr95
from analysis.representation_ablation import calc_fpr95


# ───────────────────────── golden AUROC ─────────────────────────

def test_auroc_golden_ranking():
    # y=[0,0,1,1], scores=[.1,.4,.35,.8] -> 3/4 correctly ranked pairs = 0.75
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    auroc, _ = auroc_fpr95(y, s)
    assert auroc == pytest.approx(0.75, abs=1e-9)


def test_auroc_perfect_and_inverted():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert auroc_fpr95(y, np.array([0, 0, 0, 1, 1, 1]))[0] == pytest.approx(1.0)
    assert auroc_fpr95(y, np.array([1, 1, 1, 0, 0, 0]))[0] == pytest.approx(0.0)


def test_auroc_all_ties_is_half():
    y = np.array([0, 0, 1, 1])
    auroc, _ = auroc_fpr95(y, np.array([0.5, 0.5, 0.5, 0.5]))
    assert auroc == pytest.approx(0.5)


# ─────────────────── single-class / empty guards ───────────────────

def test_auroc_fpr95_single_class_returns_nan_not_crash():
    # The guarded version degrades to (nan, nan) rather than raising.
    auroc, fpr = auroc_fpr95(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(auroc) and np.isnan(fpr)


def test_calc_fpr95_single_class_returns_nan():
    # After the fix, calc_fpr95 guards single-class input like auroc_fpr95
    # (previously it raised, and callers' bare `except: continue` silently
    # dropped the metric).
    assert np.isnan(calc_fpr95(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


# ─────────────────────────── FPR@95 ───────────────────────────

def test_fpr95_perfect_separation_is_zero():
    y = np.array([0] * 50 + [1] * 50)
    s = np.array([0.0] * 50 + [1.0] * 50)
    _, fpr95 = auroc_fpr95(y, s)
    assert fpr95 == pytest.approx(0.0, abs=1e-9)


def test_fpr95_within_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = rng.normal(size=200)
    if len(np.unique(y)) == 2:
        _, fpr95 = auroc_fpr95(y, s)
        assert 0.0 <= fpr95 <= 1.0


def test_calc_fpr95_matches_auroc_fpr95_when_two_class():
    # The two FPR95 implementations should agree on well-posed input.
    rng = np.random.default_rng(1)
    y = np.array([0] * 100 + [1] * 100)
    s = np.concatenate([rng.normal(0, 1, 100), rng.normal(1.5, 1, 100)])
    _, fpr_a = auroc_fpr95(y, s)
    fpr_b = float(calc_fpr95(y, s))
    assert fpr_a == pytest.approx(fpr_b, abs=1e-9)


def test_auroc_matches_sklearn_reference():
    rng = np.random.default_rng(2)
    y = np.array([0] * 80 + [1] * 120)            # class imbalance
    s = np.concatenate([rng.normal(0, 1, 80), rng.normal(0.7, 1, 120)])
    auroc, _ = auroc_fpr95(y, s)
    assert auroc == pytest.approx(roc_auc_score(y, s), abs=1e-12)
