"""Manifold-Decomposition Detector: branches, calibrated-max fusion, edges."""
import io

import joblib
import numpy as np
import pytest

from analysis.mdd import MDD, l4_columns
from analysis.vmem_utils import TOTAL_PHI_DIM, auroc_fpr95


@pytest.fixture
def clean_split():
    rng = np.random.default_rng(0)
    D = TOTAL_PHI_DIM
    clean = rng.normal(0, 1, (4000, D)).astype(np.float32)
    return clean[:3000], clean[3000:3500], clean[3500:]   # fit, calib, eval


def test_l4_columns_selects_deepest_block():
    cols = l4_columns(TOTAL_PHI_DIM)
    assert cols[0] == TOTAL_PHI_DIM - 3 * 256 and cols[-1] == TOTAL_PHI_DIM - 1
    # falls back gracefully for a truncated phi
    assert l4_columns(100)[0] == 0


def test_branch_names_without_spatial(clean_split):
    fit, cal, _ = clean_split
    mdd = MDD(use_spatial=False, n_ref=2000, k_nn=32).fit(fit, cal)
    assert mdd.branch_names == ["radius", "rcf", "l4"]


def test_spatial_branch_activates_only_with_phi_spatial(clean_split):
    fit, cal, ev = clean_split
    rng = np.random.default_rng(7)
    sp_fit = rng.normal(0, 1, (len(fit), 8)).astype(np.float32)
    sp_cal = rng.normal(0, 1, (len(cal), 8)).astype(np.float32)
    sp_ev = rng.normal(0, 1, (len(ev), 8)).astype(np.float32)

    assert "spatial" not in MDD(use_spatial=True, n_ref=2000, k_nn=32).fit(fit, cal).branch_names
    mdd = MDD(use_spatial=True, n_ref=2000, k_nn=32).fit(fit, cal, sp_fit, sp_cal)
    assert "spatial" in mdd.branch_names
    out = mdd.score_branches(ev, sp_ev)
    assert "spatial" in out and "fused" in out


def test_fused_detects_dilation(clean_split):
    fit, cal, ev = clean_split
    rng = np.random.default_rng(1)
    dilation = (rng.normal(0, 1, (500, TOTAL_PHI_DIM)) + 6.0).astype(np.float32)
    mdd = MDD(use_spatial=False, n_ref=2000, k_nn=32).fit(fit, cal)
    a_clean, a_ood = mdd.score(ev), mdd.score(dilation)
    auroc, _ = auroc_fpr95(np.r_[np.zeros(len(a_clean)), np.ones(len(a_ood))],
                           np.r_[a_clean, a_ood])
    assert auroc > 0.95


def _shell(n, radius, seed, jitter=0.01):
    """Points on a radius-`radius` sphere shell in TOTAL_PHI_DIM dims."""
    g = np.random.default_rng(seed)
    u = g.normal(0, 1, (n, TOTAL_PHI_DIM))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    return (u * radius + g.normal(0, jitter, u.shape)).astype(np.float32)


def test_rcf_two_sided_flags_smaller_radius():
    # Clean lives on a radius-10 shell. A point on a radius-3 shell (a
    # contraction toward the mode) must score HIGHER on RCF than an
    # in-distribution radius-10 point — the two-sided property that lets RCF
    # catch contractions, which one-sided distance detectors invert on.
    clean = _shell(4000, 10.0, seed=1)
    mdd = MDD(use_spatial=False, n_ref=2000, k_nn=64).fit(clean[:3000], clean[3000:3500])
    rcf_contraction = mdd.score_branches(_shell(200, 3.0, seed=2))["rcf"]
    rcf_indist = mdd.score_branches(_shell(200, 10.0, seed=3))["rcf"]
    assert np.median(rcf_contraction) > np.median(rcf_indist)


def test_scores_finite_and_deterministic(clean_split):
    fit, cal, ev = clean_split
    mdd = MDD(use_spatial=False, n_ref=2000, k_nn=32).fit(fit, cal)
    s1, s2 = mdd.score(ev), mdd.score(ev)
    assert np.isfinite(s1).all()
    assert np.array_equal(s1, s2)              # deterministic, no RNG in scoring


def test_knn_eq_one_no_nan():
    # k_nn resolving to 1 (tiny reference) must not produce NaN (unbiased-std fix).
    rng = np.random.default_rng(4)
    clean = rng.normal(0, 1, (60, TOTAL_PHI_DIM)).astype(np.float32)
    mdd = MDD(use_spatial=False, n_ref=1, k_nn=1).fit(clean[:40], clean[40:50])
    out = mdd.score(clean[50:])
    assert np.isfinite(out).all()


def test_serialization_roundtrip(clean_split):
    fit, cal, ev = clean_split
    mdd = MDD(use_spatial=False, n_ref=2000, k_nn=32).fit(fit, cal)
    ref = mdd.score(ev)
    buf = io.BytesIO(); joblib.dump(mdd, buf); buf.seek(0)
    assert np.allclose(joblib.load(buf).score(ev), ref, atol=1e-4)


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        MDD().score_branches(np.zeros((2, TOTAL_PHI_DIM), np.float32))
