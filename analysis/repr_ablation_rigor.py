"""Representation ablation, rigor pass: is phi/phi_spatial's separation from
the ANN/spike/logit baselines statistically real, and does it hold up without
Mahalanobis's Gaussian-clean-class assumption?

`representation_ablation.py` (the canonical pipeline stage, unmodified) reports
one Mahalanobis AUROC per (representation, corruption, severity) and nothing
else -- no uncertainty, no significance test, no phi_spatial, and every number
still depends on Mahalanobis's parametric assumption. This script is a
SEPARATE, heavier analysis pass that adds:

  1. phi_spatial as a compared representation (extracted alongside phi but
     never actually compared against ANN/spike/logits before).
  2. Cluster (per-recording) bootstrap 95% CIs on AUROC -- resampling frames
     directly would treat correlated within-sequence frames as independent
     and understate the true uncertainty.
  3. A PAIRED cluster-bootstrap significance test: phi/phi_spatial vs the
     ANN/spike/logit baselines, and phi_spatial vs phi. Same resampled
     recordings score both representations each draw, so shared run-to-run
     variance cancels and what's left isolates the representation's effect.
  4. Jensen-Shannon divergence (bits) between the clean/corrupt SCORE
     distributions -- complements AUROC (rank-order only) with how separated
     the two distributions actually are.
  5. MMD (Gretton et al., 2012, "A Kernel Two-Sample Test", JMLR): a
     detector-free comparison of the RAW representation vectors -- no
     Mahalanobis fit at all, no Gaussian assumption, RBF kernel with a
     median-heuristic bandwidth, significance via a recording-level
     permutation test (not a frame-level one, for the same cluster-structure
     reason as #2/#3).
  6. Cliff's delta (Cliff, 1993): `2*AUROC - 1` is an exact identity for a
     continuous-score two-sample comparison (AUROC = P(X>Y) + 0.5*P(X=Y);
     Cliff's delta = P(X>Y) - P(X<Y)), so this needs no separate O(n^2) pass --
     reported alongside AUROC as the ordinal effect size S5 asks for.
  7. Benjamini-Hochberg FDR (1995) q-values over the pairwise-test p-value
     grid (S1) -- there are ~5 pairs x 6 corruptions x 5 severities = 150
     tests in `pairwise_tests.csv`; uncorrected p-values there would
     overstate significance.
  8. TOST-style equivalence (Schuirmann, 1987), read off the delta's existing
     95% cluster-bootstrap CI (S3): the pair is equivalence-flagged at margin
     +/-0.02 AUROC iff the WHOLE CI sits inside that band. Note this reuses
     the already-computed 95% CI (alpha=0.025/side), more conservative than
     TOST's conventional 90%-CI/alpha=0.05 pairing -- flagged, not silently
     mismatched.
  9. Ross (2014, PLOS ONE 9(2):e87357) k-NN mutual information between the RAW
     representation and the clean/corrupt label -- detector-free like MMD, but
     answers a different question (I1): how much information about the label
     this representation carries at all, in bits. Matches scikit-learn's
     `_compute_mi_cd` source exactly (generalized to multivariate X; see
     `ross_mi_bits` docstring for why -- an earlier version transcribed from
     an AI-summarized PDF fetch had the wrong formula and failed the
     identical-distributions sanity check). Swept over k in {3,5,10} and
     reported as mean+-std (I1's own caveat: KSG-family estimators are biased
     in high-D, so std here is the honest uncertainty, not a formal CI).
     Feeds Fano's inequality (I3) for a representation-intrinsic error-
     probability floor: no detector on this representation, however good, can
     beat that floor.

Output (curated/committed numbers, not the churn-y outputs/results/ working
dir): unified_numbers/repr_ablation_metrics.csv,
unified_numbers/repr_ablation_pairwise_tests.csv.
"""
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from collections.abc import Mapping
from scipy.spatial.distance import jensenshannon
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.optimize import brentq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    slice_phi_stat, split_train_eval, load_phi_seq_lens,
    chunked_apply, device_for, load_pt, materialize_f32, auroc_aupr_fpr95,
    split_boundary, seq_lens_after_cut, FAST_MODE,
    pool_ranges, POOL_NAMES,
)

# ── phi_spatial: the 1488-D re-extraction, NOT the 1408-D legacy block ───────
# vmem_utils.load_phi_spatial reads phi_spatial out of outputs/phi/<run>.pt,
# which is the ORIGINAL 1408-D version (spatial_var + participation ratio only).
# The organizational stats (flatness + causal persistence, commit 34af9dd) that
# drive the shipped spatial branch live in the top-level phi_spatial/ folder as
# 1488-D. Scoring the 1408-D block here would report the pre-fix representation
# -- the one that sits BELOW chance on spatial_dropout -- next to results
# produced with the 1488-D one. Same override the unified_numbers drivers use.
SPATIAL_DIR = cfg.REPO_ROOT / "phi_spatial"


def load_phi_spatial(run_name="clean", rows=None):
    """1488-D phi_spatial for one run, or None. A couple of files carry a
    ' (1)' download suffix.

    `rows` (a slice) is applied to the MMAP-BACKED tensor before materializing,
    so only the requested pages are faulted. This matters: the full array is
    1.9 GB, and `phi_plus_spatial` would otherwise need phi (2.9 GB) + spatial
    (1.9 GB) + the concat (4.8 GB) + a standardized copy (4.8 GB) resident at
    once, which OOMs. Callers that only need one pool must pass its slice.
    """
    for name in (f"{run_name}.pt", f"{run_name} (1).pt"):
        p = SPATIAL_DIR / name
        if p.exists():
            ps = load_pt(p).get("phi_spatial", None)
            if ps is None:
                return None
            return materialize_f32(ps if rows is None else ps[rows])
    return None


# ── which runs are real ──────────────────────────────────────────────────────
# outputs/phi/ also holds throwaway probe artifacts from the free-rider ablation
# (random-weight SNN, raw-input) -- random_*.pt, raw_*.pt, _tmp_*. Globbing the
# directory picked up 18 of them and wrote rows with invented corruption names
# like "random_hot_pixel" straight into the results CSV. Only clean + the
# configured corruption x severity grid are real runs.
#
# phi/phi_spatial for L2 and L4 exist on disk for every corruption; only the
# scoring config was ever trimmed to [1,3,5]. Score all five (matches
# unified_numbers).
cfg.SEVERITIES = [1, 2, 3, 4, 5]
VALID_RUNS = {"clean"} | {f"{c}_L{s}" for c in cfg.CORRUPTIONS for s in cfg.SEVERITIES}
from analysis.gpu_fit import empirical_precision

UNIFIED_DIR = Path(__file__).resolve().parent.parent / "unified_numbers"
# --output-dir redirects every CSV this script writes. Without it a smoke test
# (`--fast`, tiny bootstrap/permutation counts) overwrites the committed
# unified_numbers/ tables with throwaway numbers that look real. Any test run
# MUST pass a scratch directory.
for _i, _a in enumerate(sys.argv):
    if _a == "--output-dir" and _i + 1 < len(sys.argv):
        UNIFIED_DIR = Path(sys.argv[_i + 1])
    elif _a.startswith("--output-dir="):
        UNIFIED_DIR = Path(_a.split("=", 1)[1])
if FAST_MODE and UNIFIED_DIR.name == "unified_numbers":
    raise SystemExit(
        "Refusing to write FAST_MODE results into unified_numbers/.\n"
        "--fast uses tiny bootstrap/permutation counts; those numbers are not "
        "reportable and must not land next to the real tables.\n"
        "Pass --output-dir <scratch-dir> for smoke tests.")

SEED = 0
BOOT_B = 50 if FAST_MODE else 300          # bootstrap draws for CI / paired tests
POOL_CAP = 20000                           # per-class frame cap inside each bootstrap draw
# ^ speed-only: point estimates (auroc/aupr/fpr95/js) always use the FULL
# held-out data. Capping the pooled frames INSIDE each bootstrap draw keeps
# the per-draw AUROC recompute fast; the quantity the CI actually measures is
# between-RECORDING resampling variance, which this barely affects as long as
# the cap stays well above a typical recording's frame count.

PAIRS = [
    ("full_membrane", "ANN"),
    ("full_membrane", "logits"),
    ("full_membrane", "spike"),
    ("phi_spatial", "ANN"),
    ("phi_spatial", "full_membrane"),
]


def fast_auroc(y, s):
    """Rank-based AUROC (Mann-Whitney U). ~2-3x faster than sklearn's
    roc_auc_score at this scale (no multiclass/threshold bookkeeping), which
    matters when it's called O(B x reps x runs) times for the bootstrap."""
    order = np.argsort(s, kind="quicksort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n_pos = y.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def held_out_lens_for(arr_len, seq_lens):
    """Per-recording frame counts for the held-out [cut:] tail of an array of
    length `arr_len`, mirroring the cut `held_out_eval` applies internally."""
    if not seq_lens:
        return None
    cut = split_boundary(arr_len, seq_lens=seq_lens)
    return seq_lens_after_cut(seq_lens, cut)


# ── canonical pools (2026-08-29) ─────────────────────────────────────────────
# This file used to use its own sequence-aware 70/30 split, so its numbers were
# not comparable with the MDD tables or anything in unified_numbers/README.md.
# It now uses the project's single 4-way split: fit 50% / calib 10% /
# sensitivity 10% / final 30%. Detectors fit on `fit`; every reported metric
# comes from `final`. `calib` and `sensitivity` are deliberately left unused
# here (a one-sided Mahalanobis has nothing to calibrate and no knob to select)
# rather than quietly folded back into fit -- keeping the eval data identical
# to MDD's is the whole point.
def fit_rows(arr, seq_lens):
    a, b = pool_ranges(len(arr), seq_lens)["fit"]
    return arr[a:b]


def final_rows(arr, seq_lens):
    a, b = pool_ranges(len(arr), seq_lens)["final"]
    return arr[a:b]


def final_lens(arr_len, seq_lens):
    """Per-recording frame counts for the `final` pool."""
    if not seq_lens:
        return None
    a, _ = pool_ranges(arr_len, seq_lens)["final"]
    return seq_lens_after_cut(seq_lens, a)


def group_by_seq(scores, seq_lens):
    """Split flat per-frame scores into one array per recording (the cluster
    bootstrap resampling unit). Falls back to a single cluster when seq_lens
    is unavailable/mismatched (legacy files) -- CI then collapses to a point
    since there's nothing to resample."""
    scores = np.asarray(scores)
    if not seq_lens or int(np.sum(seq_lens)) != len(scores):
        return [scores]
    out, i = [], 0
    for L in seq_lens:
        out.append(scores[i:i + L]); i += L
    return out


def _pool(recs, idx, rng, cap=POOL_CAP):
    pool = np.concatenate([recs[k] for k in idx]) if len(idx) else np.array([])
    if cap and len(pool) > cap:
        pool = rng.choice(pool, cap, replace=False)
    return pool


def cluster_bootstrap_auroc_ci(clean_recs, corr_recs, rng, B=BOOT_B):
    """95% CI for AUROC via cluster (per-recording) bootstrap: resample
    recordings with replacement, pool their frames, recompute AUROC."""
    nc, nt = len(clean_recs), len(corr_recs)
    if nc < 2 or nt < 2:
        return float("nan"), float("nan")
    stats = np.empty(B)
    for b in range(B):
        neg = _pool(clean_recs, rng.integers(0, nc, nc), rng)
        pos = _pool(corr_recs, rng.integers(0, nt, nt), rng)
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        stats[b] = fast_auroc(y, np.r_[neg, pos])
    lo, hi = np.nanpercentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def paired_cluster_bootstrap(clean_a, corr_a, clean_b, corr_b, rng, B=BOOT_B):
    """Paired cluster bootstrap for AUROC_a - AUROC_b: the SAME resampled
    recording indices score both representations each draw, so within-run
    variability shared by both cancels out and the delta isolates the
    representation effect. Returns (delta_ci_lo, delta_ci_hi, p_two_sided);
    p is the bootstrap-distribution sign-flip fraction (2*min(P(d<=0), P(d>=0)))."""
    nc, nt = len(clean_a), len(corr_a)
    if nc < 2 or nt < 2 or len(clean_b) != nc or len(corr_b) != nt:
        return float("nan"), float("nan"), float("nan")
    deltas = np.empty(B)
    for b in range(B):
        ci = rng.integers(0, nc, nc)
        ti = rng.integers(0, nt, nt)
        neg_a, pos_a = _pool(clean_a, ci, rng), _pool(corr_a, ti, rng)
        neg_b, pos_b = _pool(clean_b, ci, rng), _pool(corr_b, ti, rng)
        y_a = np.r_[np.zeros(len(neg_a)), np.ones(len(pos_a))]
        y_b = np.r_[np.zeros(len(neg_b)), np.ones(len(pos_b))]
        auc_a = fast_auroc(y_a, np.r_[neg_a, pos_a])
        auc_b = fast_auroc(y_b, np.r_[neg_b, pos_b])
        deltas[b] = auc_a - auc_b
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return float(lo), float(hi), float(min(p, 1.0))


def js_divergence_bits(a, b, n_bins=50):
    """Jensen-Shannon divergence (bits, in [0,1]) between the clean/corrupt
    score distributions -- a histogram-based, threshold-free separability
    measure that (unlike AUROC) is sensitive to distribution SHAPE, not just
    rank order."""
    a, b = np.asarray(a), np.asarray(b)
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(a, bins=bins)
    q, _ = np.histogram(b, bins=bins)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    dist = jensenshannon(p / p.sum(), q / q.sum(), base=2)
    return float(dist ** 2) if np.isfinite(dist) else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Detector-free comparison: MMD (Gretton et al., 2012). Mahalanobis assumes
# clean phi is roughly Gaussian with one covariance -- a strong parametric
# assumption every AUROC/CI/JS number above inherits. MMD compares the two
# RAW vector sets directly via an RKHS mean-embedding distance: no detector is
# fit, no Gaussian assumption, and it comes with its own permutation-based
# significance test. The permutation reshuffles whole RECORDINGS (not
# frames), matching the cluster structure respected everywhere else here.
#
# Exact MMD needs an n x n kernel matrix, so both sides are subsampled down to
# a bounded number of recordings x frames/recording before the O(n^2) kernel
# build -- this is a speed cap only (mirrors the GMM_FIT_SAMPLES /
# OCSVM_FIT_SAMPLES pattern in vmem_utils), not a change to the point AUROC
# metrics above, which still see the full held-out data.
# ─────────────────────────────────────────────────────────────────────────────

MMD_REC_CAP = 24                            # recordings sampled per side
MMD_FRAME_CAP = 40                          # frames sampled per sampled recording
MMD_PERM_B = 30 if FAST_MODE else 200       # recording-level permutations for the p-value


def _mmd_subsample(feat, ho_lens, rng, rec_cap=MMD_REC_CAP, frame_cap=MMD_FRAME_CAP):
    """Subsample up to `rec_cap` recordings and `frame_cap` frames/recording
    from a held-out representation. Returns (rows, sizes) where `sizes` is
    the per-sampled-recording row count, so the permutation test can
    reshuffle whole recordings intact. (None, None) if there's no recording
    structure to subsample from (legacy files without seq_lens)."""
    if not ho_lens or len(ho_lens) < 2:
        return None, None
    starts = np.cumsum([0] + list(ho_lens))[:-1]
    rec_idx = rng.choice(len(ho_lens), size=min(rec_cap, len(ho_lens)), replace=False)
    chunks = []
    for r in rec_idx:
        L = ho_lens[r]
        take = min(frame_cap, L)
        rows = np.sort(rng.choice(L, size=take, replace=False)) + starts[r]
        chunks.append(feat[rows])
    return np.concatenate(chunks, axis=0), [len(c) for c in chunks]


def _rbf_kernel_matrix(X, device):
    """Full n x n RBF Gram matrix with a median-heuristic bandwidth."""
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).to(device)
    sq = (Xt * Xt).sum(1)
    d2 = (sq[:, None] + sq[None, :] - 2 * (Xt @ Xt.T)).clamp(min=0)
    off_diag = d2[~torch.eye(len(X), dtype=torch.bool, device=device)]
    bw2 = torch.median(off_diag).clamp(min=1e-12)
    return torch.exp(-d2 / (2 * bw2)).cpu().numpy()


def _mmd2_unbiased(K, n_a):
    """Unbiased MMD^2 (Gretton et al. 2012, eq. 3) from a precomputed Gram
    matrix `K` whose first `n_a` rows/cols are group A, the rest group B."""
    n = K.shape[0]
    a, b = n_a, n - n_a
    Kxx, Kyy, Kxy = K[:a, :a], K[a:, a:], K[:a, a:]
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (a * (a - 1)) if a > 1 else 0.0
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (b * (b - 1)) if b > 1 else 0.0
    sum_xy = Kxy.sum() / (a * b) if a > 0 and b > 0 else 0.0
    return float(sum_xx + sum_yy - 2 * sum_xy)


def mmd_test(clean_X, clean_sizes, corr_feat, corr_ho_lens, rng, B=MMD_PERM_B):
    """Classifier/detector-free comparison of two held-out representation
    samples: MMD^2 effect size + a recording-level permutation p-value.
    `clean_X`/`clean_sizes` are a FIXED subsample computed once per
    representation (see main()) and reused against every corruption run.
    Returns (mmd2, p_value, n_clean, n_corrupt); all-NaN/0 if there isn't
    enough recording structure to subsample/permute."""
    corr_X, corr_sizes = _mmd_subsample(corr_feat, corr_ho_lens, rng)
    if clean_X is None or corr_X is None or len(clean_sizes) < 2 or len(corr_sizes) < 2:
        return float("nan"), float("nan"), 0, 0

    device = device_for("MMD kernel", verbose=False)
    X = np.concatenate([clean_X, corr_X], axis=0)
    K = _rbf_kernel_matrix(X, device)
    n_a = len(clean_X)
    obs = _mmd2_unbiased(K, n_a)

    all_sizes = clean_sizes + corr_sizes
    n_recs, n_a_recs = len(all_sizes), len(clean_sizes)
    rec_bounds = np.cumsum([0] + all_sizes)
    row_idx = np.arange(len(X))
    perm_stats = np.empty(B)
    for i in range(B):
        rec_perm = rng.permutation(n_recs)
        row_perm = np.concatenate([row_idx[rec_bounds[r]:rec_bounds[r + 1]] for r in rec_perm])
        a_frames = sum(all_sizes[r] for r in rec_perm[:n_a_recs])
        perm_stats[i] = _mmd2_unbiased(K[np.ix_(row_perm, row_perm)], a_frames)
    p = float((perm_stats >= obs).mean())
    return obs, p, len(clean_X), len(corr_X)


# ─────────────────────────────────────────────────────────────────────────────
# Detector-free representation ranking, take 2: mutual information (Ross 2014)
# + Fano's inequality (I1/I3). Like MMD, this never fits a detector -- it asks
# how much label information the raw representation carries at all, which is
# the ceiling every detector on this representation is bounded by.
# ─────────────────────────────────────────────────────────────────────────────

MI_K_VALUES = (3, 5, 10)


def ross_mi_bits(X0, X1, k_values=MI_K_VALUES):
    """Ross (2014, PLOS ONE 9(2):e87357) k-NN mutual information between a
    continuous multivariate representation and a binary label.

    NOTE ON PROVENANCE: an earlier version of this function used a formula
    transcribed from an AI-summarized PDF fetch, which turned out to be wrong
    (it failed the basic sanity check of MI(X, label) -> ~0 for two identical
    distributions, giving ~7 bits instead). This version instead matches
    scikit-learn's `_compute_mi_cd` (`sklearn/feature_selection/_mutual_info.py`,
    read directly from source, not summarized), which implements the same
    Ross 2014 estimator and is generalized here from sklearn's 1-D-only
    `c.reshape((-1,1))` case to the full multivariate representation:

        I(X;Y) = psi(N) + <psi(k)> - <psi(N_{y_i})>_i - <psi(m_i)>_i   (nats)

    `m_i` is the count of POOLED-data points (including the point itself)
    within a radius `nextafter(r_k, 0)` of point i, where `r_k` is the exact
    distance to its k-th nearest neighbor WITHIN its own class -- the
    infinitesimal downward shift excludes ties at that boundary, which is
    what makes the same-class contribution to `m_i` come out to exactly `k`
    (self + the k-1 strictly-closer same-class neighbors), matching sklearn's
    tie-breaking exactly rather than reinventing it. Euclidean distance
    throughout (sklearn's default for both searches). Per-k estimates are
    clipped to >=0 (sklearn's own convention: a negative MI estimate means
    "close to 0", not a real negative MI). High-D KSG-style MI is biased (see
    docstring caveat), so this sweeps k and returns (mean_bits, std_bits).

    Implementation note: this computes an exact GPU brute-force N x N
    pairwise-distance matrix (same pattern as `_rbf_kernel_matrix` above)
    rather than a cKDTree. `phi`/`membrane_*` representations are
    700-2112-D; a k-d tree's partitioning gives no speedup at that
    dimensionality (curse of dimensionality) and was ~90s/run in practice.
    N here is bounded by the MMD subsample caps (<=~1000/side), so the
    O(N^2) brute force is both exact and fast."""
    X = np.concatenate([X0, X1], axis=0)
    y = np.concatenate([np.zeros(len(X0)), np.ones(len(X1))])
    N = len(X)
    N_y = {0: len(X0), 1: len(X1)}
    if min(N_y.values()) < 2:
        return float("nan"), float("nan")

    device = device_for("Ross MI pairwise distances", verbose=False)
    # float64 Gram: the expansion ||a||^2 + ||b||^2 - 2a.b suffers catastrophic
    # cancellation in float32 when the norms are large relative to the true
    # separation -- phi's norms are O(1e3) while nearby frames differ by O(1e-2).
    # Symptom, observed: D[i,i] came out slightly POSITIVE instead of 0, so a
    # point could fall outside its own k-NN radius, giving m_i = 0 and
    # digamma(0) = -inf, i.e. a reported MI of +inf. The explicit diagonal zero
    # below makes the self-distance exact regardless.
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64)).to(device)
    sq = (Xt * Xt).sum(1)
    D = (sq[:, None] + sq[None, :] - 2 * (Xt @ Xt.T)).clamp(min=0).sqrt()  # exact N x N Euclidean
    D.fill_diagonal_(0.0)   # self-distance is 0 by definition, not by arithmetic

    idx = {0: np.where(y == 0)[0], 1: np.where(y == 1)[0]}
    label_counts = np.where(y == 0, N_y[0], N_y[1])

    estimates = []
    for k in k_values:
        k_eff = min(k, min(N_y.values()) - 1)
        if k_eff < 1:
            continue
        radius = np.empty(N, dtype=np.float64)
        for cls in (0, 1):
            ci = torch.from_numpy(idx[cls]).to(device)
            sub = D.index_select(0, ci).index_select(1, ci).clone()
            sub.fill_diagonal_(float("inf"))  # exclude self from the k-NN search
            kth = torch.kthvalue(sub, k_eff, dim=1).values.cpu().numpy()
            radius[idx[cls]] = np.nextafter(kth, 0)
        radius_t = torch.from_numpy(radius).to(device)
        m = (D <= radius_t[:, None]).sum(dim=1).double().cpu().numpy()  # includes self, matches sklearn
        # m counts the point itself, so m >= 1 always holds mathematically. The
        # floor is a belt-and-braces guard: digamma(0) = -inf would silently
        # turn one bad row into a reported MI of +inf (and a NaN Fano floor).
        m = np.maximum(m, 1.0)
        mi_nats = (digamma(N) + digamma(k_eff)
                   - np.mean(digamma(label_counts)) - np.mean(digamma(m)))
        estimates.append(max(0.0, float(mi_nats)) / np.log(2))  # nats -> bits
    if not estimates:
        return float("nan"), float("nan")
    return float(np.mean(estimates)), float(np.std(estimates))


def fano_error_floor(mi_bits, p1):
    """Fano's inequality, binary case: H_b(P_e) >= H(Y) - I(X;Y), solved for
    the minimum error probability ANY detector on this representation could
    achieve (I3) -- a representation-intrinsic ceiling, not specific to
    Mahalanobis or any other density model. H(Y) uses the ACTUAL class
    balance p1 = P(corrupt) in the scored sample, not an assumed 0.5. Returns
    0.0 if the bound is non-binding (I >= H(Y): perfect detection is not
    ruled out) and NaN if the bound is degenerate (I well below 0, i.e. MI
    estimation noise swamped the signal -- not evidence of anything)."""
    if not np.isfinite(mi_bits) or p1 <= 0 or p1 >= 1:
        return float("nan")
    def h_b(p):
        if p <= 0 or p >= 1:
            return 0.0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    target = h_b(p1) - mi_bits
    if target <= 0:
        return 0.0
    if target > 1.0:   # only reachable via a negative MI estimate (noise, not signal)
        return float("nan")
    return float(brentq(lambda p: h_b(p) - target, 1e-12, 0.5))


def mi_fano_test(clean_X, corr_feat, corr_ho_lens, rng, k_values=MI_K_VALUES):
    """Wraps ross_mi_bits/fano_error_floor for one (representation, run) cell,
    reusing the SAME fixed clean subsample as the MMD test (`clean_X`, from
    `_mmd_subsample` in main()) and drawing an independent corrupt-side
    subsample the same way. Returns (mi_bits, mi_std_bits, fano_error_floor)."""
    corr_X, _ = _mmd_subsample(corr_feat, corr_ho_lens, rng)
    if clean_X is None or corr_X is None or len(clean_X) < 5 or len(corr_X) < 5:
        return float("nan"), float("nan"), float("nan")
    mi_bits, mi_std = ross_mi_bits(clean_X, corr_X, k_values)
    p1 = len(corr_X) / (len(clean_X) + len(corr_X))
    return mi_bits, mi_std, fano_error_floor(mi_bits, p1)


# ── C2ST: the paper's own TV ceiling, measured instead of asserted ───────────
C2ST_FOLDS = 5


def _groups_from_sizes(sizes):
    """Recording id per row, from `_mmd_subsample`'s per-recording row counts."""
    if not sizes:
        return np.zeros(0, dtype=int)
    return np.repeat(np.arange(len(sizes)), sizes)


def c2st_tv(clean_X, corr_X, clean_groups, corr_groups, rng):
    """Classifier two-sample test -> a LOWER bound on TV(P, Q), the quantity the
    paper's Proposition `prop:tv-ceiling` bounds AUROC by and then explicitly
    declines to estimate ("We do not estimate delta = TV(P,Q) directly").

    TWO SEPARATE SOURCES, kept straight because they are often conflated:

      * the PROCEDURE is C2ST (Lopez-Paz & Oquab, ICLR 2017): label sample P as
        0 and Q as 1, train a binary classifier, and use held-out accuracy as
        the test statistic.
      * the TV IDENTITY is Le Cam's two-point lemma, NOT from that paper --
        verified, C2ST never mentions total variation. For equal priors the
        minimum average error of ANY test between P and Q is (1 - TV(P,Q))/2,
        so the Bayes-optimal balanced accuracy is acc* = (1 + TV)/2, i.e.

            TV(P, Q) = 2 * acc* - 1.

    A trained classifier's held-out balanced accuracy under-estimates acc*
    (logistic regression is not the Bayes rule), so `TV_hat = max(0, 2*acc - 1)`
    is a LOWER bound on TV. That is the useful direction: it certifies how much
    separability actually exists in the representation, independently of
    Mahalanobis or any other fitted density model. It can never prove a
    corruption is undetectable -- only that it is at least this detectable.

    THREE DEVIATIONS FROM C2ST AS PUBLISHED, all deliberate:

      1. GROUPED folds instead of the paper's random shuffle. C2ST assumes
         i.i.d. samples; our frames are strongly correlated within a recording,
         so a random shuffle puts near-duplicate frames on both sides and
         drives accuracy toward 1 for EVERY representation. Splitting by
         recording is the same leakage guard the rest of this repo applies. Not
         optional -- a random split here measures memorisation, not separation.
      2. BALANCED accuracy instead of plain accuracy. Le Cam's identity assumes
         equal priors; a grouped split does not guarantee balanced fold counts,
         and plain accuracy would then silently absorb the class imbalance.
      3. K-FOLD CV instead of the paper's single train/test split, averaged over
         folds. Reduces the variance of a statistic we report as a point
         estimate rather than testing. CONSEQUENCE: the paper's null
         distribution N(1/2, 1/(4*n_te)) does NOT apply here -- it assumes one
         split of i.i.d. samples, and both assumptions are broken above. No
         p-value is emitted for this column; the MMD test in this same file is
         the file's recording-level significance test.

    WHY THIS IS NOT THE SAME AS THE AUROC COLUMN. The AUROC elsewhere in this
    file comes from an unsupervised, corruption-blind detector fitted on clean
    only -- the deployable setting. This is the opposite: a SUPERVISED probe
    that sees both classes. The gap between them is exactly the
    "information-limited vs detector-limited" question. A corruption with
    TV_hat ~ 0 is invisible in this representation to any detector (supporting
    the paper's invariance claim for polarity_flip); one with large TV_hat but
    chance unsupervised AUROC is a DETECTOR failure, not an information limit.
    Reported as `c2st_tv_lower`, never as an OOD-detection result.

    GROUPING. Folds are split by RECORDING, never by frame: neighbouring frames
    within a sequence are near-duplicates, and a frame-level split lets the
    probe memorize the recording instead of learning the corruption, inflating
    accuracy towards 1 for every representation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    if clean_X is None or corr_X is None or len(clean_X) < 20 or len(corr_X) < 20:
        return float("nan"), float("nan")
    X = np.concatenate([clean_X, corr_X]).astype(np.float64)
    y = np.concatenate([np.zeros(len(clean_X)), np.ones(len(corr_X))]).astype(int)
    # Offset the corrupt-side group ids so a recording index shared between the
    # two sides cannot merge into one group.
    g = np.concatenate([np.asarray(clean_groups),
                        np.asarray(corr_groups) + int(np.max(clean_groups)) + 1])

    n_folds = int(min(C2ST_FOLDS, len(np.unique(clean_groups)),
                      len(np.unique(corr_groups))))
    if n_folds < 2:
        return float("nan"), float("nan")

    accs = []
    for tr, te in StratifiedGroupKFold(n_splits=n_folds).split(X, y, groups=g):
        if len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        # BALANCED accuracy: Le Cam's identity assumes equal priors, and the
        # fold's class counts are not guaranteed balanced after a group split.
        per_class = [np.mean(pred[y[te] == c] == c) for c in (0, 1)
                     if np.any(y[te] == c)]
        accs.append(float(np.mean(per_class)))
    if not accs:
        return float("nan"), float("nan")
    acc = float(np.mean(accs))
    return max(0.0, 2 * acc - 1), acc


def bh_fdr(pvals):
    """Benjamini-Hochberg (1995) step-up FDR q-values: q_(i) =
    min_{j>=i} p_(j)*m/j, m = number of non-NaN tests (S1). NaN p-values pass
    through as NaN and are excluded from m and the ranking."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    m = int(valid.sum())
    if m == 0:
        return q
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order] * m / (np.arange(m) + 1)
    q_sorted = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0, 1)
    q[order] = q_sorted
    return q


TOST_MARGIN = 0.02  # AUROC units; see docstring item 8


def twonn_intrinsic_dim(X, cap=5000, seed=0):
    """TwoNN intrinsic-dimension estimator (Facco et al., Sci. Rep. 7:12140,
    2017; M4), verified against the paper: uses only the 1st/2nd nearest-
    neighbor distances per point. mu_i = r2_i/r1_i ~ Pareto(I_d); I_d is the
    zero-intercept least-squares fit of -log(1-F_emp(mu)) on log(mu)."""
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    if len(X) < 10:
        return float("nan")
    d, _ = cKDTree(X).query(X, k=3)  # self, 1st NN, 2nd NN
    r1, r2 = d[:, 1], d[:, 2]
    keep = r1 > 0
    mu = np.sort(r2[keep] / r1[keep])
    n = len(mu)
    if n < 10:
        return float("nan")
    F_emp = (np.arange(1, n + 1) - 0.5) / n
    x, y = np.log(mu), -np.log(1 - F_emp)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    return float((x * y).sum() / (x * x).sum())


def fit_mahalanobis(train_feat):
    """Fit mean + precision once and return a score(test_feat) closure.

    The covariance fit/inversion is O(d^3); callers that score many runs
    against the same train split must fit once and reuse the closure. Scoring
    runs chunked on the GPU (the closure captures only mu/P, not train_feat)."""
    try:
        cov = empirical_precision(train_feat, op="repr-ablation-rigor Mahalanobis fit")
        mu = cov.location_
        P = cov.precision_
    except Exception as e:
        print(f"Warning: Covariance fit failed ({e}). Using simple L2.")
        mu = train_feat.mean(axis=0)
        P = np.eye(len(mu))

    device = device_for("repr-ablation-rigor Mahalanobis scoring", verbose=False)
    mu_t = torch.from_numpy(np.ascontiguousarray(mu, dtype=np.float32)).to(device)
    P_t = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float32)).to(device)

    def score(test_feat):
        def fn(chunk):
            diff = chunk - mu_t
            return ((diff @ P_t) * diff).sum(dim=1)
        return chunked_apply(fn, np.ascontiguousarray(test_feat, dtype=np.float32),
                             device, n_ref=P_t.shape[0])
    return score


class LazyFeatures(Mapping):
    """Lazy, memory-bounded {phi, ann, spike, fused} loader (see
    representation_ablation.LazyFeatures -- same design, duplicated here so
    this script has no import-order coupling to the canonical pipeline stage).
    phi_spatial is NOT preloaded (~2 GB/run); fetched on demand per rep/run
    via `extract_representation`."""

    def __init__(self, cache_size: int = 2, verbose: bool = True):
        self._phi_files = {f.stem: f for f in cfg.PHI_DIR.glob("*.pt")
                           if f.stem in VALID_RUNS}
        self._ann_files = ({f.stem: f for f in cfg.ANN_DIR.glob("*.pt")}
                           if cfg.ANN_DIR.exists() else {})
        self._spike_files = ({f.stem: f for f in cfg.SPIKE_DIR.glob("*.pt")}
                             if cfg.SPIKE_DIR.exists() else {})
        fused_dir = cfg.OUTPUT_DIR / "features/fused"
        self._fused_files = ({f.stem: f for f in fused_dir.glob("*.pt")}
                             if fused_dir.exists() else {})
        self._cache_size = max(1, cache_size)
        self._verbose = verbose
        self._cache = OrderedDict()
        self._clean = None

        runs = set(self._phi_files)
        for label, files in (("ANN", self._ann_files), ("spike", self._spike_files)):
            if files:
                missing = sorted(runs - set(files))
                if missing:
                    print(f"Warning: {len(missing)} run(s) have phi but no {label} "
                          f"features ({', '.join(missing[:5])}"
                          f"{', ...' if len(missing) > 5 else ''}); "
                          f"{label}-based representations will skip them.")

    def _build(self, run):
        if self._verbose:
            print(f"  [load] {run}", flush=True)
        feats = {
            'phi': materialize_f32(load_pt(self._phi_files[run])['phi']),
            'ann': {}, 'spike': {},
        }
        if run in self._ann_files:
            feats['ann'] = {k: v.numpy()
                            for k, v in load_pt(self._ann_files[run]).items()
                            if isinstance(v, torch.Tensor)}
        if run in self._spike_files:
            d = load_pt(self._spike_files[run])
            sp = {k: v.numpy() for k, v in d.items() if isinstance(v, torch.Tensor)}
            if "spike_entropy" in sp:
                sp["spike_entropy"] = np.nan_to_num(sp["spike_entropy"], nan=0.0)
            feats['spike'] = sp
        if run in self._fused_files:
            d = load_pt(self._fused_files[run])
            feats['fused'] = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                              for k, v in d.items() if v is not None}
        return feats

    def __getitem__(self, run):
        if run not in self._phi_files:
            raise KeyError(run)
        if run == 'clean':
            if self._clean is None:
                self._clean = self._build('clean')
            return self._clean
        if run in self._cache:
            self._cache.move_to_end(run)
            return self._cache[run]
        feats = self._build(run)
        self._cache[run] = feats
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return feats

    def __iter__(self):
        return iter(self._phi_files)

    def __len__(self):
        return len(self._phi_files)

    def __contains__(self, run):
        return run in self._phi_files


def load_all_features(cache_size: int = 2, verbose: bool = True):
    return LazyFeatures(cache_size=cache_size, verbose=verbose)


def extract_representation(feats, rep_name, run_name=None):
    """Given a dict of {phi, ann, spike} features for a run, return the
    specific representation as a 2D numpy array."""
    if rep_name == "full_membrane":
        return feats['phi']
    elif rep_name == "phi_spatial":
        # Not preloaded into `feats` (it's ~2 GB/run and every other
        # representation here never touches it) -- loaded on demand straight
        # from disk, same access pattern as LazyPhiDict.get_phi_spatial.
        return load_phi_spatial(run_name) if run_name else None
    elif rep_name == "membrane_mean":
        return slice_phi_stat(feats['phi'], 'mu')
    elif rep_name == "membrane_var":
        return slice_phi_stat(feats['phi'], 'var')
    elif rep_name == "membrane_kurtosis":
        return slice_phi_stat(feats['phi'], 'kurtosis')
    elif rep_name == "ANN":
        return feats['ann'].get('last_ann_gap', feats['ann'].get('asab_gap'))
    elif rep_name == "logits":
        return feats['ann'].get('head_cls_L0_gap')
    elif rep_name == "spike":
        return feats.get('spike', {}).get('spike_rate')
    elif rep_name == "spike_entropy":
        return feats.get('spike', {}).get('spike_entropy')
    elif rep_name == "membrane_fused" and 'fused' in feats:
        return feats['fused'].get('membrane_fused')

    return None


# Column mean/std for `phi_plus_spatial`, fitted ONCE on the clean fit pool.
# It MUST NOT be re-derived per run: standardizing a corrupted run by its own
# statistics subtracts out exactly the distribution shift the detector is
# supposed to see. (Observed directly: per-run scaling drove phi_plus_spatial to
# AUROC 0.478 / C2ST-TV 0.15 on spatial_dropout while phi_spatial alone reached
# 0.597 / 0.99 -- the concat was destroying its own signal.)
_PAIR_STATS = {}


def extract_pool(feats, rep_name, run_name, rows, fit_stats=False):
    """One representation, restricted to the row range `rows` (a slice).

    Exists because `phi_plus_spatial` cannot be built full-length: phi (2.9 GB)
    + phi_spatial (1.9 GB) + concat (4.8 GB) + standardized copy (4.8 GB) is
    ~14 GB resident for a single representation. Slicing each half FIRST -- and
    faulting only those pages out of the mmap-backed spatial file -- keeps the
    concat to the size of one pool.

    `fit_stats=True` (used once, on the clean fit pool) records the z-scoring
    statistics that every later call reuses.
    """
    if rep_name == "phi_plus_spatial":
        # The union the shipped MDD actually sees. Its delta over each half
        # answers "are the moment and dispersion read-outs complementary, or is
        # one subsumed by the other?" -- which no table has ever reported.
        # The two halves are z-scored because raw phi and raw phi_spatial differ
        # in magnitude by orders of magnitude, so an un-normalised concat would
        # let whichever block is larger dominate any Euclidean/Mahalanobis fit
        # for a UNITS reason rather than an information one. The transform is
        # fitted on CLEAN FIT ONLY and then held fixed -- see _PAIR_STATS.
        sp = load_phi_spatial(run_name, rows=rows) if run_name else None
        if sp is None:
            return None
        both = np.concatenate([feats['phi'][rows], sp], axis=1)
        del sp
        if fit_stats or not _PAIR_STATS:
            _PAIR_STATS["mu"] = both.mean(0)
            _PAIR_STATS["sd"] = both.std(0) + 1e-8
        both -= _PAIR_STATS["mu"]
        both /= _PAIR_STATS["sd"]
        return both
    if rep_name == "phi_spatial":
        return load_phi_spatial(run_name, rows=rows) if run_name else None
    full = extract_representation(feats, rep_name, run_name)
    return None if full is None else full[rows]


def main():
    print("Running representation ablation (rigor pass: CI + paired tests + JS + MMD)...")
    all_feats = load_all_features()
    if 'clean' not in all_feats:
        print("Error: 'clean' run not found. Run extract.py first.")
        return

    reps = [
        "logits", "ANN", "spike", "spike_entropy",
        "membrane_mean", "membrane_var", "membrane_kurtosis", "full_membrane",
        "phi_spatial", "phi_plus_spatial", "membrane_fused"
    ]

    results = []
    pair_results = []
    rng = np.random.default_rng(SEED)
    clean_seq_lens = load_phi_seq_lens("clean")

    print(f"Fitting {len(reps)} representation scorers on clean...")
    fitted = {}   # rep -> (scorer, clean_scores, clean_recs, train_width, mmd_clean_X, mmd_clean_sizes)
    # Canonical 4-way pools (sequence-aligned): fit on `fit` (50%), score the
    # `final` pool (30%) as clean negatives. Identical boundaries to every MDD
    # table, so the two are directly comparable. Row ranges are derived once,
    # from phi's length, and every representation is sliced to them at load
    # time -- see extract_pool for why full-length is not an option.
    clean_n = len(all_feats['clean']['phi'])
    _cr = pool_ranges(clean_n, clean_seq_lens)
    clean_fit_sl, clean_fin_sl = slice(*_cr["fit"]), slice(*_cr["final"])

    for rep in reps:
        train_feat_fit = extract_pool(all_feats['clean'], rep, 'clean', clean_fit_sl,
                                      fit_stats=True)
        clean_test_feat = extract_pool(all_feats['clean'], rep, 'clean', clean_fin_sl)
        if train_feat_fit is None or clean_test_feat is None:
            print(f"Skipping {rep} (not found)")
            continue
        scorer = fit_mahalanobis(train_feat_fit)
        clean_scores = scorer(clean_test_feat)
        clean_ho_lens = final_lens(clean_n, clean_seq_lens)
        clean_recs = group_by_seq(clean_scores, clean_ho_lens)
        # Fixed MMD reference subsample of the RAW clean vectors (not scores):
        # reused against every corruption run so it's computed once per rep.
        mmd_clean_X, mmd_clean_sizes = _mmd_subsample(clean_test_feat, clean_ho_lens, rng)
        fitted[rep] = (scorer, clean_scores, clean_recs, train_feat_fit.shape[1],
                       mmd_clean_X, mmd_clean_sizes)

    # M4: intrinsic dimensionality is a property of the clean manifold, not of
    # any one corruption run -- computed once per representation, reusing the
    # same fixed clean subsample already drawn for MMD.
    dim_rows = [{"representation": rep, "ambient_dim": w,
                 "intrinsic_dim_twonn": twonn_intrinsic_dim(mmd_clean_X)}
                for rep, (_, _, _, w, mmd_clean_X, _) in fitted.items()
                if mmd_clean_X is not None]
    if dim_rows:
        dim_dest = UNIFIED_DIR / "repr_intrinsic_dim.csv"
        UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(dim_rows).to_csv(dim_dest, index=False)
        print(f"  {dim_dest} ({len(dim_rows)} rows): TwoNN intrinsic dimension per representation")

    run_names = sorted(name for name in all_feats if name != 'clean')
    for run_name in tqdm(run_names, desc="Representation Ablation (rigor)"):
        feats = all_feats[run_name]
        parts = run_name.rsplit('_L', 1)
        corruption = parts[0]
        severity = int(parts[1]) if len(parts) > 1 else 0
        run_seq_lens = load_phi_seq_lens(run_name)
        # Positives = the run's `final` pool only, cut at the SAME
        # sequence-aligned boundary as the clean negatives, so no frame whose
        # clean twin was fitted on can ever appear as a positive.
        run_n = len(feats['phi'])
        run_fin_sl = slice(*pool_ranges(run_n, run_seq_lens)["final"])

        per_rep_recs = {}    # rep -> (clean_recs, corr_recs), reused by the pairwise tests below
        per_rep_auroc = {}
        for rep, (scorer, clean_scores, clean_recs, train_width,
                  mmd_clean_X, mmd_clean_sizes) in fitted.items():
            test_feat_ho = extract_pool(feats, rep, run_name, run_fin_sl)
            if test_feat_ho is None:
                continue
            if test_feat_ho.shape[1] != train_width:
                continue   # representation width differs for this run — skip

            ho_lens = final_lens(run_n, run_seq_lens) if run_seq_lens else None
            corr_scores = scorer(test_feat_ho)

            metrics = auroc_aupr_fpr95(clean_scores, corr_scores)
            if metrics is None:
                continue
            auroc, aupr, fpr95 = metrics

            corr_recs = group_by_seq(corr_scores, ho_lens)
            ci_lo, ci_hi = cluster_bootstrap_auroc_ci(clean_recs, corr_recs, rng)
            js = js_divergence_bits(clean_scores, corr_scores)
            # Detector-free comparison of the RAW vectors (no Mahalanobis fit
            # involved at all): MMD^2 effect size + recording-permutation p-value.
            mmd2, mmd_p, mmd_n_clean, mmd_n_corr = mmd_test(
                mmd_clean_X, mmd_clean_sizes, test_feat_ho, ho_lens, rng)
            # Second detector-free lens on the same raw vectors: mutual
            # information (Ross 2014) + the Fano error-probability floor it
            # implies (I1/I3).
            mi_bits, mi_std, fano_floor = mi_fano_test(
                mmd_clean_X, test_feat_ho, ho_lens, rng)
            # Third: the supervised decodability ceiling. TV_hat = 2*acc-1
            # instantiates the paper's own Proposition prop:tv-ceiling, which
            # currently states the bound and declines to measure it.
            corr_X_c, corr_sizes_c = _mmd_subsample(test_feat_ho, ho_lens, rng)
            tv_lo, c2st_acc = c2st_tv(
                mmd_clean_X, corr_X_c,
                _groups_from_sizes(mmd_clean_sizes),
                _groups_from_sizes(corr_sizes_c), rng)
            per_rep_recs[rep] = (clean_recs, corr_recs)
            per_rep_auroc[rep] = auroc

            results.append({
                "model": "hybrid",
                "representation": rep,
                "detector": "mahalanobis",
                "corruption": corruption,
                "severity": severity,
                "auroc": auroc,
                "auroc_ci_lo": ci_lo,
                "auroc_ci_hi": ci_hi,
                "aupr": aupr,
                "fpr95": fpr95,
                "js_divergence_bits": js,
                "n_clean_recordings": len(clean_recs),
                "n_corrupt_recordings": len(corr_recs),
                "mmd2": mmd2,
                "mmd_p_value": mmd_p,
                "mmd_n_clean": mmd_n_clean,
                "mmd_n_corrupt": mmd_n_corr,
                "cliffs_delta": 2 * auroc - 1,
                "mi_bits": mi_bits,
                "mi_std_bits": mi_std,
                "fano_error_floor": fano_floor,
                # Supervised probe -- a decodability CEILING, never an OOD result.
                "c2st_tv_lower": tv_lo,
                "c2st_balanced_acc": c2st_acc,
                # The gap that separates "no information" from "wrong detector".
                "c2st_minus_auroc": (tv_lo / 2 + 0.5) - auroc
                                    if np.isfinite(tv_lo) else float("nan"),
            })

        # Paired significance tests: is phi/phi_spatial's AUROC significantly
        # different from the ANN/spike/logit baselines' (and from each other)
        # ON THIS RUN, controlling for shared recording-level variance.
        for rep_a, rep_b in PAIRS:
            if rep_a not in per_rep_recs or rep_b not in per_rep_recs:
                continue
            clean_a, corr_a = per_rep_recs[rep_a]
            clean_b, corr_b = per_rep_recs[rep_b]
            d_lo, d_hi, p = paired_cluster_bootstrap(clean_a, corr_a, clean_b, corr_b, rng)
            pair_results.append({
                "corruption": corruption,
                "severity": severity,
                "rep_a": rep_a,
                "rep_b": rep_b,
                "auroc_a": per_rep_auroc[rep_a],
                "auroc_b": per_rep_auroc[rep_b],
                "delta_auroc": per_rep_auroc[rep_a] - per_rep_auroc[rep_b],
                "delta_ci_lo": d_lo,
                "delta_ci_hi": d_hi,
                "p_value": p,
            })

    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    metrics_dest = UNIFIED_DIR / "repr_ablation_metrics.csv"
    df.to_csv(metrics_dest, index=False)

    pair_df = pd.DataFrame(pair_results)
    if len(pair_df):
        # S1: BH-FDR q-values over the whole pairwise-test grid (not per
        # corruption/severity subset -- the doc's 240-cell multiple-comparison
        # problem is about the grid as a whole).
        pair_df["p_value_bh_q"] = bh_fdr(pair_df["p_value"].values)
        # S3: TOST-style equivalence read off the already-computed 95% CI (see
        # docstring item 8 for the alpha caveat).
        pair_df[f"tost_equivalent_{TOST_MARGIN}"] = (
            (pair_df["delta_ci_lo"] > -TOST_MARGIN) & (pair_df["delta_ci_hi"] < TOST_MARGIN)
        )
    pairs_dest = UNIFIED_DIR / "repr_ablation_pairwise_tests.csv"
    pair_df.to_csv(pairs_dest, index=False)

    print(f"Representation ablation (rigor pass) complete.\n"
          f"  {metrics_dest} ({len(df)} rows): auroc/ci/aupr/fpr95/js_divergence_bits/"
          f"mmd2/mmd_p_value/cliffs_delta/mi_bits/fano_error_floor\n"
          f"  {pairs_dest} ({len(pair_df)} rows): paired AUROC significance "
          f"+ BH-FDR q-values + TOST equivalence, phi/phi_spatial vs baselines")


if __name__ == "__main__":
    main()
