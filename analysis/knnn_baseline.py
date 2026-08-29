"""k-NNN baseline (Nizan & Tal, arXiv:2305.17695) on membrane phi, as a
geometry-aware contraction baseline requested by the reviewer (vs the RCF branch).

Faithful to the paper's Eq. (2):
    AS(f) = sum_{i=1}^{k}  sum_{s=1}^{S}  sum_{j=1}^{n}
            | (f_s - f_{i,s}) . v_{ij,s} | * 1/sqrt(e_{ij,s})
where for each NORMAL reference point f_i we precompute, per feature partition s
(dim L), the eigenvectors v_{ij,s} and eigenvalues e_{ij,s} of the covariance of
f_i's K_NN neighbours-of-neighbours; small eigenvalues (anomaly directions) get
up-weighted by 1/sqrt(e). Paper settings: k=3 test neighbours, 25 neighbours-of-
neighbours, partition dim L=5 with greedy correlation-based feature reordering.

Run unsupervised on clean phi (no labels). 50-sequence subset (like the other
diagnostics); per-sequence AUROC at L5. Compares k-NNN vs plain k-NN (k=3) and
recalls the RCF-branch number. CPU.
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"d:/Perdue")))
from vmem_benchmark import benchmark_config as cfg
from analysis.free_rider_ablation import _slice_first_seqs
from analysis.vmem_utils import (split_boundary, auroc_fpr95, aggregate_by_seq,
                                 seq_lens_after_cut, pool_ranges)

MAX_SEQ = 50
K_TEST = 3        # nearest reference neighbours of a test point
K_NN = 25         # neighbours-of-neighbours for each reference point's local cov
L = 5             # feature-partition dimensionality
N_REF = 2000      # reference subsample (normal set)
# Was a bespoke 0.85 cut; now the project's canonical 4-way pool split, so
# this baseline is scored on exactly the data MDD reports on.
TRAIN_RATIO = 0.85  # retained only for legacy call sites; unused below
SEED = 0
CORR = ["hot_pixel", "event_rate_shift", "event_flood",
        "temporal_jitter", "spatial_dropout", "polarity_flip"]
rng = np.random.default_rng(SEED)


def load_run(run):
    p = cfg.PHI_DIR / f"{run}.pt"
    if not p.exists():
        return None, None
    d = torch.load(p, map_location="cpu", weights_only=True)
    sliced, k = _slice_first_seqs(d["phi"].float(), d.get("seq_lens"), MAX_SEQ)
    sl = d.get("seq_lens")
    sl = [int(x) for x in sl[:k]] if sl is not None else None
    return sliced.numpy().astype(np.float32), sl


def knn_dist(query, ref, k):
    """Indices + distances of the k nearest ref rows for each query row (euclidean)."""
    # ||q-r||^2 = |q|^2 + |r|^2 - 2 q.r
    qn = (query ** 2).sum(1)[:, None]
    rn = (ref ** 2).sum(1)[None, :]
    d2 = qn + rn - 2.0 * query @ ref.T
    np.maximum(d2, 0, out=d2)
    idx = np.argpartition(d2, kth=min(k, ref.shape[0] - 1), axis=1)[:, :k]
    # sort the k by true distance
    rows = np.arange(len(query))[:, None]
    order = np.argsort(d2[rows, idx], axis=1)
    return idx[rows, order]


def greedy_corr_order(X):
    """Greedy correlation-based feature ordering: walk to the most-correlated
    unused feature so each length-L partition groups correlated dimensions."""
    C = np.abs(np.corrcoef(X.T))
    np.fill_diagonal(C, -1.0)
    C = np.nan_to_num(C, nan=-1.0)
    D = X.shape[1]
    used = np.zeros(D, bool)
    order = [0]; used[0] = True
    for _ in range(D - 1):
        last = order[-1]
        c = C[last].copy(); c[used] = -1.0
        nxt = int(np.argmax(c))
        order.append(nxt); used[nxt] = True
    return np.array(order)


def fit_reference(ref):
    """Per reference point, per partition: eigvecs/eigvals of its K_NN neighbourhood."""
    N, D = ref.shape
    parts = [np.arange(s, min(s + L, D)) for s in range(0, D, L)]
    nn = knn_dist(ref, ref, K_NN + 1)[:, 1:]          # drop self
    V_store, e_store = [], []                          # per partition: (N, L, L), (N, L)
    for cols in parts:
        sub = ref[:, cols]                             # (N, l)
        nbr = sub[nn]                                  # (N, K_NN, l)
        mu = nbr.mean(1, keepdims=True)
        c = nbr - mu
        cov = np.einsum("nki,nkj->nij", c, c) / (K_NN - 1)  # (N, l, l)
        cov += 1e-6 * np.eye(cov.shape[1])
        e, V = np.linalg.eigh(cov)                     # ascending eigenvalues
        V_store.append(V.astype(np.float32))
        e_store.append(np.maximum(e, 1e-8).astype(np.float32))
    return parts, nn, V_store, e_store


def knnn_score(test, ref, parts, V_store, e_store):
    """Eq. (2): anisotropic, eigenvalue-weighted distance to K_TEST ref neighbours."""
    nn = knn_dist(test, ref, K_TEST)                   # (M, K_TEST)
    score = np.zeros(len(test), np.float32)
    for cols, V, e in zip(parts, V_store, e_store):
        ts = test[:, cols]                             # (M, l)
        diff = ts[:, None, :] - ref[:, cols][nn]       # (M, K_TEST, l)
        Vn = V[nn]                                      # (M, K_TEST, l, l)
        en = e[nn]                                       # (M, K_TEST, l)
        proj = np.abs(np.einsum("mki,mkij->mkj", diff, Vn))  # |diff . v_j|
        score += (proj / np.sqrt(en)).sum((1, 2))
    return score


def plain_knn_score(test, ref):
    d = knn_dist(test, ref, K_TEST)
    qn = (test ** 2).sum(1)[:, None]; rn = (ref ** 2).sum(1)[None, :]
    d2 = qn + rn - 2.0 * test @ ref.T
    rows = np.arange(len(test))[:, None]
    return np.sqrt(np.maximum(d2[rows, d], 0)).mean(1)


def main():
    clean, csl = load_run("clean")
    cut = pool_ranges(len(clean), csl)["final"][0]
    fit, evalc = clean[:cut], clean[cut:]
    eval_sl = seq_lens_after_cut(csl, cut)

    mu = fit.mean(0); sd = fit.std(0) + 1e-9
    std = lambda a: (a - mu) / sd
    fit_s = std(fit)
    order = greedy_corr_order(fit_s)               # reorder so partitions group correlated dims
    fit_s = fit_s[:, order]
    if len(fit_s) > N_REF:
        ref = fit_s[rng.choice(len(fit_s), N_REF, replace=False)]
    else:
        ref = fit_s
    print(f"reference {ref.shape}, partitions of L={L} -> {int(np.ceil(ref.shape[1]/L))}; "
          f"k={K_TEST}, k_nn={K_NN}")
    parts, _, V_store, e_store = fit_reference(ref)

    evalc_s = std(evalc)[:, order]
    knnn_clean = knnn_score(evalc_s, ref, parts, V_store, e_store)
    knn_clean = plain_knn_score(evalc_s, ref)

    print(f"\n{'corruption':<17} | {'k-NNN':>7} | {'plain-kNN':>9}   (per-seq AUROC, L5, 50-seq)")
    print("-" * 52)
    res = {}
    for c in CORR:
        run = f"{c}_L5"
        phi, sl = load_run(run)
        if phi is None:
            continue
        rcut = pool_ranges(len(phi), sl)["final"][0]
        ph = std(phi[rcut:])[:, order]
        rsl = seq_lens_after_cut(sl, rcut)
        a_knnn = knnn_score(ph, ref, parts, V_store, e_store)
        a_knn = plain_knn_score(ph, ref)

        def au(clean_s, corr_s, csl_, rsl_):
            ca = aggregate_by_seq(clean_s, csl_); ta = aggregate_by_seq(corr_s, rsl_)
            y = np.r_[np.zeros(len(ca)), np.ones(len(ta))]
            return auroc_fpr95(y, np.r_[ca, ta])[0]

        r1 = au(knnn_clean, a_knnn, eval_sl, rsl)
        r2 = au(knn_clean, a_knn, eval_sl, rsl)
        res[c] = (r1, r2)
        print(f"{c:<17} | {r1:7.3f} | {r2:9.3f}")
    print("-" * 52)
    print(f"{'MEAN':<17} | {np.mean([v[0] for v in res.values()]):7.3f} | "
          f"{np.mean([v[1] for v in res.values()]):9.3f}")


if __name__ == "__main__":
    main()
