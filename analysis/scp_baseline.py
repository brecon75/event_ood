"""SCP baseline — Martinez-Seras et al. 2023 (Information Fusion 100:101943;
arXiv 2210.00894), the only prior SNN-OOD method portable to our host.

Faithful to Algorithm 1 / Eqs. (2)-(4) and the configuration in their Sec. 4:

  1. Sample P=1000 in-distribution instances for characterization.
  2. Collect spike counts at the LAST layer:  q(n) = sum_t s_{n,t}   (Eq. 2)
  3. Pairwise L1 / Manhattan distance                                (Eq. 3)
  4. Agglomerative hierarchical clustering (class-conditional)
  5. Cluster archetype = MEDIAN (their f_agg, chosen for outlier robustness)
  6. Score = min_m L1(q_x, archetype_m) over the predicted class      (Eq. 4)
  7. Class-conditional threshold lambda calibrated on a SEPARATE 1000-sample
     subset. AUROC is threshold-free, so step 7 only matters for FPR95.

Two deviations from the published method, forced by the host and recorded in
`Docs/ablation_attack_plan.md` (they belong in the paper, not just here):

  D1  class conditioning.  Their host is a classifier: one predicted yhat per
      sample, archetypes and thresholds both conditioned on it. Ours is YOLOX
      (0..N boxes over 2 classes, no per-frame label), so we run C=1 — a single
      unconditional archetype bank. Clustering, median archetypes and min-L1 are
      preserved exactly; only the conditioning drops. Defensible because their
      class assignment already uses model *predictions*, not ground truth.
  D2  per-neuron vs per-channel.  SCP counts per neuron at layer L; VmemMonitor
      GAPs over HxW, so we get per-*channel* counts (SNN Block 4, 256 ch).

NOTE their AUROC convention treats in-distribution as POSITIVE, inverted
relative to ours. We report in the repo's convention (corrupt = positive) so the
numbers sit alongside every other table; `--their-convention` flips it.

This module is signal-agnostic on purpose: it scores whatever matrix it is
given. That is what makes the 2x2 possible — SCP's *rule* on phi (runnable now)
vs on true spike counts (needs the ~11 GB last-layer spike extraction).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    LazyPhiDict, TRAIN_RATIO, split_boundary, seq_lens_after_cut,
    aggregate_by_seq, auroc_aupr_fpr95, slice_phi_layer, _get_present,
)

P_CHARACTERIZE = 1000   # their |D^c_tr|
N_CLUSTERS = 8          # M^c; their paper lets the clustering decide -- swept via --clusters


class SCP:
    """Spike Count Pattern detector (Martinez-Seras et al. 2023), C=1 variant.

    D3 -- standardization. Published SCP takes raw spike counts, which are
    already homogeneous units (all neurons, same [0,T] range), so L1 needs no
    rescaling. phi's three moment blocks (mu, var, kurtosis) are NOT
    homogeneous -- on layer 3 kurtosis alone carries ~75% of the raw L1 mass
    (mu ~17%, var ~8%), so unstandardized L1/clustering on phi is nearly blind
    to mean/variance shifts for a scale reason, not an algorithmic one. MDD
    z-scores every phi dimension before touching it (mdd.py:229-232); SCP must
    do the same for the rule-ablation cell to isolate the *detector*, not a
    unit mismatch. standardize=True (default) applies this for --signal phi;
    pass False to reproduce the literal (unstandardized) published rule.
    """

    def __init__(self, n_clusters: int = N_CLUSTERS, p: int = P_CHARACTERIZE, seed: int = 42,
                 standardize: bool = True):
        self.n_clusters = n_clusters
        self.p = p
        self.seed = seed
        self.standardize = standardize
        self.archetypes = None
        self.mu_ = None
        self.sd_ = None

    def fit(self, counts: np.ndarray):
        """counts: (N, D) in-distribution count vectors. Steps 1-5."""
        if self.standardize:
            self.mu_ = counts.mean(axis=0)
            self.sd_ = counts.std(axis=0) + 1e-8
            counts = (counts - self.mu_) / self.sd_

        rng = np.random.default_rng(self.seed)
        n = min(self.p, len(counts))
        sub = counts[np.sort(rng.choice(len(counts), n, replace=False))]

        # Eq. 3: L1. AgglomerativeClustering with a precomputed L1 metric and
        # average linkage -- 'ward' is Euclidean-only and would silently
        # contradict their stated distance.
        m = min(self.n_clusters, len(sub))
        labels = AgglomerativeClustering(
            n_clusters=m, metric="manhattan", linkage="average"
        ).fit_predict(sub)

        # Step 5: archetype = median of each cluster (their f_agg).
        self.archetypes = np.stack(
            [np.median(sub[labels == c], axis=0) for c in range(m)]
        ).astype(np.float32)
        return self

    def score(self, counts: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """Eq. 4: min_m L1(q_x, archetype_m). Higher = more OOD."""
        if self.standardize:
            counts = (counts - self.mu_) / self.sd_
        out = np.empty(len(counts), dtype=np.float32)
        for i in range(0, len(counts), chunk):
            block = counts[i:i + chunk].astype(np.float32)
            out[i:i + chunk] = cdist(block, self.archetypes, metric="cityblock").min(axis=1)
        return out


def _counts_from_phi(phi: np.ndarray, layer: int) -> np.ndarray:
    """SCP's input slot, filled with phi's last-layer block.

    This is the *rule* ablation cell of the 2x2, NOT 'SCP as published': phi is
    sub-threshold membrane, not spike counts. Never present a phi-fed SCP as the
    Martinez-Seras baseline -- and in particular never synthesize counts from phi
    analytically (r = Fbar((theta-mu)/sigma) makes them a deterministic function
    of phi, so the comparison is rigged by the paper's own DPI corollary).
    """
    return slice_phi_layer(phi, layer)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signal", choices=["phi", "spike"], default="phi",
                    help="phi = the rule-ablation cell (runs now); spike = SCP as "
                         "published (needs the last-layer spike extraction).")
    ap.add_argument("--layer", type=int, default=3,
                    help="PLIF block index for the 'last layer L' (default 3 = SNN Block 4).")
    ap.add_argument("--clusters", type=int, default=N_CLUSTERS)
    ap.add_argument("--no-standardize", action="store_true",
                    help="Skip the D3 z-score standardization and use raw L1 (literal "
                         "published rule). Only meaningful for --signal phi -- phi's "
                         "moments are not homogeneous units, so this reproduces the "
                         "kurtosis-dominated confound the standardization fixes.")
    ap.add_argument("--their-convention", action="store_true",
                    help="Report AUROC with in-distribution as positive (their paper's "
                         "convention) instead of the repo's corrupt-as-positive.")
    ap.add_argument("--output", type=Path,
                    default=cfg.OUTPUT_DIR / "results" / "scp_baseline.csv")
    args = ap.parse_args()

    if args.signal == "spike":
        print("SCP-as-published needs last-layer spike counts; outputs/spike/ is empty.\n"
              "Re-extract with spikes ON (~11 GB, last layer only), then rerun with "
              "--signal spike.")
        return

    all_phi = LazyPhiDict(cache_size=1)
    if "clean" not in all_phi:
        print(f"Error: clean.pt missing from {cfg.PHI_DIR}.")
        return

    clean = all_phi["clean"]
    clean_seq_lens = all_phi.get_seq_lens("clean")
    cut = split_boundary(len(clean), TRAIN_RATIO, clean_seq_lens)

    fit_counts = _counts_from_phi(clean[:cut], args.layer)
    eval_counts = _counts_from_phi(clean[cut:], args.layer)
    eval_seq_lens = seq_lens_after_cut(clean_seq_lens, cut)

    standardize = not args.no_standardize
    print(f"SCP (C=1, M={args.clusters}, P={P_CHARACTERIZE}, L1/median, "
          f"standardize={standardize}) on {args.signal}, layer {args.layer}: "
          f"fit={len(fit_counts)} eval={len(eval_counts)} dim={fit_counts.shape[1]}")

    scp = SCP(n_clusters=args.clusters, standardize=standardize).fit(fit_counts)
    clean_scores = scp.score(eval_counts)
    del clean

    rows = []
    present = _get_present(all_phi)
    runs = [f"{c}_L{s}" for c in present for s in cfg.SEVERITIES if f"{c}_L{s}" in all_phi]

    for run in runs:
        phi_full = all_phi[run]
        run_seq_lens_full = all_phi.get_seq_lens(run)
        run_cut = split_boundary(len(phi_full), TRAIN_RATIO, run_seq_lens_full)
        corr = _counts_from_phi(np.ascontiguousarray(phi_full[run_cut:]), args.layer)
        all_phi._cache.pop(run, None)
        del phi_full

        corr_scores = scp.score(corr)
        run_seq_lens = seq_lens_after_cut(run_seq_lens_full, run_cut)
        corruption, sev = run.rsplit("_L", 1)

        for gran, cs, ts in (
            ("frame", clean_scores, corr_scores),
            ("sequence", aggregate_by_seq(clean_scores, eval_seq_lens),
             aggregate_by_seq(corr_scores, run_seq_lens)),
        ):
            if cs is None or ts is None:
                continue
            m = auroc_aupr_fpr95(cs, ts)
            if m is None:
                continue
            auroc, aupr, fpr95 = m
            if args.their_convention:
                auroc = 1.0 - auroc
            rows.append({"method": "SCP", "signal": args.signal, "corruption": corruption,
                         "severity": int(sev), "granularity": gran, "auroc": auroc,
                         "aupr": aupr, "fpr95": fpr95})
        print(f"  {run:28s} frame={rows[-2]['auroc']:.3f} seq={rows[-1]['auroc']:.3f}")

    import pandas as pd
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nWrote {args.output} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
