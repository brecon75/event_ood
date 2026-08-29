"""95% bootstrap confidence intervals for the plain-`max` combiner's AUROC,
L5, across the full window sweep (1/8/16/32/64/128/256/full).

Cluster bootstrap: per-frame scores within a recording are correlated, so
recordings (sequences) are resampled with replacement, not frames -- same
methodology as `bootstrap_ci.py`, reused here (`per_recording_windows`,
`boot_ci`) but rebuilt on the current fit/calib/sensitivity/final split
instead of the old cached pickle, and scored with the plain max (alpha=0)
rather than the shipped alpha=0.5 fusion.

--pool sensitivity (default): dev-pool check, corruption labels used here,
    FINAL never read.
--pool final: THE one-shot reported number. Run this exactly once, after
    every combiner/hyperparameter choice is frozen from the sensitivity-pool
    exploration -- that is the entire point of holding this pool back.

Plain max IS the shipped detector now (mdd.fusion_alpha default reverted to
0.0 2026-08-29 -- the org-branch spatial fix made the alpha-median push
unnecessary), so this script's plain-max scoring already matches the paper's
tab:mdd/tab:window/tab:mdd-aupr headline numbers; no separate fused variant
needed. Also reports AUPR/FPR@95 (point estimate, no CI) alongside AUROC for
tab:mdd-aupr.

Output: outputs/results/mdd_max_ci_<pool>.csv
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import LazyPhiDict, split_boundaries, auroc_aupr_fpr95
from analysis.mdd import MDD
from analysis.mdd_split_sensitivity import POOL_NAMES, POOL_FRACS, _seq_ids

SEV = 5
B = 2000
SEED = 0
DEFAULT_WINDOWS = [1, 8, 16, 32, 64, 128, 256, None]


def per_recording_windows(scores, seq_lens, W):
    out, i = [], 0
    for L in seq_lens:
        seq = scores[i:i + L]; i += L
        if W is None:
            out.append(np.array([seq.mean()]))
        elif W == 1:
            out.append(seq.copy())
        else:
            w = [seq[j:j + W].mean() for j in range(0, L, W)
                 if len(seq[j:j + W]) >= max(1, W // 2)]
            out.append(np.array(w))
    return out


def auroc(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return roc_auc_score(y, np.r_[neg, pos])


def boot_ci(clean_recs, corr_recs, rng, B):
    nc, nt = len(clean_recs), len(corr_recs)
    stats = np.empty(B)
    for b in range(B):
        ci = rng.integers(0, nc, nc)
        ti = rng.integers(0, nt, nt)
        neg = np.concatenate([clean_recs[k] for k in ci])
        pos = np.concatenate([corr_recs[k] for k in ti])
        stats[b] = auroc(neg, pos)
    return np.percentile(stats, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", choices=["sensitivity", "final"], default="sensitivity")
    args = ap.parse_args()
    pool = args.pool
    windows = DEFAULT_WINDOWS

    print(f"Bootstrap CI for plain max, {pool} pool, L{SEV}, B={B} resamples, "
          f"windows={['full' if w is None else w for w in windows]}.")
    all_phi = LazyPhiDict(cache_size=1)
    clean = all_phi["clean"]
    seq_lens = all_phi.get_seq_lens("clean")
    n = len(clean)
    cuts = split_boundaries(n, POOL_FRACS, seq_lens)
    bounds = [0] + cuts + [n]
    ranges = {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(POOL_NAMES)}
    a, b = ranges[pool]
    fa, fb = ranges["fit"]
    ca, cb = ranges["calib"]

    sids = _seq_ids(seq_lens)
    seq_ids_sens = sorted(set(sids[a:b].tolist()))
    seq_lens_sens = [seq_lens[i] for i in seq_ids_sens]

    spatial = all_phi.get_phi_spatial("clean")
    use_spatial = spatial is not None
    sp_fit = spatial[fa:fb] if use_spatial else None
    sp_calib = spatial[ca:cb] if use_spatial else None

    mdd = MDD(use_spatial=use_spatial).fit(clean[fa:fb], clean[ca:cb], sp_fit, sp_calib)
    # score_branches() folds 'org' into 'spatial' (mdd.py:508-509) whenever both are
    # present, so it never appears in a scored dict -- branch_names (fit-time) does
    # list it, so it must be excluded here or the s[c] index below raises KeyError.
    branch_cols = [b for b in mdd.branch_names if b != "org"]
    print(f"Branches: {branch_cols}; {pool} pool = {b - a} frames / {len(seq_ids_sens)} sequences")

    def _max_score(phi_pool, sp_pool):
        s = mdd.score_branches(phi_pool, sp_pool)
        s.pop("fused", None)
        Z = np.stack([s[c] for c in branch_cols], axis=1)
        return Z.max(axis=1)

    clean_max = _max_score(clean[a:b], spatial[a:b] if use_spatial else None)
    rng = np.random.default_rng(SEED)

    rows = []
    for corruption in cfg.CORRUPTIONS:
        run = f"{corruption}_L{SEV}"
        if run not in all_phi:
            continue
        phi_run = all_phi[run][a:b]
        sp_run = all_phi.get_phi_spatial(run)
        sp_run = sp_run[a:b] if (use_spatial and sp_run is not None) else None
        corr_max = _max_score(phi_run, sp_run)

        for W in windows:
            clean_recs = per_recording_windows(clean_max, seq_lens_sens, W)
            corr_recs = per_recording_windows(corr_max, seq_lens_sens, W)
            neg_all, pos_all = np.concatenate(clean_recs), np.concatenate(corr_recs)
            metrics = auroc_aupr_fpr95(neg_all, pos_all)
            pt, aupr_v, fpr95_v = metrics if metrics else (float("nan"),) * 3
            lo, hi = boot_ci(clean_recs, corr_recs, rng, B)
            rows.append({"corruption": corruption, "window": ("full" if W is None else W),
                         "auroc": pt, "aupr": aupr_v, "fpr95": fpr95_v,
                         "ci_lo": lo, "ci_hi": hi})
        print(f"  {corruption} done")

    df = pd.DataFrame(rows)
    res_dir = cfg.OUTPUT_DIR / "results"
    dest = res_dir / f"mdd_max_ci_{pool}.csv"
    df.to_csv(dest, index=False)
    print(f"\nWrote {dest} ({len(df)} rows)")

    for W in windows:
        Wkey = "full" if W is None else W
        sub = df[df.window == Wkey]
        print(f"\n=== max AUROC + 95% bootstrap CI (window={Wkey}, L{SEV}, B={B}) ===")
        print(f"{'corruption':<17} | {'AUROC':>6} | {'95% CI':>16}")
        print("-" * 46)
        for _, r in sub.iterrows():
            print(f"{r.corruption:<17} | {r.auroc:6.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}]")

    # Plain point-estimate AUROC (NO bootstrap) across ALL severities -- cheap
    # reuse of clean_max/mdd already fitted above. CI stays L5-only (expensive);
    # this just fills in the severity curve for the same plain-max combiner.
    sev_rows = []
    for corruption in cfg.CORRUPTIONS:
        for sev in cfg.SEVERITIES:
            run = f"{corruption}_L{sev}"
            if run not in all_phi:
                continue
            phi_run = all_phi[run][a:b]
            sp_run = all_phi.get_phi_spatial(run)
            sp_run = sp_run[a:b] if (use_spatial and sp_run is not None) else None
            corr_max = _max_score(phi_run, sp_run)
            for W in windows:
                clean_recs = per_recording_windows(clean_max, seq_lens_sens, W)
                corr_recs = per_recording_windows(corr_max, seq_lens_sens, W)
                neg_all, pos_all = np.concatenate(clean_recs), np.concatenate(corr_recs)
                metrics = auroc_aupr_fpr95(neg_all, pos_all)
                pt, aupr_v, fpr95_v = metrics if metrics else (float("nan"),) * 3
                sev_rows.append({"corruption": corruption, "severity": sev,
                                  "window": ("full" if W is None else W),
                                  "auroc": pt, "aupr": aupr_v, "fpr95": fpr95_v})

    sev_df = pd.DataFrame(sev_rows)
    sev_dest = res_dir / f"mdd_max_auroc_allsev_{pool}.csv"
    sev_df.to_csv(sev_dest, index=False)
    print(f"\nWrote {sev_dest} ({len(sev_df)} rows, all severities {cfg.SEVERITIES}, no CI)")


if __name__ == "__main__":
    main()
