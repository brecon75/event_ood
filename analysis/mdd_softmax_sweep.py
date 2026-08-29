"""Stage: sweep a softmax/LogSumExp combiner (smooth max) on the SENSITIVITY
pool. Same dev-set caveat as `mdd_combine_auroc.py` / `mdd_alpha_sweep.py` --
corruption labels used here, FINAL never read.

S_tau(z) = tau * log(sum_i exp(z_i / tau))
         = max(z) + tau * log(sum_i exp((z_i - max(z)) / tau))   [stable form]

As tau -> 0, S_tau -> max(z) (the plain-max combiner already swept). As tau
grows, weaker branches contribute more (tau -> inf behaves like a shifted
mean). This fills in the smooth interpolation between max and mean that the
discrete `max`/`mean`/`top2_mean` combiners in mdd_combine_experiments.py
only sample at fixed points.

Output: outputs/results/mdd_softmax_sweep.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import LazyPhiDict, split_boundaries, auroc_fpr95
from analysis.mdd import MDD
from analysis.mdd_split_sensitivity import POOL_NAMES, POOL_FRACS, _seq_ids
from analysis.evaluate_mdd_windows import aggregate_by_windows

TAUS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
WINDOWS = [1, 8, 16, 32, 64, 128, 256, None]


def logsumexp_combine(Z, tau):
    m = Z.max(axis=1)
    return m + tau * np.log(np.exp((Z - m[:, None]) / tau).sum(axis=1))


def _auroc_row(corruption, severity, tau, window, clean_s, corr_s):
    y = np.concatenate([np.zeros(len(clean_s)), np.ones(len(corr_s))])
    s = np.concatenate([clean_s, corr_s])
    auroc, fpr95 = auroc_fpr95(y, s)
    return {"corruption": corruption, "severity": severity, "tau": tau,
            "window": ("full" if window is None else window), "auroc": auroc, "fpr95": fpr95}


def main():
    print("MDD softmax/LogSumExp sweep on the SENSITIVITY pool.")
    all_phi = LazyPhiDict(cache_size=1)
    clean = all_phi["clean"]
    seq_lens = all_phi.get_seq_lens("clean")
    n = len(clean)
    cuts = split_boundaries(n, POOL_FRACS, seq_lens)
    bounds = [0] + cuts + [n]
    ranges = {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(POOL_NAMES)}
    a, b = ranges["sensitivity"]
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
    print(f"Branches: {branch_cols}; sensitivity pool = {b - a} frames / {len(seq_ids_sens)} sequences")

    def _branches_only(phi_pool, sp_pool):
        s = mdd.score_branches(phi_pool, sp_pool)
        s.pop("fused", None)
        return np.stack([s[c] for c in branch_cols], axis=1)

    Z_clean = _branches_only(clean[a:b], spatial[a:b] if use_spatial else None)

    rows = []
    for corruption in cfg.CORRUPTIONS:
        for sev in cfg.SEVERITIES:
            run = f"{corruption}_L{sev}"
            if run not in all_phi:
                continue
            phi_run = all_phi[run][a:b]
            sp_run = all_phi.get_phi_spatial(run)
            sp_run = sp_run[a:b] if (use_spatial and sp_run is not None) else None
            Z_corr = _branches_only(phi_run, sp_run)

            for tau in TAUS:
                cs = logsumexp_combine(Z_clean, tau)
                ts = logsumexp_combine(Z_corr, tau)
                for W in WINDOWS:
                    cs_agg = aggregate_by_windows(cs, seq_lens_sens, W)
                    ts_agg = aggregate_by_windows(ts, seq_lens_sens, W)
                    if cs_agg is not None and ts_agg is not None:
                        rows.append(_auroc_row(corruption, sev, tau, W, cs_agg, ts_agg))
            print(f"  scored {run}")

    df = pd.DataFrame(rows)
    res_dir = cfg.OUTPUT_DIR / "results"
    dest = res_dir / "mdd_softmax_sweep.csv"
    df.to_csv(dest, index=False)
    print(f"\nWrote {dest} ({len(df)} rows)")

    top_sev = int(df.severity.max())
    order = ["full" if w is None else w for w in WINDOWS]
    sub = df[df.severity == top_sev]
    macro = sub.groupby(["tau", "window"])["auroc"].mean().unstack("window").reindex(columns=order)
    macro["MACRO"] = macro.mean(axis=1)
    print(f"\nMacro AUROC vs tau x window @ L{top_sev} (sensitivity pool):")
    print(macro.round(3).to_string())


if __name__ == "__main__":
    main()
