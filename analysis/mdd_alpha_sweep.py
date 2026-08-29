"""Stage: sweep MDD's own studentized-max alpha (max - alpha*median) on the
SENSITIVITY pool. Same dev-set caveat as `mdd_combine_auroc.py` -- corruption
labels are used here, so this informs (rather than stays blind to) the choice
of alpha; FINAL is never read.

alpha=0 reproduces plain `max` (already in mdd_combine_window_sweep.csv);
alpha=1 reproduces `max_minus_median` (ditto). This script fills in the
values between and beyond those two to see whether some alpha beats both
endpoints, mirroring how `mdd.py`'s shipped alpha=0.5 was originally chosen.

Output: outputs/results/mdd_alpha_sweep.csv
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

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
WINDOWS = [1, 8, 16, 32, 64, 128, 256, None]


def _auroc_row(corruption, severity, alpha, window, clean_s, corr_s):
    y = np.concatenate([np.zeros(len(clean_s)), np.ones(len(corr_s))])
    s = np.concatenate([clean_s, corr_s])
    auroc, fpr95 = auroc_fpr95(y, s)
    return {"corruption": corruption, "severity": severity, "alpha": alpha,
            "window": ("full" if window is None else window), "auroc": auroc, "fpr95": fpr95}


def main():
    print("MDD alpha sweep (max - alpha*median) on the SENSITIVITY pool.")
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

            for alpha in ALPHAS:
                cs = Z_clean.max(axis=1) - alpha * np.median(Z_clean, axis=1)
                ts = Z_corr.max(axis=1) - alpha * np.median(Z_corr, axis=1)
                for W in WINDOWS:
                    cs_agg = aggregate_by_windows(cs, seq_lens_sens, W)
                    ts_agg = aggregate_by_windows(ts, seq_lens_sens, W)
                    if cs_agg is not None and ts_agg is not None:
                        rows.append(_auroc_row(corruption, sev, alpha, W, cs_agg, ts_agg))
            print(f"  scored {run}")

    df = pd.DataFrame(rows)
    res_dir = cfg.OUTPUT_DIR / "results"
    dest = res_dir / "mdd_alpha_sweep.csv"
    df.to_csv(dest, index=False)
    print(f"\nWrote {dest} ({len(df)} rows)")

    top_sev = int(df.severity.max())
    order = ["full" if w is None else w for w in WINDOWS]
    sub = df[df.severity == top_sev]
    macro = sub.groupby(["alpha", "window"])["auroc"].mean().unstack("window").reindex(columns=order)
    macro["MACRO"] = macro.mean(axis=1)
    print(f"\nMacro AUROC vs alpha x window @ L{top_sev} (sensitivity pool):")
    print(macro.round(3).to_string())


if __name__ == "__main__":
    main()
