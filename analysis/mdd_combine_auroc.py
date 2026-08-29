"""Stage: AUROC of every combining method on the SENSITIVITY pool only.

This is the deliberate crossing of the line drawn in `mdd_combine_experiments.py`:
that script's clean-only diagnostics never touch a corruption label, so a
combiner chosen from it can be claimed as corruption-free by design. This
script computes AUROC against real corruption runs on the SENSITIVITY pool to
inform that choice anyway (a standard dev-set evaluation) -- meaning the
combiner selection is no longer label-free, even though the FIT/CALIB pools
and the deployed detector's fit-time behavior still are. The FINAL pool is
never read here and stays reserved for the one-shot reported number.

Rebuilds the identical fit/calib/sensitivity split as `mdd_split_sensitivity.py`
(same POOL_FRACS, same sequence-aligned cuts -- deterministic given clean.pt),
refits one MDD on fit+calib, then for every (corruption, severity) run in
`cfg.CORRUPTIONS` x `cfg.SEVERITIES`:
  - scores the SENSITIVITY-pool rows of that run (positives)
  - reuses the clean SENSITIVITY-pool scores (negatives, computed once)
  - computes AUROC/FPR95 for the 4 raw branches AND the 8 combining methods
    from `mdd_combine_experiments.combine_all`, swept over aggregation window
    sizes (`evaluate_mdd_windows.aggregate_by_windows`, non-overlapping
    W-frame mean-pooling within each sequence). W=1 is per-frame, W=None
    ("full") is whole-sequence pooling -- the two granularities the previous
    version of this script computed are just the two endpoints of this sweep.

Output: outputs/results/mdd_combine_window_sweep.csv
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
from analysis.mdd_combine_experiments import _percentile, combine_all, METHODS
from analysis.evaluate_mdd_windows import aggregate_by_windows

DEFAULT_WINDOWS = [1, 8, 16, 32, 64, 128, 256, None]   # None = full sequence


def _auroc_row(corruption, severity, method, kind, clean_s, corr_s, window):
    y = np.concatenate([np.zeros(len(clean_s)), np.ones(len(corr_s))])
    s = np.concatenate([clean_s, corr_s])
    auroc, fpr95 = auroc_fpr95(y, s)
    return {"corruption": corruption, "severity": severity, "method": method, "kind": kind,
            "window": ("full" if window is None else window), "auroc": auroc, "fpr95": fpr95,
            "n_clean": len(clean_s), "n_corrupt": len(corr_s)}


def _score_all(Z, P, branch_cols):
    """Raw branches + all 8 combined methods -> {name: array}."""
    scored = {c: Z[:, i] for i, c in enumerate(branch_cols)}
    scored.update(combine_all(Z, P))
    return scored


def main():
    print("MDD combining-method AUROC on the SENSITIVITY pool (corruption labels used here).")
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
    assert sum(seq_lens_sens) == (b - a), "sensitivity pool is not a whole-sequence block"

    spatial = all_phi.get_phi_spatial("clean")
    use_spatial = spatial is not None
    phi_fit, phi_calib = clean[fa:fb], clean[ca:cb]
    sp_fit = spatial[fa:fb] if use_spatial else None
    sp_calib = spatial[ca:cb] if use_spatial else None

    mdd = MDD(use_spatial=use_spatial).fit(phi_fit, phi_calib, sp_fit, sp_calib)
    # score_branches() folds 'org' into 'spatial' (mdd.py:508-509) whenever both are
    # present, so it never appears in a scored dict -- branch_names (fit-time) does
    # list it, so it must be excluded here or the s[c] index below raises KeyError.
    branch_cols = [k for k in mdd.branch_names if k != "org"]
    print(f"Branches: {branch_cols}; sensitivity pool = {b - a} frames / {len(seq_ids_sens)} sequences")

    def _branches_only(phi_pool, sp_pool):
        s = mdd.score_branches(phi_pool, sp_pool)
        s.pop("fused", None)
        return np.stack([s[c] for c in branch_cols], axis=1)

    Z_calib = _branches_only(clean[ca:cb], sp_calib)
    Z_clean = _branches_only(clean[a:b], spatial[a:b] if use_spatial else None)
    P_clean = np.stack([_percentile(Z_calib[:, i], Z_clean[:, i]) for i in range(len(branch_cols))], axis=1)
    clean_scored = _score_all(Z_clean, P_clean, branch_cols)

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
            P_corr = np.stack([_percentile(Z_calib[:, i], Z_corr[:, i]) for i in range(len(branch_cols))], axis=1)
            corr_scored = _score_all(Z_corr, P_corr, branch_cols)

            for name in list(branch_cols) + METHODS:
                kind = "branch" if name in branch_cols else "combiner"
                cs, ts = clean_scored[name], corr_scored[name]
                for W in DEFAULT_WINDOWS:
                    cs_agg = aggregate_by_windows(cs, seq_lens_sens, W)
                    ts_agg = aggregate_by_windows(ts, seq_lens_sens, W)
                    if cs_agg is not None and ts_agg is not None:
                        rows.append(_auroc_row(corruption, sev, name, kind, cs_agg, ts_agg, W))
            print(f"  scored {run}")

    df = pd.DataFrame(rows)
    res_dir = cfg.OUTPUT_DIR / "results"
    dest = res_dir / "mdd_combine_window_sweep.csv"
    df.to_csv(dest, index=False)
    print(f"\nWrote {dest} ({len(df)} rows)")

    top_sev = int(df.severity.max())
    order = ["full" if w is None else w for w in DEFAULT_WINDOWS]
    for kind in ("combiner", "branch"):
        sub = df[(df.kind == kind) & (df.severity == top_sev)]
        macro = (sub.groupby(["method", "window"])["auroc"].mean().unstack("window")
                 .reindex(columns=[c for c in order if c in sub.window.unique()]))
        macro["MACRO"] = macro.mean(axis=1)
        macro = macro.sort_values("MACRO", ascending=False)
        print(f"\nMacro AUROC vs window, {kind}s @ L{top_sev} (sensitivity pool):")
        print(macro.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
