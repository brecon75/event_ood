"""Stage: try unsupervised combining methods on the standardized branch scores.

Reads the CALIB and SENSITIVITY pools written by `mdd_split_sensitivity.py`
(both clean-only, no corruption runs loaded here either). CALIB is used only
as the reference population for the percentile/probability transforms below
-- it is the same disjoint clean split MDD itself calibrated against, so this
introduces no new leakage. SENSITIVITY is where every combining method is
evaluated. FINAL is never read by this script.

Branch columns (radius/rcf/l4/spatial) are already standardized z-scores
(mean 0, std 1 on CALIB) -- see MDD.score_branches. All combiners below run
directly on those; none refit or re-standardize anything.

Combining methods (all unsupervised, no corruption label used anywhere):
  max            plain OR -- any one branch firing is enough
  mean           plain AND-ish -- averages all branches
  chi2           sum(z_i^2) -- principled for independent evidence (~chi-sq
                 under the clean null if branches are roughly Gaussian)
  top2_mean      mean of the 2 highest branch z's -- OR/AND compromise
  max_minus_median   max(z) - median(z) -- studentized max with NO tunable
                 alpha (fixed at 1), the fixed version of MDD's own fusion rule
  pct_max        max of per-branch percentiles against the CALIB empirical
                 distribution (distribution-free; robust to the heavy right
                 tail visible in the calib branch stats)
  pct_mean       mean of per-branch percentiles (same reference)
  noisy_or       1 - prod(1 - p_i), p_i = per-branch CALIB percentile

Output: outputs/results/mdd_split_sensitivity_combined.csv (raw branch
z-scores + one column per combining method) plus a console summary of each
method's clean-only distribution (mean/std/min/max) -- no AUROC, no
corruption label is computed anywhere in this script.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg

ID_COLS = {"row", "seq_id"}
METHODS = ["max", "mean", "chi2", "top2_mean", "max_minus_median",
           "pct_max", "pct_mean", "noisy_or"]


def _percentile(calib_vals, test_vals):
    """Fraction of CALIB values <= each test value, in (0, 1)."""
    sorted_calib = np.sort(calib_vals)
    n = len(sorted_calib)
    ranks = np.searchsorted(sorted_calib, test_vals, side="right")
    return ranks / (n + 1)


def combine_all(Z, P):
    """All 8 combining methods, given branch z-scores `Z` (N, n_branches) and
    their CALIB percentiles `P` (same shape). Shared by the clean-only stats
    script and the AUROC script so both compute identical combined scores."""
    out = {}
    out["max"] = Z.max(axis=1)
    out["mean"] = Z.mean(axis=1)
    out["chi2"] = (Z ** 2).sum(axis=1)
    sorted_desc = np.sort(Z, axis=1)[:, ::-1]
    out["top2_mean"] = sorted_desc[:, :2].mean(axis=1)
    out["max_minus_median"] = Z.max(axis=1) - np.median(Z, axis=1)
    out["pct_max"] = P.max(axis=1)
    out["pct_mean"] = P.mean(axis=1)
    P_clip = np.clip(P, 1e-6, 1 - 1e-6)
    out["noisy_or"] = 1 - np.prod(1 - P_clip, axis=1)
    return out


def main():
    res_dir = cfg.OUTPUT_DIR / "results"
    calib = pd.read_csv(res_dir / "mdd_split_calib.csv")
    sens = pd.read_csv(res_dir / "mdd_split_sensitivity.csv")

    branch_cols = [c for c in sens.columns if c not in ID_COLS]
    assert branch_cols == [c for c in calib.columns if c not in ID_COLS], \
        "calib/sensitivity branch columns differ"
    print(f"Combining {branch_cols} on sensitivity pool ({len(sens)} frames), "
          f"percentiles referenced against calib pool ({len(calib)} frames).")

    Z = sens[branch_cols].to_numpy()
    P = np.stack([_percentile(calib[c].to_numpy(), sens[c].to_numpy()) for c in branch_cols], axis=1)

    out = sens[["row", "seq_id"]].copy()
    for c in branch_cols:
        out[c] = sens[c]
    for name, vals in combine_all(Z, P).items():
        out[name] = vals

    dest = res_dir / "mdd_split_sensitivity_combined.csv"
    out.to_csv(dest, index=False)
    print(f"Wrote {dest} ({len(out)} rows, columns: {list(out.columns)})")

    stats = out[METHODS].describe().T[["mean", "std", "min", "max"]]
    stats_dest = res_dir / "mdd_combine_clean_stats.csv"
    stats.to_csv(stats_dest)
    print(f"Wrote {stats_dest}")
    print("\nClean-only distribution per combining method (sensitivity pool):")
    print(stats)


if __name__ == "__main__":
    main()
