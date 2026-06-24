"""Stage (diagnostic): MDD sensitivity to its hyperparameters.

Answers the reviewer question "how robust is MDD to the PCA dimensionality, the k
in cosine-kNN (RCF), and the clean-calibration sample size?" by sweeping ONE knob
at a time around the shipped defaults and reporting fused AUROC per corruption.

It runs entirely on the EXISTING static-phi (no re-extraction): it caches the
held-out clean negatives and each corruption run's held-out positives once, then
refits the MDD for every hyperparameter value and re-scores from the cache.

Knobs swept (defaults in brackets, held fixed while another varies):
  k_pca  [64]   : PCA dimensionality of the denoised manifold (B1/B2 live here)
  k_nn   [64]   : neighbours in the cosine-kNN that conditions the RCF branch
  n_ref  [15000]: size of the clean reference used to estimate conditional radii

Aggregation window defaults to W=64 frames (the deployable, bounded-latency
operating point) plus W=1 (per-frame) for reference. Leakage-safe: identical
sequence-aware clean train/calib/eval cut as evaluate_mdd.py / _windows.py.

Outputs (under outputs/results/):
  mdd_sensitivity.csv   fused AUROC / FPR95 per (param, value, corruption, severity, window)
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    LazyPhiDict, TRAIN_RATIO, split_boundary, seq_lens_after_cut, _get_present,
)
from analysis.mdd import MDD
from analysis.evaluate_mdd_windows import aggregate_by_windows, _fused, _auroc_row

CALIB_FRAC = 0.15
DEFAULTS = {"k_pca": 64, "k_nn": 64, "n_ref": 15000}
GRID = {
    "k_pca": [16, 32, 64, 128, 256],
    "k_nn":  [8, 16, 32, 64, 128],
    "n_ref": [2000, 5000, 15000, 40000],
}


def _score_config(params, cache, windows, use_spatial):
    """Fit one MDD with `params` and return AUROC rows over the cached runs."""
    cl = cache["clean"]
    mdd = MDD(use_spatial=use_spatial, **params).fit(
        cl["fit"], cl["cal"], cl["sp_fit"], cl["sp_cal"])
    clean_branches = mdd.score_branches(cl["eval"], cl["sp_eval"])
    clean_branches.pop("fused", None)
    rows = []
    for run, d in cache["runs"].items():
        corr_branches = mdd.score_branches(d["phi"], d["sp"])
        corr_branches.pop("fused", None)
        common = [k for k in mdd.branch_names
                  if k in clean_branches and k in corr_branches]
        cb_fused = _fused(clean_branches, common)
        tb_fused = _fused(corr_branches, common)
        corruption, sev = run.rsplit("_L", 1)
        for W in windows:
            cs = aggregate_by_windows(cb_fused, cl["eval_seq_lens"], W)
            ts = aggregate_by_windows(tb_fused, d["seq_lens"], W)
            if cs is None or ts is None:
                continue
            row = _auroc_row(corruption, int(sev), "fused", W, cs, ts)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--params", type=str, default="k_pca,k_nn,n_ref",
                    help="comma-separated knobs to sweep (subset of k_pca,k_nn,n_ref).")
    ap.add_argument("--windows", type=str, default="1,64",
                    help="aggregation windows; 'full'/'seq' = whole recording. Default '1,64'.")
    ap.add_argument("--severities", type=str, default="5",
                    help="comma-separated severities to score (default '5').")
    ap.add_argument("--runs", type=str, default=None,
                    help="substring filter on corruption runs (e.g. 'event_flood,spatial_dropout').")
    args = ap.parse_args()

    sweep_params = [p.strip() for p in args.params.split(",") if p.strip() in GRID]
    windows = [None if t.strip().lower() in ("full", "seq", "none", "")
               else int(t) for t in args.windows.split(",")]
    sevs = [int(s) for s in args.severities.split(",") if s.strip()]

    all_phi = LazyPhiDict()
    if "clean" not in all_phi:
        print(f"Error: clean.pt missing from {cfg.PHI_DIR}. Run extract.py first.")
        return
    clean = all_phi["clean"]
    clean_seq_lens = all_phi.get_seq_lens("clean")
    if not clean_seq_lens:
        print("Error: clean.pt has no seq_lens (legacy extraction); re-extract first.")
        return

    cut = split_boundary(len(clean), TRAIN_RATIO, clean_seq_lens)
    fit_end = max(1, int(cut * (1.0 - CALIB_FRAC)))
    if fit_end >= cut:
        fit_end = max(1, cut - 1)
    clean_spatial = all_phi.get_phi_spatial("clean")
    use_spatial = clean_spatial is not None

    # cache clean fit/calib/eval (held out) once
    cache = {"clean": {
        "fit": clean[:fit_end], "cal": clean[fit_end:cut], "eval": clean[cut:],
        "sp_fit": clean_spatial[:fit_end] if use_spatial else None,
        "sp_cal": clean_spatial[fit_end:cut] if use_spatial else None,
        "sp_eval": clean_spatial[cut:] if use_spatial else None,
        "eval_seq_lens": seq_lens_after_cut(clean_seq_lens, cut),
    }, "runs": {}}

    present = _get_present(all_phi)
    runs = [f"{c}_L{s}" for c in present for s in sevs if f"{c}_L{s}" in all_phi]
    if args.runs:
        pats = [p.strip() for p in args.runs.split(",") if p.strip()]
        runs = [r for r in runs if any(p in r for p in pats)]
    print(f"Caching held-out phi for {len(runs)} runs "
          f"(spatial branch {'ON' if use_spatial else 'OFF'})...")
    for run in tqdm(runs, desc="cache runs"):
        phi = all_phi[run]
        sp = all_phi.get_phi_spatial(run) if use_spatial else None
        run_seq_lens_full = all_phi.get_seq_lens(run)
        run_cut = split_boundary(len(phi), TRAIN_RATIO, run_seq_lens_full)
        cache["runs"][run] = {
            "phi": phi[run_cut:],
            "sp": sp[run_cut:] if sp is not None else None,
            "seq_lens": seq_lens_after_cut(run_seq_lens_full, run_cut),
        }

    rows = []
    for param in sweep_params:
        for v in GRID[param]:
            params = dict(DEFAULTS, **{param: v})
            for r in _score_config(params, cache, windows, use_spatial):
                r.update({"param": param, "value": v})
                rows.append(r)
        print(f"swept {param}: {GRID[param]}")

    if not rows:
        print("No rows produced (missing seq_lens for every run?).")
        return
    res_dir = cfg.OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = res_dir / "mdd_sensitivity.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows).")

    # console summary: fused AUROC range per knob at the top window/severity
    W = max([w for w in windows if w is not None], default=None)
    top = df[(df.window == (W if W is not None else "full")) & (df.severity == max(sevs))]
    if not top.empty:
        for param in sweep_params:
            sub = top[top.param == param]
            if sub.empty:
                continue
            piv = sub.pivot_table(index="corruption", columns="value", values="auroc")
            print(f"\nFused AUROC vs {param} @ W={W}, L{max(sevs)}:")
            print(piv.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
