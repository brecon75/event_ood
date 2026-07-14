"""Stage (diagnostic): MDD AUROC as a function of aggregation WINDOW length.

Fits ONE unsupervised MDD on clean static-phi (identical split/calibration to
`evaluate_mdd.py`) and scores every corruption run, but instead of the two fixed
granularities (per-frame + per-sequence) it sweeps a range of window sizes:

    per-frame  (W=1)  →  short windows (8, 16, 32, ...)  →  full sequence (W=full)

The point is to expose *how much pooling* a result needs. A detector that is
already strong at W=8 is a real, low-latency signal; one that only reaches 1.0
near full-sequence length is a marginal mean-shift that pooling manufactures
(e.g. the event_flood `spatial` branch, which is below chance per-frame yet
1.0 per-sequence). The curve also raises N: each sequence yields ~L/W window
points instead of one, tightening the AUROC estimate.

Window aggregation = mean of per-frame branch scores within each non-overlapping
W-frame window (tail windows shorter than W//2 are dropped). W=full reproduces
`evaluate_mdd.py`'s per-sequence number exactly; W=1 reproduces per-frame.

Leakage-safe: same sequence-aware clean train/calib/eval cut as evaluate_mdd.py;
positives are the held-out tail of each run, cut at the matching boundary.

Outputs (under outputs/results/):
  mdd_window_sweep.csv   AUROC / FPR95 per (corruption, severity, branch, window)
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
    LazyPhiDict, TRAIN_RATIO, split_boundary, auroc_fpr95,
    seq_lens_after_cut, _get_present,
)
from analysis.mdd import MDD

CALIB_FRAC = 0.15          # clean TRAIN fraction held out to calibrate branch scales
DEFAULT_WINDOWS = [1, 8, 16, 32, 64, 128, 256, None]   # None = full sequence


def aggregate_by_windows(scores, seq_lens, W):
    """Mean of per-frame `scores` over non-overlapping W-frame windows, within
    each sequence so windows never straddle a recording boundary.

    W is None  -> one value per whole sequence (matches aggregate_by_seq).
    W == 1     -> the per-frame scores unchanged.
    Tail windows shorter than W//2 are dropped so a stray 1-frame remainder
    does not become its own noisy "window".

    Returns a 1-D array, or None when seq_lens is missing / inconsistent.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not seq_lens or int(np.sum(seq_lens)) != len(scores):
        return None
    if W == 1:
        return scores.copy()

    out, start = [], 0
    for L in seq_lens:
        seq = scores[start:start + L]
        start += L
        if W is None:                       # whole-sequence pooling
            if L > 0:
                out.append(seq.mean())
            continue
        for i in range(0, L, W):
            chunk = seq[i:i + W]
            if len(chunk) >= max(1, W // 2):
                out.append(chunk.mean())
    return np.asarray(out) if out else None


def _auroc_row(corruption, severity, branch, window, clean_s, corr_s):
    y = np.concatenate([np.zeros(len(clean_s)), np.ones(len(corr_s))])
    s = np.concatenate([clean_s, corr_s])
    auroc, fpr95 = auroc_fpr95(y, s)
    return {"corruption": corruption, "severity": severity, "branch": branch,
            "window": ("full" if window is None else window),
            "auroc": auroc, "fpr95": fpr95,
            "n_clean": len(clean_s), "n_corrupt": len(corr_s)}


def _fused(branch_dict, keys, alpha=0.5):
    """Fused score over `keys` using the SHIPPED studentized max
    (max - alpha*median), identical to MDD.score_branches. alpha=0 restores the
    legacy plain max. Callers pass mdd.fusion_alpha so this tracks the detector
    default and every table reports the deployed detector."""
    stack = np.stack([branch_dict[k] for k in keys], axis=1)
    mx = np.max(stack, axis=1)
    return mx - alpha * np.median(stack, axis=1) if alpha else mx


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=str, default=None,
                    help="comma-separated window sizes, e.g. '1,16,32,full'. "
                         "'full' or 'seq' = whole-sequence pooling. "
                         f"Default: {DEFAULT_WINDOWS}")
    ap.add_argument("--runs", type=str, default=None,
                    help="comma-separated substrings; keep only corruption runs "
                         "matching any (e.g. 'event_flood,spatial_dropout'). "
                         "Avoids loading the multi-GB hot_pixel files. Default: all.")
    ap.add_argument("--max-runs", type=int, default=None,
                    help="cap the number of corruption runs scored (smoke test).")
    args = ap.parse_args()

    if args.windows:
        windows = []
        for tok in args.windows.split(","):
            tok = tok.strip().lower()
            windows.append(None if tok in ("full", "seq", "none", "") else int(tok))
    else:
        windows = DEFAULT_WINDOWS
    win_label = ["full" if w is None else w for w in windows]
    print(f"MDD window sweep over: {win_label}")

    all_phi = LazyPhiDict(cache_size=1)
    if "clean" not in all_phi:
        print(f"Error: clean.pt missing from {cfg.PHI_DIR}. Run extract.py first.")
        return

    clean = all_phi["clean"]
    clean_seq_lens = all_phi.get_seq_lens("clean")
    if not clean_seq_lens:
        print("Error: clean.pt has no seq_lens (legacy extraction). Window "
              "aggregation needs per-sequence frame counts; re-extract first.")
        return

    cut = split_boundary(len(clean), TRAIN_RATIO, clean_seq_lens)
    fit_end = max(1, int(cut * (1.0 - CALIB_FRAC)))
    if fit_end >= cut:
        fit_end = max(1, cut - 1)

    phi_fit, phi_cal, phi_eval = clean[:fit_end], clean[fit_end:cut], clean[cut:]
    if len(phi_cal) == 0 or len(phi_eval) == 0:
        print(f"Error: degenerate clean split (fit={len(phi_fit)}, cal={len(phi_cal)}, "
              f"eval={len(phi_eval)}). Need more clean frames.")
        return

    clean_spatial = all_phi.get_phi_spatial("clean")
    use_spatial = clean_spatial is not None
    sp_fit = clean_spatial[:fit_end] if use_spatial else None
    sp_cal = clean_spatial[fit_end:cut] if use_spatial else None
    sp_eval = clean_spatial[cut:] if use_spatial else None

    print(f"Clean split: fit={len(phi_fit)} / calib={len(phi_cal)} / eval={len(phi_eval)}; "
          f"spatial branch {'ON' if use_spatial else 'OFF (no phi_spatial)'}.")

    mdd = MDD(use_spatial=use_spatial).fit(phi_fit, phi_cal, sp_fit, sp_cal)
    print(f"MDD branches: {mdd.branch_names}")

    # clean held-out negatives, scored once (reused for every run and window)
    clean_branches = mdd.score_branches(phi_eval, sp_eval)
    clean_branches.pop("fused", None)
    eval_seq_lens = seq_lens_after_cut(clean_seq_lens, cut)
    # Free large clean arrays — no longer needed after this point
    del clean, clean_spatial, phi_fit, phi_cal, phi_eval, sp_fit, sp_cal, sp_eval

    rows = []
    present = _get_present(all_phi)
    runs = [f"{c}_L{s}" for c in present for s in cfg.SEVERITIES
            if f"{c}_L{s}" in all_phi]
    if args.runs:
        pats = [p.strip() for p in args.runs.split(",") if p.strip()]
        runs = [r for r in runs if any(p in r for p in pats)]
    if args.max_runs is not None:
        runs = runs[:max(0, args.max_runs)]
    print(f"Scoring MDD on {len(runs)} corruption runs across {len(windows)} windows...")

    for run in tqdm(runs, desc="MDD window sweep"):
        phi_full = all_phi[run]
        sp_full = all_phi.get_phi_spatial(run) if use_spatial else None
        run_seq_lens_full = all_phi.get_seq_lens(run)
        run_cut = split_boundary(len(phi_full), TRAIN_RATIO, run_seq_lens_full)
        phi = np.ascontiguousarray(phi_full[run_cut:], dtype=np.float32)
        sp = np.ascontiguousarray(sp_full[run_cut:], dtype=np.float32) if sp_full is not None else None
        all_phi._cache.pop(run, None)
        del phi_full, sp_full
        run_seq_lens = seq_lens_after_cut(run_seq_lens_full, run_cut)

        corr_branches = mdd.score_branches(phi, sp)
        corr_branches.pop("fused", None)

        # symmetric fused: branches present for BOTH clean and this run
        common = [k for k in mdd.branch_names if k in clean_branches and k in corr_branches]
        clean_branches["fused"] = _fused(clean_branches, common, mdd.fusion_alpha)
        corr_branches["fused"] = _fused(corr_branches, common, mdd.fusion_alpha)

        corruption, sev = run.rsplit("_L", 1)
        sev = int(sev)

        for branch in list(corr_branches.keys()):
            cs, ts = clean_branches[branch], corr_branches[branch]
            for W in windows:
                cs_agg = aggregate_by_windows(cs, eval_seq_lens, W)
                ts_agg = aggregate_by_windows(ts, run_seq_lens, W)
                if cs_agg is None or ts_agg is None:
                    continue
                rows.append(_auroc_row(corruption, sev, branch, W, cs_agg, ts_agg))

    if not rows:
        print("No rows produced (missing seq_lens for every run?).")
        return

    res_dir = cfg.OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out_csv = res_dir / "mdd_window_sweep.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv} ({len(df)} rows).")

    # console summary: fused AUROC vs window, at the top severity, as a pivot
    top_sev = int(df.severity.max())
    fused = df[(df.branch == "fused") & (df.severity == top_sev)]
    if not fused.empty:
        order = [w if w is not None else "full" for w in windows]
        pivot = (fused.pivot_table(index="corruption", columns="window", values="auroc")
                 .reindex(columns=[c for c in order if c in fused.window.unique()]))
        print(f"\nFused AUROC vs window @ L{top_sev} (rows=corruption, cols=window):")
        print(pivot.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
