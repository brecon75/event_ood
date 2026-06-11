"""Stage: evaluate the Manifold-Decomposition Detector (MDD).

Fits ONE unsupervised MDD on clean static-phi and scores every corruption run,
reporting each branch (radius / rcf / l4 / spatial) and the fused max, at two
decision granularities:

  * PER-FRAME    — the proxy-free headline number.
  * PER-SEQUENCE — per-frame scores averaged within each recording (needs
                   `seq_lens`); the lever that rescues consistent-bias
                   corruptions like event_flood.

Leakage-safe: the clean stream is cut sequence-aware into train / eval; the MDD
is fit on the first 85% of train and its branch scales are calibrated on the
remaining 15% (both disjoint from the eval negatives).

Outputs (under outputs/results/):
  mdd_metrics.csv            per-frame AUROC / FPR95 per (corruption, severity, branch)
  mdd_metrics_aggregated.csv per-sequence equivalents (omitted if seq_lens absent)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    LazyPhiDict, TRAIN_RATIO, split_boundary, auroc_fpr95,
    seq_lens_after_cut, aggregate_by_seq, _get_present,
)
from analysis.mdd import MDD

CALIB_FRAC = 0.15   # fraction of the clean TRAIN portion held out to calibrate branch scales


def _auroc_row(corruption, severity, branch, clean_s, corr_s, granularity):
    y = np.concatenate([np.zeros(len(clean_s)), np.ones(len(corr_s))])
    s = np.concatenate([clean_s, corr_s])
    auroc, fpr95 = auroc_fpr95(y, s)
    return {"corruption": corruption, "severity": severity, "branch": branch,
            "granularity": granularity, "auroc": auroc, "fpr95": fpr95,
            "n_clean": len(clean_s), "n_corrupt": len(corr_s)}


def _fused(branch_dict, keys):
    return np.max(np.stack([branch_dict[k] for k in keys], axis=1), axis=1)


def main():
    print("Evaluating MDD (Manifold-Decomposition Detector)...")
    all_phi = LazyPhiDict()
    if "clean" not in all_phi:
        print(f"Error: clean.pt missing from {cfg.PHI_DIR}. Run extract.py first.")
        return

    clean = all_phi["clean"]
    clean_seq_lens = all_phi.get_seq_lens("clean")
    cut = split_boundary(len(clean), TRAIN_RATIO, clean_seq_lens)
    fit_end = max(1, int(cut * (1.0 - CALIB_FRAC)))
    if fit_end >= cut:                       # tiny clean set: borrow one calib row
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

    print(f"Clean split: fit={len(phi_fit)} / calib={len(phi_cal)} / eval={len(phi_eval)} "
          f"({'sequence-aligned' if clean_seq_lens else 'contiguous'}); "
          f"spatial branch {'ON' if use_spatial else 'OFF (no phi_spatial)'}.")

    mdd = MDD(use_spatial=use_spatial).fit(phi_fit, phi_cal, sp_fit, sp_cal)
    print(f"MDD branches: {mdd.branch_names}")

    # clean held-out negatives, scored once
    clean_branches = mdd.score_branches(phi_eval, sp_eval)
    clean_branches.pop("fused", None)
    eval_seq_lens = seq_lens_after_cut(clean_seq_lens, cut)

    per_frame, per_seq = [], []
    present = _get_present(all_phi)
    runs = [f"{c}_L{s}" for c in present for s in cfg.SEVERITIES
            if f"{c}_L{s}" in all_phi]

    for run in runs:
        phi = all_phi[run]
        sp = all_phi.get_phi_spatial(run) if use_spatial else None
        corr_branches = mdd.score_branches(phi, sp)
        corr_branches.pop("fused", None)

        # symmetric fused: only branches present for BOTH clean and this run
        common = [k for k in mdd.branch_names if k in clean_branches and k in corr_branches]
        clean_branches["fused"] = _fused(clean_branches, common)
        corr_branches["fused"] = _fused(corr_branches, common)

        corruption, sev = run.rsplit("_L", 1)
        sev = int(sev)
        run_seq_lens = all_phi.get_seq_lens(run)

        for branch in list(corr_branches.keys()):
            cs, ts = clean_branches[branch], corr_branches[branch]
            per_frame.append(_auroc_row(corruption, sev, branch, cs, ts, "frame"))

            cs_agg = aggregate_by_seq(cs, eval_seq_lens)
            ts_agg = aggregate_by_seq(ts, run_seq_lens)
            if cs_agg is not None and ts_agg is not None:
                per_seq.append(_auroc_row(corruption, sev, branch, cs_agg, ts_agg, "sequence"))

    res_dir = cfg.OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_frame).to_csv(res_dir / "mdd_metrics.csv", index=False)
    print(f"  Wrote {res_dir / 'mdd_metrics.csv'} ({len(per_frame)} rows).")
    if per_seq:
        pd.DataFrame(per_seq).to_csv(res_dir / "mdd_metrics_aggregated.csv", index=False)
        print(f"  Wrote {res_dir / 'mdd_metrics_aggregated.csv'} ({len(per_seq)} rows).")
    else:
        print("  Per-sequence aggregation skipped (no seq_lens — legacy extraction).")

    # console summary: fused, per-frame, severity 5 (or max available)
    df = pd.DataFrame(per_frame)
    fused = df[df.branch == "fused"]
    if not fused.empty:
        top_sev = fused.severity.max()
        view = fused[fused.severity == top_sev].set_index("corruption")["auroc"]
        print(f"\nMDD fused per-frame AUROC @ L{top_sev}:")
        for c, a in view.items():
            print(f"    {c:18s} {a:.3f}")
        print(f"    {'MACRO':18s} {view.mean():.3f}")


if __name__ == "__main__":
    main()
