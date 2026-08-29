"""Diagnostic: why do evaluate_mdd_windows.py (canonical 85/15 split) and
mdd_max_ci.py --pool final (fit/calib/sensitivity/final split) disagree on
temporal_jitter (W=64,L5: 0.875 vs 0.761) when both score the EXACT SAME
held-out eval frames (both cut at the same 70% sequence boundary -- confirmed:
canonical fit(204096)+calib(36018)=240114 == split-scheme fit(171576)+
calib(33858)+sensitivity(34680)=240114)?

Hypothesis: the two scripts train MDD on DIFFERENT fit/calib data even though
they evaluate on the identical final 30% (102985 rows):
  - canonical: fit=first 85% of [0,240114), calib=last 15% of [0,240114)
               -> fit+calib uses ALL of [0, 240114) contiguously
  - mdd_max_ci "final" pool: fit=[0,171576), calib=[171576,205434)
               -> the [205434,240114) "sensitivity" chunk (34680 rows) is
                  EXCLUDED from both fit and calib entirely (reserved for a
                  different, unrelated use)

This script fits both variants and scores them on the IDENTICAL eval slice
(clean[240114:] vs temporal_jitter_L5[240114:], W=64) to isolate whether the
fit/calib composition is really what's driving the gap.
"""
import re
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    LazyPhiDict, split_boundary, split_boundaries, seq_lens_after_cut,
    load_pt, materialize_f32,
)
from analysis.mdd import MDD
from analysis.evaluate_mdd import _fused
from analysis.evaluate_mdd_windows import aggregate_by_windows
from analysis.mdd_split_sensitivity import POOL_NAMES, POOL_FRACS

SEV, W = 5, 64
CORRUPTION = "temporal_jitter"

SPATIAL_DIR = cfg.REPO_ROOT / "phi_spatial"
_dedupe = re.compile(r" \(\d+\)$")
_spatial_files = {_dedupe.sub("", f.stem): f for f in SPATIAL_DIR.glob("*.pt")}


def get_phi_spatial_override(self, key):
    if self.fast_mode:
        return None
    f = _spatial_files.get(key)
    if f is None:
        return None
    d = load_pt(f)
    ps = d.get("phi_spatial", None)
    return materialize_f32(ps) if ps is not None else None


LazyPhiDict.get_phi_spatial = get_phi_spatial_override


def score(mdd, phi, sp, seq_lens, w):
    b = mdd.score_branches(phi, sp)
    b.pop("fused", None)
    fused = _fused(b, [k for k in mdd.branch_names if k in b], mdd.fusion_alpha)
    return aggregate_by_windows(fused, seq_lens, w)


def main():
    all_phi = LazyPhiDict(cache_size=1)
    clean = all_phi["clean"]
    seq_lens = all_phi.get_seq_lens("clean")
    n = len(clean)
    clean_spatial = all_phi.get_phi_spatial("clean")
    use_spatial = clean_spatial is not None

    cut = split_boundary(n, 0.7, seq_lens)
    fit_end = max(1, int(cut * 0.85))
    cuts4 = split_boundaries(n, POOL_FRACS, seq_lens)
    bounds = [0] + cuts4 + [n]
    ranges = {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(POOL_NAMES)}
    print(f"canonical cut={cut}, fit_end={fit_end}")
    print(f"split-scheme ranges={ranges}")
    assert cut == ranges["final"][0], "eval/final boundary mismatch -- hypothesis invalid"

    run = f"{CORRUPTION}_L{SEV}"
    run_full = all_phi[run]
    run_sp_full = all_phi.get_phi_spatial(run) if use_spatial else None
    run_seq_lens = all_phi.get_seq_lens(run)

    eval_seq_lens = seq_lens_after_cut(seq_lens, cut)

    variants = {
        "canonical (fit=0:{}, calib={}:{})".format(fit_end, fit_end, cut): (0, fit_end, fit_end, cut),
        "split-scheme (fit=0:{}, calib={}:{}, gap={}:{} unused)".format(
            ranges["fit"][1], ranges["calib"][0], ranges["calib"][1],
            ranges["calib"][1], ranges["sensitivity"][1]
        ): (ranges["fit"][0], ranges["fit"][1], ranges["calib"][0], ranges["calib"][1]),
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")
    results = {}

    for label, (fa, fb, ca, cb) in variants.items():
        sp_fit = clean_spatial[fa:fb] if use_spatial else None
        sp_cal = clean_spatial[ca:cb] if use_spatial else None
        mdd = MDD(use_spatial=use_spatial).fit(clean[fa:fb], clean[ca:cb], sp_fit, sp_cal)

        cs = score(mdd, clean[cut:], clean_spatial[cut:] if use_spatial else None, eval_seq_lens, W)
        ts = score(mdd, run_full[cut:], run_sp_full[cut:] if run_sp_full is not None else None,
                   seq_lens_after_cut(run_seq_lens, cut), W)

        y = np.concatenate([np.zeros(len(cs)), np.ones(len(ts))])
        s = np.concatenate([cs, ts])
        fpr, tpr, _ = roc_curve(y, s)
        auroc = roc_auc_score(y, s)
        results[label] = auroc
        print(f"{label}: fit_n={fb-fa}, calib_n={cb-ca}, AUROC(W={W})={auroc:.4f}")
        ax.plot(fpr, tpr, lw=2, label=f"{label}\nAUROC={auroc:.3f}")

    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"temporal_jitter fused ROC, W={W}, L{SEV}\nsame eval frames, different fit/calib pools")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    out_dir = cfg.OUTPUT_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnose_temporal_jitter.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")
    print(f"\nDelta = {results[list(variants)[0]] - results[list(variants)[1]]:.4f}")


if __name__ == "__main__":
    main()
