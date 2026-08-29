"""One-off: ROC curve (FPR vs TPR) for the fused MDD score, W=64, L5, ONE
individual subplot per corruption (not overlaid) -- same canonical 85/15
split + new phi_spatial (org branch) + plain-max fusion as
`evaluate_mdd_windows.py` / `run_mdd_paper_tables.py`, just exposing the raw
ROC curve instead of collapsing to AUROC/FPR95 points.

Output: vmem_benchmark/outputs/plots/roc_w64_l5.png (2x3 grid, one axes per corruption)
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
    LazyPhiDict, split_boundaries, seq_lens_after_cut, load_pt, materialize_f32,
)
from analysis.mdd import MDD
from analysis.evaluate_mdd import _fused
from analysis.evaluate_mdd_windows import aggregate_by_windows
from analysis.mdd_split_sensitivity import POOL_NAMES, POOL_FRACS

SEV = 5
W = 64

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


def main():
    all_phi = LazyPhiDict(cache_size=1)
    clean = all_phi["clean"]
    clean_seq_lens = all_phi.get_seq_lens("clean")
    cuts = split_boundaries(len(clean), POOL_FRACS, clean_seq_lens)
    bounds = [0] + cuts + [len(clean)]
    ranges = {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(POOL_NAMES)}
    fa, fb = ranges["fit"]
    ca, cb = ranges["calib"]
    cut = ranges["final"][0]

    phi_fit, phi_cal, phi_eval = clean[fa:fb], clean[ca:cb], clean[cut:]
    clean_spatial = all_phi.get_phi_spatial("clean")
    use_spatial = clean_spatial is not None
    sp_fit = clean_spatial[fa:fb] if use_spatial else None
    sp_cal = clean_spatial[ca:cb] if use_spatial else None
    sp_eval = clean_spatial[cut:] if use_spatial else None

    mdd = MDD(use_spatial=use_spatial).fit(phi_fit, phi_cal, sp_fit, sp_cal)
    print(f"MDD branches: {mdd.branch_names}; fusion_alpha={mdd.fusion_alpha}")

    clean_branches = mdd.score_branches(phi_eval, sp_eval)
    clean_branches.pop("fused", None)
    eval_seq_lens = seq_lens_after_cut(clean_seq_lens, cut)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()

    for i, corruption in enumerate(cfg.CORRUPTIONS):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="chance")

        run = f"{corruption}_L{SEV}"
        if run not in all_phi:
            print(f"skip {run} (missing)")
            continue
        phi_full = all_phi[run]
        sp_full = all_phi.get_phi_spatial(run) if use_spatial else None
        run_seq_lens_full = all_phi.get_seq_lens(run)
        run_cut = split_boundaries(len(phi_full), POOL_FRACS, run_seq_lens_full)[-1]
        phi = np.ascontiguousarray(phi_full[run_cut:], dtype=np.float32)
        sp = np.ascontiguousarray(sp_full[run_cut:], dtype=np.float32) if sp_full is not None else None
        run_seq_lens = seq_lens_after_cut(run_seq_lens_full, run_cut)

        corr_branches = mdd.score_branches(phi, sp)
        corr_branches.pop("fused", None)
        common = [k for k in mdd.branch_names if k in clean_branches and k in corr_branches]
        cs = _fused(clean_branches, common, mdd.fusion_alpha)
        ts = _fused(corr_branches, common, mdd.fusion_alpha)

        cs_w = aggregate_by_windows(cs, eval_seq_lens, W)
        ts_w = aggregate_by_windows(ts, run_seq_lens, W)

        y = np.concatenate([np.zeros(len(cs_w)), np.ones(len(ts_w))])
        s = np.concatenate([cs_w, ts_w])
        fpr, tpr, _ = roc_curve(y, s)
        auroc = roc_auc_score(y, s)
        print(f"{corruption}: AUROC={auroc:.4f}, n_windows clean/corrupt = {len(cs_w)}/{len(ts_w)}")

        ax.plot(fpr, tpr, lw=2, color="C0")
        ax.set_title(f"{corruption}  (AUROC={auroc:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

    fig.suptitle(f"MDD fused ROC per corruption, W={W}, L{SEV}")
    fig.tight_layout()

    out_dir = cfg.OUTPUT_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "roc_w64_l5.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
