"""One-off driver: regenerate the CANONICAL Stage-8 MDD outputs (the single
85/15 train/eval split used by `evaluate_mdd.py` / `evaluate_mdd_windows.py`,
i.e. the scripts the main paper tables -- tab:mdd, tab:window, tab:mdd-branch,
tab:mdd-loo, tab:mdd-aupr -- were built from) using:

  1. the new 1488-D phi_spatial re-extraction in the top-level `phi_spatial/`
     folder (org stats: flatness + causal persistence), same monkeypatch as
     `run_unified_numbers_newspatial.py`;
  2. cfg.SEVERITIES overridden to [1, 2, 3, 4, 5] instead of the trimmed
     [1, 3, 5] -- L2/L4 phi/phi_spatial already exist on disk for every
     corruption, they were just never scored;
  3. mdd.py's fusion_alpha default reverted to 0.0 (plain max) as of
     2026-08-29 -- the org-branch fix makes the studentized alpha-median
     fusion unnecessary, so no separate override needed here, `evaluate_mdd`'s
     `_fused(..., mdd.fusion_alpha)` calls already pick up plain max.

Runs, in dependency order:
  1. evaluate_mdd_windows -> outputs/results/mdd_window_sweep.csv
     (per-branch + fused AUROC/AUPR/FPR95, every corruption x severity x
     window in [1,8,16,32,64,128,256,full] -- covers tab:mdd (frame=W1,
     w64=W64), tab:window (full sweep), tab:mdd-branch (window=full == L5
     per-sequence), tab:mdd-aupr (branch=fused, S5, W in {1,64}))
  2. evaluate_mdd -> outputs/results/mdd_metrics.csv (per-frame),
     mdd_metrics_aggregated.csv (per-sequence) -- includes leave-one-out
     (`fused_no_<branch>` rows) -- covers tab:mdd-loo

Caller copies the resulting CSVs into unified_numbers/ afterward -- this
script does not touch that directory.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import LazyPhiDict, load_pt, materialize_f32

cfg.SEVERITIES = [1, 2, 3, 4, 5]

SPATIAL_DIR = cfg.REPO_ROOT / "phi_spatial"
_dedupe = re.compile(r" \(\d+\)$")
_spatial_files = {_dedupe.sub("", f.stem): f for f in SPATIAL_DIR.glob("*.pt")}
print(f"[run_mdd_paper_tables] SEVERITIES={cfg.SEVERITIES}; "
      f"{len(_spatial_files)} runs found in {SPATIAL_DIR}: {sorted(_spatial_files)}")


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

if __name__ == "__main__":
    import analysis.evaluate_mdd_windows as s1
    import analysis.evaluate_mdd as s2

    print("\n===== [1/2] evaluate_mdd_windows =====")
    s1.main()
    print("\n===== [2/2] evaluate_mdd (leave-one-out) =====")
    s2.main()

    print("\nAll stages complete.")
