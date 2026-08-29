"""Trimmed continuation of `run_unified_numbers_allsev.py`: stages 1-3 already
completed and wrote their CSVs (mdd_split_*, mdd_combine_clean_stats,
mdd_combine_window_sweep) before the prior run was interrupted. Stages 4-5
(mdd_alpha_sweep, mdd_softmax_sweep) are now OBSOLETE -- 2026-08-29,
mdd.py's fusion_alpha default reverted to 0.0 (plain max), the alpha/tau
fusion exploration they supported is no longer part of the shipped detector.

Runs only the remaining stages, same overrides (new phi_spatial,
SEVERITIES=[1..5]) as the full driver:
  6. mdd_max_ci --pool sensitivity -> mdd_max_ci_sensitivity.csv, mdd_max_auroc_allsev_sensitivity.csv
  7. mdd_max_ci --pool final       -> mdd_max_ci_final.csv, mdd_max_auroc_allsev_final.csv
  8. mdd_hparam_sensitivity        -> mdd_hparam_sensitivity_sensitivity.csv
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
print(f"[run_unified_numbers_allsev_ci] SEVERITIES={cfg.SEVERITIES}; "
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
    import analysis.mdd_max_ci as s6
    import analysis.mdd_hparam_sensitivity as s7

    sys.argv = ["mdd_max_ci.py", "--pool", "sensitivity"]
    print("\n===== [6/8] mdd_max_ci --pool sensitivity =====")
    s6.main()
    sys.argv = ["mdd_max_ci.py", "--pool", "final"]
    print("\n===== [7/8] mdd_max_ci --pool final =====")
    s6.main()

    sys.argv = ["mdd_hparam_sensitivity.py"]
    print("\n===== [8/8] mdd_hparam_sensitivity =====")
    s7.main()

    print("\nAll stages complete.")
