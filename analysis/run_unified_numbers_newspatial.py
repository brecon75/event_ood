"""One-off driver: regenerate the entire `unified_numbers/` pipeline using the
1488-D phi_spatial re-extraction in the top-level `phi_spatial/` folder (org
stats: flatness + causal persistence) instead of the stale 1408-D phi_spatial
embedded in outputs/phi/<run>.pt. Base phi (2112-D) still comes from
outputs/phi/ as usual; only the spatial branch source is swapped, via the
same LazyPhiDict.get_phi_spatial monkeypatch as run_mdd_newspatial.py, so none
of the individual stage scripts are touched.

Runs, in the dependency order documented in unified_numbers/README.md:
  1. mdd_split_sensitivity   -> mdd_split_{fit,calib,sensitivity,final}.csv
  2. mdd_combine_experiments -> mdd_split_sensitivity_combined.csv, mdd_combine_clean_stats.csv
  3. mdd_combine_auroc       -> mdd_combine_window_sweep.csv
  4. mdd_alpha_sweep         -> mdd_alpha_sweep.csv
  5. mdd_softmax_sweep       -> mdd_softmax_sweep.csv
  6. mdd_max_ci --pool sensitivity -> mdd_max_ci_sensitivity.csv, mdd_max_auroc_allsev_sensitivity.csv
  7. mdd_max_ci --pool final       -> mdd_max_ci_final.csv, mdd_max_auroc_allsev_final.csv
  8. mdd_hparam_sensitivity  -> mdd_hparam_sensitivity_sensitivity.csv

All writes land in outputs/results/ (each stage's own default). Caller copies
into unified_numbers/ afterward -- this script does not touch that directory.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import LazyPhiDict, load_pt, materialize_f32

SPATIAL_DIR = cfg.REPO_ROOT / "phi_spatial"
_dedupe = re.compile(r" \(\d+\)$")
_spatial_files = {_dedupe.sub("", f.stem): f for f in SPATIAL_DIR.glob("*.pt")}
print(f"[run_unified_numbers_newspatial] {len(_spatial_files)} runs found in {SPATIAL_DIR}: "
      f"{sorted(_spatial_files)}")


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
    import analysis.mdd_split_sensitivity as s1
    import analysis.mdd_combine_experiments as s2
    import analysis.mdd_combine_auroc as s3
    import analysis.mdd_alpha_sweep as s4
    import analysis.mdd_softmax_sweep as s5
    import analysis.mdd_max_ci as s6
    import analysis.mdd_hparam_sensitivity as s7

    print("\n===== [1/8] mdd_split_sensitivity =====")
    s1.main()
    print("\n===== [2/8] mdd_combine_experiments =====")
    s2.main()
    print("\n===== [3/8] mdd_combine_auroc =====")
    s3.main()
    print("\n===== [4/8] mdd_alpha_sweep =====")
    s4.main()
    print("\n===== [5/8] mdd_softmax_sweep =====")
    s5.main()

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
