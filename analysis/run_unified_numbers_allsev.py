"""One-off driver: same as `run_unified_numbers_newspatial.py` (new 1488-D
phi_spatial override) but ALSO overrides `cfg.SEVERITIES` from the trimmed
[1, 3, 5] to the full [1, 2, 3, 4, 5] -- data for L2/L4 already exists in both
outputs/phi/ and phi_spatial/ for every corruption, it was just never scored
because SEVERITIES was cut down. This fills the S2/S4 gaps in the paper
tables reconstructed in unified_numbers/README.md.

Only the four severity-dependent stages actually change behavior:
mdd_combine_auroc, mdd_alpha_sweep, mdd_softmax_sweep, mdd_max_ci
(mdd_split_sensitivity, mdd_combine_experiments, mdd_hparam_sensitivity do
not iterate over cfg.SEVERITIES).

All writes land in outputs/results/ (each stage's own default), same as the
newspatial driver -- this script does not touch unified_numbers/ itself.
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
print(f"[run_unified_numbers_allsev] SEVERITIES={cfg.SEVERITIES}; "
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
