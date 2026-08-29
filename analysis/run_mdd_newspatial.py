"""One-off: run evaluate_mdd using the 1488-D phi_spatial re-extraction that
landed in the top-level phi_spatial/ folder (org stats: flatness + causal
persistence) instead of the stale 1408-D phi_spatial embedded in
outputs/phi/<run>.pt. Base phi (2112-D) still comes from outputs/phi/ as
usual; only the spatial branch source is swapped, via a monkeypatch of
LazyPhiDict.get_phi_spatial so evaluate_mdd.py itself is untouched.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import LazyPhiDict, load_pt, materialize_f32
import analysis.evaluate_mdd as em

SPATIAL_DIR = cfg.REPO_ROOT / "phi_spatial"
_dedupe = re.compile(r" \(\d+\)$")
_spatial_files = {_dedupe.sub("", f.stem): f for f in SPATIAL_DIR.glob("*.pt")}
print(f"[run_mdd_newspatial] {len(_spatial_files)} runs found in {SPATIAL_DIR}")


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
    em.main()
