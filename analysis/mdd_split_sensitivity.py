"""Stage: corruption-free MDD split + per-branch score dump.

Splits the CLEAN phi stream into four sequence-aligned, non-overlapping pools
(no corrupted run is loaded anywhere in this script):

    fit (50%)         branch statistics (covariance, PCA basis, RCF reference)
    calib (10%)       branch score standardization (median/MAD)
    sensitivity (10%) hyperparameter sensitivity checks -- clean-only diagnostics,
                      never scored against a corruption label
    final (30%)       reserved for the one-shot reported result (untouched here)

Every cut lands on a whole-sequence boundary (`split_boundaries`), so no
sequence contributes frames to more than one pool. Fits ONE MDD on
fit+calib, then scores every pool (including fit/calib themselves, for
reference) and writes one CSV per pool with the per-frame, per-branch
CALIBRATED scores (radius/rcf/l4/spatial) plus a `seq_id` column so the
split can be audited directly from the CSVs. The `fused` score MDD.
score_branches computes internally is deliberately dropped -- these CSVs are
raw per-branch scores only, so combining methods can be tried later without
MDD's built-in studentized-max fusion baked in.

Outputs (under outputs/results/): mdd_split_<fit|calib|sensitivity|final>.csv
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    LazyPhiDict, split_boundaries, POOL_NAMES, POOL_FRACS)
from analysis.mdd import MDD

# POOL_NAMES/POOL_FRACS now live in vmem_utils (the canonical home, so the
# representation/detector/information stages can share them without importing
# an MDD module). Re-exported here: every existing `from analysis
# .mdd_split_sensitivity import POOL_NAMES, POOL_FRACS` keeps working.
__all__ = ["POOL_NAMES", "POOL_FRACS", "main"]


def _seq_ids(seq_lens):
    """Global 0-based sequence index per frame, e.g. [0,0,0,1,1,2,...]."""
    return np.repeat(np.arange(len(seq_lens)), seq_lens)


def main():
    print("MDD split + sensitivity score dump (clean-only, no corruption runs loaded)...")
    all_phi = LazyPhiDict(cache_size=1)
    if "clean" not in all_phi:
        print(f"Error: clean.pt missing from {cfg.PHI_DIR}. Run extract.py first.")
        return

    clean = all_phi["clean"]
    seq_lens = all_phi.get_seq_lens("clean")
    n = len(clean)
    if not seq_lens:
        print("Error: clean.pt has no seq_lens (legacy extraction) -- cannot "
              "guarantee a leak-free sequence-aligned split.")
        return

    cuts = split_boundaries(n, POOL_FRACS, seq_lens)
    edges = np.cumsum(seq_lens)
    for c in cuts:
        assert c in edges, f"cut {c} does not land on a sequence boundary"
    bounds = [0] + cuts + [n]
    ranges = {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(POOL_NAMES)}

    sids = _seq_ids(seq_lens)
    spatial = all_phi.get_phi_spatial("clean")
    use_spatial = spatial is not None

    # leak check: every sequence id appears in exactly one pool.
    seen = {}
    print(f"Clean split ({n} frames, {len(seq_lens)} sequences):")
    for name, (a, b) in ranges.items():
        pool_sids = set(sids[a:b].tolist())
        for sid in pool_sids:
            assert sid not in seen, f"sequence {sid} leaked into both {seen[sid]} and {name}"
            seen[sid] = name
        print(f"  {name:12s} frames={b - a:6d}  sequences={len(pool_sids):4d}")
    assert len(seen) == len(seq_lens), "not every sequence was assigned to a pool"
    print("  Leak check passed: every sequence appears in exactly one pool.")

    phi_fit, phi_calib = clean[ranges["fit"][0]:ranges["fit"][1]], clean[ranges["calib"][0]:ranges["calib"][1]]
    sp_fit = spatial[ranges["fit"][0]:ranges["fit"][1]] if use_spatial else None
    sp_calib = spatial[ranges["calib"][0]:ranges["calib"][1]] if use_spatial else None

    mdd = MDD(use_spatial=use_spatial).fit(phi_fit, phi_calib, sp_fit, sp_calib)
    print(f"MDD branches: {mdd.branch_names}")

    res_dir = cfg.OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    for name, (a, b) in ranges.items():
        phi_pool = clean[a:b]
        sp_pool = spatial[a:b] if use_spatial else None
        scores = mdd.score_branches(phi_pool, sp_pool)
        scores.pop("fused", None)   # raw per-branch scores only -- fusion happens downstream
        df = pd.DataFrame({"row": np.arange(a, b), "seq_id": sids[a:b]})
        for branch, vals in scores.items():
            df[branch] = vals
        out = res_dir / f"mdd_split_{name}.csv"
        df.to_csv(out, index=False)
        print(f"  Wrote {out} ({len(df)} rows, columns: {list(df.columns)})")


if __name__ == "__main__":
    main()
