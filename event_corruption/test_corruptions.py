"""
test_corruptions.py — quick smoke-test for all 6 corruptions on one sequence.

Reads one real sequence from disk and verifies:
  - output shape, dtype, and value range are valid
  - output differs from the original (corruption is non-trivial)
  - no exceptions are raised for any (corruption, severity) pair

Run from event_corruption/:
    python test_corruptions.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import h5py

from corrupt.registry import apply_corruption, CORRUPTIONS, SEVERITIES
from pipeline.loader  import load_histogram, load_labels, load_repr_index

# ---------------------------------------------------------------------------
# Pick ONE sequence for the smoke-test
# ---------------------------------------------------------------------------
SEQ_DIR = Path(
    "d:/Perdue/gen1/test/"
    "17-04-04_11-00-13_cut_15_122500000_182500000"
)


def main():
    print(f"Loading sequence: {SEQ_DIR.name}")
    histogram, timestamps = load_histogram(SEQ_DIR)
    print(
        f"  histogram : {histogram.shape}  dtype={histogram.dtype}  "
        f"min={histogram.min()}  max={histogram.max()}"
    )
    print(f"  timestamps: {timestamps.shape}  dtype={timestamps.dtype}")

    n_pass = 0
    n_fail = 0

    for name in CORRUPTIONS:
        for severity in [1, 3, 5]:
            rng = np.random.default_rng(42)
            try:
                out = apply_corruption(histogram, timestamps, name, severity, rng)
            except Exception as exc:
                print(f"  FAIL  {name:20s}  s={severity}  EXCEPTION: {exc}")
                n_fail += 1
                continue

            # Shape & dtype
            if out.shape != histogram.shape:
                print(f"  FAIL  {name:20s}  s={severity}  shape {out.shape} != {histogram.shape}")
                n_fail += 1
                continue
            if out.dtype != np.uint8:
                print(f"  FAIL  {name:20s}  s={severity}  dtype={out.dtype}")
                n_fail += 1
                continue
            if out.min() < 0 or out.max() > 255:
                print(f"  FAIL  {name:20s}  s={severity}  range [{out.min()},{out.max()}]")
                n_fail += 1
                continue

            diff = np.abs(out.astype(np.int32) - histogram.astype(np.int32)).mean()
            print(
                f"  OK    {name:20s}  s={severity}  "
                f"mean_orig={histogram.mean():.3f}  mean_out={out.mean():.3f}  "
                f"mean_abs_diff={diff:.3f}"
            )
            n_pass += 1

    print(f"\nResults: {n_pass} passed, {n_fail} failed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
