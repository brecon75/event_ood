"""
config.py — Central configuration for the Vmem robustness benchmark.

=============================================================================
DEFAULT RUN CONFIGURATION DEFINITION:
-----------------------------------------------------------------------------
- Hooking Layers       : All layers hooked by default (PLIF_LAYERS = None)
- Dataset Split        : 'test' split (SPLIT = "test")
- Dataset Capping      : Full test split (MAX_SEQUENCES = 470)
- Benchmark Permutations: All 6 corruptions x 5 severity levels = 30 runs
                         (+1 clean baseline run = 31 total runs)
=============================================================================
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# !! EDIT THESE THREE PATHS !!
# ---------------------------------------------------------------------------
GEN1_ROOT  = Path("d:/Perdue/gen1")           # root of the Gen1 dataset
CKPT_PATH  = Path("d:/Perdue/HybridDetection/gen1_mAP36.ckpt")
HYBRID_DIR = Path("d:/Perdue/HybridDetection") # repo root (for sys.path)

# ---------------------------------------------------------------------------
# Output directory (auto-created)
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("d:/Perdue/vmem_benchmark/outputs")
PHI_DIR    = OUTPUT_DIR / "phi"
TRAJ_DIR   = OUTPUT_DIR / "trajs"
PLOT_DIR   = OUTPUT_DIR / "plots"

# ---------------------------------------------------------------------------
# Inference settings
# ---------------------------------------------------------------------------
DEVICE      = "cuda"   # or "cpu"
BATCH_SIZE  = 1        # histograms per batch (each ~5.6 MB on GPU)
TRAJ_SAVE_N = 50       # save trajectories for first N histogram frames only

# ---------------------------------------------------------------------------
# Sequence cap — stop after this many sequences. Set to None or 470 for the full split.
# ---------------------------------------------------------------------------
MAX_SEQUENCES  = 470    # process all 470 sequences in the test split

# ---------------------------------------------------------------------------
# Advanced memory management & saving:
# - PHI_SAVE_EVERY: Save partial results every N sequences. Enables
#   aggressive GC cleanup and mid-run recovery/resume on crashes.
# ---------------------------------------------------------------------------
PHI_SAVE_EVERY = 5     # save a phi chunk after every 5 sequences

# ---------------------------------------------------------------------------
# PLIF layers to monitor (0-indexed over all MultiStepParametricLIFNode in the
# backbone).  The backbone has exactly 4 PLIF nodes:
#   0 → features_01[0].neuron  (64 ch, 1/2 res)
#   1 → features_01[1].neuron  (128 ch, 1/4 res)
#   2 → features_23[0].neuron  (256 ch, 1/8 res)
#   3 → features_23[1].neuron  (256 ch, 1/8 res)
# None means hook all of them.
# ---------------------------------------------------------------------------
PLIF_LAYERS = None   # hook all 4; change to e.g. [2, 3] to reduce memory

# ---------------------------------------------------------------------------
# Corruption config — 6 types × 5 severity levels
# ---------------------------------------------------------------------------
CORRUPTIONS = [
    "hot_pixel",
    "event_flood",
    "temporal_jitter",
    "polarity_flip",
    "event_rate_shift",
    "spatial_dropout",
]

SEVERITIES = [1, 2, 3, 4, 5]

# Gen1 split to benchmark against
SPLIT = "test"
