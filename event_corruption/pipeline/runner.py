"""
Orchestrates all 6 corruptions × 5 severities across every sequence.

GPU strategy (when CUDA is available):
    - Upload histogram to GPU once per sequence
    - Run all (corruption, severity) variants on-device
    - Download each result to CPU only for saving
    - This avoids 30× redundant CPU array ops on ~1.74 GB arrays

CPU fallback is automatic when CuPy / CUDA is not available.
"""
import hashlib
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from corrupt.registry   import CORRUPTIONS, SEVERITIES, apply_corruption
from corrupt.cuda_utils import cuda_available, to_gpu, to_cpu, clear_gpu_memory
from pipeline.loader    import (
    load_histogram,
    load_labels,
    load_repr_index,
    load_label_timestamps,
)
from pipeline.saver     import save_corrupted

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

def _make_seed(base_seed: int, corruption: str, severity: int, seq_name: str) -> int:
    """
    MD5-based seed derivation — portable across Python versions and
    PYTHONHASHSEED settings (unlike built-in hash()).
    """
    key = f"{base_seed}|{corruption}|{severity}|{seq_name}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % (2 ** 31)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all_corruptions(
    input_dir     : Path,
    output_dir    : Path,
    seed          : int        = 42,
    split         : str        = "",
    corruptions   : list | None = None,
    severities    : list | None = None,
    skip_existing : bool       = True,
) -> None:
    """
    Apply all (or a subset of) corruptions × severities to every sequence.

    Parameters
    ----------
    input_dir     : Gen1 split directory containing timestamped sequence dirs
    output_dir    : root output path; sub-dirs created automatically
    seed          : base RNG seed (default 42)
    split         : label for log messages (e.g. "test")
    corruptions   : subset of CORRUPTIONS to run; None → all 6
    severities    : subset of SEVERITIES to run; None → all [1..5]
    skip_existing : skip (corruption, severity, seq) if output already exists
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)

    corruptions = corruptions or list(CORRUPTIONS.keys())
    severities  = severities  or list(SEVERITIES)

    # Discover sequences by looking for labels.npz files
    label_files = sorted(input_dir.glob("*/labels_v2/labels.npz"))
    seq_dirs    = [p.parent.parent for p in label_files]

    if not seq_dirs:
        raise FileNotFoundError(
            f"No sequences found under {input_dir}. "
            "Expected '<seq>/labels_v2/labels.npz' pattern."
        )

    use_cuda = cuda_available()
    device   = "GPU (CuPy)" if use_cuda else "CPU (NumPy)"

    print(
        f"[{split or input_dir.name}]  "
        f"{len(seq_dirs)} sequences | "
        f"{len(corruptions)} corruptions | "
        f"{len(severities)} severities | "
        f"device: {device}"
    )

    total_seqs = len(seq_dirs)

    pbar = tqdm(seq_dirs, desc=f"[{split or input_dir.name}]", unit="seq")
    for seq_idx, seq_dir in enumerate(pbar, 1):
        seq_name = seq_dir.name
        t0 = time.perf_counter()

        # --- Load from disk (always CPU) ---
        histogram_cpu, timestamps = load_histogram(seq_dir)
        labels, label_idx         = load_labels(seq_dir)
        repr_idx                  = load_repr_index(seq_dir)
        label_ts                  = load_label_timestamps(seq_dir)

        # --- Upload to GPU once (if available) ---
        histogram = to_gpu(histogram_cpu)
        t_load = time.perf_counter() - t0

        n_done = 0
        for corruption_name in corruptions:
            for severity in severities:
                out_dir = output_dir / corruption_name / str(severity) / seq_name

                if skip_existing and (out_dir / "labels_v2" / "labels.npz").exists():
                    logger.debug("Skip (exists): %s", out_dir)
                    continue

                rng = np.random.default_rng(
                    _make_seed(seed, corruption_name, severity, seq_name)
                )

                # --- Corruption runs on GPU (or CPU if no CUDA) ---
                try:
                    corrupted_device = apply_corruption(
                        histogram, timestamps, corruption_name, severity, rng
                    )
                    # --- Download to CPU for saving ---
                    corrupted_cpu = to_cpu(corrupted_device)
                    del corrupted_device  # Explicitly free GPU reference
                except Exception as e:
                    if "out of memory" in str(e).lower():
                        logger.warning(f"GPU OOM for {seq_name} {corruption_name} s={severity}. Falling back to CPU.")
                        # Fallback to CPU for this variant
                        corrupted_cpu = apply_corruption(
                            histogram_cpu, timestamps, corruption_name, severity, rng
                        )
                    else:
                        raise e

                save_corrupted(
                    corrupted_cpu, timestamps,
                    labels, label_idx,
                    repr_idx, label_ts,
                    out_dir,
                )
                del corrupted_cpu
                n_done += 1
                
                # Periodically clear GPU memory to prevent fragmentation
                if n_done % 5 == 0:
                    clear_gpu_memory()

        # Cleanup GPU memory after each sequence
        del histogram
        clear_gpu_memory()

        t_total = time.perf_counter() - t0
        pbar.set_description(f"[{split or input_dir.name}] {seq_name} ({t_total:.1f}s)")

    print("Done.")
