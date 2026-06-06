"""
run_test_pipeline.py — End-to-end integration test runner.

Runs all pipeline steps on a minimal dataset (1 sequence, 1 corruption, severity 5)
to verify the complete extract → analyse → report chain works correctly.

Strategy: Rather than rewriting benchmark_config.py in place, we:
  1. Append test overrides to benchmark_config.py (Python last-write wins)
  2. Always restore the backup in the finally block
  3. Pass explicit CLI flags to extract.py so it doesn't depend on config at all
"""
import subprocess
import sys
import shutil
from pathlib import Path

PYTHON    = str(Path("d:/Perdue/vmem_benchmark/.venv/Scripts/python.exe"))
CWD       = "d:/Perdue"
TEST_OUT  = Path("d:/Perdue/vmem_benchmark/test_outputs")
LOG_PATH  = Path("d:/Perdue/test_pipeline_live.log")
CFG_PATH  = Path("vmem_benchmark/benchmark_config.py")
BAK_PATH  = Path("vmem_benchmark/benchmark_config.py.bak")


def run_cmd(args, log_file):
    log_file.write(f"\n{'='*70}\n")
    log_file.write(f"STEP: {' '.join(str(a) for a in args)}\n")
    log_file.write(f"{'='*70}\n")
    log_file.flush()
    print(f"Running: {' '.join(str(a) for a in args)}")

    res = subprocess.run(
        args, stdout=log_file, stderr=log_file,
        text=True, cwd=CWD
    )
    if res.returncode != 0:
        print(f"\n[ERROR] Exit code {res.returncode} — check {LOG_PATH}")
        raise RuntimeError(f"Step failed: {' '.join(str(a) for a in args)}")


def write_test_config_overrides(cfg_path: Path):
    """Append test-mode overrides to benchmark_config.py (Python last-write wins)."""
    with open(cfg_path, 'a', encoding='utf-8') as f:
        f.write("\n\n# --- TEST PIPELINE OVERRIDES (auto-appended by run_test_pipeline.py) ---\n")
        f.write('OUTPUT_DIR       = Path("d:/Perdue/vmem_benchmark/test_outputs")\n')
        f.write('PHI_DIR          = OUTPUT_DIR / "phi"\n')
        f.write('TRAJ_DIR         = OUTPUT_DIR / "trajs"\n')
        f.write('PLOT_DIR         = OUTPUT_DIR / "plots"\n')
        f.write('TEMPORAL_PHI_DIR = OUTPUT_DIR / "temporal_phi"\n')
        f.write('ANN_DIR          = OUTPUT_DIR / "ann_features"\n')
        f.write('SPIKE_DIR        = OUTPUT_DIR / "spike"\n')
        f.write('DETECTOR_DIR     = OUTPUT_DIR / "detectors"\n')
        f.write('MAX_SEQUENCES    = 1\n')
        f.write('CORRUPTIONS      = ["hot_pixel"]\n')
        f.write('SEVERITIES       = [5]\n')


def main():
    print("=" * 80)
    print(" VMEM BENCHMARK - FULL PIPELINE INTEGRATION TEST")
    print("=" * 80)

    # ── CUDA Device Check ─────────────────────────────────────────────────────
    try:
        import torch
        import sys
        sys.path.insert(0, "d:/Perdue")
        from vmem_benchmark import benchmark_config as cfg
        cuda_avail = torch.cuda.is_available()
        configured_device = cfg.DEVICE
        print(f"[CUDA Status] PyTorch CUDA available: {cuda_avail}")
        print(f"[CUDA Status] Configured device: {configured_device}")
        if configured_device == "cuda" and cuda_avail:
            print("[CUDA Status] CUDA is active and will be used for GPU-accelerated steps.")
        else:
            print("[CUDA Status] WARNING: CUDA is NOT active or NOT available; running steps on CPU.")
    except Exception as e:
        print(f"[CUDA Status] Could not check CUDA availability: {e}")

    # ── 1. Fresh log ──────────────────────────────────────────────────────────
    LOG_PATH.write_text("=== VMEM BENCHMARK INTEGRATION PIPELINE TEST LOG ===\n",
                        encoding='utf-8')
    print(f"Live output -> {LOG_PATH}")

    # ── 2. Clean test output directory ───────────────────────────────────────
    if TEST_OUT.exists():
        print(f"Cleaning {TEST_OUT} ...")
        shutil.rmtree(TEST_OUT, ignore_errors=True)
    TEST_OUT.mkdir(parents=True, exist_ok=True)

    # ── 3. Backup + override config ──────────────────────────────────────────
    print("Backing up benchmark_config.py ...")
    shutil.copy2(CFG_PATH, BAK_PATH)

    log_file = open(LOG_PATH, 'a', encoding='utf-8')
    try:
        write_test_config_overrides(CFG_PATH)
        print("Config overrides appended.")

        steps = [
            # Comprehensive unit/sanity tests
            [PYTHON, "vmem_benchmark/test_pipeline.py"],

            # extract.py accepts CLI flags directly — no config rewriting needed
            [PYTHON, "vmem_benchmark/extract.py",
             "--max-seq", "1",
             "--corruptions", "hot_pixel",
             "--severities", "5",
             "--output-dir", str(TEST_OUT)],

            # All analysis scripts read cfg which now has test_outputs baked in
            [PYTHON, "analysis/extract_offline_features.py"],
            [PYTHON, "analysis/fusion_features.py"],
            [PYTHON, "analysis/extract_ann_baselines.py"],
            [PYTHON, "analysis/evaluate_ann_baselines.py"],
            [PYTHON, "analysis/fit_detectors.py"],
            [PYTHON, "analysis/evaluate_detectors.py"],
            [PYTHON, "analysis/representation_ablation.py"],
            [PYTHON, "analysis/severity.py"],
            [PYTHON, "analysis/reliability.py"],
            [PYTHON, "analysis/cross_corruption.py"],
            [PYTHON, "analysis/free_rider_ablation.py"],
            [PYTHON, "analysis/analyse.py", "--fast"],
            [PYTHON, "reporting/build_paper_tables.py"],
            [PYTHON, "reporting/build_paper_figures.py"],
        ]

        for i, step in enumerate(steps, 1):
            print(f"\n>>> [{i}/{len(steps)}] Running step ...")
            run_cmd(step, log_file)

        print("\n" + "=" * 80)
        print("SUCCESS: Full pipeline integration test passed!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[FAILED] {e}")
        sys.exit(1)

    finally:
        log_file.close()
        print("\nRestoring benchmark_config.py ...")
        if BAK_PATH.exists():
            shutil.copy2(BAK_PATH, CFG_PATH)
            BAK_PATH.unlink()
            print("Config restored.")


if __name__ == "__main__":
    main()
