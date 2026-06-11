import sys
import os
import argparse
import subprocess
from pathlib import Path
import time

# Resolve paths
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import benchmark_config as cfg


def plan_extraction(gpus, workers_per_gpu, corruptions, vram_fraction=None):
    """Pure scheduling policy, separated from main() so it is unit-testable
    without spawning subprocesses.

    Returns (workers, chunks, vram_frac):
      workers    list of {"gpu_id", "device"} (gpus x workers_per_gpu, or
                 workers_per_gpu CPU workers when gpus is empty)
      chunks     corruptions round-robin'd across workers (len == len(workers))
      vram_frac  per-process VRAM fraction (None on CPU unless overridden)
    """
    if gpus:
        workers = [{"gpu_id": g, "device": "cuda"}
                   for g in gpus for _ in range(workers_per_gpu)]
    else:
        workers = [{"gpu_id": None, "device": "cpu"}
                   for _ in range(workers_per_gpu)]

    num_workers = len(workers)
    chunks = [[] for _ in range(num_workers)]
    for idx, corr in enumerate(corruptions):
        chunks[idx % num_workers].append(corr)

    vram_frac = vram_fraction
    if vram_frac is None and gpus:
        vram_frac = max(0.1, min(1.0, 0.95 / workers_per_gpu))
    return workers, chunks, vram_frac


def main():
    parser = argparse.ArgumentParser(
        description="Parallel extraction runner for Vmem robustness benchmark on multiple GPUs/processes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gpus", type=int, nargs="+", help="List of GPU IDs to use (e.g. 0 1 2 3). If omitted, auto-detects via PyTorch.")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Number of parallel processes to launch per GPU")
    parser.add_argument("--max-seq", type=int, default=-2, help="Max sequences to process per run (set -1 or 0 for no cap)")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--vram-fraction", type=float, help="Fraction of VRAM allocated per process (defaults to 0.95 / workers-per-gpu)")
    
    args, unknown = parser.parse_known_args()
    
    # 1. Determine GPU list
    import torch
    n_gpus = torch.cuda.device_count()
    if args.gpus is not None:
        gpus = args.gpus
    else:
        if n_gpus > 0:
            gpus = list(range(n_gpus))
        else:
            gpus = [] # Will run on CPU
            
    # 2-4. Workers, corruption partition, VRAM fraction (pure policy helper)
    workers, chunks, vram_frac = plan_extraction(
        gpus, args.workers_per_gpu, cfg.CORRUPTIONS, args.vram_fraction)
    num_workers = len(workers)
    if num_workers == 0:
        print("Error: No workers defined.")
        return

    print("=" * 60)
    print("  VMEM BENCHMARK: PARALLEL EXTRACTION LAUNCHER")
    print("=" * 60)
    print(f"  Available GPUs   : {n_gpus}")
    print(f"  Target GPUs      : {gpus if gpus else 'CPU'}")
    print(f"  Workers per GPU  : {args.workers_per_gpu}")
    print(f"  Total Workers    : {num_workers}")
    if gpus:
        print(f"  VRAM per Process : {vram_frac * 100:.1f}%")
    print("-" * 60)

    # 5. Create logs directory
    log_dir = cfg.OUTPUT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    processes = []
    
    # Path to Python interpreter and extract.py
    python_exe = sys.executable
    extract_py = _HERE / "extract.py"
    
    for w_idx, worker in enumerate(workers):
        corr_subset = chunks[w_idx]
        
        # Build command arguments
        cmd_args = [python_exe, str(extract_py)]
        cmd_args.extend(unknown)
        
        if corr_subset:
            cmd_args.extend(["--corruptions"] + corr_subset)
        else:
            # If a worker has no corruptions assigned, it only runs clean (if worker 0) or nothing (if worker > 0)
            if w_idx > 0:
                print(f"  [Worker {w_idx}] No corruptions assigned, skipping worker process.")
                continue
            # argparse nargs='+' rejects an empty --corruptions list, so use
            # the dedicated flag for a clean-only pass.
            cmd_args.append("--clean-only")
            
        # If the clean run or corruptions are already present on disk, they will be skipped automatically by extract.py.
            
        if args.max_seq != -2:
            cmd_args.extend(["--max-seq", str(args.max_seq)])
        if args.split:
            cmd_args.extend(["--split", args.split])
        if worker["device"] == "cuda":
            cmd_args.extend(["--device", "cuda"])
            if vram_frac is not None:
                cmd_args.extend(["--vram-fraction", f"{vram_frac:.4f}"])
        else:
            cmd_args.extend(["--device", "cpu"])
            
        if w_idx > 0:
            cmd_args.append("--skip-clean")
            
        # Set environment variables for this worker (e.g. CUDA_VISIBLE_DEVICES)
        env = os.environ.copy()
        if worker["gpu_id"] is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(worker["gpu_id"])
            
        log_file = log_dir / f"worker_{w_idx}.log"
        print(f"  [Worker {w_idx}] Device: {worker['device']}:{worker['gpu_id'] if worker['gpu_id'] is not None else ''} | Corruptions: {corr_subset if corr_subset else 'None'} | Log: {log_file.name}")
        
        # Open log file and spawn subprocess
        f_log = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd_args,
            env=env,
            stdout=f_log,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append((proc, f_log, w_idx, log_file))
        
    print("=" * 60)
    print("  Workers spawned. Monitoring progress...")
    print("  (Check worker logs in outputs/logs/ for detailed output.)\n")
    
    # 6. Monitor processes
    active_processes = list(processes)
    start_time = time.time()

    try:
        while active_processes:
            time.sleep(5)
            still_active = []
            for proc, f_log, w_idx, log_path in active_processes:
                ret = proc.poll()
                if ret is None:
                    still_active.append((proc, f_log, w_idx, log_path))
                else:
                    f_log.close()
                    elapsed = time.time() - start_time
                    if ret == 0:
                        print(f"  [Worker {w_idx}] Finished successfully in {elapsed:.1f}s.")
                    else:
                        print(f"  [Worker {w_idx}] FAILED with exit code {ret} (check {log_path.name}).")
            active_processes = still_active
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers...")
        for proc, f_log, w_idx, _ in active_processes:
            if proc.poll() is None:
                proc.terminate()
        for proc, f_log, w_idx, _ in active_processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
            f_log.close()
            print(f"  [Worker {w_idx}] terminated.")
        return

    print("\nAll parallel extraction workers completed.")

if __name__ == "__main__":
    main()
