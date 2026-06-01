"""
gpu_test.py — GPU vs CPU benchmark for all 6 corruptions on one real sequence.

Run with:
    .venv\Scripts\python.exe gpu_test.py
"""
import sys, time
sys.path.insert(0, '.')

import numpy as np
import cupy as cp

from pathlib import Path
from pipeline.loader   import load_histogram
from corrupt.registry  import apply_corruption, CORRUPTIONS
from corrupt.cuda_utils import to_gpu, to_cpu, cuda_available

SEQ = Path(
    "d:/Perdue/gen1/test/"
    "17-04-04_11-00-13_cut_15_122500000_182500000"
)
SEVERITY = 3


def main():
    print("=== GPU CORRUPTION TEST ===")
    print(f"CUDA available : {cuda_available()}")
    if not cuda_available():
        print("ERROR: CuPy / CUDA not found. Install cupy-cuda12x.")
        sys.exit(1)

    props = cp.cuda.runtime.getDeviceProperties(0)
    name  = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    vram  = props["totalGlobalMem"] / 1024**3
    print(f"GPU            : {name}  ({vram:.1f} GB VRAM)")
    print()

    # ------------------------------------------------------------------ Load
    t0 = time.perf_counter()
    hist_cpu, ts = load_histogram(SEQ)
    t_load = time.perf_counter() - t0
    mb = hist_cpu.nbytes / 1024**2
    print(f"Load from disk : {t_load:.2f}s   shape={hist_cpu.shape}   {mb:.0f} MB")

    # --------------------------------------------------------- Upload to GPU
    t0 = time.perf_counter()
    hist_gpu = to_gpu(hist_cpu)
    cp.cuda.Stream.null.synchronize()
    t_up = time.perf_counter() - t0
    print(f"Upload to GPU  : {t_up*1000:.1f} ms")
    print()

    # ----------------------------------------- Per-corruption GPU vs CPU ---
    header = f"{'Corruption':<22}  {'GPU (ms)':>9}  {'CPU (ms)':>9}  {'Speedup':>8}  {'Match':>5}"
    print(header)
    print("-" * len(header))

    all_ok    = True
    total_gpu = 0.0
    total_cpu = 0.0

    for corruption_name in CORRUPTIONS:
        # -- GPU run --
        rng_g = np.random.default_rng(42)
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        out_gpu = apply_corruption(hist_gpu, ts, corruption_name, SEVERITY, rng_g)
        cp.cuda.Stream.null.synchronize()
        t_gpu = (time.perf_counter() - t0) * 1000

        # -- CPU run --
        rng_c = np.random.default_rng(42)
        t0 = time.perf_counter()
        out_cpu = apply_corruption(hist_cpu, ts, corruption_name, SEVERITY, rng_c)
        t_cpu = (time.perf_counter() - t0) * 1000

        # -- Verify GPU == CPU --
        out_gpu_np = to_cpu(out_gpu)
        match  = np.array_equal(out_gpu_np, out_cpu)
        ok_str = "YES" if match else "NO !"
        if not match:
            all_ok = False

        speedup = t_cpu / t_gpu if t_gpu > 0 else float("inf")
        total_gpu += t_gpu
        total_cpu += t_cpu

        print(
            f"{corruption_name:<22}  {t_gpu:>9.1f}  {t_cpu:>9.1f}"
            f"  {speedup:>7.1f}x  {ok_str:>5}"
        )

    # --------------------------------------------------------- Totals ------
    print("-" * len(header))
    overall_speedup = total_cpu / total_gpu if total_gpu > 0 else float("inf")
    print(
        f"{'TOTAL (all 6 corruptions)':<22}  {total_gpu:>9.1f}  {total_cpu:>9.1f}"
        f"  {overall_speedup:>7.1f}x"
    )
    print()

    # Download timing
    t0 = time.perf_counter()
    _ = to_cpu(out_gpu)
    cp.cuda.Stream.null.synchronize()
    t_down = (time.perf_counter() - t0) * 1000
    print(f"Download result: {t_down:.1f} ms")
    print()

    # Estimate full run time
    n_severities   = 5
    n_corruptions  = 6
    n_variants     = n_severities * n_corruptions      # = 30
    save_est_ms    = 800                               # ~800ms per variant (write ~1.62GB)

    # GPU path: upload once + n_variants*(gpu_per_corruption + download + save)
    gpu_per_variant_ms = (total_gpu / 6) * 1         # avg per corruption, 1 severity
    gpu_total_ms = (t_up * 1000) + n_variants * (gpu_per_variant_ms + t_down + save_est_ms)
    # CPU path: n_variants * cpu_per_corruption
    cpu_per_variant_ms = total_cpu / 6
    cpu_total_ms = n_variants * (cpu_per_variant_ms + save_est_ms)

    print(f"Estimated time per sequence ({n_variants} variants):")
    print(f"  GPU path : {gpu_total_ms/1000:.1f}s")
    print(f"  CPU path : {cpu_total_ms/1000:.1f}s")
    print(f"  Speedup  : {cpu_total_ms/gpu_total_ms:.1f}x")
    print()
    print("RESULT:", "ALL PASSED" if all_ok else "SOME FAILURES — check Match column")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
