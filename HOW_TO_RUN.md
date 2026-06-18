# HOW TO RUN THE SNN Robustness Benchmark

This guide outlines how to install dependencies, configure the environment, and execute the robustness evaluation pipeline.

---

## 1. Installation

Execute the following commands to initialize the Python virtual environment and install dependencies:

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Upgrade pip and install compatible setuptools/PyTorch
python -m pip install --upgrade pip wheel
pip install "setuptools==81.0.0"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install benchmark dependencies
pip install -r requirements.txt
```

---

## 2. Running the Full Pipeline

The benchmark can be executed end-to-end using the native runner scripts in the root directory. They run Stage 1 in parallel to speed up extraction, then run the downstream analysis and reporting scripts.

### On Windows (PowerShell):
```powershell
# Run with default settings (automatically uses parallel extraction on visible GPUs)
.\run_full_benchmark.ps1

# Run with custom paths and settings forwarded to the extraction engine
.\run_full_benchmark.ps1 --gpus 0 --workers-per-gpu 2 --gen1-root "C:/path/to/gen1" --output-dir "C:/path/to/outputs"
```

### On Linux / Bash:
```bash
# Run with default settings
./run_full_benchmark.sh

# Run with custom paths and settings
./run_full_benchmark.sh --gpus 0 1 2 3 --max-seq 200
```

### Quick validation (smoke test)
Before a full run, validate the whole pipeline end-to-end on **1 sequence × 2 corruptions × 1 severity** (~minutes). It temporarily caps the config, runs all stages, then restores the config:
```powershell
.\run_test_pipeline.ps1 --gpus 0 --workers-per-gpu 1
```

### Pipeline stages & key outputs
The runner executes 16 stages: parallel φ extraction → offline/temporal features → fusion → ResNet ANN baselines → OOD detector fit/eval → **MDD evaluation (Stage 8)** → ablations/severity/reliability/cross-corruption/free-rider → analysis plots → paper tables/figures. Results land under `outputs/results/`, including:
- `ood_metrics.csv` — the 7 fitted OOD detectors per corruption/severity.
- `mdd_metrics.csv` / `mdd_metrics_aggregated.csv` — the Manifold-Decomposition Detector (radius + RCF + deep-layer + spatial branches), per-frame and per-recording.

> **Note:** Stage 1 extraction now also saves `phi_spatial` (spatial-dispersion stats, float32) alongside `phi`, plus per-sequence `seq_lens`, in each `outputs/phi/<run>.pt`. `phi` is unchanged (2112-D), so existing analyses are unaffected; the spatial features require a fresh extraction.

> **Resourcing the full-data run (Stages 6–7).** The fitting/scoring stages train on the **full** clean set (no data caps; only GMM EM and the PCA-SVD keep fixed-size compute caps, which barely affect a 5-Gaussian / 64-direction estimate). This is a host-RAM and CPU fact, not a bug:
> - `fit_detectors.py` runs LedoitWolf, PCA and the kNN reference build on the full `X_train` in **host RAM / CPU** — `chunked_apply` only tiles GPU loops, so provision enough host RAM (full φ per run ≈ 2.9 GB plus sklearn copies). For the brute-force kNN query a faiss backend speeds it up *without* shrinking the reference.
> - `evaluate_detectors.py` now scores **every** detector (mahalanobis, gmm, ocsvm, pca, knn, ae, flow) on the **GPU** via VRAM-aware chunking. Previously ocsvm/gmm/mahalanobis/pca ran single-threaded on CPU — the OCSVM RBF `decision_function` alone dominated multi-day runtimes; the GPU path collapses that to GPU-bound time. The full reference is streamed to the GPU in blocks (never resident whole), so a small GPU won't OOM.

---

## 3. Running Parallel Extraction Manually

If you only want to run the parallel extraction stage (`Stage 1`) to generate feature files:

```powershell
# Run across 4 GPUs on a cluster node (one worker per GPU)
python vmem_benchmark/run_parallel_extract.py --gpus 0 1 2 3

# Run on a single GPU with 3 parallel processes (maximizes GPU utilization)
python vmem_benchmark/run_parallel_extract.py --gpus 0 --workers-per-gpu 3

# Override dataset paths and split
python vmem_benchmark/run_parallel_extract.py --gpus 0 --gen1-root "/path/to/gen1" --split test
```

*Logs for individual parallel worker processes are saved under `vmem_benchmark/outputs/logs/worker_<id>.log` to prevent interleaved console prints.*
