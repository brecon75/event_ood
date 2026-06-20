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

> **GPU utilization (extraction).** Stage 1 runs the SNN forward pass at **batch size 1** (a hard SpikingJelly constraint — the batch axis is the time axis), so a *single* extraction process cannot saturate a large GPU. You fill the GPU by **packing multiple workers per GPU** with `--workers-per-gpu`; each process splits the card's VRAM (`0.95 / workers`). Raise it while watching `nvidia-smi` until GPU-util saturates and before CPU-RAM / disk-IO becomes the limit. (The downstream scoring stages are already GPU-offloaded — see §5.)

*Logs for individual parallel worker processes are saved under `vmem_benchmark/outputs/logs/worker_<id>.log` to prevent interleaved console prints.*

---

## 4. Combined φ-Analysis Runner (read φ once)

Each φ-consuming analysis stage normally launches as its own process and **independently re-reads every run's φ from disk** — at full scale the same ~90 GB of φ is read 5–7× over. `analysis/run_phi_stages.py` runs those stages **inside one process sharing a single resident loader**, so each run's φ is read from disk **exactly once** and reused by every stage (verified: every run loads once across all 8 stages). It modifies no stage script — it builds one shared loader and injects it into each stage at runtime.

It covers the eight φ-scoring/fit stages: `fit_detectors`, `eval_detectors`, `mdd`, `representation_ablation`, `severity`, `reliability`, `cross_corruption`, `analyse` (always run in this dependency order).

```powershell
# All eight stages, φ read once (full residency)
python analysis/run_phi_stages.py

# Quick end-to-end smoke test (subsamples φ)
python analysis/run_phi_stages.py --fast

# WHITELIST — run only these stages
python analysis/run_phi_stages.py --stages eval_detectors mdd analyse

# BLACKLIST — run everything EXCEPT these
python analysis/run_phi_stages.py --skip analyse reliability

# Combine them (whitelist applied first, then blacklist)
python analysis/run_phi_stages.py --stages eval_detectors mdd analyse --skip analyse

# Cap RAM on a smaller node (LRU; 'clean' always pinned)
python analysis/run_phi_stages.py --max-resident 8
```

| Flag | Effect |
|---|---|
| `--stages A B C …` | **Whitelist** — run only these stages (default: all). |
| `--skip X Y …` | **Blacklist** — run every stage except these (applied after `--stages`). |
| `--output-dir PATH` | Run directory holding `phi/` and receiving all outputs (`detectors/`, `tables/`, `results/`, `plots/`). Rebases every path. |
| `--ram-frac F` | Fraction of **available host RAM** the resident φ cache may use (default `0.85`). Higher = more runs held in RAM = fewer disk re-reads. |
| `--vram-frac F` | Fraction of **free VRAM** each GPU chunk is sized to (default `0.40`). Higher = bigger chunks = better GPU saturation; self-tunes via OOM-halving. |
| `--max-resident N` | Keep at most `N` runs resident (LRU, `clean` pinned). Overrides the RAM-aware default. Evicted runs are re-read from disk (mmap). |
| `--no-color` | Disable the coloured / de-duplicated console output. |
| `--fast` | Smoke-test subsample (propagates to every stage). |

The console output is colourised and **collapses consecutive duplicate lines** (e.g. a detector's per-run fit message that repeats dozens of times becomes one line + a `repeated xN` summary). Colour auto-disables when output is redirected to a file, so logs stay clean.

Both stage selectors validate against the known stage names, so a typo errors out instead of silently running nothing; if the filters leave no stages it prints a clean message and exits. A failing stage is reported but does **not** abort the rest, and a per-stage timing summary is printed at the end.

**Notes & prerequisites:**
- **RAM-aware by default (host-RAM OOM guard).** Full residency holds every run's φ in **host RAM** (~90 GB at full scale). The GPU scoring path is VRAM-OOM-safe (auto-chunked), but host RAM is **not** auto-protected — so on startup the runner estimates the φ footprint (from the `.pt` headers, no data read), detects available RAM, and if full residency would exceed ~70% of it, **auto-caps residency** to fit and prints the decision. Evicted runs are re-read from disk (mmap). It prints `phi footprint`, `host RAM available`, and the chosen `resident cache` so you can see the call; override anytime with `--max-resident`. If RAM can't be detected it holds all and warns. Note this bounds the φ *cache* only — each stage still allocates its own transient RAM, which the 30% headroom reserves; it is a strong guard, not an absolute guarantee. The standalone per-stage scripts also still work unchanged.
- **Run producers first:** this runner only covers the φ-scoring/fit stages. The producer stages (extract, offline/temporal features, fusion, ANN baselines — §2/§3) must already have written their artifacts.

---

## 5. I/O Behaviour & Optimizations

The φ files store both `phi` (~3 GB/run) and `phi_spatial` (~2 GB/run, used only by the MDD) in one archive. All φ loads are **memory-mapped** (`torch.load(mmap=True)` with a graceful fallback), so a stage faults in **only the tensor it actually uses**:
- φ-only stages no longer read the ~2 GB/run of `phi_spatial` they never touch (~40% fewer bytes/run).
- `seq_lens` metadata is read without deserializing the multi-GB φ tensors, and is memoized in-process.

Combined with §4's single-load runner, this removes both the per-stage waste (unused `phi_spatial`, whole-file metadata reads) and the cross-stage re-reads. See `Docs/bottlenecks.md` for the full per-stage I/O / GPU / RAM breakdown.
