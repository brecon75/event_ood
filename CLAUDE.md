# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Vmem-φ** is a research benchmark for out-of-distribution (OOD) detection in Spiking Neural Networks (SNNs) using membrane potential statistics. The core idea: the sub-threshold membrane potential V_mem(t) from Parametric Leaky Integrate-and-Fire (PLIF) neurons is extracted at **zero additional compute cost** and used as an OOD signal in a Hybrid SNN–ANN event-camera object detector trained on the Prophesee Gen1 dataset (cars/pedestrians, 240×304, 36% mAP).

The full benchmark evaluates 7 OOD detectors against 6 event-camera corruption types at 5 severities.

## Commands

All commands assume the `vmem_benchmark/.venv` is activated.

```powershell
# Activate virtual environment
vmem_benchmark\.venv\Scripts\activate

# Quick validation (2 corruptions × 1 sequence each)
.\run_test_pipeline.ps1

# Full benchmark — 31 runs across all 470 test sequences
.\run_full_benchmark.ps1 --gpus 0 1 --workers-per-gpu 2

# Single stage: parallel feature extraction only
python vmem_benchmark/run_parallel_extract.py --gpus 0 --workers-per-gpu 2

# Single stage: single-process extraction (debugging)
python vmem_benchmark/extract.py --device cuda

# Fit OOD detectors on clean data
python analysis/fit_detectors.py

# Score all detectors on corrupted runs
python analysis/evaluate_detectors.py

# Run all analysis scripts and generate plots
python analysis/analyse.py

# Generate paper tables and figures
python reporting/build_paper_tables.py
python reporting/build_paper_figures.py
```

### Installation

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install "setuptools==81.0.0"
pip install -r vmem_benchmark/requirements.txt
```

## Architecture

### 15-Stage Pipeline

```
Stage 1:  run_parallel_extract.py          → φ features (31 runs × 470 sequences)
Stage 2:  extract_offline_features.py      → Temporal AE + margin histograms
Stage 3:  fusion_features.py               → Concatenate representations
Stage 4:  extract_ann_baselines.py         → ResNet-18 event-image / voxel-grid
Stage 5:  evaluate_ann_baselines.py        → ANN baseline scoring
Stage 6:  fit_detectors.py                 → Fit 7 OOD detectors on clean φ
Stage 7:  evaluate_detectors.py            → Score detectors → detector_metrics.csv
Stage 8:  evaluate_mdd.py                  → MDD: per-frame + per-sequence AUROC (all branches)
Stage 9:  representation_ablation.py       → μ vs σ² vs κ ablation
Stage 10: severity.py                      → Spearman ρ monotonicity
Stage 11: reliability.py                   → OOD score vs mAP degradation
Stage 12: cross_corruption.py              → Zero-shot generalization
Stage 13: free_rider_ablation.py           → Trained vs Random SNN vs Raw Input
Stage 14: analyse.py                       → All analysis, heatmaps, PCA, conformal
Stage 15: reporting/build_paper_tables.py  → LaTeX tables
Stage 16: reporting/build_paper_figures.py → Publication-ready figures
```

**MDD (Stage 8)** is the single unsupervised detector from `Docs/novel.md`: radius +
direction-conditioned RCF + deep-layer + (when `phi_spatial` exists) spatial branches, fused by a
calibrated max. `analysis/mdd.py` holds the class; `evaluate_mdd.py` writes `results/mdd_metrics.csv`
(per-frame) and `results/mdd_metrics_aggregated.csv` (per-sequence).

### Key Components

**`vmem_benchmark/`** — Core inference and extraction:
- `benchmark_config.py` — Single source of truth for all paths, corruption list, severities, device, and batch size. **Edit this file to change dataset/model paths or add corruptions.**
- `monitor.py` — `VmemMonitor` registers `forward_hook` on all `MultiStepParametricLIFNode` layers. Extracts V_mem tensor (shape `T×1×C×H×W`), applies Global Average Pooling, and computes `[μ, σ², κ]` per channel → 2112-D φ vector per frame (3 moments × 704 channels across 4 PLIF layers). `collect_phi_spatial()` additionally computes the spatial-dispersion stats GAP discards (`spatial_var`, participation-ratio `spatial_pr`) → **1408-D `phi_spatial`** per frame (2 stats × 704 channels), the per-frame signal for spatial corruptions (`spatial_dropout`, `event_flood`). Stored **float32** (~+60 GB across 31 runs; float16 overflows `spatial_var` under high-activity corruptions like `event_flood`).
- `extract.py` — Main inference loop. Runs model on clean + 30 corrupted variants. Saves φ as `.pt` files under `outputs/phi/`; each file holds `phi`, `phi_spatial` (same rows/`seq_lens`), `done_seqs`, `seq_lens`. Strictly `BATCH_SIZE=1` — SpikingJelly treats the batch dimension as time; B>1 causes membrane state cross-leakage across samples. **`phi_spatial` requires a fresh extraction** — resuming on top of a pre-spatial `phi` file is refused by the merge guard (row-count mismatch).
- `run_parallel_extract.py` — Distributes 6 corruptions across N workers on M GPUs. Each worker gets its own log file under `outputs/logs/`.

**`analysis/`** — Post-hoc analysis (14 scripts):
- `vmem_utils.py` — `LazyPhiDict` lazy-loads φ tensors on demand (full dataset = ~343k frames; loading all at once is impractical). Also defines `LAYER_SPECS` metadata and shared metrics (AUROC, FPR@95%).
- `vmem_scorers.py` — Scorer factories that return closures over fitted detectors. Supports Mahalanobis, kNN, GMM, RealNVP, Autoencoder, OCSVM.
- `vmem_models.py` — Neural models: `RealNVP` (4 coupling layers), `Autoencoder` (3-layer MLP), `TemporalAutoencoder` (1D CNN).
- `analyse.py` — Orchestrator that calls all comparison, plotting, and temporal analysis functions.

**`event_corruption/`** — Event-stream corruption library:
- `corrupt/registry.py` — Maps corruption name strings to implementation functions.
- `corruption_wrap.py` (in `vmem_benchmark/`) — Bridge that converts PyTorch histogram tensors `(N,20,H,W)` to the corruption library's expected format and back.

**`HybridDetection/`** — The upstream SNN–ANN model (separate git repo, Hydra-configured). Do not modify; load via `vmem_benchmark/model_loader.py`.

### Data Flow

```
Gen1 sequences (HDF5)
    └─► [event_corruption] → corrupted histogram tensors (N×20×H×W)
            └─► [HybridDetection backbone] → PLIF forward pass
                    └─► [VmemMonitor hooks] → V_mem(t) per PLIF layer
                            └─► GAP + [μ,σ²,κ] → φ (2112-D) per frame
                                    └─► saved to outputs/phi/<run_name>.pt
```

### Representations

| Name | Dimensionality | Source |
|---|---|---|
| Static φ (membrane stats) | 2112-D | μ, σ², κ of V_mem via GAP |
| Margin histograms | configurable bins | V_mem − θ quantized |
| Temporal AE latents | latent dim | 1D CNN AE on V(t) trajectories |
| Temporal φ | 7 statistics | Autocorrelation, CUSUM, HF energy, etc. |
| ANN baselines | ResNet-18 embedding | Event-image or voxel-grid inputs |
| Fused | concatenation | All above + LogisticRegression meta-classifier |

### OOD Detectors (fitted on clean data, Stage 6)

Mahalanobis (Ledoit-Wolf), kNN (k=5), GMM (5 components), OCSVM (RBF), PCA (50 components), MLP Autoencoder, RealNVP normalizing flow. Fitted detectors are saved as `.joblib` / `.pt` under `outputs/detectors/`.

**Mahalanobis is the reference detector** in all ablation studies due to computational efficiency.

### Corruptions

Behavior below uses the **reference detector (Mahalanobis) on static φ, full 343k data, L5**. Note three corruptions are **below chance** (informative but inverted — corrupted sits closer to the clean mean than held-out clean).

| Corruption | Static-φ AUROC (L5) | Key Behavior |
|---|---|---|
| `hot_pixel` | 1.000 | Perfect monotonicity; persistent spurious events saturate the moments |
| `temporal_jitter` | 0.709 | Moderate; standardized static / static-AE reach ~0.82–0.84 |
| `event_rate_shift` | 0.675 | Phase transition ~severity 3; a global activity scalar reaches ~0.85 |
| `polarity_flip` | 0.429 | **Below chance** — model learned polarity-symmetric features |
| `event_flood` | 0.408 | **Below chance** — flood preserves per-channel moment structure |
| `spatial_dropout` | 0.286 | **Anti-detectable** (Spearman ρ = −1.0) — fewer events → quieter membrane → looks *more* normal |

> **Temporal note:** earlier docs claimed temporal features "rescue" event_flood/spatial_dropout to ~0.85. **That result was an artifact (leakage + 50-sample noise) and does not reproduce.** Leakage-safe temporal gives only a *modest* ~0.05–0.08 gain (handcrafted temporal ≈ 0.63 on event_flood, ≈ 0.54 on spatial_dropout). See `Docs/performance_brief.md` and `Docs/Findings.md` §5. Performance levers (two-sided scoring, activity scalar, sequence aggregation, meta-fusion) are catalogued in `Docs/performance_brief.md`.

## Configuration

All paths and hyperparameters live in `vmem_benchmark/benchmark_config.py`. Key constants:

```python
BATCH_SIZE  = 1      # Must stay 1 — SpikingJelly batch=time convention
TRAJ_SAVE_N = 50     # Trajectory storage cap (full dataset ≈ 15 TB)
PHI_SAVE_EVERY = 5   # Checkpoint φ every N sequences
PLIF_LAYERS = None   # None = hook all 4 layers
```

## Outputs

All results land under `vmem_benchmark/outputs/`:

| Subdirectory | Contents |
|---|---|
| `phi/` | Raw φ tensors, one `.pt` per run |
| `trajs/` | V(t) trajectories for first 50 samples |
| `temporal_phi/` | Temporal handcrafted features |
| `ann_features/` | ResNet-18 baseline embeddings |
| `detectors/` | Fitted detector objects (`.joblib` / `.pt`) |
| `logs/` | Per-worker extraction logs |
| `tables/` | CSV result tables |
| `plots/` | PDF/PNG figures |

## Key Constraints

- **B=1 strictly**: SpikingJelly's `MultiStepParametricLIFNode` treats the batch axis as the time axis. Running B>1 mixes membrane states across sequences, corrupting φ entirely.
- **HybridDetection is read-only**: Loaded via checkpoint; train config resolved via Hydra from `HybridDetection/config/`. Do not modify model code.
- **CuPy required for GPU-accelerated corruption**: `cupy-cuda12x` must match the CUDA version. CPU-only fallback exists but is slow for large sequences.
