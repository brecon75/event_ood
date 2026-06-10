# Vmem-φ: OOD Detection for SNNs via Membrane Potential

## What This Is

An out-of-distribution (OOD) detection benchmark for a **Hybrid SNN-ANN event-camera object detector** (CVPR 2025, Ahmed et al.). The core idea: the sub-threshold **membrane potential V_mem(t)** of PLIF (Parametric Leaky Integrate-and-Fire) neurons is extracted during normal inference at **zero additional compute cost**, and used as an OOD signal.

---

## Model

- **Architecture**: Hybrid SNN–ANN (SpikingJelly PLIF backbone + YOLOX detection head)
- **Dataset**: Prophesee Gen1 event camera dataset (cars + pedestrians, 240×304 resolution)
- **Checkpoint**: `gen1_mAP36.ckpt` — 36% mAP on clean test set
- **SNN backbone**: 4 PLIF layers across 2 blocks (`features_01`, `features_23`)
  - Block 1: 64 channels, 1/2 resolution
  - Block 2: 128 channels, 1/4 resolution
  - Block 3: 256 channels, 1/8 resolution
  - Block 4: 256 channels, 1/8 resolution
- **Batch size**: Strictly B=1 (SpikingJelly treats batch dim as time axis; B>1 causes cross-sample membrane leakage)

---

## The Feature Vector φ

During every forward pass, hooks on all 4 PLIF layers capture `V_mem(t)` with shape `(T=10, 1, C, H, W)`. Three temporal moments are computed per channel after Global Average Pooling:

- **μ** (mean), **σ²** (variance), **κ** (excess kurtosis)

Concatenated across all 4 layers:

```
φ = [μ, σ², κ]_layer1 ++ [μ, σ², κ]_layer2 ++ [μ, σ², κ]_layer3 ++ [μ, σ², κ]_layer4
φ ∈ ℝ^2112    (3 × (64 + 128 + 256 + 256) = 2112)
```

---

## Temporal Features (φ_τ)

For each PLIF layer, from the spatially-pooled scalar trace `v(t)`:

1. Threshold-margin mean: `mean(θ - v[t])`
2. Minimum threshold margin: `min(θ - v[t])`
3. Margin variance
4. First-difference absolute mean: `mean(|v[t] - v[t-1]|)`
5. First-difference variance
6. Lag-1 autocorrelation
7. High-frequency energy ratio (FFT)

→ φ_τ ∈ ℝ^28 (7 features × 4 layers)

Also: a **Conv1D Temporal Autoencoder** trained on clean trajectories; reconstruction error = OOD score.

**Limitation**: Raw trajectory storage capped at 50 samples per run (full dataset = ~15 TB). All temporal results are from a 50-sample pilot subset.

---

## Benchmark Setup

### Dataset Scale
- **Test split**: 470 sequences, ~343,000 histogram frames
- **Runs**: 31 total (1 clean + 6 corruptions × 5 severities)
- **Frames per run**: ~343k (static φ) / 50 (temporal features)

### Corruptions (6 types, 5 severity levels each)

| Corruption | What it does | Params (L1→L5) |
|---|---|---|
| **hot_pixel** | Adds constant count to fixed random pixels (defective sensor pixels) | 10→300 pixels, count 20→200 |
| **event_flood** | Adds uniform random noise to every pixel/time-bin | scale 0.1→0.5 |
| **temporal_jitter** | Randomly permutes time bin ordering | ±1→±8 bins |
| **polarity_flip** | Randomly flips event polarity | p=0.1→0.5 |
| **event_rate_shift** | Scales total event count down | factor 0.8→0.1 |
| **spatial_dropout** | Zeroes random spatial locations per frame | p=0.1→0.5 |

### OOD Detectors (7, all fitted on clean φ training split)

| Detector | Method |
|---|---|
| Mahalanobis | Ledoit-Wolf regularised covariance; Mahal distance |
| kNN (k=5) | Distance to 5th nearest clean neighbour |
| GMM | Gaussian Mixture Model, 5 components, neg log-likelihood |
| PCA-Mahal | PCA to 50 components, then Mahalanobis |
| One-Class SVM | RBF kernel OCSVM |
| Normalizing Flow | RealNVP, 4 coupling layers on PCA-reduced φ |
| MLP Autoencoder | 3-layer MLP encoder-decoder, 128-dim latent, L2 recon error |

### Metrics
- **AUROC** (primary), **AUPR**, **FPR@95** (False Positive Rate at 95% TPR)
- **Spearman ρ** between OOD score and severity level
- **R²** for severity regression

---

## Results

### 1. Detector Comparison (avg over all 6 corruptions × 5 severities)

| Detector | Avg AUROC | Avg FPR@95 |
|---|---|---|
| **MLP Autoencoder** | **0.673** | **0.695** |
| Normalizing Flow | 0.650 | 0.798 |
| PCA-Mahal | 0.650 | 0.707 |
| Mahalanobis | 0.638 | 0.726 |
| kNN (k=5) | 0.637 | 0.747 |
| GMM | 0.617 | 0.817 |
| One-Class SVM | 0.598 | 0.801 |

**Key insight**: Autoencoder wins on both metrics → φ space has non-Gaussian, non-linear manifold structure. Parametric methods underfit. NF gets good AUROC but terrible FPR (good ranker, bad binary alarm).

---

### 2. Per-Layer AUROC (Mahalanobis scorer, avg over 5 severities)

| Layer | hot_pixel | event_flood | temporal_jitter | polarity_flip | event_rate_shift | spatial_dropout | AVG |
|---|---|---|---|---|---|---|---|
| Block 1 | **0.871** | 0.523 | 0.570 | 0.547 | 0.430 | 0.488 | 0.571 |
| Block 2 | 0.677 | 0.517 | 0.609 | 0.520 | 0.568 | 0.481 | 0.562 |
| Block 3 | 0.771 | 0.512 | 0.794 | 0.532 | 0.662 | 0.478 | 0.625 |
| Block 4 | 0.815 | 0.511 | **0.865** | 0.529 | **0.739** | 0.481 | **0.656** |
| ALL concat | 0.804 | 0.523 | 0.768 | 0.549 | 0.701 | 0.480 | 0.638 |

**Key insights**:
- Block 1 (closest to sensor) = best hot_pixel detector. DC injection saturates the first layer before diffusing.
- Blocks 3–4 = best for temporal/rate corruptions. Larger receptive fields + more timestep integration.
- event_flood: undetectable at ALL layers (0.511–0.523). Not a per-layer failure — fundamentally invisible.
- spatial_dropout: below 0.50 at ALL layers. Detector gives *lower* OOD score to corrupted frames.
- ALL concat (0.638) **worse** than Block 4 alone (0.656). Concatenation dilutes signal.

---

### 3. Per-Corruption Severity Breakdown

#### hot_pixel — Perfect saturation ramp
| Severity | AUROC (best det) |
|---|---|
| 1 | ~0.50 |
| 2 | ~0.59 |
| 3 | ~0.93 |
| 4 | ~1.00 |
| 5 | **1.000** (all 7 detectors) |

#### event_flood — Completely flat across all detectors
| Severity | AUROC |
|---|---|
| 1 | 0.504 |
| 2 | 0.509 |
| 3 | 0.517 |
| 4 | 0.532 |
| 5 | 0.554 |

Total gain: only +0.05 over 5 severity levels. All 7 detectors nearly identical. Fundamental indistinguishability from busy clean scene.

#### temporal_jitter — Strong ramp, Autoencoder wins
- Autoencoder: 0.806 → **0.947** (L5) ← best non-trivial result
- Normalizing Flow: 0.834 → 0.928
- PCA-Mahal: 0.787 → 0.930

#### event_rate_shift — Phase transition at Severity 3
| Severity | Mahal AUROC | kNN AUROC | GMM AUROC |
|---|---|---|---|
| 1 | 0.491 | 0.485 | 0.496 |
| 2 | 0.496 | 0.496 | 0.544 |
| **3** | **0.806** | **0.819** | 0.452 |
| 4 | 0.810 | 0.820 | 0.473 |
| 5 | 0.902 | 0.927 | **0.139** |

GMM collapses at L5 (0.139) — corrupted distribution falls outside GMM support, log-likelihood inverts.

#### polarity_flip — Monotone but weak
L1: ~0.51 → L5: ~0.60–0.67. Network learned polarity-symmetric features.

#### spatial_dropout — Perfect anti-detection
| Severity | AUROC |
|---|---|
| 1 | 0.499 |
| 3 | 0.490 |
| 5 | **0.439** |

At L5, Mahalanobis AUROC = 0.439. Detector actively labels corrupted frames as MORE normal than clean.  
Mechanism: fewer events → less input current → V_mem closer to resting state → smaller Mahal distance from clean mean → lower OOD score.

---

### 4. Spearman Severity Correlation

| Corruption | ρ | p-value | Notes |
|---|---|---|---|
| hot_pixel | +1.000 | 0.000 | Perfect monotone |
| event_flood | +1.000 | 0.000 | Monotone but AUROC barely moves |
| polarity_flip | +1.000 | 0.000 | Weak but consistent |
| event_rate_shift | +1.000 | 0.000 | Phase jump at L3 drives it |
| temporal_jitter | +0.700 | 0.188 | **Not significant** (p>0.05, n=5 too small) |
| **spatial_dropout** | **-1.000** | **0.000** | **Perfect inverse — corrupted looks more normal** |

spatial_dropout ρ = -1.000 with p = 0.000 is a key finding: statistically significantly *anti*-correlated.

---

### 5. Temporal Features vs Static φ (50-sample pilot, Severity 5)

| Corruption | Static φ (best det, full dataset) | Handcrafted Temporal | Temporal AE |
|---|---|---|---|
| hot_pixel | 1.000 | 1.000 | 1.000 |
| event_rate_shift | 0.902 | 0.952 | **0.960** |
| temporal_jitter | **0.947** | 0.853 | 0.905 |
| polarity_flip | 0.666 | 0.830 | **0.850** |
| **event_flood** | **0.554 (broken)** | **0.830** | **0.848 (rescued)** |
| **spatial_dropout** | **0.439 (inverted)** | **0.817** | **0.846 (rescued)** |

**Most important finding**: Temporal features rescue both hard corruptions:
- event_flood: +0.294 AUROC (0.554 → 0.848)
- spatial_dropout: +0.407 AUROC (0.439 → 0.846, from anti-detection to strongly detectable)

With only 50 training samples. If run on all 343k frames (requires online feature computation instead of raw traj saving), temporal results would dominate across all corruptions.

---

### 6. Free-Rider Ablation (5 sequences, Mahalanobis, Severity 5)

| Condition | hot_pixel AUROC | event_flood AUROC |
|---|---|---|
| **A — Trained SNN** | **0.9980** | 0.5522 |
| B — Random SNN (same arch, random weights) | 0.8217 | 0.4978 |
| C — Raw Input Stats (μ, σ², κ on raw histogram) | 0.7164 | 0.5571 |

- Trained > Random > Raw Input: trained weights encode genuinely discriminative representations
- A vs B gap (+0.176): learning matters, not just architecture
- A vs C gap (+0.282): SNN processing adds real signal beyond input statistics
- All cluster near chance for event_flood → theoretical indistinguishability confirmed

---

### 7. Representation Ablation (Mahalanobis, all corruptions)

Which moment contributes most?

- **σ² (variance) alone**: single strongest individual feature
- **μ (mean) alone**: best for event_flood (only corruption where global mean shifts)
- **κ (kurtosis) alone**: weakest overall; unique signal on temporal_jitter
- **Combined [μ, σ², κ]**: outperforms all individual statistics on average (complementary)

Other representations tested: ANN GAP features, logits, spike rate, spike entropy, fused membrane.

---

### 8. Conformal Prediction (Mahalanobis calibration)

Thresholds from clean calibration split:
- q_90 = 90th percentile → "Clean at 90% confidence"
- q_99 = 99th percentile → "OOD at 99% confidence"
- Between → "Ambiguous"

Results:
- hot_pixel L5: OOD fraction >95%
- event_flood (all severities): OOD fraction <15% → not confidently detected by static φ

---

### 9. Severity Regression (Ridge regression, φ → severity level 1–5)

| Corruption | R² |
|---|---|
| hot_pixel | >0.7 |
| temporal_jitter | >0.7 |
| event_rate_shift | >0.7 |
| event_flood | ~0 |
| spatial_dropout | ~0 |

φ encodes corruption intensity *continuously* for detectable corruptions.

---

### 10. Corruption Classification (7-class: clean + 6 corruptions)

LinearSVC / LogisticRegression on φ achieves >50% top-1 accuracy.  
Confusion: event_flood and high-severity spatial_dropout frequently confused with clean.

---

## Why φ Succeeds / Fails: PLIF Theory

PLIF update rule:
```
V[t] = (1 - 1/τ) · V[t-1] + W · X[t]
```
Fire when V[t] ≥ θ, then reset V ← 0.

| Corruption | Effect on V_mem | Explains |
|---|---|---|
| hot_pixel | Constant DC added to X[t] at fixed pixels → V saturates toward θ | AUROC = 1.000 (mathematically inevitable) |
| event_flood | Uniform noise shifts mean of ALL neurons equally → indistinguishable from busy clean scene | AUROC ≈ 0.554 (cannot separate from high-activity clean) |
| spatial_dropout | Zeros channels of X[t] → reduces σ² but not μ proportionally → V closer to resting state | Mahal distance DECREASES → anti-detection (ρ = -1.0) |
| temporal_jitter | Scrambles time ordering of X[t] → destroys temporal autocorrelation of V(t) | Lag-1 autocorr feature effective; deeper layers more sensitive |
| event_rate_shift | Scales all X[t] → firing regime changes nonlinearly | Phase transition at severity 3 |
| polarity_flip | Flips sign of events → network learned polarity-symmetric features | Weak perturbation to V |

---

## Neuromorphic Hardware Argument

On neuromorphic chips (Intel Loihi, BrainScaleS, SpiNNaker), V_mem is a **native hardware register** — read cost = 0. Every other OOD method is incompatible:

| Method | Overhead | Neuromorphic? |
|---|---|---|
| **Vmem-φ** | **None** | **✓** |
| MSP / Energy | Softmax over logits | ✗ |
| ODIN | Temperature-scaled softmax | ✗ |
| ReAct | Feature clipping + re-forward | ✗ |
| ViM | SVD of feature matrix | ✗ |
| GradNorm | Gradient backpropagation | ✗ |
| kNN / Mahal on ANN | Auxiliary ANN forward pass | ✗ |

---

## Pipeline Architecture

```
run_full_benchmark
  ├── run_parallel_extract.py        → 31 inference runs (parallel GPU)
  │     └── extract.py               → VmemMonitor hooks, saves φ tensors
  ├── extract_offline_features.py    → margin histograms, temporal AE latents
  ├── fusion_features.py             → concatenate all feature types
  ├── extract_ann_baselines.py       → ResNet-18 on event-image + voxel-grid
  ├── evaluate_ann_baselines.py      → MSP, Energy, ODIN, Mahal, kNN on ANN feats
  ├── fit_detectors.py               → fit 7 OOD detectors on clean φ
  ├── evaluate_detectors.py          → AUROC, AUPR, FPR@95 for all 210 combos
  ├── representation_ablation.py     → μ vs σ² vs κ vs combined vs ANN vs spike
  ├── severity.py                    → Spearman ρ (score vs severity)
  ├── reliability.py                 → OOD score vs mAP degradation correlation
  ├── cross_corruption.py            → zero-shot: train on hot_pixel, test on others
  ├── free_rider_ablation.py         → Trained vs Random SNN vs Raw Input Stats
  ├── analyse.py                     → all high-level analysis + plots
  └── reporting/
        ├── build_paper_tables.py
        └── build_paper_figures.py
```

Key output directories:
- `vmem_benchmark/outputs/phi/` — raw φ tensors per run
- `vmem_benchmark/outputs/trajs/` — raw V(t) trajectories (capped at 50 samples)
- `vmem_benchmark/outputs/temporal_phi/` — handcrafted temporal features
- `vmem_benchmark/outputs/tables/` — all CSV result tables
- `vmem_benchmark/outputs/plots/` — all figures

---

## Key Numbers Summary

| Metric | Value |
|---|---|
| φ dimensionality | 2112 |
| Test frames | ~343,000 |
| Corruptions × severities | 6 × 5 = 30 |
| OOD detectors evaluated | 7 |
| Best overall AUROC (avg) | 0.673 (MLP Autoencoder) |
| Best single result | 1.000 (hot_pixel L5, all detectors) |
| Worst static result | 0.439 (spatial_dropout L5, Mahalanobis) |
| Temporal rescue — event_flood | 0.554 → 0.848 (+0.294) |
| Temporal rescue — spatial_dropout | 0.439 → 0.846 (+0.407) |
| Free-rider gap (trained vs raw) | +0.282 AUROC on hot_pixel |
| Trajectory sample cap | 50 (disk constraint; full = ~15 TB) |

---

## Open Problems / Future Work

1. **Fix temporal limitation**: Compute online temporal φ during extraction (saves ~15 TB, runs on all 343k frames)
2. **Layer attention**: Concatenation hurts — learn weights over layers
3. **DSEC transfer**: Test on outdoor driving dataset (DVXplorer sensor)
4. **event_flood fix**: Static φ provably cannot detect it; need temporal features or different representation
5. **Closed-loop adaptation**: Use OOD score to trigger NMS threshold adjustments at runtime
6. **Contrastive encoder**: Train φ-space with corruption-supervised contrastive loss
