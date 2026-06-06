# Analysis Test Suite — Overview

**Project:** Vmem-based Out-of-Distribution (OOD) Detection for Spiking Neural Networks  
**Dataset:** GEN1 Prophesee event-camera dataset  
**Model:** Hybrid SNN–ANN detector (SpikingJelly + YOLOX head)  
**Location:** `d:\Perdue\analysis\`

---

## Executive Summary

The `analysis/` directory implements a complete post-hoc OOD evaluation suite.
Raw event-camera data is corrupted at five severity levels across multiple corruption types.
The SNN backbone's membrane potential trajectories (`φ`) are extracted for each corrupted run and
fed through a battery of detectors and statistical tests.

The pipeline is orchestrated end-to-end by `run_test_pipeline.py`, which chains the following stages in order:

```
test_pipeline.py  →  extract.py  →  extract_offline_features.py
→  fusion_features.py  →  extract_ann_baselines.py  →  evaluate_ann_baselines.py
→  fit_detectors.py   →  evaluate_detectors.py   →  representation_ablation.py
→  severity.py        →  reliability.py           →  cross_corruption.py
→  free_rider_ablation.py  →  analyse.py
```

---

## Stage-by-Stage Description

### Stage 1 — Feature Extraction (`extract.py`)

Runs the trained hybrid SNN–ANN backbone on clean and corrupted input sequences.
For each sequence the `VmemMonitor` hooks into all PLIF (Parametric Leaky Integrate-and-Fire) layers
and records membrane potential statistics (mean, variance, kurtosis) after global-average-pooling.

**Output:** `φ` tensors (one `.pt` per run) saved to `cfg.PHI_DIR`.  
**Shape:** `(N_frames, 3 × n_PLIF_layers)` — concatenated `[μ, σ², κ]` per layer.

---

### Stage 2 — Offline Feature Extraction (`extract_offline_features.py`)

Processes the compressed temporal-GAP trajectories saved by Stage 1 to compute two additional feature types:

| Feature | Description |
|---|---|
| **Margin Histogram** | Per-layer histogram of `(Vmem − θ)` over `[-2θ, +2θ]` with 20 bins; shape `(N, n_layers × 20)` |
| **Trajectory Latent** | Latent code of the temporal GAP trajectory encoded by a `TemporalAutoencoder` |

**Output:** Saved as `.pt` files in `features/margin_hist/` and `features/trajectory_latent/`.

---

### Stage 3 — Feature Fusion (`fusion_features.py`)

Concatenates all available representations — membrane stats, margin histogram, temporal latent,
and temporal-phi — into a single fused representation per run.

Uses a `LogisticRegression` meta-classifier to assess which combination of features yields the
highest OOD discriminability.

**Output:** `features/fused/<run>.pt` files and a `fusion_scores.csv` result table.

---

### Stage 4 — ANN Baseline Extraction (`extract_ann_baselines.py`)

Extracts features from two pretrained **ANN-only** baselines for comparison with the SNN-based detector:

| Baseline | Backbone | Input |
|---|---|---|
| Event Image | ResNet-18 (2-channel) | 2-ch polarity-summed event image |
| Voxel Grid | ResNet-18 (20-channel) | 20-bin temporal voxel grid |

Both backbones use ImageNet-pretrained weights adapted for event-camera channel counts.

**Output:** `cfg.ANN_DIR/event_image/<run>.pt` and `cfg.ANN_DIR/voxel_grid/<run>.pt`.

---

### Stage 5 — ANN Baseline Evaluation (`evaluate_ann_baselines.py`)

Scores each ANN representation with five OOD detectors derived from the clean training split:

| Detector | Method |
|---|---|
| MSP | Maximum Softmax Probability (negated) |
| Energy | Free-energy score from logits |
| ODIN | Temperature-scaled softmax (T=1000) |
| Mahalanobis | Ledoit-Wolf covariance on penultimate features |
| KNN | Distance to k-th nearest clean neighbour |

Metrics computed: **AUROC**, **AUPR**, **FPR@95% TPR** per corruption × severity.  
**Output:** `results/ann_baselines_metrics.csv` and plots in `figures/`.

---

### Stage 6 — Detector Fitting (`fit_detectors.py`)

Fits six OOD detectors on the **clean membrane-fused representation** extracted in Stage 3:

| Detector | Algorithm |
|---|---|
| Mahalanobis | Empirical covariance + precision matrix (scikit-learn) |
| KNN | k-Nearest Neighbours (k=5) distance scorer |
| GMM | Gaussian Mixture Model (8 components, negative log-likelihood) |
| OCSVM | One-Class SVM (RBF kernel) |
| PCA | PCA reconstruction error (top-50 components) |
| Autoencoder (AE) | `SimpleAE` — 3-layer MLP encoder + decoder; trained 50 epochs on CUDA |

**Outputs:** `.joblib` files for sklearn models; `ae.pt` for the PyTorch AE, all saved in `cfg.DETECTOR_DIR`.

---

### Stage 7 — Detector Evaluation (`evaluate_detectors.py`)

Evaluates all fitted detectors (from Stage 6) against every corruption × severity combination.

Scoring functions:

| Name | Scoring Strategy |
|---|---|
| `score_mahalanobis` | Mahalanobis distance from fitted mean/precision |
| `score_knn` | Distance to k-th nearest training neighbour |
| `score_gmm` | Negative log-likelihood under GMM |
| `score_ocsvm` | Negated OCSVM decision function |
| `score_pca` | L2 reconstruction error after PCA projection |
| `score_ae` | L2 reconstruction error from trained AE (GPU-accelerated) |

Metrics: **AUROC**, **AUPR**, **FPR@95%** per (detector, corruption, severity), plus an aggregate over severity ≥ 3.  
**Output:** `results/detector_metrics.csv`, bar charts, and ROC curve PDFs.

---

### Stage 8 — Representation Ablation (`representation_ablation.py`)

Tests nine distinct feature representations using a fixed **Mahalanobis** detector to isolate which part of the SNN's internal signal carries the OOD information:

| Representation | Description |
|---|---|
| `full_membrane` | Full φ vector: concatenated `[μ, σ², κ]` for all layers |
| `membrane_mean` | Mean component only |
| `membrane_var` | Variance component only |
| `membrane_kurtosis` | Kurtosis component only |
| `ANN` | Last ANN GAP feature (penultimate YOLOX head) |
| `logits` | Classification head output (head_cls_L0_gap) |
| `spike` | Spike rate (average firing frequency per neuron) |
| `spike_entropy` | Entropy of spike distribution |
| `membrane_fused` | Fused representation from Stage 3 |

**Metrics:** AUROC, AUPR, FPR@95 per (representation, corruption, severity).  
**Output:** `results/representation_metrics.csv`, heatmap at `figures/representation_heatmap.pdf`.

---

### Stage 9 — Severity Monotonicity (`severity.py`)

Tests whether OOD scores **increase monotonically** with corruption severity — a key desideratum for a trustworthy detector.

Computes **Spearman's ρ** between detector score and severity level (0 = clean, 1–5 = corrupted):

- **All representations** × Mahalanobis detector
- **Fused membrane** × all six fitted detectors (from Stage 6)

For each (corruption, representation, detector) triple.

**Output:** `results/severity_metrics.csv`, bar chart at `figures/severity_curves.pdf`.

---

### Stage 10 — Reliability Prediction (`reliability.py`)

Measures whether the SNN's Mahalanobis OOD score can **predict performance degradation** of the downstream object detector.

For each corrupted run:
1. Computes **degradation** = difference in confident detection counts (clean − corrupt, thresholded at 0.3).
2. Computes per-frame OOD score from the membrane-fused representation.
3. Reports **Spearman ρ**, **Pearson r**, and **R²** between OOD score and degradation.
4. Computes **AURC** (Area Under the Risk-Coverage Curve) using OOD score as the uncertainty signal.

**Output:** `results/reliability_metrics.csv`, boxplot at `figures/reliability_curve.pdf`, risk-coverage curve at `figures/risk_coverage.pdf`.

---

### Stage 11 — Cross-Corruption Generalisation (`cross_corruption.py`)

Tests **zero-shot generalisation**: a binary logistic classifier is trained on clean vs. `hot_pixel` (all severities) using the fused representation, and then evaluated on five **unseen** corruption types:

- `event_flood`, `temporal_jitter`, `polarity_flip`, `event_rate_shift`, `spatial_dropout`

Metric: **AUROC** per (eval_corruption, severity).  
**Output:** `results/cross_corruption.csv`.

---

### Stage 12 — Free Rider Ablation (`free_rider_ablation.py`)

This is the core **"Idea 9"** ablation, answering the question:

> *Does the SNN actually learn something useful, or is any signal a free rider from the input structure?*

Three conditions are compared using a Mahalanobis scorer on clean-vs-corrupted L5 sequences:

| Condition | Description |
|---|---|
| **A — Trained SNN** | Vmem statistics from the trained backbone |
| **B — Random SNN** | Vmem statistics from a randomly re-initialised backbone (same architecture) |
| **C — Raw Input Stats** | Pixel-level `[μ, σ², κ]` computed directly from the input histogram tensor |

Applied to two corruption types at max severity (L5): `hot_pixel` and `event_flood`.  

**Sample Result (5-sequence val subset):**

| Condition | hot_pixel_L5 | event_flood_L5 |
|---|---|---|
| Trained SNN | **0.9980** | 0.5522 |
| Random SNN | 0.8217 | 0.4978 |
| Raw Input Stats | 0.7164 | 0.5571 |

The gap between Trained and Random confirms the model has learned a **discriminative internal representation** beyond the trivial input statistics.

**Output:** `tables/free_rider_ablation.csv`, bar chart at `cfg.PLOT_DIR/free_rider_ablation.pdf`.

---

### Stage 13 — Analysis Orchestrator (`analyse.py`)

Runs all high-level analysis and plotting passes after feature extraction is complete.
Internally calls sub-modules in `analyse_comparisons.py`, `analyse_plots.py`, and `analyse_temporal.py`.

| Level | Function | Description |
|---|---|---|
| L1 | `plot_sensitivity_heatmap` | Global AUROC heatmap per corruption × layer |
| L1 | `plot_auroc_vs_severity` | AUROC vs severity line plot |
| L1 | `plot_all_trajectories` | Vmem trajectory visualisations |
| L2 | `run_per_layer_auroc_table` | Per-layer × per-corruption AUROC breakdown (Mahalanobis) |
| L2 | `run_statwise_ablation` | Ablation: which stat (mean/var/kurt) contributes most |
| L3 | `run_detector_comparison` | Compare all detectors: Mahalanobis, KNN, GMM, OCSVM, NF, AE |
| L4 | `run_temporal_analysis` | Temporal autocorrelation, change-point detection, CUSUM |
| L5 | `run_spearman_severity` | Spearman ρ across corruptions and severities |
| L5 | `save_full_results_table` | Combined CSV of all AUROC results |
| Ideas | `plot_pca_subspaces` | PCA of clean vs corrupt φ distributions |
| Ideas | `run_severity_regression` | Ridge regression to predict severity from φ |
| Ideas | `run_corruption_classification` | Multi-class classification of corruption type from φ |
| Ideas | `run_conformal_prediction` | Conformal prediction sets for uncertainty quantification |

**Output:** All figures → `cfg.PLOT_DIR`, all tables → `TABLE_DIR`.

---

## Support Modules

| File | Role |
|---|---|
| `vmem_models.py` | Defines `TemporalAutoencoder`, `NormalizingFlow`, and training helpers used by Stage 2 and 3 |
| `vmem_scorers.py` | Functional OOD scorer factories: `mahalanobis_scorer`, `knn_scorer`, `gmm_scorer`, `normalizing_flow_scorer`, `autoencoder_scorer`, `pca_mahalanobis_scorer`, `ocsvm_scorer` |
| `vmem_utils.py` | Utilities: `LazyPhiDict` (on-demand φ loader), `auroc_fpr95`, `LAYER_SPECS`, `slice_phi_layer`, `slice_phi_stat`, `TABLE_DIR` |
| `analyse_comparisons.py` | Hosts all Level 2–7 analysis functions called from `analyse.py` |
| `analyse_plots.py` | All matplotlib/seaborn plotting code (heatmaps, ROC curves, bar charts, etc.) |
| `analyse_temporal.py` | Temporal analysis: autocorrelation, CUSUM change-point detection, temporal AUROC |

---

## Output Artefacts

| Path (relative to `cfg.OUTPUT_DIR`) | Contents |
|---|---|
| `features/phi/<run>.pt` | Raw membrane stats φ per run |
| `features/margin_hist/<run>.pt` | Margin histogram features |
| `features/trajectory_latent/<run>.pt` | Temporal AE latent codes |
| `features/fused/<run>.pt` | Fused multi-representation features |
| `ann_features/event_image/<run>.pt` | ResNet-18 event-image features |
| `ann_features/voxel_grid/<run>.pt` | ResNet-18 voxel-grid features |
| `detectors/*.joblib` | Fitted sklearn detectors |
| `detectors/ae.pt` | Fitted PyTorch autoencoder |
| `results/detector_metrics.csv` | Full detector evaluation results |
| `results/ann_baselines_metrics.csv` | ANN baseline evaluation results |
| `results/representation_metrics.csv` | Representation ablation AUROC table |
| `results/severity_metrics.csv` | Severity monotonicity (Spearman ρ) |
| `results/reliability_metrics.csv` | Reliability prediction (Spearman ρ, Pearson r, R², AURC) |
| `results/cross_corruption.csv` | Cross-corruption generalisation AUROC |
| `tables/free_rider_ablation.csv` | Free rider ablation AUROC table |
| `figures/*.pdf` | All generated plots (heatmaps, ROC curves, etc.) |

---

## Key Metrics Used

| Metric | Meaning |
|---|---|
| **AUROC** | Area Under ROC Curve — primary ranking metric (higher = better) |
| **AUPR** | Area Under Precision-Recall Curve — useful for imbalanced data |
| **FPR@95** | False Positive Rate when TPR = 95% — lower is better |
| **Spearman ρ** | Rank correlation between OOD score and severity/degradation |
| **AURC** | Area Under Risk-Coverage Curve — selective prediction quality |
| **R²** | Coefficient of determination for OOD-score vs degradation regression |

---

*Document created: June 2026.*  
*See also: `Docs/implementation_status.tex` for the full benchmark feature checklist, and `Docs/Findings.md` for empirical findings.*
