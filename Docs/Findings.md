# Benchmark Results — Findings & Interpretation

> Results from `vmem_benchmark/outputs/tables/` and `outputs/plots/`.
> 6 corruptions × 5 severity levels × 7 detectors = 210 data points.

> [!CAUTION]
> **CORRECTION (2026-06-11): the headline "temporal rescues the hard corruptions to ~0.85" result in Section 5 does NOT reproduce.** It was computed on the 50-sample raw-trajectory path with a train/eval split that pre-dates the `unified train/eval split` fix, and was inflated by leakage + tiny-sample noise. Re-running the exact same method now gives **0.29 / 0.04 / 0.06** (below chance), and the leakage-safe per-frame temporal features give only **~0.54–0.62** — a modest, significant improvement over static, *not* a rescue. Section 5 and the summary tables below have been corrected. Any AUROC in this doc generated **before the split fix** should be treated as provisional until regenerated. See `Docs/performance_brief.md` for the full audit and `temporal_vs_static.py` / `reproduce_findings_temporal.py` for the reproduction.

---

## 1. Detector Comparison — The Headline Numbers

| Detector | Avg AUROC | Avg FPR@95 | Verdict |
|:---------|:---------:|:----------:|:--------|
| **Autoencoder** | **0.6727** | **0.6947** | Best overall — AUROC *and* FPR |
| Normalizing Flow | 0.6496 | 0.7982 | Good AUROC, badly calibrated FPR |
| PCA-Mahal | 0.6495 | 0.7073 | Tied 2nd, much better FPR than NF |
| Mahalanobis | 0.6375 | 0.7258 | Solid baseline |
| kNN (k=5) | 0.6372 | 0.7467 | Tied with Mahal |
| GMM | 0.6172 | 0.8174 | Consistently weakest |
| One-Class SVM | 0.5979 | 0.8008 | Worst overall |

**Interpretation:**

The MLP Autoencoder wins on both AUROC and FPR@95. This tells us the phi feature space has a **non-Gaussian, non-linear manifold structure** that parametric methods (Mahalanobis, GMM) cannot capture well. The Autoencoder learns the manifold implicitly through reconstruction. 

The Normalizing Flow gets a high AUROC (good at *ranking*) but terrible FPR@95 — its score threshold is poorly calibrated. It should not be used as a binary alarm system. 

GMM at 0.6172 suggests that 5 Gaussian components is not enough to model the multi-modal phi distribution. The phi space has more complex topology than a mixture of 5 Gaussians.

---

## 2. Per-Layer AUROC — The Architectural Finding

| Layer | hot_pixel | event_flood | temporal_jitter | polarity_flip | event_rate_shift | spatial_dropout | AVG |
|:------|:---------:|:-----------:|:---------------:|:-------------:|:----------------:|:---------------:|:---:|
| SNN Block 1 | **0.871** | 0.523 | 0.570 | 0.547 | 0.430 | 0.488 | 0.571 |
| SNN Block 2 | 0.677 | 0.517 | 0.609 | 0.520 | 0.568 | 0.481 | 0.562 |
| SNN Block 3 | 0.771 | 0.512 | 0.794 | 0.532 | 0.662 | 0.478 | 0.625 |
| SNN Block 4 | 0.815 | 0.511 | **0.865** | 0.529 | **0.739** | 0.481 | **0.656** |
| ALL (concat) | 0.804 | 0.523 | 0.768 | 0.549 | 0.701 | 0.480 | 0.638 |

**Key findings:**

- **Block 1 is the best hot_pixel detector (0.871).** Hot pixels saturate the very first layer most dramatically — they inject a constant DC signal before it gets diffused and integrated through deeper processing. Block 1 is literally the closest layer to the sensor noise.

- **Blocks 3 & 4 dominate temporal_jitter (0.794/0.865) and event_rate_shift (0.662/0.739).** These corruptions disrupt the *temporal dynamics* of event sequences. Deeper layers integrate over more computational timesteps and have larger receptive fields, making them sensitive to timing-based and rate-based distortions that early layers miss.

- **event_flood is undetectable at all layers (~0.511–0.523).** This is not a per-layer problem. The corruption is fundamentally indistinguishable from a high-activity clean scene at every depth of the SNN.

- **spatial_dropout is below 0.50 at all layers.** The detector is giving *lower* anomaly scores to spatially dropped inputs than to clean ones. The SNN interprets "fewer events = quieter scene = more normal". This is a systematic inversion.

- **ALL (concat) does not beat Block 4 (0.638 vs 0.656).** Concatenating all four layers dilutes the signal from the best layers with noise from the weaker ones. This directly motivates layer attention / fusion weighting (the `fusion_features.py` experiment).

---

## 3. Full Severity Breakdown — The Most Revealing Table

### hot_pixel — Perfect saturation ramp
Severity 1: ~0.50 (invisible), L2: ~0.59, L3: ~0.93, L4: ~1.00, L5: **1.000 across all 7 detectors**.

This is exactly what the PLIF theory predicts. At low severity, only a few pixels are corrupted and the membrane barely shifts. As severity increases, the constant DC injection pushes V toward θ everywhere until the separation is total.

### event_flood — Completely flat, all detectors
L1: 0.504, L2: 0.509, L3: 0.517, L4: 0.532, L5: 0.554 — a gain of only **0.05 AUROC across 5 severity levels**, and all 7 detectors return nearly identical scores. This is the clearest possible evidence that event_flood is **fundamentally invisible to membrane potential statistics**. The corruption mimics a high-activity clean scene at the neuron level.

### temporal_jitter — Strong ramp, Autoencoder wins
- Autoencoder: 0.806 → **0.947** at L5 ← the single best result across all non-trivial corruptions
- Normalizing Flow: 0.834 → 0.928
- PCA-Mahal: 0.787 → 0.930 — comparable to NF but with much better FPR

Temporal jitter clearly disrupts the internal statistics of phi — deeper layers and more expressive detectors pick it up strongly at high severity.

### event_rate_shift — Phase transition at Severity 3

| Severity | Mahal AUROC | kNN AUROC | GMM AUROC |
|:--------:|:-----------:|:---------:|:---------:|
| 1 | 0.491 | 0.485 | 0.496 |
| 2 | 0.496 | 0.496 | 0.544 |
| **3** | **0.806** | **0.819** | **0.452** |
| 4 | 0.810 | 0.820 | 0.473 |
| 5 | 0.902 | 0.927 | **0.139** |

There is a **phase transition between Severity 2 and Severity 3.** At low severity, the SNN adapts and the phi distribution barely shifts. Above a critical rate reduction, the SNN enters a different firing regime that is sharply distinguishable. 

The **GMM completely collapses at high severity** (0.139 at L5). The corrupted distribution at L5 is so far from the training distribution that it falls outside the GMM's support and the log-likelihood wraps around to very high values (i.e. scores become very negative, which the negation inverts). This is a known GMM failure mode in very high-dimensional spaces.

### polarity_flip — Monotone but weak
L1: ~0.51 → L5: ~0.60–0.67. The network has learned polarity-symmetric or polarity-agnostic features, so flipping the sign of events doesn't strongly perturb V. This makes sense — the detector model was trained on real event streams where both polarities appear naturally.

### spatial_dropout — Perfect anti-detection (ρ = -1.0)
AUROC goes *down* as severity increases: L1: 0.499, L3: 0.490, L5: **0.439**.

At L5, Mahalanobis AUROC = 0.439 — **the detector is actively labelling corrupted frames as more normal than clean frames**. Spatially sparse input → fewer events → neurons receive less input current → membrane stays closer to its resting state → Mahalanobis distance from the clean-mean *decreases*. The detector interprets silence as safety.

---

## 4. Spearman Severity Correlation — Monotonicity

| Corruption | ρ | p-value | Interpretation |
|:-----------|:-:|:-------:|:---------------|
| hot_pixel | **+1.000** | 0.000 | Perfect monotone, certain |
| event_flood | **+1.000** | 0.000 | Monotone but AUROC barely moves (0.504→0.554) |
| temporal_jitter | +0.700 | **0.188** | Positive trend, **not statistically significant** |
| polarity_flip | **+1.000** | 0.000 | Monotone, weak but consistent |
| event_rate_shift | **+1.000** | 0.000 | Monotone (phase jump at L3 drives this) |
| spatial_dropout | **-1.000** | 0.000 | **Perfect inverse — corrupted looks more normal** |

> [!IMPORTANT]
> `spatial_dropout ρ = -1.000` is a paper-level finding. It is not just failing — it is perfectly, statistically significantly anti-correlated. This is fully explainable by the PLIF dynamics and must be prominently discussed.

> [!NOTE]
> `temporal_jitter ρ = 0.700, p = 0.188` — **not significant** at α = 0.05. With only 5 severity levels, you need ρ > 0.9 for significance. This weakens the monotonicity claim for temporal_jitter specifically.

---

## 5. Temporal Features — CORRECTED (the "rescue to 0.85" does not reproduce)

> [!CAUTION]
> The original version of this section claimed temporal features rescue event_flood and spatial_dropout to ~0.83–0.85 and called it *"the most important finding in the entire benchmark."* **That result is an artifact and is retracted.** What follows is the corrected, reproducible picture.

### 5a. What was claimed vs. what reproduces

The original 0.85 numbers came from the **50-sample raw-trajectory path** (`load_traj_as_temporal_phi` on `trajs/`, capped at 50), with a train/eval split that pre-dated the `unified train/eval split` fix. Re-running the **exact same 50-sample method** on freshly extracted data:

| Corruption | Originally claimed (50-traj) | Reproduction (same 50-traj method, leakage-safe split) | Per-frame temporal (handcrafted, proper split) |
|:-----------|:----------------------------:|:------------------------------------------------------:|:----------------------------------------------:|
| `event_flood` | 0.830 / 0.848 | **0.291** | 0.625 |
| `spatial_dropout` | 0.817 / 0.846 | **0.037** | 0.541 |
| `temporal_jitter` | 0.853 / 0.905 | **0.061** | 0.602 |

**Diagnosis — two compounding causes:**
1. **Broken sample regime:** 50 samples = 35 train / 15 test, all from the *first 50 frames of one sequence* — tiny, autocorrelated, unrepresentative. A 28-D Mahalanobis fit on 35 points is near-degenerate (AUROC swings to 0.04).
2. **Train/eval leakage, since fixed:** the original temporal path almost certainly scored clean frames that were in its own fit set (no held-out split), so clean looked perfectly in-distribution by construction → inflated AUROC. The `unified train/eval split` commit added the held-out split, which deflated it to the honest level.

### 5b. The honest temporal result (leakage-safe, larger sample)

A 4-detector head-to-head on a freshly extracted 5-sequence subset (5,160 frames; all detectors fit on the same clean-train; 95% bootstrap CIs):

| Corruption | STATIC Maha | STATIC MLP-AE | TEMPORAL handcrafted | TEMPORAL AE |
|:-----------|:-----------:|:-------------:|:--------------------:|:-----------:|
| `event_flood` | 0.578 | 0.50–0.56 | **0.625** [.610,.639] | 0.520 |
| `spatial_dropout` | 0.468 | 0.457 | **0.541** [.526,.556] | 0.399 |
| `temporal_jitter` | 0.732 | **0.839** | 0.602 | 0.575 |

- **Handcrafted temporal significantly beats static on event_flood and spatial_dropout** (non-overlapping CIs) — but by **~0.05–0.08**, not a rescue to 0.85.
- The **Temporal AE is the *weakest* detector** — the handcrafted stats do the temporal work, not the learned AE.
- **Caveat:** this subset is **not representative** — its static event_flood AUROC is 0.578 vs the full-data 0.409 (extraction caps to the *first* N sequences, a biased sample). The relative ranking is internally valid; absolute numbers are not. Only 5 distinct scenes, so between-scene uncertainty exceeds the frame-level CIs.

**The corrected implication:** temporal features give a *modest, reproducible* improvement on the two hardest corruptions — they do **not** dominate static across the board. The paper cannot be framed as "temporal rescues everything." Establishing a genuine temporal advantage would require (a) a representative multi-sequence extraction (not first-N), (b) full-dataset temporal coverage, and (c) likely sequence-level aggregation. See `Docs/performance_brief.md`.

---

## Summary Table — What Works and What Doesn't (CORRECTED)

Reference-detector (Mahalanobis) static-φ AUROC on full 343k data at L5, vs. the honest leakage-safe temporal best. (The old "Temporal (50 samples)" column has been removed — those numbers were artifacts; see §5.)

| Corruption | Static-φ Mahal (full, L5) | Temporal best (honest) | Verdict |
|:-----------|:-------------------------:|:----------------------:|:--------|
| hot_pixel | **1.000** | 1.000 | Trivial — both perfect |
| temporal_jitter | 0.709 | ~0.60 | Static wins (standardized static AE ~0.84) |
| event_rate_shift | 0.675 | n/a | Static + activity-scalar → 0.85 |
| polarity_flip | 0.429 (below chance) | ~0.55 (two-sided) | Hard; modest gains only |
| **event_flood** | **0.408 (below chance)** | **~0.63 (handcrafted temporal)** | Temporal helps modestly, not a rescue |
| **spatial_dropout** | **0.286 (anti-detectable)** | **~0.54 (handcrafted temporal)** | Temporal lifts above chance, but weak |

> Note: §1–§4 "best detector" static numbers (e.g. event_flood 0.554, event_rate_shift 0.902) mix detectors and some pre-date the split fix; the reference-detector full-data values above are the trustworthy baseline. Regenerate §1–§4 tables before citing them in the paper.

---

## What These Results Mean for the Paper

| Finding | What to Do With It |
|:--------|:-------------------|
| Autoencoder wins on static phi | Argue phi has non-linear manifold structure; parametric methods insufficient |
| Block 1 best for sensor noise, Block 4 best for dynamics | Supports corruption classification (Ideas.md Idea 3); motivates layer selection |
| event_flood invisible to static phi | Must be explained theoretically (uniform noise = indistinguishable from busy scene) |
| spatial_dropout ρ = -1.0 (anti-detection) | Must be explained; actually strengthens the theoretical story (silence = low Mahal) |
| ~~Temporal features rescue all hard corruptions~~ **RETRACTED** | Temporal gives only a modest (~0.05–0.08) reproducible gain; do **not** frame the paper as a temporal rescue (see §5) |
| Phase transition in event_rate_shift at L3 | Footnote or supplementary; suggests SNN firing regime threshold |
| Honest hard-corruption story | event_flood/spatial_dropout/polarity_flip are genuinely hard for membrane statistics; the contribution is the *theory* (why) + neuromorphic-hardware framing, not a magic temporal fix |
