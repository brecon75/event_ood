# Research Ideas to Strengthen the Paper

> Based on a full read of the codebase and the actual results in `implementation_status.tex`:
> kNN AUROC=0.754, PCA-Mahal=0.736, hot_pixel=1.000, event_flood=0.554, spatial_dropout=0.439–0.573.

---

## The Core Narrative Problem to Fix First

The current results show a massive gap between corruptions:

| Corruption | Best AUROC | Interpretation |
|:-----------|:-----------|:---------------|
| `hot_pixel` | **1.000** | Trivial — saturates neurons completely |
| `temporal_jitter` | ~0.70 | Detectable |
| `event_flood` | **0.554** | Near-random guess |
| `spatial_dropout` | **0.439–0.573** | Below random at some severities |

Any reviewer will ask: *"Why does your method fail on half the corruptions?"*
The ideas below either explain this theoretically or fix it empirically.

---

## Idea 1 — Theory: Why Does Vmem Work? (Leaky Integrator Analysis)

**Effort**: None (pure math). **Impact**: Very High.

The PLIF update rule is:

```
V[t] = (1 - 1/τ) · V[t-1] + W · X[t]
```

Under each corruption type, the effect on the steady-state distribution of V is predictable:

- **hot_pixel** adds a constant DC offset to `X[t]` for specific spatial locations → `V` saturates toward `θ` → explains AUROC = 1.000 (mathematically inevitable, not lucky)
- **temporal_jitter** scrambles the time ordering of `X[t]` → destroys the temporal autocorrelation structure of `V(t)` → explains why the lag-1 autocorr feature is effective
- **spatial_dropout** zeros out some channels of `X[t]` → reduces `σ²` but not `μ` → explains why variance is the best single moment feature
- **event_flood** adds uniform noise across all pixels → shifts the mean of **all** neurons equally → indistinguishable from a "busy, high-activity scene" → explains AUROC = 0.554 (cannot separate from a noisy clean scene)

A one-page theoretical section with equations and a diagram transforms an empirical benchmark into a **principled framework**. This is the single strongest addition to the paper.

---

## Idea 2 — The Neuromorphic Hardware Advantage Argument

**Effort**: None (writing only). **Impact**: Very High — unique positioning.

On Intel Loihi, BrainScaleS, and SpiNNaker: **the membrane potential V is a native hardware register**. Reading it costs approximately zero additional operations — it is computed as a side effect of the forward pass that already happens.

Every competing ANN-based OOD method requires extra computation that is impossible on neuromorphic chips:

| Method | Overhead | Neuromorphic Compatible? |
|:-------|:---------|:------------------------|
| Vmem-phi (ours) | 0 (native register read) | **Yes** |
| MSP / Energy | Softmax over logits | No |
| ODIN | Temperature-scaled softmax | No |
| ReAct | Feature clipping + re-forward | No |
| ViM | SVD of features | No |
| GradNorm | Gradient backpropagation | No |
| kNN / Mahalanobis on ANN features | Auxiliary ANN forward pass | No |

This reframes the paper's claim from *"Vmem is slightly better than ANN methods on Gen1"* to **"Vmem-phi is the only OOD detection method that works at all on the target deployment hardware."** That is a fundamentally stronger and more unique contribution.

---

## Idea 3 — Corruption Classification (Multi-Class, Not Binary)

**Effort**: ~15 lines of sklearn code. **Impact**: High — practical utility.

Instead of just binary OOD detection (clean vs. corrupt), use phi to classify **which of the 6 corruption types** is occurring. This is a 7-class problem: clean + hot_pixel + event_flood + temporal_jitter + polarity_flip + event_rate_shift + spatial_dropout.

Run a lightweight `LinearSVC` on phi. If it achieves >50% top-1 accuracy:
- phi encodes **corruption-type-specific signatures** in its geometry
- the model can trigger **corruption-specific mitigations** (spatial filtering for hot pixels, temporal averaging for jitter)
- this is the bridge to Point 8 (closed-loop adaptation) from the rescue ladder

Bonus: the confusion matrix tells you which corruptions look the same to the SNN (likely event_flood and spatial_dropout will be confused).

---

## Idea 4 — Fix the 50-Sample Temporal Analysis Limitation

**Effort**: Medium. **Impact**: High — unlocks Level 4 properly.

The biggest current limitation: temporal analysis (Level 4) only uses **50 samples** because storing raw trajectories for 343k frames would need ~15 TB. The temporal AE trained on 50 samples is scientifically weak.

**Fix**: Move the temporal phi computation *online* into `monitor.py` during `extract.py`. The features in `load_traj_as_temporal_phi` (11 scalars per channel per layer) are already written — just run them during extraction instead of saving the raw `V(t)` tensor:

```
Raw trajectory:  (T=10, N, C, H, W)  ← 15 TB for all runs
Temporal phi:    (N, 22 * n_layers)   ← ~50 MB for all runs
```

This makes the handcrafted temporal features run on all 343k frames instead of 50, which is likely one of the strongest quantitative results in the paper if it works well.

---

## Idea 5 — Severity Regression (Not Just Binary Detection)

**Effort**: ~5 lines of sklearn. **Impact**: High — strong empirical evidence.

Instead of only treating severity as an ordinal label for Spearman correlation, train a **Ridge regression** model to predict severity level (1–5) from phi and report `R²` and `MSE`.

This tests a much stronger hypothesis: not just *"does OOD score go up as severity increases"* (monotonicity), but *"can we predict exactly which severity level this is?"*

If a simple linear regressor on phi achieves `R² > 0.7`, it means phi encodes corruption intensity **continuously**, not just as a discrete alarm. That is extraordinary evidence for the quality of the representation.

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
# X = phi for all corrupted runs, y = severity labels
```

---

## Idea 6 — Conformal Prediction Intervals on OOD Scores

**Effort**: ~20 lines using `mapie` library. **Impact**: Medium — safety community appeal.

Instead of reporting a single point estimate AUROC, compute **conformal prediction sets** on the OOD scores. For each test frame, output one of:
- "Clean at 95% confidence" (accept)
- "OOD at 95% confidence" (alarm)  
- "Ambiguous" (abstain)

Using the clean calibration set (343k clean frames) to calibrate the non-conformity threshold gives **statistically guaranteed false positive rates** — not just empirical ones.

For safety-critical autonomous driving, a decision with *"I am 95% confident this is OOD"* is far more useful than a raw AUROC score. Conformal prediction is also very trendy in the safety-ML community (2023–2025 publications).

---

## Idea 7 — PCA Subspace Scatter Plots (Best Paper Figure)

**Effort**: ~20 lines of code. **Impact**: High — best possible visualization.

Run PCA on phi separately for each corruption. Plot a 2×3 grid of scatter plots showing clean (blue) vs. corrupt (red) in PCA-2D space, one subplot per corruption.

This will visually show **exactly** why:
- hot_pixel is trivially separated (two non-overlapping clouds)
- event_flood heavily overlaps with clean (two overlapping blobs)
- deeper layers separate better than shallower layers

Reviewers will immediately understand the AUROC numbers by looking at the figure — much more compelling than a table alone.

---

## Idea 8 — Online Streaming OOD with Exponential Moving Average

**Effort**: Medium. **Impact**: Medium — deployment story.

All current analysis is offline (fit on clean, evaluate on corrupt). Implement an **online** version:

For each incoming histogram frame at time `t`:
1. Compute phi (already done during forward pass)
2. Update EMA: `mu_ema[t] = alpha * phi[t] + (1 - alpha) * mu_ema[t-1]`  with `alpha = 0.1`
3. Compute Mahalanobis distance from current phi to EMA mean
4. Alarm if distance > threshold

This simulates a deployed online OOD detector with no retraining needed, and demonstrates that phi-based OOD detection works **causally** (no future information required). Directly addresses the reviewer question *"does this work in a real deployed system?"*

---

## Idea 9 — The "Free Rider" Ablation (Critical Validity Check)

**Effort**: Medium-Hard (requires re-running extraction with random weights). **Impact**: Very High — validates the core claim.

Compare three conditions:
- **(A)** Vmem-phi from the **trained** SNN (current results)
- **(B)** Vmem-phi from a **randomly initialized** SNN (same architecture, no training)
- **(C)** Raw input statistics — mean, variance, kurtosis computed directly on the event histogram, never going through any network

If **(A) >> (C)**: The SNN's learned representations are genuinely informative. Core claim validated.  
If **(A) ≈ (B)**: The SNN architecture matters, but not the learned weights — any SNN would work.  
If **(A) ≈ (C)**: The membrane potentials aren't adding anything beyond raw input statistics. **This would be a major problem for the paper's premise.**

This is the most important unasked question in the paper. Running it either strongly validates or forces you to rethink the framing before submission.

---

## Idea 10 — DSEC Dataset Transfer

**Effort**: Hard. **Impact**: High if it works.

The `reporting/build_paper_tables.py` script already references `dsec_transfer.csv`, suggesting this was planned. Run the same benchmark on the **DSEC dataset** (stereo event camera, outdoor urban driving scenes).

This tests generalization across:
- Different event camera sensors (DAVIS346 on Gen1 vs. DVXplorer on DSEC)
- Different scenes (indoor object detection → outdoor driving)
- Potentially different model architectures

A single-dataset paper is always vulnerable to *"this only works on Gen1"* reviewer criticism. DSEC is the de facto standard for event camera outdoor driving and is highly cited.

---

## Priority Ranking

| Priority | Idea | Effort | Paper Impact |
|:---------|:-----|:-------|:-------------|
| **#1** | Theory: PLIF leaky integrator derivation | None (pure math) | Very High — turns benchmark into framework |
| **#2** | Neuromorphic hardware argument | None (writing) | Very High — unique positioning claim |
| **#3** | PCA subspace scatter plots | Very Low | High — best paper figure |
| **#4** | Severity regression (R² score) | Very Low | High — strong empirical result |
| **#5** | Corruption classification (which type?) | Low | High — practical utility |
| **#6** | Fix 50-sample temporal limitation | Medium | High — unlocks Level 4 |
| **#7** | Free rider ablation | Medium-Hard | Very High — validates core claim |
| **#8** | Conformal prediction intervals | Medium | Medium — safety community appeal |
| **#9** | Online streaming OOD (EMA) | Medium | Medium — deployment story |
| **#10** | DSEC dataset transfer | Hard | High if it works |

---

## Quick Wins (Can Be Done Today)

1. **Idea 7** — PCA scatter plots, ~20 lines in `analysis/analyse.py`
2. **Idea 5** — Severity regression, ~5 lines in `analysis/analyse.py`
3. **Idea 3** — Corruption classification, ~15 lines in `analysis/analyse.py`
4. **Ideas 1 & 2** — Pure writing/math, no code at all
