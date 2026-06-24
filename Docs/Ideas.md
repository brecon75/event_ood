# Research Ideas to Strengthen the Paper

> Based on a full read of the codebase and the actual results in `implementation_status.tex`:
> kNN AUROC=0.754, PCA-Mahal=0.736, hot_pixel=1.000, event_flood=0.554, spatial_dropout=0.439–0.573.

> [!CAUTION]
> **Update (2026-06-11):** The temporal-feature "rescue to ~0.85" that motivated Idea 4 was found to be **non-reproducible** (leakage + 50-sample noise; see `Docs/Findings.md` §5 and `Docs/performance_brief.md`). Honest temporal gain is only ~0.05–0.08. Idea 4 is re-scoped below. New, empirically-tested performance levers (two-sided scoring, activity scalar, sequence aggregation, meta-fusion) are catalogued in `Docs/performance_brief.md` §6–7 — read that before proposing detector changes.

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

## Idea 4 — Re-establish the Temporal Result Honestly (RE-SCOPED)

**Effort**: Medium. **Impact**: Now uncertain — was over-sold.

**Status:** The online temporal-phi computation this idea proposed has **already been implemented** — `monitor.py` now computes `temporal_phi` (28-D) and `temporal_gap` online for all frames, not just 50. When measured properly (leakage-safe split, thousands of frames), temporal beats static by only **~0.05–0.08** on the hard corruptions — a real but modest effect, not the 0.85 rescue the original Findings.md claimed (that was an artifact; see §5 there).

**What's actually left to do** to get a defensible temporal result:
1. **Representative extraction.** `extract.py --max-seq N` caps to the *first* N sequences (`seq_dirs[:N]`), which is a biased sample (a 5-seq subset gave static event_flood 0.578 vs full-data 0.409). Add stride/random sequence sampling so the subset reflects the dataset.
2. **Full-dataset temporal coverage.** Extract `temporal_gap`/`temporal_phi` for all 6 corruptions over the full data; the Temporal AE has only ever trained on ≤4k frames.
3. **Sequence-level aggregation** (see Idea 11) — likely the biggest lever, since the corruption is consistent per-sequence.
4. Only then claim a temporal advantage, with bootstrap CIs across *sequences*, not frames.

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

## Idea 11 — Sequence-Level Score Aggregation (biggest untested lever)

**Effort**: Low. **Impact**: Potentially Very High.

Corruption is applied to a *whole sequence*, so every frame carries the same bias. Averaging the per-frame OOD score within a sequence integrates that consistent bias and tightens class separation by ~√(frames-per-sequence) — a weak 0.55 per-frame signal can become strongly separable per *recording*. Needs `seq_lens` (now saved). Changes the decision granularity to per-sequence (which is what you actually want for "is this recording corrupted?"). For the below-chance corruptions, aggregate the *two-sided* score. Not yet measured — should be the first thing tried.

## Idea 12 — Two-Sided / Folded Scoring for Anti-Detectable Corruptions

**Effort**: Very Low. **Impact**: Medium (fixes the inversion).

`event_flood`, `spatial_dropout`, `polarity_flip` are **below chance** for static φ — informative but inverted (corrupted sits *closer* to the clean mean). A folded score `|dist − mean_clean_dist| / std` flags "too typical" as well as "too far," recovering the wasted signal: spatial_dropout 0.286→0.531, event_flood 0.408→0.577, polarity 0.429→0.547 (measured). Trade-off: it *degrades* already-well-detected corruptions, so it must be combined per-corruption, not applied globally (see Idea 13).

## Idea 13 — Many-Cheap-Views → Meta-Classifier Fusion

**Effort**: Low. **Impact**: High (banks the per-corruption envelope).

No single transform wins on all corruptions (they live in different feature geometries). Measured per-corruption winners: activity-scalar → event_rate_shift 0.85; standardized static → jitter 0.82; two-sided → the inverted trio; handcrafted-temporal → flood/dropout. Feed *all* cheap views (baseline distance, two-sided distance, global activity scalar, standardized φ, handcrafted temporal) as features into the existing Stage-3 LogReg meta-classifier so it learns per-corruption which signal to trust — capturing the oracle envelope without any single view's regressions. Also bump `MAX_FIT_SAMPLES` (3000 → 30k+; the covariance is currently fit on 1.4% of available clean data — free accuracy).

*(Full measured numbers for Ideas 11–13 are in `Docs/performance_brief.md` §6–7.)*

---

## Priority Ranking

| Priority | Idea | Effort | Paper Impact |
|:---------|:-----|:-------|:-------------|
| **#1** | Theory: PLIF leaky integrator derivation | None (pure math) | Very High — turns benchmark into framework |
| **#2** | Neuromorphic hardware argument | None (writing) | Very High — unique positioning claim |
| **#3** | Sequence-level aggregation (Idea 11) | Low | Potentially Very High — biggest untested performance lever |
| **#4** | Many-views → meta-fusion + more fit samples (Idea 13) | Low | High — banks measured per-corruption envelope |
| **#5** | PCA subspace scatter plots | Very Low | High — best paper figure |
| **#6** | Severity regression (R² score) | Very Low | High — strong empirical result |
| **#7** | Two-sided scoring (Idea 12) | Very Low | Medium — fixes the below-chance inversion |
| **#8** | Corruption classification (which type?) | Low | High — practical utility |
| **#9** | Free rider ablation | Medium-Hard | Very High — validates core claim |
| **#10** | Re-establish temporal honestly (Idea 4, re-scoped) | Medium | Uncertain — was over-sold; modest gain only |
| **#11** | Conformal prediction intervals | Medium | Medium — safety community appeal |
| **#12** | Online streaming OOD (EMA) | Medium | Medium — deployment story |
| **#13** | DSEC dataset transfer | Hard | High if it works |

---

## Quick Wins (Can Be Done Today)

1. **Idea 7** — PCA scatter plots, ~20 lines in `analysis/analyse.py`
2. **Idea 5** — Severity regression, ~5 lines in `analysis/analyse.py`
3. **Idea 3** — Corruption classification, ~15 lines in `analysis/analyse.py`
4. **Ideas 1 & 2** — Pure writing/math, no code at all

---

# Addendum (2026-06-11) — Performance levers for the below-chance corruptions

> [!IMPORTANT]
> **Empirical update (2026-06-11):** these ideas were tested on the full 343k static-φ data
> (`outputs_stale_20260611_104023/phi`); scripts and numbers in `test_ideas/RESULTS.md`. Summary:
> - **A1 (χ² two-sided) is REFUTED** — ≈0 effect; clean d² is not a tight band (median ~14, not
>   D=2112), so there is nothing to exploit. Dropped.
> - **A2b kNN** is a real free win on event_rate_shift (0.62→0.78).
> - **A2a per-layer** is a big free win on temporal_jitter (deep **layer-4 only = 0.93** vs 0.78
>   pooled) — but the *early*-layer hypothesis for flood/dropout was wrong (no help).
> - **Idea 11 sequence-aggregation** rescues **event_flood to ~1.0** but *hurts* event_rate_shift
>   and spatial_dropout — it only works when the per-frame bias is consistent and same-sign.
> - The brief's strong below-chance numbers (flood 0.408 / dropout 0.286) **did not reproduce**
>   here (0.554 / 0.438) under either split — treat like the 0.85.
> - **Still unsolved cheaply: polarity_flip and spatial_dropout** (~0.47–0.59) — only these
>   genuinely need idea A3 (spatial stats), which needs re-extraction and was not tested.

Added after a full read of `Docs/performance_brief.md` §6–7. These are scoped specifically at
lifting AUROC on `event_flood` / `spatial_dropout` / `polarity_flip` (below chance for static φ)
and pushing `temporal_jitter` / `event_rate_shift` toward >0.9, without regressing `hot_pixel`
and without violating the constraints (BATCH_SIZE=1; per-frame φ cheap; raw trajectories 15 TB).
Ideas A1–A2 are genuinely new; A3–A5 are new twists/extensions on existing Ideas 11–13.

## Central thesis the brief slightly gets wrong

"Below chance for static φ" is treated as a property of the **representation**. It is mostly a
property of the **scoring rule**. One-sided Mahalanobis `d²(x) = (x−μ)ᵀΣ⁻¹(x−μ)` ranks "far from
the clean mean" as OOD. flood / dropout / polarity make the membrane *quieter / more regular*, so
corrupted frames land at **smaller** `d²` than held-out clean → AUROC < 0.5. The signal is present
in φ; only the sign is flipped. **Exhaust free re-scoring of existing φ (A1, A2, and Idea 12)
before extracting anything new.**

### Corrections / skeptical flags
- **polarity_flip at 0.429 contradicts "polarity-symmetric features"** (§3). Perfect symmetry →
  identical φ → AUROC exactly 0.5. The 0.071 below chance means a small, *consistent* asymmetric
  shift exists — a detectable direction, not noise.
- **The √(frames-per-sequence) estimate for Idea 11 is optimistic.** Consecutive frames are
  heavily autocorrelated, so effective N ≪ N. Variance reduction scales with √N_eff. Still a
  large win, but measure N_eff (lag-1 autocorrelation of per-frame scores) — don't assume it.
- **Do not rank any idea on first-N subsets.** `extract.py --max-seq N` is `seq_dirs[:N]`, biased
  (flood 0.578 vs full 0.409). That is the sampling pathology behind the non-reproducible 0.85.
  Switch the dev harness to a stratified random sample of sequences first (also noted in Idea 4).
- **Reliability/mAP story is currently void** until clean `det_outputs` has nonzero confident
  detections (memory: `det-outputs-all-zero`). Not an AUROC idea, but a hole in the paper.

## Idea A1 — Whole-vector χ² two-sided score `|d² − d²_clean_median|`
**Effort**: Very Low (scalar transform on existing scores). **Impact**: High — fixes the inversion
*without* the jitter regression that per-feature folding (Idea 12) causes.

Clean `d²` ~ χ²_D (D≈2112), tightly concentrated; both "too far" and "too close to the mean" are
anomalous. Per-*feature* folding (Idea 12) hurt jitter (0.709→0.615) because it discards the
consistent directional shift on every dimension. Doing it at the **whole-vector** level keeps it:
jitter has `d² > D` so `|d²−center|` stays monotonic with `d²` (jitter preserved), while
dropout/flood have `d² < D` so the same score flips them up.

Expected per-frame: hot_pixel ~1.0, jitter ~0.71 (preserved), rate_shift ~0.68 (preserved), flood
~0.41→~0.59, polarity ~0.43→~0.57, dropout ~0.29→~0.71. Every corruption ≥ chance, easy ones
untouched. **This is the per-frame complement to Idea 11** — combine them and every corruption
becomes strongly separable per sequence.

**Kill experiment.** Center on the *empirical* held-out-clean median `d²` (not theoretical D — Σ
fit on ≤50k samples makes the clean band wider than χ² predicts). Recompute AUROC. Killed if the
empirical clean `d²` distribution overlaps both corrupted tails, or it regresses jitter like
per-feature folding did.

## Idea A2 — Per-layer / per-moment slicing of existing φ + non-Mahalanobis detectors
**Effort**: Free (re-score existing φ / fitted detectors). **Impact**: Medium–High — possible free
win, and fills a blind spot.

Two free analyses on data you already have:
1. **Per-layer slicing.** φ pools 4 PLIF layers (64/128/256/256). Deep layers carry the learned
   invariances (polarity symmetry, activity normalization) that *create* the inversion; the
   earliest layer is closest to raw event density, so flood (more events → higher σ²) and dropout
   (fewer events → lower σ²) should shift early-layer moments strongly and monotonically. Stage-8
   ablation splits μ/σ²/κ but not per-layer. Slice columns, refit Mahalanobis per layer.
2. **Report kNN / GMM / RealNVP / OCSVM on the hard corruptions.** The brief shows only
   Mahalanobis. kNN local density scores "is this in a sparsely-populated region of clean space" —
   a different geometry that may not invert. These detectors are already fitted and saved.

**Kill experiment.** Tabulate per-layer Mahalanobis AUROC and per-detector AUROC for flood/dropout
at L5. Killed if no single-layer slice beats pooled-φ AND all density detectors also go below 0.5
(→ inversion is intrinsic to GAP'd φ, not the detector — weight shifts to Ideas 11/A1 and A3).

## Idea A3 — Cheap spatial summary stats at extraction (new twist on §7 "spatial info")
**Effort**: Medium (touches `monitor.py`/`extract.py`, needs re-extraction). **Impact**: High on
dropout/flood — the most mechanism-targeted shot at the two worst ones.

§7 frames spatial info as needing 15 TB of raw V(t). It doesn't. GAP destroys exactly the
structure these corruptions live in: spatial_dropout zeros spatial *regions* (GAP mean can stay
flat while spatial heterogeneity changes); event_flood fills space (spatial entropy / active-pixel
fraction rise). Compute a handful of **spatial scalars per channel** at extraction — active-pixel
fraction, spatial variance of the per-pixel mean, quadrant means, center-of-mass displacement,
max-pixel — a few scalars per channel, not H×W maps. Respects B=1 and storage.

**Kill experiment.** Add spatial-moment φ on ~20 stratified sequences, refit Mahalanobis. Killed if
spatial-φ AUROC on dropout/flood doesn't beat GAP-φ (→ spatial structure already normalized out of
V_mem, unrecoverable downstream).

## Idea A4 — Make the meta-fusion (Idea 13) honest via outlier exposure across corruption *types*
**Effort**: Low. **Impact**: High if transfer holds; the brief's own "different geometries" finding
predicts it may not — medium confidence.

Idea 13 (LogReg over cheap views) needs negatives to train — it can't be fit clean-only. State this
explicitly and keep it honest by holding out corruption *types*: train the combiner on {hot_pixel,
jitter, rate_shift}, test on {flood, dropout, polarity}. This is standard outlier exposure and uses
the existing Stage-11 cross_corruption split. Feed A1's two-sided score as one of the views.

**Kill experiment.** Cross-type CV as above. Killed if held-out-type AUROC ≤ the best single
unsupervised view (the "no single transform wins" finding actively predicts this).

## Idea A5 — Temporal, scoped to the reproduced gain (extends Idea 4)
**Effort**: Cheap extraction (28-D temporal_phi, no 15 TB). **Impact**: Modest (~0.05–0.08) — do
not oversell; this is the lever that produced the non-reproducible 0.85.

Extract temporal_phi at full scale, fuse its *score* with static (separate detectors — don't drown
28 dims inside 2112-D Mahalanobis). Lag-1 autocorrelation of V across the 10 SNN steps is exactly
what temporal_jitter perturbs.

**Kill experiment.** Score-fuse on the stratified subset, compare to static with leave-one-scene-out
CIs. Killed if the fused CI overlaps static-alone.

## Recommended order
Run **A1 + A2 + Idea 12** today (free re-scoring — reveals how much of "below chance" is a scoring
artifact). Layer **Idea 11** on top of A1's two-sided per-frame score (most likely path to high
AUROC, cheap, reproducible). Only if those plateau, spend re-extraction budget on **A3**, then
**A4 / A5**. Before trusting any dev number: fix first-N sampling and confirm `det_outputs` is
nonzero — otherwise you risk minting a second 0.85.

---

# Addendum Round 2 (2026-06-11) — grounded in the `test_ideas/` results

> [!IMPORTANT]
> **Empirical update — R1–R5 + R10 tested** (`test_ideas/run_tier1.py`, `test_polarity_r10.py`,
> numbers in `RESULTS.md`). Outcome:
> - **R1 RMD is the first win on spatial_dropout: 0.44 → 0.60** (RMD-iso). It strips the magnitude
>   term, which inverts the magnitude-driven corruptions but rescues the structure-driven one.
> - This exposes a **two-class taxonomy**: *magnitude* corruptions (hot_pixel, rate_shift, flood,
>   jitter) — detected by raw d²/kNN/GMM, destroyed by RMD; vs *structure* corruption
>   (spatial_dropout) — detected only by RMD.
> - **R3 GMM lifts temporal_jitter to 0.877** (per-layer-L4 Maha 0.93 still best).
> - **R4 PCA-residual** modestly helps jitter (0.83) / polarity (0.59).
> - **R5 whitened kNN — REFUTED** (worse than plain kNN; inverts rate_shift).
> - **R10: polarity_flip is undetectable from V_mem (~0.55, expected ceiling)**; a near-free *input*
>   ON/OFF-balance scalar gives ~0.69 per-frame (input-side, not membrane — scope accordingly).
> - Net: spatial_dropout now has a working detector direction (RMD → R6 next); polarity is a V_mem
>   dead end.

After the empirical run (`test_ideas/RESULTS.md`): A1 is dead, the cheap static-φ space is largely
mined out (kNN, deep-layer slicing, flood-aggregation are the wins), and the two genuinely
unsolved corruptions are **polarity_flip (~0.59)** and **spatial_dropout (~0.47)**. Two facts now
anchor everything: (i) clean d² is tiny and non-stationary (median ~14 ≪ D=2112, mean ~45 across
scenes) → φ lives on a low-dim manifold and raw Mahalanobis is dominated by scene magnitude;
(ii) flood/dropout/polarity **preserve the marginal moments** [μ,σ²,κ], so the signal that's left
must be in structure GAP and marginal-moments throw away (spatial layout, channel co-activation),
or on the input side.

## Tier 1 — testable now on the full static-φ (no re-extraction)

### R1. Relative Mahalanobis Distance (RMD)
- **Mechanism.** Clean d² is non-stationary across scenes (calib median 14 vs eval mean 45), so raw
  Maha is swamped by scene/background magnitude — the exact thing that produces apparent inversions
  and washes out weak corruption signal. RMD = d²_foreground(x) − d²_background(x), with background
  a single broad Gaussian; subtracting the background term isolates corruption-specific deviation
  (Ren et al. 2021, the standard fix for magnitude-dominated Mahalanobis).
- **Impact.** The near-chance trio (flood, dropout, polarity); may also stabilize rate_shift.
- **Cost.** One extra Gaussian, existing data. Free. **Kill:** RMD ≤ raw Maha on the trio.

### R2. kNN on the layer-4 slice (and per-layer kNN)
- **Mechanism.** Layer-4 carried jitter (0.93 under Maha) and kNN beat Maha globally; restricting
  local-density scoring to the most-informative layer drops noise from uninformative layers.
- **Impact.** temporal_jitter (possibly >0.93), event_rate_shift. **Cost.** Free.
- **Kill:** per-layer kNN never beats both per-layer Maha and pooled kNN.

### R3. GMM and RealNVP density (finish what A2b started)
- **Mechanism.** kNN > Maha proves the clean manifold is non-Gaussian; a 5-component GMM / flow
  models multi-modality a single Gaussian misses. Both are already fitted in the pipeline but were
  never reported per-corruption (only kNN was tested). **Impact.** the trio. **Cost.** Free.
- **Kill:** neither beats kNN.

### R4. PCA off-manifold residual score
- **Mechanism.** d²≈14 ≪ D=2112 means clean φ lives on a low-dim manifold; most dims are redundant.
  Fit PCA on clean, score = reconstruction-residual energy in the *discarded* subspace (not
  Mahalanobis within the top-k). Corruptions that nudge φ slightly off the clean manifold surface
  in the residual even when invisible to in-subspace distance.
- **Impact.** dropout, polarity (subtle off-manifold pushes). **Cost.** Free (PCA detector exists).
- **Kill:** residual-energy AUROC ≤ raw Maha.

### R5. Whitened kNN
- **Mechanism.** Whiten φ by the clean covariance (Mahalanobis geometry), *then* kNN — couples the
  global metric that normalizes feature scales with the local density that handles non-Gaussianity.
- **Impact.** the trio + rate_shift. **Cost.** Free. **Kill:** ≤ plain kNN.

## Tier 2 — one cheap, targeted re-extraction pass (per-frame, B=1, small output)

Do all of these in a single extraction run so the GPU cost is paid once.

### R6. Cross-channel covariance summary  *(highest-value new representation)*
- **Mechanism.** The strongest empirical fact is that flood/dropout/polarity preserve the *marginal*
  moments — so the discriminative signal must be in *which channels co-activate together*, which GAP
  + per-channel moments discard entirely. Add cheap per-frame summaries of the channel×channel
  structure: off-diagonal correlation energy, participation ratio (effective # active channels),
  top-k eigenvalue ratios of the per-frame channel covariance. A small vector per layer, B=1-safe,
  tiny storage.
- **Impact.** Directly targets the two unsolved ones — polarity_flip, spatial_dropout.
- **Kill:** covariance-summary φ doesn't beat marginal φ on those two → the joint structure is also
  preserved, and the signal really isn't in the membrane.

### R7. Spatial pooling beyond GAP (= idea A3, reaffirmed)
- Active-pixel fraction, spatial std/max, quadrant means, center-of-mass per channel. Targets
  spatial_dropout and event_flood. Same re-extraction pass as R6.

### R8. Spike rate & entropy at full scale
- **Mechanism.** The pipeline *already* extracts `spike_rate`/`spike_entropy` (704-D) but only for
  clean/flood/hot_pixel at one sequence. Firing rate responds directly to event density (flood ↑,
  dropout ↓) and is a different signal from sub-threshold V_mem moments. Just run the existing
  extractor for all corruptions at full scale — no new code.
- **Impact.** flood, dropout, rate_shift. **Kill:** spike features ≤ static φ on those.

### R9. Input-side polarity-balance scalar (for polarity_flip) — see R10.

## Tier 3 — conceptual / scoping

### R10. Polarity_flip is probably undetectable from V_mem *by construction* — prove it, then scope it
- **Mechanism.** The brief's own diagnosis is that the model learned polarity-*symmetric* features.
  If true, V_mem is near-invariant to a polarity flip → φ barely moves → **AUROC ≈ 0.5 is the
  expected ceiling, not a fixable deficiency.** The only signal lives on the *input* side: the
  positive/negative event-count balance and its spatial pattern, which the flip inverts. Add a
  near-free input-statistic channel (Σ positive bins vs Σ negative bins per frame from the
  histogram). Honest caveat: this is input-side, not membrane-side — it broadens the framing
  slightly but stays ~zero-compute and is complementary.
- **Kill experiment (do this first, it's cheap).** Load a few clean sequences, apply polarity_flip,
  and measure whether per-frame pos/neg balance separates clean from flipped. If balanced scenes
  give ~0.5, polarity_flip is genuinely undetectable from this sensor representation and should be
  scoped *out* of the V_mem claim rather than chased.

### R11. Detector routing via a cheap corruption-type classifier (sharpens A4)
- **Mechanism.** Empirically no single detector wins — kNN(rate_shift), L4-Maha(jitter),
  aggregation(flood). A cheap multinomial classifier on φ predicts the corruption family, then
  routes to that family's best detector (outlier-exposure + held-out types, per A4).
- **Impact.** Captures the oracle envelope. **Kill:** held-out-type routing accuracy too low to beat
  the best single detector.

## Round-2 priority
R1 (RMD) and R3 (GMM/flow) are the highest-value free tests — run with R2/R4/R5 in one sitting.
If the unsolved trio is still stuck, the re-extraction pass should bundle **R6 + R7 + R8** together
(one GPU run), with **R10's cheap input-balance check done first** to decide whether polarity_flip
is even worth re-extracting for.
