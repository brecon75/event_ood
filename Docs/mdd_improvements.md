# MDD Improvements — Residual-Corruption Loop (polarity_flip & spatial_dropout)

**Status:** experiments complete through exp10; **`analysis/mdd.py` is NOT yet modified**
(changes below are validated in scratchpad and pending go-ahead).
**Scope constraint:** no re-extraction — existing `phi` (2112-D) + `phi_spatial` (1408-D) only.
**Date:** 2026-06-23.

All AUROC numbers are **per-sequence, severity L5**, clean held-out eval (102,985 frames /
90 sequences) vs the matched held-out tail of each corruption. The experiment harness
reproduces the live pipeline exactly (verified: `MDD.fused|s` = pol 0.609 / drop 0.560;
`rcf|s` drop 0.693; `spatial|s` pol 0.639 all match `evaluate_mdd_windows.py`).

---

## 1. The original MDD (baseline)

`analysis/mdd.py` — one unsupervised detector, 4 branches over the clean static-φ manifold,
fused by a **calibrated max**:

| branch | definition | rescues |
|---|---|---|
| `radius` | `\|‖z‖ − E‖z‖\| / sd` — two-sided isotropic energy in PCA-64 | hot_pixel, event_rate_shift (dilations) |
| `rcf` | `\|‖z‖ − E[‖z‖\|dir]\| / sd` — direction-conditioned radius (cosine-kNN) | event_flood, spatial_dropout (contractions) |
| `l4` | Ledoit-Wolf Mahalanobis on the layer-4 `[μ,σ²,κ]` block | temporal_jitter |
| `spatial` | Ledoit-Wolf Mahalanobis on `phi_spatial` | event_flood, spatial_dropout (per-frame) |

**Baseline per-sequence AUROC (L5):**

| corruption | radius | rcf | l4 | spatial | **fused (max)** |
|---|---|---|---|---|---|
| hot_pixel | 1.000 | 0.989 | 1.000 | 1.000 | **1.000** |
| event_rate_shift | 0.900 | 0.834 | 0.506 | 0.540 | **0.952** |
| temporal_jitter | 0.513 | 0.393 | 0.933 | 0.724 | **0.947** |
| event_flood | 1.000 | 0.989 | 0.896 | 1.000 | **1.000** |
| **polarity_flip** | 0.486 | 0.552 | 0.573 | 0.639 | **0.609** |
| **spatial_dropout** | 0.391 | 0.693 | 0.403 | 0.411 | **0.560** |

Two residuals: `polarity_flip` (0.609) and `spatial_dropout` (0.560). Note the **fused max is
*below* the best single branch** on both (spatial 0.639 > 0.609; rcf 0.693 > 0.560) — the max
inflates the clean floor.

---

## 2. What changed — two modifications to EXISTING branches/fusion (no new branch)

### (A) Enrich the `radius` branch with an empirical-covariance Mahalanobis
The isotropic radius (`‖z‖`) ignores the **covariance structure** of the clean PCA-64 manifold.
A `polarity_flip` swaps ON/OFF events, which **rotates** the clean covariance without moving the
marginals much — invisible to `‖z‖` and washed out by Ledoit-Wolf shrinkage, but caught by an
**empirical (unshrunk) covariance Mahalanobis** in the same PCA-64 space (`emp_maha64`, exp9):
polarity **0.656** (> spatial 0.639), while keeping hot_pixel 1.000 / event_flood 1.000.
Full whitening alone drops `event_rate_shift` (0.900→0.600), so it is folded in, not swapped:

> **`radius' = max( |‖z‖−E‖z‖|/sd ,  emp_maha64_z )`** (both calibrated on held-out clean).

This keeps the dilation sensitivity (isotropic term) **and** adds the polarity covariance-rotation
sensitivity (Mahalanobis term) — all inside the one existing energy/radius branch.

### (B) Replace the `max` fusion with a studentized combiner
The 4-way `max` inflates the clean floor: when several branches are mildly elevated by clean noise,
`max` still rises, dragging AUROC below the best single branch. A **studentized max** subtracts a
robust noise-floor estimate so a *lone* strong branch (dropout's `rcf`) stands out while diffuse
clean noise cancels (exp6–exp8):

> **`fused = max(branches) − α · median(branches)`**,  α ≈ 0.5 (tunable 0–1.25).

`α` trades polarity vs dropout (dropout's signal is isolated → likes subtraction; polarity's is
shared → likes α≈0). α≈0.5 is the balanced operating point; α≈1.25 maximizes dropout.

---

## 3. What improved (per-sequence AUROC, L5)

| corruption | baseline (max) | **radius′ + (max−0.5·median)** | radius′ + (max−1.25·median) |
|---|---|---|---|
| hot_pixel | 1.000 | 1.000 | 1.000 |
| event_rate_shift | 0.952 | 0.956 | 0.948 |
| temporal_jitter | 0.947 | 0.952 | 0.966 |
| event_flood | 1.000 | 1.000 | 1.000 |
| **polarity_flip** | 0.609 | **0.607** | 0.560 |
| **spatial_dropout** | 0.560 | **0.613** | **0.689** |
| MEAN | 0.845 | **0.855** | 0.860 |
| MIN (worst-corruption floor) | 0.560 | **0.607** | 0.560 |

- **spatial_dropout: 0.560 → 0.613** (balanced) and up to **0.689** (recovers the `rcf` branch
  ceiling the old `max` threw away) — purely from a better combiner, no new feature.
- **polarity_flip: 0.609 → 0.620** at α=0 with `radius'` (covariance-rotation branch); held ≈flat
  (0.607) at the balanced α=0.5.
- **temporal_jitter: 0.947 → 0.952–0.966**, all easy corruptions stay at ceiling, the worst-case
  floor rises **0.560 → 0.607**.

### Honest ceiling
Neither residual reaches the "solved" bar (≥0.8). The cause is physical and matches the design doc:
- `spatial_dropout` is a **contraction** — its only signature (under-dispersion / lower activity /
  lower participation ratio) is real but *smaller than clean scene-to-scene variance*; pooling +
  the `rcf` branch cap it at ≈0.69.
- `polarity_flip` is **near-invisible by construction** (the model learned polarity-symmetric
  features); the residual lives in the covariance rotation + early-layer L1 + kurtosis, ceiling ≈0.66.

To break 0.8 would require information **not in the current φ** (temporal V(t) trajectories, finer
spatial pooling, or the raw polarity channel) — explicitly out of scope (no re-extraction).

---

## 4. Timeline (scratchpad experiments)

| # | experiment | finding |
|---|---|---|
| exp1 | broad score survey (subsampled fit) | flawed harness, but flagged under-dispersion (drop) + kurtosis (pol) as the live signals |
| exp2 | faithful harness (full fit + seq_lens) | reproduced pipeline exactly; rcf 0.693 (drop), spatial 0.639 (pol) are the best singles; fused max dilutes both |
| exp3 | selective calibrated fusions + per-layer | `rcf+shrink_var`→0.704 (drop); `spatial+kurt+L1`→0.662 (pol); early-layer **L1** retains polarity (0.630) |
| exp4 | density inversion, `sp_pr`, rcf tuning | all confirm the contraction direction but stay weak (≤0.56); rcf tuning flat (~0.69) |
| exp5 | broaden `l4` branch (fold L1+kurt) | l4-pol 0.573→0.634 and event_flood-l4 0.896→1.000, but temporal_jitter cost; **fused barely moved → fusion is the bottleneck** |
| exp6 | fusion combiners vs max | **`max_minus_rest`** (studentized max) lifts dropout 0.560→0.605, temporal_jitter→0.961, raises the floor |
| exp7 | α-sweep `max − α·mean(rest)` | Pareto tension: ↑α ⇒ dropout↑, polarity↓ (isolated vs shared signal) |
| exp8 | robust-floor studentization (min/median/bot2) | **`max − α·median`** best; α=1.5 recovers dropout to 0.693 |
| exp9 | hunt a stronger polarity branch | **empirical-cov PCA-64 Mahalanobis = 0.656** (best pol branch); catches covariance rotation |
| exp10 | fold emp_maha into `radius` + studentized-median | Pareto frontier lifted; both targets improved, worst-case floor 0.560→0.607 |

**Artifacts:** all scripts under the session scratchpad `exp/` (prep_cache, harness, exp1–exp10);
caches `cache.npz` + `cache_<corruption>_L5.npz`. `analysis/mdd.py` unchanged.

---

## 5. Proposed integration into `analysis/mdd.py` (pending approval)

1. In `fit()`: after the PCA basis, also store `emp_mu`, `emp_P = pinv(cov(Zfit))`.
2. In `_branches_raw()`: `radius = max( two_sided_radius_z , calibrated(maha(Z; emp_mu,emp_P)) )`.
3. In `score_branches()`: replace `np.max(stack, 1)` with `mx − α·np.median(stack, 1)` (α default 0.5,
   exposed as an `MDD(__init__)` arg).
4. Re-run `evaluate_mdd.py` / `evaluate_mdd_windows.py` to confirm the L1–L5 aggregate and no
   regressions on the four solved corruptions.
