# Vmem-φ — Noteworthy Results Digest

> **One place** that accumulates every interesting / noteworthy result, finding, and
> negative result from across the project, with **provenance** and a **reliability flag** so
> nothing is cited blind. Assembled by sweeping all of `Docs/*`, `vmem_benchmark/outputs/*`,
> and the result CSVs (2026-06-23).

### Reliability legend
- ✅ **AUTHORITATIVE** — leakage-safe, real `seq_lens`, current extraction (`outputs/results/final_results.csv`, `mdd_improvements.md`). Quote freely.
- 🟡 **PROVISIONAL** — leakage-safe but L5-only / 8k-subsample / single split / no bootstrap CIs (`test_ideas/`, `novel.md`). Direction solid, absolute ±several points.
- 🟠 **PILOT/BIASED** — small (5-seq) or first-N biased subset, or pre-`seq_lens` proxy (`EXPERIMENT.md`, `project_update.md`). Internally-valid ranking, absolute numbers unreliable.
- ❌ **RETRACTED** — does not reproduce; do **not** cite (leakage / 50-sample noise).
- 🚧 **BLOCKED/PENDING** — needs a GPU run not yet banked.

---

## 0. The headline numbers (what we stand behind)

| Claim | Number | Flag | Source |
|---|---|---|---|
| One unsupervised, corruption-blind MDD score, per-frame macro (excl. polarity) | **≈ 0.77** | 🟡 | `novel.md §4` |
| Same, with per-recording aggregation, macro (excl. polarity) | **≈ 0.91**, 4/6 corruptions ≥ 0.99 | 🟡→✅ | `novel.md §4`, `final_results.csv` |
| Improved MDD per-sequence AUROC @ L5 (real `seq_lens`) | hot 1.00 · flood 1.00 · rate 0.955 · jitter 0.952 · **dropout 0.620** · **polarity 0.611** | ✅ | `final_results.csv` |
| Worst-corruption per-sequence floor (improved vs baseline) | 0.560 → **0.611** | ✅ | `mdd_improvements.md §3` |
| φ is zero-cost: native membrane register read, **0 MACs / 0 weight fetches** | — | (argument) | `ideas.md §2`, theory |
| Host model clean Gen1 mAP | ≈ **0.36** | ✅ | `gen1_mAP36.ckpt` |

---

## 1. AUTHORITATIVE current MDD results (`outputs/results/final_results.csv`)

Improved MDD = `radius′ = max(isotropic-energy, empirical-cov-Mahalanobis-PCA64)` +
studentized fusion `max − α·median`. Leakage-safe, real `seq_lens`, eval = held-out clean vs
matched held-out corrupted. **Severities present: 1, 3, 4, 5 (no L2 in this run).**

### Per-FRAME AUROC (improved)
| Corruption | S1 | S3 | S4 | S5 |
|---|---|---|---|---|
| hot_pixel | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal_jitter | 0.658 | 0.742 | 0.799 | 0.850 |
| event_rate_shift | 0.510 | 0.701 | 0.716 | 0.855 |
| event_flood | 0.504 | 0.517 | 0.531 | 0.551 |
| polarity_flip | 0.504 | 0.516 | 0.527 | 0.540 |
| spatial_dropout | 0.500 | 0.502 | 0.517 | 0.560 |

### Per-SEQUENCE AUROC (improved)
| Corruption | S1 | S3 | S4 | S5 |
|---|---|---|---|---|
| hot_pixel | 0.989 | 1.000 | 1.000 | 1.000 |
| event_flood | **0.989** | 1.000 | 1.000 | 1.000 |
| event_rate_shift | 0.516 | 0.853 | 0.852 | 0.955 |
| temporal_jitter | 0.810 | 0.892 | 0.928 | 0.952 |
| spatial_dropout | 0.497 | 0.500 | 0.524 | **0.620** |
| polarity_flip | 0.522 | 0.562 | 0.589 | **0.611** |

### W=64 window (the genuine vs manufactured separator)
- event_flood frame→W64→seq @ L5: **0.551 → 0.889 → 1.000** (crosses 0.9 around W≈64).
- hot_pixel saturated at every window; polarity (0.54→0.58→0.61) and dropout (0.56→0.58→0.62) stay flat → genuine residuals.
- Source: `final_results.csv` columns `*_w64`; window sweep `mdd_window_sweep.csv` (2041 rows, 17 windows × all branches).

> ⚠️ **Discrepancy to reconcile:** the per-frame/seq tables in `paper_figures.md` (and the paper) include an **L2 column and slightly different L5 values** (e.g. frame jitter 0.823 vs 0.850 here). Those came from an earlier `evaluate_mdd.py` run; `final_results.csv` is the **current** graph-generating run and has no L2. Pick one source before final submission.

---

## 2. Per-BRANCH decomposition @ L5, per-sequence (`mdd_improvements.md §1`) 🟡/✅

| Corruption | radius | rcf | l4 | spatial | fused(max) |
|---|---|---|---|---|---|
| hot_pixel | 1.000 | 0.989 | 1.000 | 1.000 | **1.000** |
| event_rate_shift | 0.900 | 0.834 | 0.506 | 0.540 | **0.952** |
| temporal_jitter | 0.513 | 0.393 | 0.933 | 0.724 | **0.947** |
| event_flood | 1.000 | 0.989 | 0.896 | 1.000 | **1.000** |
| polarity_flip | 0.486 | 0.552 | 0.573 | **0.639** | 0.609 |
| spatial_dropout | 0.391 | **0.693** | 0.403 | 0.411 | 0.560 |

**Key:** the plain max sits **below the best single branch** on both residuals (spatial 0.639>0.609; rcf 0.693>0.560) — clean-floor inflation. This is the entire motivation for studentized fusion.

Best-branch identity @ L5 (`paper_figures.md`): hot→radius, jitter→l4, rate→radius, polarity→spatial, flood→radius, dropout→rcf.

---

## 3. THE conceptual finding — the dilation / contraction / invisible taxonomy (`novel.md`, `project_update.md §3`)

Corruptions are classed by **how they move φ relative to the clean manifold**, and that — not the name — determines what can detect them:

| Class | Corruptions | φ moves | Why standard detectors fail |
|---|---|---|---|
| **Dilation** (louder) | hot_pixel, event_rate_shift, temporal_jitter | **outward** | nothing — distance detectors handle these |
| **Contraction** (quieter) | spatial_dropout, event_flood | **inward, toward the mode** | distance/density rate a contraction as *more normal than normal* → they **invert** (AUROC < 0.5) |
| **Invisible** | polarity_flip | barely moves | network learned polarity-symmetric features → signal absent from V_mem |

- jitter is a *dilation in a subspace*: it does **not** change pooled magnitude (radius AUROC 0.47), it distorts timing-sensitive deep layers → needs **per-layer L4** distance.
- **No single distance/likelihood is the right primitive for all of them** — dilations and contractions point in opposite directions; jitter hides in a subspace; flood has no per-frame signal. This is why every prior single method topped out ≈ 0.70 macro and inverted on contractions.

---

## 4. THE anti-detection / inversion finding (below chance) 🟡 + an important nuance

- Reference detector (Mahalanobis on static φ, full data, L5): **3 corruptions below chance** — polarity 0.429, event_flood 0.408, **spatial_dropout 0.286** (`performance_brief.md §3`).
- **AUROC < 0.5 = informative but inverted**: corrupted frames sit *closer* to the clean mean than held-out clean. The detector "interprets silence as safety."
- **spatial_dropout Spearman ρ = −1.000 (p=0.000)** — perfectly, statistically significantly anti-correlated with severity. A paper-level finding (`Findings.md §4`).
- **Crucial nuance (`ideas.md` addendum):** "below chance" is mostly a property of the **scoring rule**, not the representation. The signal is present in φ; only the sign is flipped. Two-sided / direction-conditioned scoring recovers it.
- ⚠️ **The dramatic magnitudes don't fully reproduce.** On the current extraction the below-chance trio reads flood **0.554**, dropout **0.438**, not 0.408/0.286. The *direction* (contractions invert) is robust; the exact magnitude is extraction-specific (`project_update.md §5`, `Findings.md`).

---

## 5. THE aggregation finding — rescue **and** the honest "manufactured by pooling" caveat ✅

- **Rescue:** event_flood has essentially **no per-frame signal** (0.469–0.55 on *every* detector at *every* layer) because proportional inflation mimics a busy clean scene — but its bias is **temporally consistent**, so per-recording averaging lifts it **0.47 → 1.00**. Strongest frame→sequence jump in the benchmark.
- **The honest caveat:** each corruption is applied to the **whole sequence**, so every frame carries the *same* bias; averaging integrates it at ~√(#frames). The 1.00 is **partly a property of the evaluation protocol**, not per-frame detectability — it would shrink if only some frames were corrupted or the decision were per-frame. event_flood's 1.00 is "**manufactured by pooling**"; hot_pixel is genuinely solved at every window.
- **√N is optimistic:** consecutive frames are heavily autocorrelated, so effective N ≪ N; variance reduction scales with √N_eff — measure lag-1 autocorrelation of per-frame scores, don't assume it (`ideas.md` skeptical flags).
- Aggregation does **not** rescue spatial_dropout (non-stationary per-frame bias) or polarity (no signal) — only flat residuals remain flat under pooling, confirming they're genuine.

---

## 6. THE novelty — the RCF (Radial-Conditional) branch (`novel.md §3`)

- Decompose each vector into **radius** `r=‖z‖` and **direction** `û=z/r`; for a test sample find its k nearest clean neighbours **by direction** (cosine), read off their conditional radius distribution `p(r|û)`, score `|r − E[r|û]|/std`.
- *"Given this sample's direction on the clean manifold, is its magnitude what clean data pointing this way normally has?"* — flags **under-dispersion**, which global distance/density cannot see.
- **First method to take spatial_dropout above chance per-frame (0.44 → 0.559)**, and unlike RMD-iso it aggregates cleanly. To our knowledge the first detector built specifically to catch **contractions** of a learned manifold.

---

## 7. Fusion findings (`novel.md §3`, `mdd_improvements.md`, `performance_brief.md`)

- **Plain calibrated max < best single branch** on both residuals (clean-floor inflation).
- **Studentized fusion `max − α·median` (α≈0.5)** is the Pareto winner: spatial_dropout 0.560 → **0.613** (→ 0.689 at α=1.25), worst-corruption floor 0.560 → 0.607, no easy-corruption regression. α trades polarity vs dropout (one static combiner can't max both).
- **Radius enrichment:** fold an *empirical (unshrunk) covariance* Mahalanobis in PCA-64 into B1 — catches the covariance **rotation** a polarity flip induces (polarity 0.656 > spatial 0.639); folded as `max(...)` not swapped, since full whitening alone drops event_rate_shift 0.900→0.600.
- **Combiner sweep:** rank/quantile calibration helps residuals but **collapses** easy ones (event_flood 1.00→0.61). The *fusion rule* is a first-class design axis.
- **Supervised LR fusion buys nothing per-frame** (0.763 ≈ max 0.766) and **fails to generalize** (LR leave-one-out 0.486, inverts jitter & rate_shift). The unsupervised max is simpler and safer.
- exp1–exp10 timeline of the residual loop is in `mdd_improvements.md §4`.

---

## 8. The closed-form theory (predictions matching measured AUROC ordering) (`theory`, `ideas.md §1`)

PLIF: `V[t] = (1−1/τ)V[t−1] + W·X[t]` → `V[t] = Σ λ^k W X[t−k]`. Steady state: `E[V]=τWm`, `Var[V]` has an autocovariance term depending on **time ordering**.

| Corruption | Lever pulled | Prediction | Measured |
|---|---|---|---|
| hot_pixel | constant δ → mean saturates at θ | AUROC ≈ 1, monotone (inevitable) | 1.000, ρ=+1 ✅ |
| temporal_jitter | permutation kills γ(j) / lag-1 autocorr | temporal/deep (L4) only | L4 best (0.93) ✅ |
| event_rate_shift | m→αm scales all means | 1-D activity scalar; phase transition | phase jump @ S3 ✅ |
| spatial_dropout | lowers σ², μ intact; **GAP discards it** | anti-detectable under one-sided scoring | ρ=−1 ✅ |
| event_flood | uniform m↑ = busy scene | near-chance per frame | 0.47–0.55 ✅ |
| polarity_flip | network polarity-symmetric | ≈0.5 ceiling by construction | ≈0.48–0.55 ✅ |

Every prediction follows from the state equation with **no parameters fit to corruption labels**.

---

## 9. Per-LAYER architectural finding (`Findings.md §2`) 🟠 (pre-split mix, regenerate before citing)

| Layer | hot_pixel | event_flood | temporal_jitter | polarity | event_rate_shift | spatial_dropout | AVG |
|---|---|---|---|---|---|---|---|
| Block 1 | **0.871** | 0.523 | 0.570 | 0.547 | 0.430 | 0.488 | 0.571 |
| Block 2 | 0.677 | 0.517 | 0.609 | 0.520 | 0.568 | 0.481 | 0.562 |
| Block 3 | 0.771 | 0.512 | 0.794 | 0.532 | 0.662 | 0.478 | 0.625 |
| Block 4 | 0.815 | 0.511 | **0.865** | 0.529 | **0.739** | 0.481 | **0.656** |
| ALL concat | 0.804 | 0.523 | 0.768 | 0.549 | 0.701 | 0.480 | 0.638 |

- **Block 1 best for hot_pixel** (closest to sensor; DC saturates first layer before diffusing).
- **Blocks 3–4 best for jitter & rate_shift** (larger receptive field + more timestep integration).
- **event_flood undetectable at every layer** (0.511–0.523) — not a per-layer problem.
- **spatial_dropout < 0.50 at every layer** — systematic inversion.
- **ALL-concat (0.638) < Block 4 alone (0.656)** — concatenation dilutes; motivates layer weighting.

---

## 10. Severity behaviors & monotonicity (`Findings.md §3–4`) 🟠

- **hot_pixel** — saturation ramp: S1 ~0.50 → S3 ~0.93 → S5 **1.000** (all 7 detectors).
- **event_flood** — flat: S1 0.504 → S5 0.554, **+0.05 over 5 severities, all 7 detectors identical**.
- **temporal_jitter** — strong ramp, AE wins: 0.806 → **0.947** @ L5.
- **event_rate_shift** — **phase transition between S2 and S3** (Mahal 0.496 → 0.806). **GMM collapses at L5 (0.139)** — corrupted distribution outside GMM support, log-likelihood inverts.
- **spatial_dropout** — AUROC *decreases* with severity: 0.499 → 0.490 → **0.439**.

**Spearman ρ (score vs severity):** hot_pixel +1.0, event_flood +1.0 (but AUROC barely moves), polarity +1.0, event_rate_shift +1.0, **temporal_jitter +0.700 (p=0.188, NOT significant, n=5 too small)**, **spatial_dropout −1.000 (p=0.000)**.

---

## 11. Classical detector comparison (avg over 6×5) (`Findings.md §1`, `EXPERIMENT.md`) 🟠

| Detector | Avg AUROC | Avg FPR@95 | Verdict |
|---|---|---|---|
| **MLP Autoencoder** | **0.673** | **0.695** | best both metrics → φ space is non-Gaussian, non-linear |
| Normalizing Flow | 0.650 | 0.798 | good ranker, **badly calibrated** (bad binary alarm) |
| PCA-Mahal | 0.650 | 0.707 | tied 2nd, much better FPR than NF |
| Mahalanobis | 0.638 | 0.726 | solid baseline (the reference detector) |
| kNN (k=5) | 0.637 | 0.747 | — |
| GMM | 0.617 | 0.817 | weakest classical (5 comps underfit multimodal φ) |
| One-Class SVM | 0.598 | 0.801 | worst |

The AE win → φ has **non-Gaussian, non-linear manifold structure** parametric methods underfit. (Note `implementation_status.tex` also reports kNN 0.754 / PCA-Mahal 0.736 on a different cut — provenance differs.)

---

## 12. Validity & "beyond binary" results (`EXPERIMENT.md §6–10`) 🟠 (pilot subsets)

- **Free-rider ablation** (5 seq, Mahalanobis, L5): **A trained 0.998 > B random-init 0.822 > C raw-input 0.716** on hot_pixel. A−B = +0.176 (learning matters), A−C = +0.282 (SNN adds real signal beyond input stats). The make-or-break validity check — **trained ≫ raw**. ✅-direction, 🟠-numbers.
- **Severity regression (Ridge, φ → severity 1–5):** R² > 0.7 for hot_pixel / jitter / rate_shift; **≈ 0 for event_flood / spatial_dropout** → φ encodes intensity *continuously* for detectable corruptions.
- **7-class corruption classification** (LinearSVC/LogReg on φ): > 50% top-1; **event_flood & high-severity spatial_dropout confused with clean** (as predicted).
- **Conformal prediction** (clean-calibrated thresholds): hot_pixel L5 OOD-fraction > 95%; event_flood OOD-fraction < 15% (not confidently detectable by static φ). Safety framing.
- **Representation ablation:** σ² alone = strongest single moment; **μ alone best for event_flood** (only corruption shifting global mean); κ weakest but unique on jitter; combined [μ,σ²,κ] best on average.

---

## 13. ANN baseline context (`project_update.md §4`) 🟠

- Membrane φ + **OCSVM reaches ≈ 0.81 overall AUROC** across the benchmark.
- ResNet-18 ANN baselines (event-image / voxel-grid) top out ≈ **0.74** (DICE/ViM); several energy-based ANN methods **invert badly (~0.18)**.
- → the cheap membrane signal is **competitive with a full ANN baseline** — and the ANN baselines are not even on-chip-feasible.

---

## 14. ❌ RETRACTED & negative results (do NOT cite)

- **❌ "Temporal features rescue the hard corruptions to ~0.85"** (`Findings.md §5`). Was the *most important claimed finding*; it is an **artifact** of leakage + a 50-sample degenerate split (35 train/15 test from the first 50 frames of one sequence). Re-running the exact method gives **0.29 / 0.04 / 0.06** (below chance). **Honest temporal gain is only ~0.05–0.08** (handcrafted temporal: flood ≈0.625, dropout ≈0.541). The **Temporal-AE is the weakest detector**.
- **❌ The dramatic below-chance magnitudes (flood 0.408, dropout 0.286)** don't reproduce on the current extraction (0.554 / 0.438). Direction holds, magnitude extraction-specific.
- **Negative levers** (`performance_brief.md §6`): two-sided folded Mahalanobis rescues the trio but *degrades* jitter (0.709→0.615); global activity scalar wins only rate_shift (→0.85); per-feature standardization helps jitter (→0.82) but hurts rate_shift (→0.568); naive max(two-sided+activity) worse than either; supervised LR meta-fusion no per-frame gain + fails to generalize. **No single transform wins everywhere.**

---

## 15. Ideas tested (`ideas.md` addenda, `test_ideas/RESULTS.md`) 🟡

- **R1 RMD-iso:** first win on spatial_dropout **0.44 → 0.60** (strips magnitude — inverts magnitude corruptions, rescues the structure one). Exposed the magnitude-vs-structure taxonomy.
- **A2a per-layer:** **L4-only = 0.93** on jitter (vs 0.78 pooled) — big free win.
- **A2b kNN:** rate_shift 0.62 → 0.78 free win.
- **R3 GMM** lifts jitter to 0.877; **R4 PCA-residual** modestly helps jitter/polarity.
- **R5 whitened kNN — refuted** (worse than plain kNN). **A1 χ² two-sided — refuted** (clean d² not a tight band: median ~14 ≪ D=2112).
- **R10 polarity:** undetectable from V_mem (~0.55 ceiling); a near-free **input** ON/OFF-balance scalar reaches ~0.69 (input-side, scope accordingly).
- **Idea 11 aggregation** rescues flood to ~1.0 but *hurts* rate_shift/dropout (only works when per-frame bias is consistent & same-sign).
- Clean φ geometry fact: **d² median ≈ 14, mean ≈ 45 ≪ D=2112** → φ lives on a thin low-dim manifold; raw Mahalanobis is dominated by scene magnitude.

---

## 16. phi_spatial (the GAP-recovery branch) findings 🟡 (memory `phi-spatial-extraction`)

- Mechanism check (synthetic contraction, signal only in phi_spatial): fused-without-spatial **0.496 → 1.000** with spatial; spatial branch alone 1.000 → wiring validated.
- On the banked extraction: spatial branch **NAILS event_flood (1.000)** but is **below chance on spatial_dropout (0.411)** — buys event_flood per-frame, **not** dropout. Fused max even *dilutes* the best branch (rcf 0.693 → 0.560). So phi_spatial helps flood, dropout remains a residual.

---

## 17. Figures available (`outputs/graphs/`, 21 PNGs, generator `plot_corruption_graphs.py`)

Window sweep (01), per-branch window (02), severity sweep (03), branch heatmap (04), **radial-energy KDE (05)**, **Mahalanobis d² inversion hist (06)**, shared-PCA scatter (07), **PCA drift arrows (08)**, channel-shift fingerprint (09), corruption similarity matrix (10), LDA scatter (11), log-d² violins (13), baseline-vs-improved bars (15), **α trade-off (16)**, fusion combiners (17), ROC curves (18), window heatmap (19), σ² shrinkage violin (20), **phi_spatial KDE (21)**, improvement-by-severity (22), best-by-severity (23).

---

## 18. Caveats / holes to close before submission

- 🚧 **Reliability / mAP-degradation story is void** until clean `det_outputs` has nonzero confident detections (memory `det-outputs-all-zero`). Keep out of the paper for now.
- 🚧 **Motivation mAP-degradation table** pending the GPU sweep (Neftci detector, no retrain).
- **No bootstrap CIs** anywhere — all AUROC are single-split point estimates. The per-frame macro ~0.77 is the robust figure.
- **Single dataset (Gen1)**; DSEC/Gen4 transfer untested (`dsec_transfer.csv` referenced but absent).
- **L2 missing** from `final_results.csv`; reconcile with the L2 column in `paper_figures.md`/the paper.
- **AUROC is oracle-labelled separability**, not a deployable fixed-threshold value.
- Engineering: full pipeline is now all-GPU / RAM-safe; extraction is the GPU-bound floor at B=1 (`bottlenecks.md`).

---

*Sources swept: `Findings.md`, `novel.md`, `ideas.md`, `performance_brief.md`, `mdd_improvements.md`, `project_update.md`, `EXPERIMENT.md`, `bottlenecks.md`, `implementation_status.tex`, `paper/paper_figures.md`, `paper/paper_skeleton.md`, `outputs/results/final_results.csv`, `outputs/results/mdd_window_sweep.csv`, `outputs/graphs/README.md`.*
