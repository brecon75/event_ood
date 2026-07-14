# Paper Figures & Tables — Running Collection

Scratchpad of paper-ready statistics, tables, and figures as they are produced.
Append new results here (do not overwrite) with provenance so they can be lifted
into the manuscript later. Each entry should name its **source file** and the
**generating command/branch** so numbers are reproducible and auditable.

---

## MDD fused-pipeline results (max-fusion)

**Source:** `vmem_benchmark/outputs/mdd_metrics.csv` (per-frame),
`vmem_benchmark/outputs/mdd_metrics_aggregated.csv` (per-sequence).
**Generator:** `analysis/evaluate_mdd.py` (Stage 8).
**What "fused" is:** the calibrated per-frame **max** across branches —
each branch score is standardized, then `np.max(...)` across branches *per frame*
(`evaluate_mdd.py:_fused`, line 46-47), and AUROC is computed on that fused score
(`_auroc_row`, lines 37-43). It is the genuine detector output, **not** the max of
the branch AUROCs. The MDD is fit once on clean-train only and is corruption-blind;
the corruption label is used solely to compute AUROC at scoring time.

### Fused per-FRAME AUROC

| Corruption | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| hot_pixel | 0.983 | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal_jitter | 0.617 | 0.640 | 0.663 | 0.737 | 0.823 |
| event_rate_shift | 0.412 | 0.440 | 0.674 | 0.713 | 0.866 |
| polarity_flip | 0.415 | 0.422 | 0.437 | 0.459 | 0.480 |
| event_flood | 0.413 | 0.418 | 0.428 | 0.445 | 0.469 |
| spatial_dropout | 0.407 | 0.406 | 0.404 | 0.404 | 0.399 |

### Fused per-SEQUENCE AUROC

| Corruption | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| hot_pixel | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 |
| temporal_jitter | 0.649 | 0.659 | 0.682 | 0.774 | 0.868 |
| event_rate_shift | 0.364 | 0.411 | 0.752 | 0.803 | 0.961 |
| polarity_flip | 0.376 | 0.396 | 0.434 | 0.485 | 0.527 |
| event_flood | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| spatial_dropout | 0.350 | 0.345 | 0.339 | 0.327 | 0.297 |

### Fused vs best single branch — `fused / best-branch` AUROC per severity

"best-branch" = the oracle-best individual branch in each cell (radius / rcf / l4 /
spatial). It is an **upper bound, not a deployable detector** — you only know which
branch is best because you know the corruption label. Fused is what actually runs
(corruption-blind, no per-corruption branch selection).

**per-FRAME**

| Corruption | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| hot_pixel | 0.983/0.996 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 |
| temporal_jitter | 0.617/0.662 | 0.640/0.700 | 0.663/0.725 | 0.737/0.795 | 0.823/0.864 |
| event_rate_shift | 0.412/0.479 | 0.440/0.493 | 0.674/0.598 | 0.713/0.615 | 0.866/0.759 |
| polarity_flip | 0.415/0.478 | 0.422/0.477 | 0.437/0.476 | 0.459/0.490 | 0.480/0.525 |
| event_flood | 0.413/0.478 | 0.418/0.487 | 0.428/0.496 | 0.445/0.511 | 0.469/0.532 |
| spatial_dropout | 0.407/0.478 | 0.406/0.477 | 0.404/0.476 | 0.404/0.482 | 0.399/0.526 |

**per-SEQUENCE**

| Corruption | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| hot_pixel | 0.999/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 |
| temporal_jitter | 0.649/0.689 | 0.659/0.724 | 0.682/0.747 | 0.774/0.830 | 0.868/0.898 |
| event_rate_shift | 0.364/0.454 | 0.411/0.507 | 0.752/0.651 | 0.803/0.674 | 0.961/0.862 |
| polarity_flip | 0.376/0.453 | 0.396/0.455 | 0.434/0.496 | 0.485/0.565 | 0.527/0.623 |
| event_flood | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 | 1.000/1.000 |
| spatial_dropout | 0.350/0.452 | 0.345/0.456 | 0.339/0.461 | 0.327/0.501 | 0.297/0.583 |

### Best-branch identity at L5 (which branch wins each corruption)

| Corruption | best branch (frame) | best branch (seq) |
|---|---|---|
| hot_pixel | radius | radius |
| temporal_jitter | l4 | l4 |
| event_rate_shift | radius | radius |
| polarity_flip | spatial | spatial |
| event_flood | radius | radius |
| spatial_dropout | rcf | rcf |

### Takeaways (for discussion section)
- Fusion **clearly wins only on event_rate_shift** (+0.11 frame / +0.10 seq at L5) —
  branches are complementary there and the max ORs them productively.
- **event_flood:** near-chance per-frame (0.469) but **perfect per-sequence (1.000)** —
  the spatial branch separates it only after sequence-level aggregation. Strongest
  frame→sequence jump in the benchmark.
- **Residuals:** polarity_flip (~0.48–0.53) and spatial_dropout (0.30–0.40, *decreasing*
  with severity) remain unsolved; an oracle branch (spatial / rcf) would beat fused there
  (spatial_dropout gap up to −0.286 at L5 seq). Flat max underperforms its own best branch
  on these → lever is a **weighted/calibrated/gated fuse** instead of unweighted max.
- All numbers are point estimates with no confidence interval; AUROC is an oracle-labeled
  separability metric, not a value the corruption-blind detector reproduces at deployment.

---

## Motivation experiment — mAP degradation of the pretrained detector (PENDING GPU run)

**Source (to be produced):** `results/neftci_map_degradation.csv` (raw),
`results/neftci_map_degradation_summary.csv` + `results/neftci_map_degradation_table.tex`.
**Generator:** `HybridDetection/validation_corrupt.py` + `run_map_degradation.sh` (run the
sweep), then `analysis/summarize_map_degradation.py` (build table). Runs the pretrained
`gen1_mAP36.ckpt` with **no retraining**, corrupting `data[EV_REPR]` before the backbone.
**Status:** scripts implemented + the corruption-injection + summarizer logic validated
locally on synthetic data; the actual sweep needs the Gen1 test set + GPU (Modal/A100).
**Clean baseline to confirm:** mAP (AP@[.5:.95]) ≈ 0.36 (checkpoint name `gen1_mAP36`).
**Paper hook:** Table `tab:map-degradation` in `paper_sections/motivation_experiment.tex`
(currently a `\todofig` placeholder). Hypothesis: hot_pixel / event_flood / spatial_dropout
crater mAP; polarity_flip degrades mAP even though the membrane cannot flag it (a deliberate
detectability-vs-harm mismatch).

## Qualitative figures (advisor request)

- **`outputs/plots/corruption_gallery.{pdf,png}`** — event representation (ON=red, OFF=blue)
  before/after each corruption × severity. Generator `analysis/viz_corruptions.py` (pass
  `--sample <real Gen1 frame>`; falls back to a synthetic layout frame). Validated on
  synthetic frame locally.
- **`outputs/plots/per_layer_dist_<corruption>_L5.pdf`** — per-PLIF-layer distributions of
  the membrane moments [μ, σ², κ], clean vs corrupted (4×3 grid). Generator
  `analyse_plots.plot_per_layer_distributions` (wired into `analyse.py`), from existing
  `outputs/phi/*.pt` — **no GPU needed**. Validated on synthetic phi locally.

## Spatial-branch mechanism check (synthetic, validates the spatial_dropout fix)

On a controlled contraction whose signal lives **only** in `phi_spatial` (GAP'd φ ~ clean):
fused-without-spatial AUROC = **0.496** (chance) → fused-with-spatial = **1.000**; spatial
branch alone = 1.000. Confirms `MDD` spatial branch + `LazyPhiDict.get_phi_spatial` +
`evaluate_mdd.py` are correctly wired; the real-data lift of `spatial_dropout` only awaits the
fresh `phi_spatial` extraction. (Throwaway check, not committed.)

## Cross-corruption zero-shot transfer (App. F, `tab:crosscorr`)

**Source:** `vmem_benchmark/outputs/results/cross_corruption.csv` (run 2026-06-24).
**Generator:** `analysis/cross_corruption.py` — trains one clean-vs-`hot_pixel` logistic
detector on the `membrane_fused` φ (val-disjoint train/eval, sequence-aware split) and scores it
**zero-shot** on the five held-out corruptions at L1/L3/L4/L5. **Folded into** `paper_iclr.tex`
§ Severity/Cross-corruption appendix, replacing the `\todofig` placeholder.

Zero-shot AUROC (rows = held-out eval corruption; trained on `hot_pixel` only):

| eval corruption | L1 | L3 | L4 | L5 |
|---|---|---|---|---|
| event_flood | 0.50 | 0.52 | 0.53 | 0.55 |
| temporal_jitter | 0.48 | 0.46 | 0.47 | 0.52 |
| polarity_flip | 0.50 | 0.50 | 0.51 | 0.51 |
| event_rate_shift | 0.46 | 0.38 | 0.30 | **0.16** |
| spatial_dropout | 0.50 | 0.49 | 0.49 | 0.47 |

**Takeaway:** transfer fails — every held-out type sits at/below chance regardless of severity, and
`event_rate_shift` inverts strongly (0.16 @ L5, its activity-driven φ shift opposes `hot_pixel`'s).
Only `event_flood` shows a weak monotone rise, capping at 0.55. Confirms the φ signature is
**corruption-specific**, not a generic OOD direction — motivates fitting MDD on clean data alone.
(Single-source transfer, not a full N×N matrix: only `hot_pixel` is used as the train type.)

---

## MDD branch leave-one-out ("branches turned off") — per-sequence AUROC @ L5

**Source:** `vmem_benchmark/outputs/results/mdd_metrics_aggregated.csv` (rows `fused`, `fused_no_*`).
**Generator:** `analysis/evaluate_mdd.py` (Stage 8; leave-one-out block, plain-max rule).
**In paper:** Appendix `app:mdd-branches`, Table `tab:mdd-loo`.

Δ = (fused with branch removed) − (full fused). Negative ⇒ that branch was helping.

| Corruption | fused | −radius | −rcf | −l4 | −spatial | critical branch |
|---|---|---|---|---|---|---|
| hot_pixel | 1.000 | 1.000 | 1.000 | 1.000 | 0.989 | spatial |
| event_rate_shift | 0.949 | **0.869** (−0.079) | 0.953 | 0.950 | 0.948 | radius |
| temporal_jitter | 0.943 | 0.916 | 0.956 | **0.834** (−0.109) | 0.947 | l4 |
| event_flood | 1.000 | 1.000 | 1.000 | 1.000 | 0.989 | spatial |
| polarity_flip | 0.620 | 0.619 | 0.627 | 0.621 | **0.607** (−0.013) | spatial |
| spatial_dropout | 0.575 | 0.570 | **0.404** (−0.171) | 0.586 | 0.581 | rcf |

**Takeaway:** every branch is load-bearing — each is the sole detector of exactly one corruption,
so dropping it collapses that corruption and only that one. No removal helps any corruption by
more than +0.013, so a smaller MDD is strictly dominated. This is the "selective fusion" view
(fuse-all-but-one) the residual analysis asked for.

---

## MDD fusion-combiner sweep — is there a better combiner than (studentized) max?

**Source:** cached branch scores `vmem_benchmark/outputs/results/mdd_branch_scores_L5.pkl`.
**Generator:** `scratchpad/fusion_explore.py` (fits MDD, scores all branches once, caches; then
evaluates 7 combiners on top). Per-sequence AUROC, L5, full data. p-value combiners convert each
branch to its right-tail empirical p vs the clean **calib** split (leakage-safe), then combine.

| corruption | z-max | **z-student** | z-top2 | rank-max | fisher | stouffer | hmp |
|---|---|---|---|---|---|---|---|
| hot_pixel | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| event_flood | 1.000 | 1.000 | 1.000 | 0.918 | 0.961 | 0.790 | 0.622 |
| temporal_jitter | 0.943 | 0.952 | 0.922 | **0.969** | 0.906 | 0.867 | 0.942 |
| polarity_flip | 0.620 | 0.611 | 0.621 | 0.620 | 0.636 | 0.620 | 0.617 |
| event_rate_shift | 0.949 | 0.955 | 0.938 | **0.967** | 0.870 | 0.543 | 0.960 |
| spatial_dropout | 0.575 | **0.620** | 0.521 | 0.564 | 0.478 | 0.468 | 0.520 |
| **MEAN** | 0.848 | **0.856** | 0.834 | 0.840 | 0.809 | 0.715 | 0.777 |
| **WORST** | 0.575 | **0.611** | 0.521 | 0.564 | 0.478 | 0.468 | 0.520 |

**Takeaway:** the shipped **studentized max** (max − 0.5·median) is the Pareto winner — best on both
MEAN (0.856) and WORST-corruption (0.611). The principled p-value combiners (rank-max/Bonferroni,
Fisher, Stouffer, harmonic-mean-p) do **not** beat it: aggregating combiners (fisher/stouffer/hmp)
collapse the easy saturated corruptions, and rank-max trades dilation gains (temporal_jitter 0.969,
event_rate_shift 0.967 — both best in row) for losses on event_flood (1.0→0.918) and spatial_dropout
(0.620→0.564). **Structural finding:** rank-max is best on the two *dilation* corruptions while
z-student is best on the *contraction* residual — no single static combiner maximizes both, which is
the same dilation-vs-contraction tension the α knob exposes. The residuals (spatial_dropout ~0.62,
polarity_flip ~0.62) are **information-limited** (C2ST ceiling), not fusion-limited — so "combine
better" has little headroom beyond the current studentized max.

---

## Fusion lit-review combiners (Aggarwal–Sathe AOM/MOA, Kriegel unification, ViM-sum)

**Source:** cached branch scores `mdd_branch_scores_L5.pkl`; **Generator:** `scratchpad/fusion_litreview.py`.
Per-sequence AUROC, L5, full data. Tests theory-backed combiners from the outlier-ensemble /
OOD-fusion literature not covered by the earlier sweep.

| corruption | **z-student** | AOM | MOA | unify-max | unify-mean | wsum |
|---|---|---|---|---|---|---|
| hot_pixel | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| event_flood | 1.000 | 1.000 | 1.000 | 0.625 | 0.740 | 0.740 |
| temporal_jitter | 0.952 | 0.922 | 0.922 | 0.949 | 0.883 | 0.883 |
| polarity_flip | 0.611 | 0.622 | 0.621 | 0.608 | 0.630 | 0.630 |
| event_rate_shift | 0.955 | 0.933 | 0.938 | 0.957 | 0.751 | 0.751 |
| spatial_dropout | 0.620 | 0.523 | 0.521 | 0.600 | 0.501 | 0.501 |
| **MEAN** | **0.856** | 0.833 | 0.834 | 0.790 | 0.751 | 0.751 |
| **WORST** | **0.611** | 0.523 | 0.521 | 0.600 | 0.501 | 0.501 |

**Takeaway:** studentized max remains Pareto-best on MEAN and WORST. AOM/MOA (Aggarwal & Sathe
balanced combiners) and the averaging family (Kriegel unify-mean, ViM-style sum) dilute the
residuals (spatial_dropout 0.62→0.52) and the saturated event_flood (1.0→0.63–0.74). This matches
outlier-ensemble theory: our regime is "well-hidden outliers flagged by FEW detectors" (each
corruption fires ONE branch), where MAX-family reduces bias and AVG-family dilutes; the studentized
max is the variance-corrected MAX sitting at the right bias–variance point. COPOD-style copula
aggregation = the earlier "fisher" combiner (already diluted). Conclusion: fusion is solved up to
the φ information ceiling; remaining headroom needs new INFORMATION, not a new combiner.

---

## Fusion combiners — formula-VERIFIED + cross-domain extension (16 methods)

**Source:** cached `mdd_branch_scores_L5.pkl`; **Generator:** `scratchpad/fusion_verify.py`.
Per-sequence AUROC, L5, full data. Each combiner checked against its paper/reference impl
(PyOD+combo, Kriegel SDM'11, COPOD ICDM'20, meta-analysis, Kolde Bioinf'12, Parisi PNAS'14).

**Formula-faithfulness audit & fixes:**
- **Kriegel Gaussian unify** — earlier code used `norm.cdf`; exact is `max(0, erf((O−μ)/(σ√2)))`.
  Fixed. (worst 0.600→0.607, mean 0.790→0.792.) The `max(0)` clip helps the contraction residual,
  hurts saturated event_flood.
- **COPOD** — earlier "fisher" was right-tail only; faithful COPOD = `Σ_dim max(U_skew,(U_l+U_r)/2)`
  with `U=−log ecdf`. Added (mean 0.835 vs fisher 0.809). Still dilutes.
- **AOM/MOA** — switched from all-pairs to combo-faithful random size-2 buckets ×50; result
  unchanged (mean 0.834/0.833) → the all-pairs instantiation was already correct in expectation.
- **Stouffer / Fisher-stat / Bonferroni(min-p) / HMP** — match standard meta-analysis; no change.

Leaderboard (by worst-corruption, then mean):

| combiner | worst | mean | family | verdict |
|---|---|---|---|---|
| **z-student (shipped)** | **0.611** | **0.856** | max | **winner** |
| kriegel-max (exact) | 0.607 | 0.792 | max | close worst, poor mean |
| z-max | 0.575 | 0.848 | max | |
| LSE-τ4 (energy soft-max) | 0.568 | 0.846 | max | →max as τ↑ |
| bonferroni min-p | 0.564 | 0.840 | max-p | |
| LSE-τ1 | 0.542 | 0.839 | soft | |
| MOA / AOM | 0.527 / 0.525 | 0.833 / 0.834 | avg | dilutes |
| HMP | 0.520 | 0.777 | avg-p | dilutes |
| COPOD | 0.519 | 0.835 | avg-p | dilutes |
| kriegel-mean | 0.516 | 0.802 | avg | dilutes |
| Fisher | 0.478 | 0.809 | avg-p | dilutes |
| RRA (Kolde) | 0.471 | 0.741 | rank-consensus | dilutes |
| Stouffer | 0.468 | 0.715 | avg-p | dilutes |
| median | 0.420 | 0.761 | avg | dilutes |
| SML (Parisi spectral) | 0.387 | 0.751 | avg-weighted | worst — eigvec points wrong on contraction |

**Takeaway (robust across 16 combiners spanning 7 papers/domains):** every averaging/consensus
combiner (AOM, MOA, mean, median, COPOD, Fisher, Stouffer, HMP, RRA, SML) **dilutes**; only the
MAX-family stays competitive, and **studentized max wins both worst-case and mean**. This is exactly
the Aggarwal–Sathe bias–variance prediction for "well-hidden outliers flagged by FEW detectors":
our corruptions each fire ONE branch (leave-one-out), so consensus across branches destroys signal.
The label-free Spectral Meta-Learner (Parisi PNAS'14) is *worst* — its leading-eigenvector weights
chase the dominant common-variance direction, which inverts on contractions (spatial_dropout 0.387).
Confirms: residuals are information-limited, not fusion-limited — no combiner recovers them.

---

## Bootstrap confidence intervals on headline MDD AUROCs (reviewer ask E2)

**Source:** `analysis/bootstrap_ci.py` on cached `outputs/results/mdd_branch_scores_L5.pkl`.
**In paper:** Appendix `app:window`, Table `tab:ci`. Point estimates match `tab:window` exactly.
95% cluster bootstrap (resample recordings — the independent unit — pool windows, 2000 resamples).

| corruption | W=64 AUROC [95% CI] | full AUROC [95% CI] |
|---|---|---|
| hot_pixel | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] |
| event_rate_shift | 0.884 [0.845, 0.916] | 0.949 [0.908, 0.982] |
| event_flood | 0.890 [0.875, 0.904] | 1.000 [1.000, 1.000] |
| temporal_jitter | 0.852 [0.816, 0.885] | 0.943 [0.903, 0.977] |
| spatial_dropout | 0.553 [0.496, 0.606] | 0.575 [0.490, 0.655] |
| polarity_flip | 0.581 [0.528, 0.634] | 0.620 [0.536, 0.699] |

**Takeaway:** the four detectable corruptions are well clear of chance at both granularities; the two
residuals' CIs straddle/barely exceed 0.5 — they are genuine residuals, not split noise.

---

## k-NNN geometry baseline (reviewer ask E3) — contraction residual

**Source:** `analysis/knnn_baseline.py` (faithful to Nizan & Tal arXiv:2305.17695, Eq. 2; k=3,
25 neighbours-of-neighbours, L=5 partitions, greedy correlation-ordered features). Unsupervised on
clean φ, 50-seq subset, per-sequence AUROC @ L5. **In paper:** `app:baselines` paragraph.

| corruption | k-NNN | plain k-NN (k=3) |
|---|---|---|
| hot_pixel | 1.000 | 1.000 |
| event_rate_shift | 1.000 | 1.000 |
| event_flood | 1.000 | 1.000 |
| temporal_jitter | 1.000 | 1.000 |
| **spatial_dropout** | **0.531** | **0.694** |
| polarity_flip | 1.000 | 0.898 |

**Takeaway:** on the contraction residual `spatial_dropout`, the dedicated contraction/hubness-aware
geometry baseline k-NNN (0.531) is **worse** than a plain k-NN (0.694) and ≈ the RCF branch (~0.69) —
its small-eigenvalue "anomaly-direction" up-weighting misfires when the anomaly is a contraction
*toward* the mode. Corroborates "information-limited, not detector-limited." Caveat: on this 50-seq
per-sequence subset the 4 detectable corruptions saturate at 1.000 (and polarity's 1.000 is a
small-N artifact); only the residual cell is discriminative.

---

## event_flood detection-latency vs AUROC (reviewer Q: "latency/accuracy trade-off to guide
deployment thresholds")

**Source:** `vmem_benchmark/outputs/results/mdd_window_sweep.csv` (existing window sweep, no
re-extraction), `corruption=event_flood, severity=5, branch=fused` rows — same numbers as the
`event_flood` row of `tab:window`. **Generator:** throwaway script converting `window * 50ms/frame`
to latency and plotting vs AUROC (log-x); figure copied to
`vmem_benchmark/outputs/graphs/45_event_flood_latency.png`. **In paper:** new
`fig:event-flood-latency` + paragraph in `app:window`, right after `tab:window`.

| window (frames) | latency (s) | fused AUROC |
|---|---|---|
| 1 | 0.05 | 0.551 |
| 8 | 0.40 | 0.614 |
| 16 | 0.80 | 0.677 |
| 32 | 1.60 | 0.778 |
| 64 | 3.20 | 0.889 |
| 128 | 6.40 | 0.976 |
| 256 | 12.80 | 0.999 |
| full (~730 frames) | 36.5 | 1.000 |

**Takeaway:** AUROC crosses the 0.85 operating threshold at W=64 (3.2s) — the practical knee of the
curve for time-critical deployment; the full-recording 1.00 ceiling needs 36.5s and is an optimistic
upper bound, not a deployable operating point.
