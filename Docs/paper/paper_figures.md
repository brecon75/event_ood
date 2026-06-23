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
