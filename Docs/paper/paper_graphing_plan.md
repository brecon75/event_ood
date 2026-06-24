# Paper Graphing Plan — full-repo scan of what's worth graphing

Scan date 2026-06-23. Inventory of every figure worth putting in the paper, mapped to the
section it serves, with **data source / generating code** and **status**. Cross-references
`Docs/paper_skeleton.md` (figure asks) and the existing plot code.

**Status legend:** ✅ generated now (in `outputs/graphs/`) · ⚙ code exists, needs a pipeline run ·
🆕 needs new plotting code · ⛔ blocked (data not producible yet).
**Data reality:** the only result CSV currently on disk is `outputs/results/mdd_window_sweep.csv`.
Every ⚙ figure below depends on a pipeline stage (`evaluate_detectors`, `evaluate_mdd`,
`representation_ablation`, `severity`, `free_rider_ablation`, `reporting/*`) that has **not been
run on this data yet** — its plot code exists but its input CSV is absent.

---

## Already generated this session — `outputs/graphs/` (✅)

### Windowed / quantitative (from `mdd_window_sweep.csv`)
| # | file | section | shows |
|---|---|---|---|
| 01 | window_sweep_fused_L5 | §9.4 | fused AUROC vs window, per corruption (pooling-manufactured vs genuine residual) |
| 02 | window_branches_L5 | §9.4 / §7 | per-corruption branch curves vs window |
| 03 | severity_sweep_seq | §9.x / §10.3 | per-seq fused AUROC vs severity |
| 04 | branch_heatmap_L5 | §7 / §9.2 | branch × corruption AUROC heatmap |

### Geometry of corruption (§6.x — the dilation/contraction thesis)
| # | file | shows |
|---|---|---|
| 05 | radial_energy_kde_L5 | ‖z‖ energy distribution clean vs corrupt (dilation→right, contraction→left) |
| 06 | mahalanobis_d2_hist_L5 | d² histograms — the contraction **inversion** (corrupt at smaller d² than clean) |
| 07 | pca_scatter_shared_L5 | shared clean-PCA(2) scatter — *weak: overlaps; superseded by 11 (LDA)* |
| 08 | pca_drift_arrows_L5 | clean→corrupt centroid drift arrows |

### Corruption comparison (§6 — "how do the corruptions differ?", the views PCA can't show)
| # | file | shows |
|---|---|---|
| 09 | corruption_fingerprint_heatmap | **per-(layer,moment) signed deviation** — each corruption's signature; the single best comparison plot |
| 10 | corruption_similarity_matrix | cosine similarity of deviation directions — contractions cluster, dilations cluster |
| 11 | lda_scatter | **supervised LDA(2)** — separates corruptions where PCA fails |
| 12 | tsne_scatter | t-SNE of φ colored by corruption |
| 13 | energy_d2_violins | ‖z‖ and d² distributions side-by-side across all corruptions |
| 14 | effectsize_heatmap | per-(layer,moment) \|Cohen d\| — where in the network each corruption lands |

Generators: `scratchpad/exp/make_graphs.py` (01–08), `make_qualitative.py` (09–14). Both at L5.

---

## Existing plot code in the repo (⚙ — needs its pipeline stage run)

| figure | section | code | input CSV needed |
|---|---|---|---|
| event-frame corruption panels (raw input, clean→sev5) | §4 | `analysis/viz_corruptions.py` | none (renders from data) — **runnable now**, high value |
| sensitivity heatmap (per-channel deviation) | §9.6 | `analyse_plots.plot_sensitivity_heatmap` | φ on disk — runnable |
| per-layer moment distributions | §3 / §5 | `analyse_plots.plot_per_layer_distributions` | φ — runnable |
| **membrane V(t) trajectory traces** | §5 | `analyse_plots.plot_all_trajectories` | `trajs/` (NOT extracted this run — ⛔) |
| AUROC vs severity | §9 | `analyse_plots.plot_auroc_vs_severity` | φ — runnable |
| per-layer AUROC heatmap | §10.2 | `_plot_per_layer_heatmap` | φ — runnable |
| PCA subspaces (per-corruption) | §9.5 | `plot_pca_subspaces` | φ — runnable (but see 07/11) |
| statwise ablation (μ/σ²/κ) | §10.2 | `plot_statwise_ablation` | `representation_ablation` run |
| detector comparison (7 OOD) | §9 | `plot_detector_comparison` | `detector_metrics.csv` |
| temporal-features comparison | §10.6 | `plot_temporal_comparison` | temporal run |
| corruption-type confusion matrix | §10.5 | `plot_corruption_confusion_matrix` | classification run |
| free-rider ablation (trained/random/raw) | §10.1 | `plot_free_rider_ablation` | `free_rider_ablation` run |
| severity curves + Spearman ρ | §10.3 | `severity.py` | `severity` run |
| representation heatmap | §10.2 | `representation_ablation.py` | run |
| reliability / risk-coverage | §11 | `reliability.py` | ⛔ `det_outputs` all-zero |
| paper figs 1–7, severity3+, ann_vs_membrane | §9–10 | `reporting/build_paper_figures.py` | their result CSVs |

---

## Priority for the paper

**P1 — core thesis, generate/run first**
- §6 geometry: figs **05, 06, 09, 11** (dilation/contraction + corruption fingerprint + LDA). ✅ done.
- §9.4 window/severity: figs **01, 03**. ✅ done.
- §4 raw-input corruption panels: `viz_corruptions.py` — runnable now, not yet run. ⚙
- §9.2 main MDD AUROC table+heatmap: fig **04** ✅; needs `evaluate_mdd.py` run for the full L1–L5 numbers. ⚙

**P2 — supporting**
- §7 branch behavior: figs **02, 14**, plus baseline-vs-improved bars + α-tradeoff curve (🆕, see graphs/README).
- §10.1 free-rider (make-or-break validity): run `free_rider_ablation.py`. ⚙
- §10.2 representation/per-layer/statwise: run `representation_ablation.py`. ⚙
- §3/§5 per-layer distributions + sensitivity heatmap. ⚙

**P3 — supplementary / blocked**
- §5 V(t) trajectories (⛔ no `trajs/` this run), §11 reliability (⛔ det_outputs zero),
  §12 DSEC transfer (⛔ no data), §9.7 model comparison (⛔ no `model_comparison.csv`).

---

## New plotting code worth adding (🆕)
- **Baseline-vs-improved bars** (fused max vs studentized `max−α·median`) per corruption — §7.x.
- **α polarity/dropout tradeoff curve** (the Pareto frontier) — §7.x.
- **Fusion-combiner comparison bars** (max/mean/rank/studentized) — §10.7.
- **ROC curves** per corruption at L5 — §9.
- **Window heatmap** corruption×window (companion to fig 01).
- **RCF conditional-band plot** (r vs direction, clean band, contractions below) — §6.x, illustrates B2.
- **σ²-shrinkage violins** and **φ_spatial (var/pr) KDE** — the contraction signature — §6/§11.
- All have data available now (φ caches + `mdd_window_sweep.csv`); only plotting code is missing.
