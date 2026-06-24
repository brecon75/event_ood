# Vmem-φ — Paper Skeleton (v2, working draft)

> Living outline for the manuscript, rewritten after a full read of the ideas docs
> (`ideas.md`), the MDD design (`novel.md`), the measured results (`paper_figures.md`),
> the corruption library, and all 23 `analysis/` scripts. Sections are listed in
> submission order. Each carries: **(a)** what it contains, **(b)** the script(s) /
> artifact(s) that produce its numbers, and **(c)** a status flag.
>
> Status legend: **DRAFTED** = prose exists (in `paper_sections_theory_hardware.tex`);
> **READY** = the analysis is implemented and has produced numbers; **PENDING** = code
> exists but the run/number is not yet banked; **BLOCKED** = cannot be produced yet.
>
> Measured numbers live in `Docs/paper_figures.md` — do **not** hard-code results here
> without a pointer back to source.

**Working title:** *Vmem-φ: Zero-Cost Out-of-Distribution Detection in Spiking Neural
Networks from Membrane-Potential Statistics*

**Three-pillar thesis** (what makes this more than a benchmark):
1. **Theory** (§5) — per-corruption detectability is a closed-form consequence of the PLIF
   state equation, not an empirical accident.
2. **Positioning** (§8) — φ is the *only* OOD signal that runs inside the neuromorphic
   compute envelope; every competing method needs compute the chip cannot provide.
3. **Method** (§6–7) — the *dilation / contraction / invisible* geometric diagnosis, and the
   **MDD** detector (the direction-conditioned RCF core) that follows from it and is the
   first method to take the contraction corruptions above chance per-frame.

> **Open placement decision:** §8 (hardware) currently sits right before Experiments because
> its drafted prose ends by telling the reader how to read the benchmark. Alternative: move it
> to §3 (right after Related Work) as an up-front stakes-setter. Flip if preferred.

---

## 0. Abstract `[TODO — write last]`

One paragraph: PLIF membrane potential V_mem(t) is a free byproduct of the SNN forward pass;
its per-channel statistics φ = [μ, σ², κ] (2112-D via GAP over 704 channels × 4 PLIF layers)
form an OOD signal at *zero additional compute*. We give (i) a closed-form leaky-integrator
account of *why* each corruption is or isn't detectable, (ii) the argument that φ is the only
OOD method that runs on the neuromorphic hardware SNNs target, and (iii) the MDD detector that
follows from a dilation/contraction geometric diagnosis. Benchmarked over 6 event-camera
corruptions × 5 severities on Prophesee Gen1. Headline: per-frame macro ≈ 0.77 (excl.
polarity), four of six corruptions ≥ 0.99 with sequence aggregation. *Pull final numbers from
`Docs/paper_figures.md`.*

---

## 1. Introduction `[TODO]`
- Event cameras + SNNs for low-power detection; deployment target is neuromorphic silicon.
- OOD detection is safety-critical for autonomous perception, but standard OOD methods assume
  an ANN with logits / gradients / feature re-forwarding — unavailable on-chip.
- Contribution list:
  1. φ: a zero-cost membrane-statistics OOD signal.
  2. A closed-form leaky-integrator theory predicting per-corruption detectability (§5).
  3. The neuromorphic-compatibility argument (§8) — φ is the *only* method that runs on target HW.
  4. The dilation/contraction diagnosis + the MDD detector with the novel RCF branch (§6–7).
  5. A multi-detector benchmark on Gen1, plus honest negatives: polarity_flip (input-side,
     membrane-blind) and spatial_dropout (GAP-destroyed signature).
  6. **Gen1-C**, a reproducible event-camera corruption benchmark (seeded, deterministic,
     model-agnostic — released as code + seeds, ImageNet-C style), plus a released
     membrane-statistics (φ) feature set so OOD detectors can be benchmarked without re-running
     SNN inference (§4.x).

---

## 2. Background & Related Work `[TODO]`
- PLIF / SpikingJelly `MultiStepParametricLIFNode`; Hybrid SNN–ANN detector on Gen1 (36% mAP),
  B=1 constraint (batch axis = time).
- OOD detection families: MSP, ODIN, Energy, ReAct, ViM, GradNorm, Mahalanobis/kNN on features.
- Neuromorphic readout: V_mem as a native hardware register (Loihi, BrainScaleS, SpiNNaker).
- Event-camera corruption literature.

---

## 3. The φ Representation `[READY — monitor.py / extract.py]`
*What we detect from, and the two properties that drive every later result.*
- VmemMonitor `forward_hook` on all 4 PLIF layers → V_mem tensor (T×1×C×H×W) → GAP over space →
  first three moments [μ, σ², κ] per channel → **2112-D φ per frame** (3 × 704).
- The `phi_spatial` extension: spatial-dispersion stats GAP discards (`spatial_var`,
  participation-ratio `spatial_pr`) → **1408-D** per frame. (Stored float32; extraction with
  `phi_spatial` + per-sequence `seq_lens` is **done** for clean + all 6 corruptions × L1/L3/L4/L5.)
- **Two consequences** (carry through the whole paper): (i) GAP discards spatial layout — φ
  knows *how much* each channel fired, not *where*; (ii) φ inherits the network's invariances.
- Clean φ lies on a **thin, curved, scene-dependent manifold** (clean d² median ≈ 14 ≪ D=2112);
  detection = deciding whether a new φ is on that manifold.

---

## 4. The Event-Camera Corruption Suite `[DRAFTED — paper_sections_theory_hardware.tex §1–2]`
- Input representation (N×20×H×W; 240×304 Gen1; 2 polarities × 10 bins of 50 µs; ON 0–9 / OFF 10–19).
- The six corruptions, design principles (seeded reproducibility, NumPy/CuPy agnostic,
  severity monotonicity), taxonomy table (failure mode → perturbed axis).
- **§4.x Corruption algorithms** — per-corruption mechanism + severity schedules (Table).
- Source of truth: `event_corruption/corrupt/*.py`, `vmem_benchmark/benchmark_config.py`.
- *Already written as compilable LaTeX.*

### §4.x Benchmark & Released Artifacts (Gen1-C) `[contribution #6]`
*Frame as a reproducible corruption benchmark + feature release — NOT a new dataset (the data
is derived from Prophesee Gen1, an existing licensed dataset).*
- **Gen1-C corruption benchmark (released as code + seeds).** 6 event-specific corruptions × 5
  monotonic severities, fully **seeded and deterministic** — a (corruption, severity, seed) triple
  regenerates the exact corrupted stream via `apply_corruption(..., rng)`. NumPy/CuPy-agnostic and
  operates on the standard `(N,20,H,W)` histogram, so it composes with *any* event model, not just
  ours (ImageNet-C / CIFAR-C model). Ships the corruption library + seed manifest + config that
  regenerate from the user's own Gen1 copy — **not** redistributed corrupted histograms (license +
  ~TB size).
- **Released φ feature set.** Pre-extracted membrane statistics for clean + 30 corrupted runs
  (with `seq_lens`). Lets the OOD community benchmark detectors on membrane statistics **without**
  GPU SNN inference. φ is per-channel moments — highly abstracted, not reconstructable to events —
  which weakens (but does not remove) Gen1-redistribution concerns.
- **Fixed evaluation protocol.** Leakage-safe fit/calib/eval split honoring `seq_lens`; per-frame
  and per-sequence AUROC; the 7-detector + MDD scoring harness.
- **Honesty / scope:** severities are synthetic transforms, not measured sensor failures; the
  contribution is the *benchmark + protocol + feature release*, not inventing corruption physics.
- **ACTION before any release claim:** verify the Prophesee Gen1 license permits distributing
  derived features / regeneration code; default to "code regenerates from your own Gen1 copy" if unsure.

---

## 5. Theory: Why Membrane Statistics Detect Corruption `[DRAFTED — .tex §3]`
- PLIF state equation → leaky exponential accumulator closed form V[t] = Σ λᵏ W X[t−k].
- Steady-state moments under stationary input: E[V] = τWm; Var[V] from input variance +
  autocovariance term (the lever temporal corruptions pull).
- Per-corruption closed-form predictions matching the measured AUROC ordering.
- §5.x "Why GAP causes the inversions" → falsifiable taxonomy → motivates phi_spatial / MDD.
- **Illustrative figure: membrane V(t) trajectories** (`analyse_plots.plot_all_trajectories`) —
  actual sub-threshold traces, clean vs corrupt, to ground the leaky-accumulator picture. `[READY]`
- *Already written.* **Note overlap with §6** — §5 is the *a priori* closed form; §6 is the
  *geometric* taxonomy that turns it into an architecture. Frame to avoid repetition.

---

## 6. The Geometric Diagnosis `[READY — novel.md §1–2]`
*The conceptual core; the architecture in §7 follows from it.*
- Corruptions classified by **how they move φ relative to the clean manifold**:
  - **Dilations** (membrane louder, φ outward): hot_pixel, event_rate_shift (radial),
    temporal_jitter (in the deep-layer subspace).
  - **Contractions** (membrane quieter, φ toward the mode): spatial_dropout, event_flood —
    these **invert** distance- and density-based detectors ("more normal than normal", AUROC < 0.5).
  - **Invisible**: polarity_flip — no signature in V_mem (network invariance).
- Why every prior single method (Mahalanobis, kNN, GMM, flow, naive max, learned LR fusion)
  topped out ≈ 0.70 macro and inverted on contractions — they collapse φ to one signed scalar.

### §6.x Figure: "Geometry of Corruption" `[DONE — analysis/plot_corruption_graphs.py → outputs/graphs/]`
*Purpose-built panel that makes the dilation/contraction/invisible thesis visible. All generated
by the single consolidated module `analysis/plot_corruption_graphs.py` (leakage-safe; L5).*
- **(a) Radial KDE** of membrane energy ‖z‖, clean vs corrupt per corruption — dilations right,
  contractions left, polarity overlaps. *Clearest visual of the core thesis.* → `05_radial_energy_kde_L5.png`
- **(b) Mahalanobis d² histogram** — contractions sit at *smaller* d² than clean → explains
  AUROC < 0.5 (the inversion). → `06_mahalanobis_d2_hist_L5.png` (+ log violins `13_energy_d2_violins.png`).
- **(c) Severity-drift arrows** in one shared clean-PCA → `08_pca_drift_arrows_L5.png`
  (+ shared-PCA scatter `07_pca_scatter_shared_L5.png`).
- **(d) Corruption similarity matrix** (cosine of mean-shift directions) → `10_corruption_similarity_matrix.png`
  — surfaces the event_rate_shift↔spatial_dropout shared axis. **RCF conditional-band view still TODO (schematic).**
- **(e) Per-layer / per-branch separation** (jitter flat on pooled radius, separating at L4) →
  per-branch window panel `02_window_branches_L5.png` + branch heatmap `04_branch_heatmap_L5.png`.
- Also: channel-shift fingerprint `09_corruption_fingerprint_heatmap.png` (residuals ≈0), LDA
  `11_lda_scatter.png`, φ_spatial KDE `21_phi_spatial_kde.png`, σ²-shrinkage `20_sigma2_shrinkage_violin.png`.
- Lives in §6 (diagnosis); quantitative AUROC stays in §9.

---

## 7. Method: The Manifold-Decomposition Detector (MDD) `[READY — analysis/mdd.py, evaluate_mdd.py]`
- Premise: stop collapsing φ to one number; decompose deviation into orthogonal axes, score each
  two-sided, fuse as a calibrated OR.
- **B1 Global radius** (two-sided): |‖z‖ − E‖z‖|/σ — dilation/contraction energy axis (wins event_rate_shift).
- **B2 RCF — Radial-Conditional** (two-sided, direction-conditioned via cosine-kNN; **the novel core**):
  flags under-dispersion → the first detector to take spatial_dropout above chance per-frame.
- **B3 Deep-layer L4 Mahalanobis** (one-sided): the subspace axis for temporal_jitter.
- **Fusion**: standardize each branch on held-out clean → S = max(B1,B2,B3), a true unsupervised
  OR (the shipped default is the **studentized** variant S = max − α·median; see §7.x). Supervised
  LR fusion buys nothing per-frame and fails to generalize (LR-LOO ≈ 0.49) — report as a negative.
- **Optional B4 per-recording aggregation** (for event_flood only — no per-frame signal).
- Pipeline: φ → standardize + PCA denoise → branches → calibrated max → optional aggregate.

### §7.x Strengthening the branches & fusion — residual-corruption pass `[DONE 2026-06-23 — wired into analysis/mdd.py (default on, toggleable); Docs/mdd_improvements.md]`
*A targeted pass on the two residuals (polarity_flip, spatial_dropout) under a strict
no-re-extraction, no-new-branch constraint: improve the EXISTING branches and the fusion rule.*
- **Radius-branch enrichment (covariance rotation).** The isotropic radius ‖z‖ ignores the
  *covariance structure* of the clean PCA-64 manifold. A polarity_flip rotates that covariance
  (ON/OFF swap) while barely moving the marginals — invisible to ‖z‖ and washed out by
  Ledoit-Wolf shrinkage, but caught by an **empirical (unshrunk) covariance Mahalanobis** in the
  same PCA-64 space (`emp_maha64`: polarity per-sequence AUROC **0.656** > spatial 0.639). Folded
  *into* B1 as `radius' = max(‖z‖-energy, emp_maha64)` so the dilation sensitivity is kept (full
  whitening alone drops event_rate_shift 0.900→0.600).
- **Studentized fusion (better than the plain max).** The calibrated max sits *below* the best
  single branch on both residuals by inflating the clean floor (≥1 branch randomly spikes on
  clean). Replace with **`fused = max(branches) − α·median(branches)`** (α≈0.5): a lone strong
  branch (dropout's RCF) survives, diffuse clean noise cancels.
- **Result difference** (per-sequence AUROC, L5; baseline max → improved default α=0.5, the
  canonical wired-in `mdd.py`): spatial_dropout 0.560→**0.620** (→0.689 at α=1.25), polarity_flip
  0.609→0.611, temporal_jitter 0.947→0.952, event_rate_shift 0.952→0.955; worst-corruption floor
  0.560→**0.611**; the four solved corruptions hold or improve at every severity (no regressions).
  α trades polarity vs dropout (one static combiner cannot maximize both). *Sources: full
  per-severity frame/W64/seq table in `outputs/results/final_results.csv`
  (`analysis/plot_corruption_graphs.py`); exp1–exp10 timeline in `Docs/mdd_improvements.md`.*
- **Honest ceiling:** neither residual reaches the 0.8 "solved" bar — spatial_dropout (contraction)
  caps ≈0.69, polarity_flip (polarity-symmetric network) ≈0.66; the φ information ceiling. This is
  itself a reportable result (§11).

---

## 8. The Neuromorphic Hardware Advantage `[DRAFTED — .tex §4]`
*Positioning section — reframes the contribution from "marginally better" to "only method that
runs at all."* (Placement flagged at top.)
- V_mem is a native, already-maintained register on Loihi / BrainScaleS / SpiNNaker; φ adds zero
  MACs and zero weight fetches.
- Table: every competing OOD family (MSP/Energy, ODIN, ReAct, ViM, GradNorm, ANN-feature
  kNN/Mahalanobis) needs a capability absent from the spiking, feed-forward, integer-event model.
- The reframed claim + the explicit simulation-not-silicon caveat.
- *Already written.* Forward-refs §7 (method) and §9 (experiments) — works in this position.

---

## 9. Experiments & Results `[READY/PENDING — evaluate_mdd.py → paper_figures.md]`
- §9.1 Setup: Gen1, 470 test sequences, 6 corruptions × 5 severities, leakage-safe
  fit/calib/eval split honoring `seq_lens` (`vmem_utils.split_train_eval`/`held_out_eval`).
  **Metrics: AUROC + AUPR + FPR@95%** (`evaluate_detectors.py`), reported per-frame and
  per-sequence; the headline aggregate uses the **severity≥3 AUROC** cut
  (`table_final_main` sorts on it) alongside the all-severity mean.
- §9.2 **Main result**: MDD fused per-frame + per-sequence AUROC tables (`mdd_metrics.csv`,
  `mdd_metrics_aggregated.csv`).  `[READY]`
- §9.3 Fused vs best-branch (oracle upper bound) — fusion clearly wins on event_rate_shift;
  the event_flood per-frame→per-sequence jump (≈0.47 → 1.00). `[READY]`
- §9.4 Severity / aggregation-window sweep (`evaluate_mdd_windows.py` →
  `outputs/results/mdd_window_sweep.csv`; 17 windows × all branches × all corr/sev). `[READY]` —
  fused AUROC vs pooling window W=1…full exposes which results are *manufactured by pooling*:
  event_flood climbs 0.55→1.00 (crosses 0.9 ≈ W=64), hot_pixel saturated at every W, while
  polarity_flip (0.54→0.61) and spatial_dropout (0.54→0.56) stay flat — the genuine residuals.
- §9.5 **PCA subspace scatter grid** (clean vs corrupt per corruption)
  (`analyse_plots.plot_pca_subspaces`). `[READY]` — *separability view only; the manifold-geometry
  story is the §6.x panel. Consider refitting this on one shared clean-PCA so panels are
  comparable, or demote it to supplementary in favor of §6.x (a)/(b).*
- Baselines for context: 7 OOD detectors (`evaluate_detectors.py`) + ANN ResNet-18
  (`evaluate_ann_baselines.py`). `[READY]`
- §9.6 **Sensitivity heatmap** — per-channel / per-corruption deviation of φ from clean
  (`analyse_plots.plot_sensitivity_heatmap`). `[READY]`
- §9.7 **Model comparison** — OOD performance across model architectures / conditions
  (`reporting` → `model_comparison.csv`, `fig7_model_comparison`, `table5_models`).
  `[PENDING — confirm model_comparison.csv is produced; reporting scripts reference it but the
  CSV may not yet exist.]`

---

## 10. Ablations & Validity Checks
- §10.1 **Free-rider** (`free_rider_ablation.py`): trained vs random-init SNN vs raw-input
  stats — *the make-or-break validity check that φ's signal is learned, not raw input*. `[PENDING run]`
- §10.2 **Representation comparison** — *is the sub-threshold membrane the right signal?*
  Scores OOD across the representations `extract_representation` defines
  (`representation_ablation.py`, `fig3_representation`): `[READY]`
  - membrane φ (`full_membrane`) and its per-moment slices μ/σ²/κ
    (`run_statwise_ablation`) and per-layer slices (`run_per_layer_auroc_table`) — L4 carries jitter;
  - **spike-rate** and **spike-entropy** (firing-rate features — a distinct signal from sub-threshold V_mem);
  - **ANN feature** (`last_ann_gap`) and **logits** (`head_cls_L0_gap`) — the ANN-side baselines;
  - **fused** (`membrane_fused`, concat + LogReg meta-classifier; `fusion_features.py`).
- §10.3 Severity monotonicity, Spearman ρ (`severity.py`, `run_spearman_severity`). `[READY]`
- §10.4 Cross-corruption zero-shot generalization (`cross_corruption.py`, held-out types). `[READY]`
- §10.5 **What φ encodes beyond binary OOD** (already implemented — promote from "future"):
  - 7-class corruption-type classification + confusion matrix
    (`run_corruption_classification`, `plot_corruption_confusion_matrix`). `[READY]`
  - Severity regression R² — φ encodes intensity *continuously* (`run_severity_regression`). `[READY]`
- §10.6 **Temporal & sequence representation** (`analyse_temporal.py`, `extract_offline_features.py`):
  handcrafted `temporal_phi` (28-D: lag-1 autocorrelation, CUSUM, HF energy, …), the
  **Temporal-AE** latents, and **margin histograms** (V_mem − θ quantized). Honest verdict:
  leakage-safe temporal beats static by only **~0.05–0.08** on the hard corruptions (lag-1
  autocorr / L4 help jitter) — *not* the retracted 0.85. Report as a real-but-modest effect with
  the retraction noted. `[READY — frame honestly per Findings.md §5]`
- §10.7 **Fusion-combiner ablation** — *is `max` the right OR?* `[DONE 2026-06-23 — studentized
  fusion wired into mdd.py; Docs/mdd_improvements.md; fig 17_fusion_combiners]` Sweep combiners
  over the calibrated branches: max, mean, top-2 mean,
  p-norm, quantile-rank max/mean, and studentized **max − α·{mean(rest), median, min}**. Findings:
  (i) plain max sits *below the best single branch* on both residuals (clean-floor inflation);
  (ii) rank/quantile calibration helps the residuals but *collapses* the easy corruptions
  (compresses their large signal → event_flood 1.00→0.61); (iii) **studentized `max − α·median`**
  is the Pareto winner (spatial_dropout 0.560→0.689, worst-case floor 0.560→0.607, no
  easy-corruption regression), at a tunable polarity/dropout tradeoff. Takeaway: the *fusion rule*,
  not just the branches, is a first-class design axis.

---

## 11. Discussion `[TODO]`
- The magnitude-vs-structure (dilation/contraction) taxonomy and which scoring rule each needs.
- The two honest residuals: **polarity_flip** (input-side only, ~0.5 membrane ceiling by
  construction) and **spatial_dropout** (GAP destroyed its spatial signature; `phi_spatial` is
  the prescribed fix).
- (Optional) Conformal prediction sets for guaranteed FPR — safety framing
  (`run_conformal_prediction`, `READY`); include if it strengthens the safety angle.

---

## 12. Limitations `[TODO]`
- Simulation (SpikingJelly on GPU), not neuromorphic silicon — cost claim is architectural.
- AUROC point estimates, single split, **no bootstrap CIs** yet (the severity sweep across
  L1/L3/L4/L5 now exists in `final_results.csv`; CIs are the remaining gap).
- Per-sequence aggregation now uses **true `seq_lens`** (no longer the within-file block proxy);
  the 0.9+ per-sequence figures are real.
- **Reliability / mAP-degradation story is void** until clean `det_outputs` has nonzero confident
  detections (`reliability.py`; memory `det-outputs-all-zero`) — keep out of the paper for now.
- Temporal gain is modest (~0.05–0.08), **not** the retracted 0.85 (that was leakage + 50-sample
  noise; the leakage-safe `held_out_eval` split is the fix). See `Findings.md` §5.
- Single dataset (Gen1); DSEC transfer is future work (`build_paper_tables.py` references a
  `dsec_transfer.csv` that does not yet exist).

---

## 13. Conclusion `[TODO]`

---

## Section / artifact status table

| § | Content | Generating script / artifact | Status |
|---|---|---|---|
| 3 | φ representation | `monitor.py`, `extract.py` | READY |
| 4 | Corruption suite + algorithms | `.tex` §1–2; `event_corruption/corrupt/*` | **DRAFTED** |
| 4.x | Gen1-C benchmark + φ feature release (contrib #6) | `event_corruption/`, `outputs/phi/` | PENDING (license check + packaging) |
| 5 | Leaky-integrator theory | `.tex` §3 | **DRAFTED** |
| 6 | Geometric diagnosis | `novel.md` §1–2 | READY (prose) |
| 7 | MDD architecture | `mdd.py`, `evaluate_mdd.py` | READY |
| 7.x | Branch enrichment (emp-cov radius') + studentized fusion | `analysis/mdd.py` (wired, default on), `Docs/mdd_improvements.md` | **DONE** (validated; no regressions) |
| 8 | Neuromorphic advantage | `.tex` §4 | **DRAFTED** |
| 9.2–9.3 | MDD AUROC tables, oracle gap | `evaluate_mdd.py` → `paper_figures.md` | READY |
| 9.4 | Severity/window sweep | `evaluate_mdd_windows.py` -> `mdd_window_sweep.csv` | READY |
| 6.x | "Geometry of Corruption" panel (radial KDE, d² inversion, drift arrows, similarity) | `analysis/plot_corruption_graphs.py` → `outputs/graphs/` | **DONE** (RCF band still schematic) |
| 9.5 | PCA subspace figure | `analyse_plots.plot_pca_subspaces` | READY |
| 10.1 | Free-rider validity check | `free_rider_ablation.py` | PENDING run |
| 10.2 | Representation comparison (membrane/spike/ANN/logits/fused + μ/σ²/κ + per-layer) | `representation_ablation.py`, `fusion_features.py`, `fig3` | READY |
| 10.3 | Severity Spearman ρ | `severity.py` | READY |
| 10.4 | Cross-corruption | `cross_corruption.py` | READY |
| 10.5 | Corruption-class + severity-regression | `analyse_comparisons.py` | READY |
| 10.6 | Temporal & sequence (temporal_phi / Temporal-AE / margin hist) | `analyse_temporal.py`, `extract_offline_features.py` | READY (modest) |
| 10.7 | Fusion-combiner ablation (max vs studentized max−α·median) | `analysis/mdd.py`, `Docs/mdd_improvements.md`, fig 17 | **DONE** (wired) |
| — | Consolidated figure generator (20 figs) + per-severity table | `analysis/plot_corruption_graphs.py` → `outputs/graphs/`, `outputs/results/final_results.csv` | **DONE** |
| 5 | Membrane V(t) trajectory figure | `analyse_plots.plot_all_trajectories` | READY |
| 9.6 | Sensitivity heatmap | `analyse_plots.plot_sensitivity_heatmap` | READY |
| 9.7 | Model comparison | `reporting` / `model_comparison.csv` | PENDING (confirm CSV) |
| 11 | Conformal sets (optional) | `analyse_comparisons.run_conformal_prediction` | READY |
| 11 | Reliability / mAP | `reliability.py` | **BLOCKED** (det_outputs zero) |
| 12 | DSEC transfer | — | future work |

## Before submission — the runs that gate the headline claims
1. **Free-rider ablation** (§10.1) — without (A) trained ≫ (C) raw-input, the whole premise is open.
2. ~~Re-extract φ with `seq_lens` (+ `phi_spatial`)~~ **DONE** — real per-sequence aggregation and
   the spatial_dropout `phi_spatial` branch are now measured (not proxy); MDD improvements wired in.
3. **Severity sweep + bootstrap CIs** on the MDD numbers — the severity sweep exists
   (`final_results.csv`, all of L1/L3/L4/L5); still need **bootstrap CIs** to replace point estimates.
