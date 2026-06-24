# Vmem-φ OOD Detection — Performance Briefing for an External Reviewer

**Purpose of this doc:** Hand this to another LLM to get *fresh ideas for improving OOD-detection performance* on the hard corruptions. It contains (1) the project setup, (2) a claim from our internal docs that turned out **not to reproduce**, (3) every lever we have already tried with real numbers, and (4) the open problem. Please read the "Already tried" section before proposing ideas so you don't repeat them.

---

## 1. Project in one paragraph

**Vmem-φ** is a benchmark for out-of-distribution (OOD) detection in Spiking Neural Networks. A Hybrid SNN–ANN event-camera object detector (Prophesee Gen1, 240×304, ~36% mAP) runs on clean and corrupted event streams. From the Parametric Leaky-Integrate-and-Fire (PLIF) neurons we extract the sub-threshold membrane potential **V_mem(t)** at *zero extra compute*, and use it as an OOD signal. The question: **can membrane statistics detect when the input is corrupted/OOD?** We evaluate 7 OOD detectors against 6 corruption types at 5 severities.

## 2. Data, representations, detectors

- **Dataset:** Gen1 test split, ~470 sequences, ~343,099 frames total. Corruptions applied per *whole sequence* at severity L1–L5.
- **Static φ (main representation):** 2112-D per frame = `[μ, σ², κ]` (mean, variance, excess kurtosis) of V_mem, **Global-Average-Pooled over space**, across 704 channels in 4 PLIF layers (64/128/256/256). GAP discards spatial detail.
- **Temporal representations** (from the V(t) trajectory over 10 SNN timesteps per frame):
  - `temporal_phi` — 28-D per frame = 7 handcrafted stats × 4 layers (autocorrelation, CUSUM, HF-energy ratio, dV mean/var, margin mean/min/var).
  - `temporal_gap` — per-frame GAP'd V(t) trajectory `(10, 704)`, fed to a 1D-CNN Temporal Autoencoder.
  - `trajs/` — raw full-resolution V(t), **capped at 50 samples** (full storage ≈ 15 TB).
- **Detectors (fit on clean only):** Mahalanobis (Ledoit-Wolf), kNN (k=5), GMM (5 comp), OCSVM (RBF), PCA (50), MLP Autoencoder, RealNVP flow. Mahalanobis is the reference.
- **Hard constraints:**
  - `BATCH_SIZE = 1` is **mandatory** — SpikingJelly treats the batch axis as the time axis; B>1 corrupts membrane state. This caps single-stream throughput; scaling = parallel workers, not bigger batches.
  - Detector fit currently uses only `MAX_FIT_SAMPLES = 3000` clean frames.
  - Evaluation is **per-frame** (each frame labeled clean/OOD), not per-sequence.

## 3. The corruptions and their static-φ behavior

Full-data static-φ Mahalanobis AUROC at L5 (343k frames, leakage-safe split, the trustworthy reference):

| Corruption | Static-φ AUROC (L5) | Behavior |
|---|---|---|
| `hot_pixel` | **1.000** | Trivial — persistent spurious events shove moments off-distribution |
| `temporal_jitter` | 0.709 | Moderate |
| `event_rate_shift` | 0.675 | Moderate; phase transition ~severity 3 |
| `polarity_flip` | 0.429 | **Below chance** — model learned polarity-symmetric features |
| `event_flood` | 0.408 | **Below chance** — flood preserves per-channel moment structure |
| `spatial_dropout` | 0.286 | **Strongly anti-detectable** — corrupted looks *more* in-distribution than clean |

**AUROC < 0.5 means the signal is informative but inverted:** corrupted frames sit *closer* to the clean mean than held-out clean frames do (the corruption quiets/regularizes the membrane).

## 4. THE CLAIM (from `Docs/Findings.md`) — and why it does NOT reproduce

Our internal `Findings.md` calls this *"the most important finding in the entire benchmark"* and concludes *"the paper's strongest version is a temporal-phi paper, not a static-phi paper."* It reports that **temporal trajectory features rescue the hardest corruptions**:

| Corruption | Static (claimed) | Temporal (claimed, 50 samples) |
|---|---|---|
| `event_flood` | 0.554 | **0.830 / 0.848** |
| `spatial_dropout` | 0.439 | **0.817 / 0.846** |
| `temporal_jitter` | — | **0.905** |

**We could not reproduce these.** The claim was computed on the **50-sample raw-trajectory path** (`trajs/`, capped at 50). Re-running that *exact* method on freshly extracted data:

| Corruption | Findings claim | Our reproduction (same 50-traj method) | Per-frame temporal (proper split) |
|---|---|---|---|
| `event_flood` | 0.83–0.85 | **0.291** | 0.625 |
| `spatial_dropout` | 0.82–0.85 | **0.037** | 0.541 |
| `temporal_jitter` | 0.85–0.91 | **0.061** | 0.602 |

**Diagnosis — the 0.85 was an artifact**, from two compounding causes:
1. **Broken sample regime:** 50 samples = 35 train / 15 test, all from the *first 50 frames of one sequence* — tiny, autocorrelated, unrepresentative. A 28-D Mahalanobis fit on 35 points is near-degenerate (AUROC swings to 0.04).
2. **Train/eval leakage, since fixed:** git history has a `refactor: unified train/eval split` commit *after* Findings.md. The original temporal code almost certainly scored clean frames that were in its own fit set (no held-out split), so clean looked perfectly in-distribution by construction → inflated AUROC. The split fix deflated it to honest levels.

## 5. The honest current performance picture

On leakage-safe, larger-sample measurements, **temporal beats static only modestly** on the hard corruptions (~0.54–0.62 vs static ~0.41–0.58). A 4-detector head-to-head on a freshly extracted 5-sequence subset (5,160 frames, all detectors fit on the same clean-train, 95% bootstrap CIs):

| Corruption | STATIC Maha | STATIC MLP-AE | TEMPORAL handcrafted | TEMPORAL AE |
|---|---|---|---|---|
| `event_flood` | 0.578 [.560,.595] | 0.50–0.56 | **0.625 [.610,.639]** | 0.520 [.501,.538] |
| `spatial_dropout` | 0.468 [.450,.486] | 0.457 [.438,.478] | **0.541 [.526,.556]** | 0.399 [.378,.419] |
| `temporal_jitter` | 0.732 [.714,.749] | **0.839 [.827,.852]** | 0.602 [.587,.619] | 0.575 [.557,.592] |

Findings:
- Handcrafted temporal *significantly* beats static on event_flood and spatial_dropout (non-overlapping CIs) — **but by ~0.05–0.08, not a rescue to 0.85.**
- The Temporal **AE** is the *weakest* detector — the handcrafted stats are doing the temporal work.
- **Caveat — this subset is not representative:** its static event_flood AUROC is 0.578 vs the full-data 0.409 (the extraction caps to the *first* N sequences, which is a biased sample). The relative ranking is internally valid; absolute numbers are not. Bootstrap CIs capture frame-level noise but there are only 5 distinct scenes, so between-scene uncertainty is larger than the CIs imply.

## 6. Levers already tried (do not re-propose these without a new twist)

| Lever | Result |
|---|---|
| **Two-sided / folded Mahalanobis** (flag "too close to mean" as well as "too far") | Rescues the 3 below-chance corruptions above chance (spatial_dropout 0.286→0.531, event_flood 0.408→0.577, polarity 0.429→0.547) but **degrades** the already-good ones (jitter 0.709→0.615). Trade-off, not free. |
| **Global activity scalar** (mean σ² energy, two-sided) | Big win for `event_rate_shift` (0.675→**0.850**); near-useless for the others. Corruption-specific. |
| **More fit samples** (3k→50k clean frames) | Free, small: jitter 0.709→0.757; rest flat. The covariance was being fit on 1.4% of available clean data. |
| **Per-feature standardization** before Ledoit-Wolf | jitter 0.709→**0.820**; but **hurts** event_rate_shift (0.675→0.568). Trade-off. |
| **Naive `max` fusion of two-sided + activity** | Worse than either alone — propagates the weak signal's false positives. |
| **Temporal AE** (1D-CNN on temporal_gap) | Weak everywhere (0.40–0.58); worse than handcrafted temporal and often worse than static. |
| **Handcrafted temporal (per-frame, proper)** | Modest significant win on event_flood/spatial_dropout (~0.54–0.62). |

**Per-corruption best achievable so far** (oracle picking the right lever): hot_pixel 1.00, event_rate_shift 0.85 (activity), temporal_jitter 0.82–0.84 (standardized static / static-AE), event_flood 0.63 (handcrafted temporal), polarity_flip 0.55 (two-sided), spatial_dropout 0.54 (handcrafted temporal). **No single transform wins everywhere** — the corruptions live in different feature geometries.

## 7. Untested / partially-tested ideas we suspect matter

- **Sequence-level aggregation (biggest untested lever).** Corruption is applied to a *whole sequence*, so every frame carries the same bias. Averaging per-frame OOD scores within a sequence integrates that consistent bias and tightens class separation by ~√(frames-per-sequence). A weak 0.55 per-frame signal could become strongly separable per *recording*. It changes the decision granularity (per-sequence, not per-frame) and needs `seq_lens` (now saved). Not yet measured.
- **Feeding multiple cheap views into the Stage-3 LogReg meta-classifier** (baseline distance + two-sided + activity + standardized + handcrafted-temporal) so it learns per-corruption which signal to trust — captures the oracle envelope without the individual regressions.
- **Full-dataset temporal extraction.** `temporal_gap`/`temporal_phi` were never extracted for all 343k frames or all corruptions; the Temporal AE has only ever trained on ≤4k frames.
- **Spatial information.** Everything is GAP'd (spatially averaged); the raw V(t) keeps H×W structure that GAP destroys — possibly where dropout/flood signal lives.

## 8. The open problem (what we want ideas on)

**Reproducibly and cheaply improve OOD AUROC on the hard corruptions — `event_flood`, `spatial_dropout`, `polarity_flip` (and ideally lift `temporal_jitter`/`event_rate_shift` to >0.9) — without breaking the easy ones.** Key realities to design around:

1. Three corruptions are **below chance** for static φ (informative but inverted).
2. No single feature transform wins across all corruptions.
3. The genuine signal is weak per-frame; the corruption is consistent per-sequence.
4. `BATCH_SIZE=1` constraint; per-frame φ is cheap, but raw trajectories are 15 TB at full scale.
5. The temporal "rescue" we hoped for is, reproducibly, only a ~0.05–0.08 improvement — we need either a better temporal representation, better aggregation, or a better detector/fusion that genuinely closes the gap.

**Concrete questions:** What detector/representation/aggregation changes would most reliably push the below-chance corruptions to high AUROC? Is there a principled fix for the "anti-detectable" inversion beyond two-sided scoring? How best to exploit the per-sequence consistency of the corruption? What temporal representation would beat the modest handcrafted-stats result?
