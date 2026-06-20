# Vmem-φ — Project Update

*Status briefing for the team. Date: 2026-06-17. Branch: `refactor-analysis`.*

This is a plain-language catch-up on what the project is, where the code stands, and
the main results we can currently defend. Read §1–2 for context, §4 for the headline
numbers, §6 for what's in flight.

---

## 1. What we're building (one paragraph)

**Vmem-φ** is an OOD-detection benchmark for a spiking event-camera object detector.
The core bet: the sub-threshold **membrane potential** `V_mem(t)` of the network's
PLIF neurons is already computed during the forward pass, so we can read it out at
**zero extra compute** and use it as an out-of-distribution signal. We extract three
moments `[μ, σ², κ]` per channel (after global-average-pooling over space), giving a
**2112-D feature vector φ per frame**, and ask: can we tell from φ alone when the input
event stream has been corrupted?

The benchmark = **6 corruption types × 5 severities** scored against a battery of OOD
detectors, on the Prophesee Gen1 dataset (470 test sequences, ~343k frames).

---

## 2. The pipeline (how the code is laid out)

Data flows in one direction:

```
Gen1 sequences (HDF5)
  └─ event_corruption/   apply corruption to event histograms
      └─ HybridDetection/ (read-only upstream SNN) forward pass
          └─ vmem_benchmark/monitor.py  hooks V_mem on each PLIF layer
              └─ GAP + [μ,σ²,κ] → φ (2112-D)  saved to outputs/phi/<run>.pt
                  └─ analysis/  fit detectors on clean φ, score on corrupted φ
```

Three directories matter:

- **`vmem_benchmark/`** — extraction. `monitor.py` does the hook + moment math;
  `extract.py` is the inference loop; `benchmark_config.py` is the single source of
  truth for paths/corruptions/severities. **Hard constraint: `BATCH_SIZE = 1`** —
  SpikingJelly treats the batch axis as the time axis, so B>1 leaks membrane state
  across sequences and corrupts φ.
- **`analysis/`** — everything post-extraction: 7 classical detectors (Mahalanobis,
  kNN, GMM, OCSVM, PCA, AE, RealNVP flow), the new **MDD** detector (see §3), plus
  severity / reliability / ablation scripts.
- **`event_corruption/`** + `HybridDetection/` — the corruption library and the
  upstream model (the latter is read-only; we only load its checkpoint).

The full run is staged (`run_full_benchmark.ps1`): extract → fit → evaluate detectors
→ evaluate MDD → ablations → reporting tables/figures.

---

## 3. The headline idea: the MDD detector

Most of the recent work converged on a single insight that's worth understanding,
because it reframes the whole problem. Full write-up is in `Docs/novel.md`; the short
version:

**Corruptions fall into three geometric classes by how they move φ relative to the
clean manifold:**

| Class | Corruptions | What happens to φ | Why standard detectors fail |
|---|---|---|---|
| **Dilation** (louder membrane) | hot_pixel, event_rate_shift, temporal_jitter | φ moves **outward** | nothing — distance detectors handle these |
| **Contraction** (quieter membrane) | spatial_dropout, event_flood | φ moves **inward, toward the clean mean** | distance/density detectors rate a contraction as *more normal than normal* → they **invert** (AUROC < 0.5) |
| **Invisible** | polarity_flip | φ barely moves | the network learned polarity-symmetric features; signal is simply absent from V_mem |

The contraction case is the real contribution. The fix is the **MDD
(Manifold-Decomposition Detector)**: instead of collapsing φ to one distance, it scores
three **orthogonal, two-sided axes** and fuses them with a calibrated `max`:

- **B1 — global radius** (two-sided ‖φ‖): catches event_rate_shift.
- **B2 — RCF, the novel core**: "given this frame's *direction* on the clean manifold,
  is its magnitude what clean data pointing this way normally has?" This is what makes
  a *contraction* detectable — it flags **under-dispersion**, which a global distance
  cannot see.
- **B3 — deep-layer (L4) Mahalanobis**: catches temporal_jitter, which hides in the
  timing-sensitive deep layers that pooled magnitude averages away.

It's **unsupervised and label-free** — no corruption labels, no trained meta-classifier
(we checked: a supervised fusion buys nothing per-frame and fails to generalize to
unseen corruption types).

---

## 4. Main results we can defend

All numbers below are **leakage-safe, full-data static-φ (343k frames, severity 5,
contiguous fit/calib/eval split)**, from the validated `test_ideas/` experiments and
`Docs/novel.md`.

### Per-corruption best (per-frame, no aggregation)

| Corruption | Class | Best AUROC | How |
|---|---|---|---|
| hot_pixel | dilation | **1.000** | any distance |
| event_rate_shift | dilation | **0.915** | two-sided radius (B1) |
| temporal_jitter | dilation | **0.930** | layer-4 Mahalanobis (B3) |
| event_flood | contraction | 0.55 per-frame → **~0.99 aggregated** | no per-frame signal; needs sequence aggregation |
| spatial_dropout | contraction | **0.56** (RCF, B2) | the residual — see §5 |
| polarity_flip | invisible | ~0.55 (ceiling) | unfixable from V_mem |

### Single MDD score (one unsupervised number, no per-corruption routing)

- **Per-frame, no aggregation: macro AUROC ≈ 0.77** (excl. polarity) — this is the
  robust, proxy-free headline figure.
- **With per-recording aggregation: macro ≈ 0.91**, with **4 of 6 corruptions ≥ 0.99**.
  ⚠️ The aggregation numbers currently use a *within-file block proxy* because the
  legacy extraction didn't save real sequence boundaries — they need the fresh
  extraction (§6) before we quote them in a paper.

### Classical detectors / ANN baselines (for context)

Membrane φ + OCSVM reaches **~0.81 overall AUROC** across the benchmark; ResNet-18 ANN
baselines (event-image / voxel-grid) top out around **0.74** (DICE/ViM), and several
energy-based ANN methods invert badly (~0.18). So the cheap membrane signal is
competitive with a full ANN baseline.

---

## 5. Honest caveats (please read before citing numbers)

A few earlier claims did **not** survive scrutiny — flagging so nobody rebuilds on them:

1. **The "temporal features rescue the hard corruptions to ~0.85" result is an
   artifact.** It was train/eval leakage + a 50-sample degenerate split. Leakage-safe
   re-measurement gives only a modest ~0.05–0.08 temporal gain. Don't cite the 0.85.
2. **The dramatic "below-chance / anti-detectable 0.29" numbers don't reproduce** on the
   current extraction (flood 0.55 not 0.41, dropout 0.44 not 0.29). The *direction*
   (contractions invert standard detectors) holds; the exact magnitudes were
   extraction-specific.
3. **Two residuals remain genuinely hard:** `spatial_dropout` (its real signature is
   spatial, and GAP throws spatial layout away) and `polarity_flip` (absent from V_mem
   by network design — should be **scoped out** of the membrane claim).
4. **Single split, severity 5, no bootstrap CIs yet.** The per-frame macro ~0.77 is
   solid; everything else carries a few points of uncertainty until we run the severity
   sweep + CIs.

---

## 6. What's in flight right now

The highest-leverage next experiment is **already coded and running**: a re-extraction
that adds a **spatial-dispersion representation** (`phi_spatial`, 1408-D — per-channel
`spatial_var` + participation ratio) alongside the existing φ. This is the cheap
spatial summary GAP normally destroys, aimed squarely at `spatial_dropout` and
`event_flood`'s per-frame signal. The same extraction saves real `seq_lens`, which lets
us replace the aggregation *proxy* with true per-recording AUROCs + bootstrap CIs.

- **Code status:** complete and committed. `monitor.py::collect_phi_spatial`,
  `analysis/mdd.py` (MDD class with the spatial branch auto-activating), and
  `analysis/evaluate_mdd.py` (Stage 8) are all in. 5 new tests pass.
- **Data status:** the full re-extraction (~+60 GB, float32 — float16 overflowed on
  event_flood) is **in progress**. The CSVs currently in `outputs/results/` are only a
  2-corruption smoke test (hot_pixel + event_flood, 1 sequence) — **not** the full
  numbers. Treat §4 (from `test_ideas/`) as authoritative until the fresh run lands.

**Expected payoff once it completes:** spatial_dropout above its 0.56 per-frame ceiling,
a real per-frame signal for event_flood, and defensible (non-proxy) per-sequence
aggregation numbers.

---

## 7. Where to look in the repo

| You want… | Go to |
|---|---|
| The conceptual story (why MDD works) | `Docs/novel.md` |
| Performance levers + the honest caveats | `Docs/performance_brief.md`, `Docs/Findings.md` §5 |
| The experiments behind §4 | `test_ideas/` (`run_arch.py`, `run_perframe.py`, `RESULTS.md`) |
| Extraction internals | `vmem_benchmark/monitor.py`, `extract.py`, `benchmark_config.py` |
| The detector | `analysis/mdd.py`, `analysis/evaluate_mdd.py` |
| How to run anything | `HOW_TO_RUN.md`, `CLAUDE.md` |

Questions → ping me. The one thing to remember: **per-frame macro ~0.77 from a single
unsupervised membrane score is the number we stand behind today**; the 0.9+ aggregated
figures are pending the fresh extraction.
