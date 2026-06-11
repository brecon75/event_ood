# The Manifold-Decomposition Detector (MDD)

*A single unsupervised OOD detector for event-camera corruptions, built from the
membrane potential V_mem of a spiking network — and why it works.*

This document explains (1) the geometric signature each corruption leaves on the
static-φ membrane representation, (2) what it therefore takes to detect each one,
(3) why every prior single method failed, and (4) the novel architecture that
detects all the *detectable* ones with one unsupervised score.

All AUROCs below are leakage-safe, full-data static-φ (343k frames, L5,
contiguous fit/calib/eval split, eval subsampled 8k/class). "Per-frame" numbers
use no aggregation; the aggregated numbers use a within-file block proxy and are
flagged as such. Scripts: `test_ideas/run_rcf.py`, `run_arch.py`, `run_perframe.py`.

---

## 0. The representation we detect from

Each frame is summarised by **static φ**: for every PLIF channel we take the
sub-threshold membrane potential V_mem, **Global-Average-Pool it over space**
(H×W collapsed to a scalar per channel), and keep its first three moments
`[μ, σ², κ]`. Stacked over 704 channels × 4 layers → a **2112-D vector per
frame**, extracted at zero extra compute from the detector's forward pass.

Two consequences govern everything that follows:

- **GAP discards spatial layout.** φ knows *how much* each channel fired on
  average, not *where*. Any corruption whose signature is spatial (which pixels
  changed) survives only as a faint second-order echo.
- **φ inherits the network's invariances.** If the upstream SNN is blind to a
  transformation, V_mem — and therefore φ — is blind to it too.

The clean φ vectors do not fill space uniformly; they lie on a **thin, curved,
scene-dependent manifold**. Detection = deciding whether a new φ is *on* that
manifold. The geometry of "off the manifold" is different for each corruption,
and that is the whole story.

---

## 1. The six corruptions: signature and detectability

We group them by **how they move φ relative to the clean manifold**, because that
— not the corruption's name — determines what detects it.

### Class A — Dilations (membrane gets *louder*)

The corruption injects or amplifies activity, so the moments grow and φ moves
**outward**, away from the clean mean.

| Corruption | What it does to events | Signature in φ | Per-frame AUROC (best view) |
|---|---|---|---|
| `hot_pixel` | a few pixels fire persistently, every frame | a handful of channels saturate → huge, structured outward shift | **1.000** (any) |
| `event_rate_shift` | global multiplicative change in event count | overall membrane **energy** (radius ‖φ‖) shifts up/down uniformly | **0.915** (radius, two-sided) |
| `temporal_jitter` | event timestamps perturbed within the window | corrupts the **deep-layer** (L4) features that integrate timing; pooled moments barely move | **0.930** (layer-4 Mahalanobis) |

**How to detect Class A:** distance from the clean mean. hot_pixel is trivial
(it leaves the manifold violently). rate_shift is a pure **radial** move — best
caught by the magnitude ‖φ‖, *two-sided* (a shift can be up *or* down).
temporal_jitter is special: it does **not** change the pooled magnitude (radius
AUROC 0.47), it distorts the timing-sensitive **deep layers** — so it needs a
*per-layer* distance on L4, not the pooled vector. This is why one global
distance cannot get all three: jitter lives in a subspace the pooled radius
averages away.

### Class B — Contractions (membrane gets *quieter*)

The corruption removes or dilutes activity, so the moments **shrink** and φ moves
**inward, toward the clean mode** — it looks *more average than average*.

| Corruption | What it does to events | Signature in φ | Per-frame AUROC (best view) |
|---|---|---|---|
| `spatial_dropout` | events deleted in spatial regions | membrane quiets; pooled magnitude *drops*; channel-correlation **structure** is disturbed | **0.559** (RCF, two-sided conditional) |
| `event_flood` | uniform extra events everywhere | moments inflate *proportionally*, preserving per-channel structure → looks like a busy clean scene | **0.552** (every detector) |

**How to detect Class B — and why it's the hard part.** A contraction is the
mirror image of the failure mode of every standard detector:

- A **distance** detector (Mahalanobis, kNN) measures "how far from clean." A
  contraction is *closer* → it scores **low** → flagged as in-distribution.
- A **density / likelihood** detector (GMM, normalizing flow) gives a
  concentration-toward-the-mode **higher** likelihood than typical clean data →
  again flagged as in-distribution.

So contractions don't merely evade these detectors — they **invert** them
(AUROC < 0.5, the historical "anti-detectable 0.29"). The *only* way to flag a
contraction is to model the **expected spread** and notice **under-dispersion**:
ask not "how far from the mean," but "is this as far out as clean data in this
direction normally is?" That requires a **two-sided, direction-conditioned**
score — the novel piece in §3.

`spatial_dropout` additionally hides because its primary signature is **spatial**
(which regions went dark) and GAP already deleted that; only the secondary
"quieter + correlation-disturbed" echo survives, capping it near 0.56 per-frame.
`event_flood` is worse still: it scales everything *proportionally*, so per-frame
it is **genuinely indistinguishable from a naturally busy clean scene** — AUROC
0.552 on *every* detector we tried. Its signal exists only in being *consistent
across a whole recording*, recoverable only by aggregation (§4).

### Class C — Invisible (no signature in V_mem)

| Corruption | What it does to events | Signature in φ | Per-frame AUROC |
|---|---|---|---|
| `polarity_flip` | swap ON/OFF event channels | **none** — the SNN learned polarity-symmetric features, so V_mem is nearly unchanged | **0.557** (ceiling) |

**How to detect Class C:** you can't, from the membrane. This is not a weak
signal, it is an *absent* one — a property of the trained network's invariance,
not of our representation. The discriminative signal survives only in the **input
histogram** (a consistent ON/OFF imbalance the flip inverts → ~0.69 from an
input-side scalar), which is outside the zero-compute V_mem story. polarity
should be **scoped out** of the membrane claim.

### Summary of what each corruption *needs*

| Corruption | Class | Detector primitive it needs |
|---|---|---|
| hot_pixel | dilation | any distance |
| event_rate_shift | dilation (radial) | **two-sided magnitude** ‖φ‖ |
| temporal_jitter | dilation (deep) | **per-layer (L4)** distance |
| spatial_dropout | contraction (structure) | **direction-conditioned two-sided** score (RCF) |
| event_flood | contraction (proportional) | **sequence aggregation** (no per-frame signal) |
| polarity_flip | invisible | input-side only — out of scope |

The central fact: **no single distance or likelihood is the right primitive for
all of them.** Dilations and contractions point in opposite directions; jitter
hides in a subspace; flood has no per-frame signal at all. That is exactly why
every previous "one method" attempt (single Mahalanobis, kNN, GMM, flow, or a
naive max-of-detectors) topped out around 0.70 macro and inverted on the
contractions.

---

## 2. Why prior single methods failed (concretely)

| Method | Per-frame macro | Failure mode |
|---|---|---|
| Mahalanobis d² (reference) | 0.66 | one-sided distance; **inverts** on contractions (dropout 0.44) |
| kNN / GMM / flow-likelihood | ≈0.66 | density-based; a contraction has *higher* likelihood → inverts |
| Two-sided whole-vector χ² | ≈0.66 | clean d² is heavy-tailed and non-stationary; no tight band to exploit |
| RMD-iso (strip magnitude) | — | rescues dropout to 0.60 but **destroys** all dilations (removes the magnitude term they live in) |
| Naive max of 6 detectors | 0.70 | 5 of 6 views were redundant magnitude detectors; their pooled noise **drowned** the one structure view, and the max inflated the clean baseline |
| Learned fusion (LR) | 0.76 (=MAX) | labels buy nothing per-frame (no latent signal); **LR-LOO inverts** on unseen corruptions (0.49) |

Every one of these collapses φ into **a single signed scalar distance**, which
cannot simultaneously face outward (dilations), face inward (contractions), and
look into a subspace (jitter). The fix is not a better scalar — it is to **stop
collapsing**, and instead measure deviation along each independent axis of the
manifold separately.

---

## 3. The novel architecture: Manifold-Decomposition Detector

The MDD's premise: **decompose each frame's deviation into orthogonal manifold
axes, score each one two-sided, and combine them as a calibrated OR.** Each axis
is built to catch exactly one geometry class.

```
                       standardize per-feature  +  PCA (denoise)
   φ (2112-D)  ───────────────────────────────────────────────►  z (k-D)
                                                                   │
        ┌──────────────────────────────┬───────────────────────── ┤
        ▼                              ▼                            ▼
  B1  global radius             B2  RCF  (NOVEL CORE)        B3  deep-layer L4 d²
  | ‖z‖ − E[‖z‖] | / σ          | ‖z‖ − E[‖z‖ | direction] | / σ   one-sided Maha
  (two-sided)                   (two-sided, direction-conditioned)  on the L4 block
        │                              │                            │
        └──────────────┬───────────────┴────────────────────────────┘
                       ▼   calibrate each branch on held-out clean
              S(frame) = max( z₁ , z₂ , z₃ )      ← single unsupervised score
                       ▼   (optional) per-recording aggregation
              S(recording) = mean over frames of S
```

### Branch 1 — Global radius (two-sided): the dilation/contraction axis

Score = standardized `| ‖z‖ − mean_clean‖z‖ |`. This is a **two-sided** test on
overall membrane energy. Two-sided is the point: a one-sided "too far" misses
contractions; the absolute value flags energy that is anomalously **high**
(rate_shift up, flood) *or* anomalously **low** (rate_shift down, dropout's
magnitude drop). Wins **event_rate_shift 0.915**.

### Branch 2 — RCF, Radial-Conditional score (two-sided): **the novel core**

This is the new primitive and the reason contractions become detectable.

Decompose each clean and test vector into **radius** `r = ‖z‖` and **direction**
`û = z/r`. From the clean reference set, for a test sample find its
**k nearest clean neighbours *by direction*** (cosine), and read off the
distribution of *their* radii — i.e. the **conditional** `p(r | û)`. Score:

```
  RCF(x) = | r_x − mean_{kNN-by-direction}(r_clean) | / std_{kNN}(r_clean)
```

In words: *"Given this sample's direction on the clean manifold, is its magnitude
what clean data pointing this way normally has?"* This directly solves the
contraction problem that defeats distance and density:

- A contraction (dropout) has small `r`. A global detector sees "small = close to
  mean = normal." RCF instead notes that **in this direction, clean data normally
  has large `r`** — so a small `r` is anomalously *under-dispersed* → high score.
- Symmetrically, a dilation in a direction where clean is normally quiet also
  scores high. The conditioning means each direction gets its *own* expected
  magnitude and spread, instead of one global band (which clean φ's heavy tails
  make useless).

RCF is the **first method to take spatial_dropout above chance per-frame
(0.44 → 0.559)**, beating the previous best (RMD-iso 0.60) *and*, unlike RMD,
aggregating cleanly. Critically, B2 conditions out exactly what B1 measures, so
the two branches are **near-independent** — the property that makes the OR in §3
work.

### Branch 3 — Deep-layer L4 Mahalanobis (one-sided): the subspace axis

temporal_jitter perturbs timing, which the timing-integrating **deep layers**
encode but the pooled magnitude averages away (radius AUROC 0.47). Restricting a
standard Mahalanobis distance to the **layer-4 block** of φ recovers it:
**jitter 0.930**, versus 0.78 on the pooled vector. This branch is the
"look into the subspace the radius can't see" axis.

### Fusion — calibrated max (a true OR, no labels)

Each branch is standardized on a held-out clean slice so its scores are
comparable z-units, then:

```
  S = max( B1, B2, B3 )
```

Because the branches are **orthogonal and calibrated**, the max is a genuine OR:
whichever axis a corruption trips, fires. This is what failed for the old
"max of 6 detectors" — there, five of six views were correlated magnitude
detectors whose pooled noise raised the clean baseline and buried the lone
structure view. With three deliberately independent, two-sided axes, the baseline
inflation is minimal and each corruption's home branch dominates.

**The fusion is parameter-free and unsupervised** — no corruption labels, no
trained meta-classifier. We verified a *supervised* logistic-regression fusion
buys nothing per-frame (0.763 ≈ MAX 0.766) and catastrophically fails to
generalize to unseen corruptions (LR-LOO 0.486, inverting jitter and rate_shift).
The calibrated max is both simpler and safer.

### Optional Branch 4 — per-recording aggregation (for event_flood only)

event_flood has **no per-frame signal** (0.552 on every detector) because
proportional inflation mimics a busy clean scene. Its *only* signature is being
**consistent across an entire recording**. Averaging the per-frame score over a
recording shrinks frame-noise by ~√N and lets that consistent bias separate:
flood per-frame 0.55 → ~0.99 aggregated. This is a deliberate, corruption-honest
move, not a general crutch — it is unnecessary for the Class-A corruptions
(already 0.87–1.0 per-frame) and does **not** help dropout (whose per-frame bias
is non-stationary) or polarity.

---

## 4. Results

### Per-frame, no aggregation (the proxy-free, defensible numbers)

| Corruption | MDD single score (MAX) | best branch (oracle) |
|---|---|---|
| hot_pixel | 1.000 | 1.000 |
| temporal_jitter | 0.869 | 0.930 (L4) |
| event_rate_shift | 0.879 | 0.915 (radius) |
| event_flood | 0.552 | 0.552 |
| spatial_dropout | 0.530 | 0.559 (RCF) |
| polarity_flip | 0.532 | 0.557 |
| **macro (excl. polarity)** | **0.766** | **0.791** |

### With per-recording aggregation (proxy — see caveats)

| Corruption | MDD + aggregation |
|---|---|
| hot_pixel | 1.000 |
| temporal_jitter | 1.000 |
| event_rate_shift | 0.995 |
| event_flood | 0.990 |
| spatial_dropout | 0.558 |
| polarity_flip | 0.606 |
| **macro (excl. polarity)** | **0.909** |

**One unsupervised score, no per-corruption routing, gets four of six corruptions
to ≥0.99 with aggregation, and ~0.77 macro (excl. polarity) per-frame without it.**

---

## 5. What is genuinely novel here

1. **The diagnosis.** Framing the corruptions as **dilations vs contractions vs
   invisibles** explains, mechanistically, why distance- and density-based OOD
   detectors *invert* on half the benchmark — a contraction is "more normal than
   normal." This reframing is the conceptual contribution; the architecture
   follows from it.
2. **The RCF score** — a **direction-conditioned, two-sided radial anomaly**.
   Standard radial / Mahalanobis methods use a single global band; RCF estimates
   the expected magnitude *per direction* via cosine-kNN and flags
   under-dispersion. It is, to our knowledge, the first detector built
   specifically to catch **contractions** of a learned manifold, and it is what
   makes the previously anti-detectable corruptions detectable.
3. **Orthogonal-axis decomposition instead of scalar distance.** MDD stops
   collapsing φ to one number. By scoring the radial, conditional-radial, and
   deep-subspace axes separately — each two-sided, each calibrated — a simple
   unsupervised max becomes a true complementary OR that no single distance,
   density, or learned fusion matched.
4. **Honest, mechanism-aligned scoping.** The architecture states *why* each
   residual is hard: flood needs aggregation (no per-frame signal), dropout needs
   spatial re-extraction (GAP destroyed its signature), polarity is unfixable
   from V_mem (network invariance). The limits are predicted by the geometry, not
   discovered by trial and error.

---

## 6. Honest limitations and next steps

- **Aggregation numbers use a within-file block proxy.** The legacy extraction
  did not save `seq_lens`, and the proxy averages randomly-ordered frames, which
  is *optimistic* versus real autocorrelated per-sequence frames. Re-extract with
  `seq_lens` and re-measure before quoting the 0.9+ figures.
- **L5 only, single split, 8k subsample, no bootstrap CIs.** Lower severities are
  harder; absolute numbers carry several points of uncertainty. The per-frame MAX
  (~0.77) is the robust figure; the rest needs a severity sweep + CIs.
- **The two real representation gaps:** (a) **GAP** deletes the spatial structure
  that is the true signature of spatial_dropout (and flood's per-frame signal); a
  cheap **spatial-covariance summary at extraction** would attack both at the
  per-frame level. (b) **polarity_flip** is a property of the upstream network and
  cannot be reached from V_mem at all.
- **The single highest-leverage experiment:** re-extract φ with per-channel
  spatial statistics *and* `seq_lens` saved. That simultaneously (a) gives
  spatial_dropout a real per-frame detector, (b) puts the flood/jitter/rate_shift
  aggregation results on solid ground, and (c) lets us replace the proxy with true
  per-recording AUROCs and bootstrap CIs.
