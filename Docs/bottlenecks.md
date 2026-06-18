# Pipeline bottleneck diagnosis & GPU/OOM hardening

*Scope: the full-scale cluster run (~343k frames/run × 31 runs, φ = 2252-D).
This catalogs what each stage is bound by (CPU / GPU / host-RAM / disk-IO),
what was fixed, and what remains.*

## TL;DR — what was slow and why

| Stage | Was bound by | Now | Risk removed |
|---|---|---|---|
| 1 · extract | **GPU** (SNN forward, B=1) | unchanged (inherent) | — |
| 6 · fit_detectors | CPU sklearn + **95 GB eager load** | lazy load (clean only); fit_ae streams; manifest | host-RAM OOM |
| 7 · evaluate_detectors | **CPU single-thread** (ocsvm/gmm/maha/pca) + **95 GB eager load** | **all scorers on GPU**, chunked; lazy load | multi-day CPU + RAM OOM |
| 8 · evaluate_mdd | GPU (already chunked) | + progress bar | — |
| 9 · representation_ablation | **CPU einsum** ×9 reps ×31 runs | **GPU Mahalanobis**; lazy load | CPU + RAM OOM |
| 10 · severity / 11 · reliability | CPU einsum (`get_mahalanobis_scores`) | **GPU**; lazy load | CPU + RAM OOM |
| 12 · cross_corruption | 1× LogisticRegression (CPU) | lazy load | RAM OOM |
| 3 · fusion_features | **own eager load (OOM)** | **still eager — see remaining** | ⚠️ open |

The headline: stage 7 ran the OCSVM RBF `decision_function` (419 SVs × 2252-D) in
**single-threaded libsvm with no BLAS/GPU** at 343k×31 frames — that one call
dominated the multi-day estimate. And six stages loaded **every run's φ into RAM
at once (~95 GB)** before scoring.

## The OOM mechanisms (how it now adapts to *any* GPU)

All GPU loops route through one VRAM-aware path so chunk sizes scale from a
laptop card to an H100 with no hardcoded constants:

- **`query_chunk_rows`** sizes the first chunk from *free* VRAM (a fraction, to
  leave room for op temporaries).
- **`chunked_apply`** halves the chunk on any CUDA OOM the estimate missed, down
  to `min_chunk`, then surfaces the error (never hangs, never corrupts).
- **`knn_score`** streams the clean reference from CPU in blocks — the full
  ~2 GB reference is *never* resident on the GPU at once; query rows are chunked
  too. (Was: whole reference uploaded before chunking → OOM on a small GPU.)
- **`fit_ae`** keeps the training set on CPU and moves only the 256-row batch to
  the device. (Was: whole ~2 GB X uploaded resident for all epochs.)
- **`LazyFeatures`** loads each run's φ on demand and keeps only `clean` (pinned)
  + a tiny LRU resident, so a single-pass stage holds ~1–2 runs, not 31.

Stress tests for all of the above live in `tests/test_gpu_scaling.py`
(OOM-halving, below-`min_chunk` surfacing, non-OOM errors propagating, empty
input, streaming-reference equivalence, lazy-loader memory bound).

## Per-stage detail

**Stage 1 — extract (GPU-bound, inherent).** The SNN forward pass at `B=1`
(SpikingJelly batch=time constraint) is the floor on the whole pipeline. Not
CPU/RAM bound. Already parallelised across GPUs/workers by
`run_parallel_extract.py`. Writes ~155 GB of φ+φ_spatial — disk throughput
matters here.

**Stage 6 — fit_detectors (CPU sklearn, now RAM-safe).** LedoitWolf (O(d³)
inversion), PCA, GMM-EM and the NearestNeighbors container fit in host RAM/CPU —
`chunked_apply` can't tile sklearn fits, so this is a resourcing fact, not a bug.
GMM (20k) and PCA-SVD caps are kept on purpose (a 5-Gaussian / 64-direction
estimate barely moves with more data). Now loads only `clean` (lazy), streams
the AE, and writes `fit_manifest.json` recording **which detector was fit on how
much data** (n_fit_samples, fraction_of_train, capped flag, fit_seconds,
convergence).

**Stage 7 — evaluate_detectors (was the multi-day killer).** All seven scorers
now run chunked on the GPU; validated against sklearn to ~1e-6 relative error.
Iterates run *names* (not `list(items())`, which would re-materialise all 31
runs) so the lazy loader's bound holds.

**Stage 9/10/11 — representation_ablation / severity / reliability.** The shared
`fit_mahalanobis` closure now scores on the GPU; `get_mahalanobis_scores` rides
the same path, so all three stages move off the single-core einsum.

## Resolved (follow-up round)

1. **`fusion_features.py` eager all-runs load — FIXED.** Refactored to load
   `clean` once (for the fusion-weight fit) then stream one run at a time via
   `_load_run`, freeing each after its fused file is written. Peak RAM ≈ clean +
   1 run instead of all 31. Added a `Fusing runs` progress bar.

2. **`representation_ablation` 9× φ re-read — FIXED.** Restructured to run-outer:
   fit all 9 representation scorers on `clean` first, then a single pass over the
   runs scoring every representation per run. Each run's φ is now read **once**
   (was 9×), still within the lazy loader's RAM bound. Results unchanged.

3. **Stale `ocsvm.joblib` — FIXED.** Re-enabled a capped RBF OCSVM fit
   (`OCSVM_FIT_SAMPLES = 20000`; fit is ~O(n²), scoring is GPU-chunked) so the
   scored model is consistent with the current data and recorded in
   `fit_manifest.json` (n_support_vectors, n_fit_samples, capped flag).

## Remaining (lower priority)

- **IO across stages.** φ is still re-read from disk by each φ-consuming stage
  (7/9/10/11). Within a stage it is now read once; across stages it is re-read.
  Options if this dominates: store φ as memory-mappable arrays, or run the
  φ-only stages in one process sharing a warm loader.

- **Host RAM for stage 6** at full scale: X_train (~2 GB) + LedoitWolf
  covariance/precision (2252² × 8 B ≈ 40 MB each) + sklearn copies. Provision a
  few GB headroom; it is not GPU-bound.
