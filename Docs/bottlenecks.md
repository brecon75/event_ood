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

## Remaining bottlenecks & recommendations

1. **`fusion_features.py` has its OWN eager `load_all_features`** (loads φ +
   margin-hist + temporal latents for *all* runs) — the one remaining ~95 GB+
   host-RAM OOM site. **Recommendation:** give it the same `LazyFeatures`
   treatment (it writes per-run fused features, so it can stream run-by-run).

2. **IO amplification.** φ is re-read from disk by every analysis stage, and
   `representation_ablation` re-reads each run **9×** (once per representation)
   because the loop is representation-outer. The lazy loader trades RAM for this
   IO. **Recommendations, in order of payoff:** (a) restructure
   `representation_ablation` to run-outer (fit all 9 rep scorers on `clean`
   first, then one pass over runs) → 9× fewer reads, still OOM-safe; (b) store φ
   as memory-mappable arrays; (c) run the φ-only stages (7/9/10/11) in a single
   process sharing one warm loader.

3. **OCSVM fitting is disabled** (`fit_detectors.py`) but a stale `ocsvm.joblib`
   is still scored by stage 7. Either re-enable a capped fit (now cheap to score
   on GPU) or drop the stale file so the benchmark column is honest.

4. **Host RAM for stage 6** at full scale: X_train (~2 GB) + LedoitWolf
   covariance/precision (2252² × 8 B ≈ 40 MB each) + sklearn copies. Provision a
   few GB headroom; it is not GPU-bound.
