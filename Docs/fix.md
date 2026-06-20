# fix.md — Outstanding issues from the analysis-refactor review

Findings from the high-effort code review of the uncommitted `refactor-analysis`
working tree (faithful ANN OOD detectors, cap removals, VRAM-aware GPU chunking,
free-rider severity loop, GMM/SVD caps). Ranked most-severe first. Line numbers
are approximate (working tree at review time).

Legend: **P0** = crash / breaks a stage · **P1** = OOM / scaling regression ·
**P2** = correctness/test robustness · **P3** = cleanup / altitude.

---

## P0 — Definite crash

### 1. Leftover legacy CSV block crashes free-rider (NameError)
- **File:** `analysis/free_rider_ablation.py:395-405`
- **Problem:** The severity-loop refactor renamed `max_sev` → `severities` and
  `corruptions` → `runs`/`run_names`, and added a new CSV writer (lines 378-391).
  But a *second, legacy* CSV-writer block survived after `plot_free_rider_ablation`
  and still references `max_sev` and `corruptions`.
- **Failure scenario:** `main()` runs the full ablation, prints results, writes
  `results/free_rider_ablation.csv`, plots, then hits
  `fieldnames = ["Condition"] + [f"{c}_AUROC_L{max_sev}" for c in corruptions]`
  → **NameError**, crashing Stage 13 *after* all compute is done. Even if the
  names were restored, `results[cond]` is now keyed by run_name (`'hot_pixel_L5'`),
  so `results[cond].get(c)` would be all-NaN. It also double-writes the same
  filename as the new block.
- **Fix:** Delete the legacy block (lines 395-405) entirely; the new block at
  378-391 already persists the CSV.

---

## P1 — OOM / scaling regressions

> **Guiding principle (user decision, 2026-06-17): TRAIN ON THE FULL DATASET.**
> Do NOT cap the data a model learns from. The fixes below are about the
> *allocation mechanism* (stream big uploads, size to VRAM), NOT about reducing
> data. Distinguish three cases:
> 1. **Learns from data → use FULL data** (kNN ref, Mahalanobis, RCF ref,
>    Ledoit-Wolf, AE, flow). Fix the *allocation*, never the data amount.
> 2. **`fit_ae` / kNN-ref OOM is an allocation bug, not a data bug** — they upload
>    ALL data to the GPU in one tensor. Fix = STREAM in batches (still 100% data).
> 3. **GMM & PCA-SVD only estimate a fixed-size thing** (5 Gaussians / 64 PCA
>    dirs), so full data barely changes them. The 20k caps there are pure
>    compute/memory bounds with ~zero faithfulness cost — KEEP unless the user
>    wants full for purity (GMM full ≈ 6 h CPU; SVD full needs a big GPU).

### 2. MDD RCF reference and `_ledoit_wolf` should fit on FULL data (just dead/misleading params)
- **File:** `analysis/mdd.py:103` (RCF ref) and `analysis/mdd.py:48` (`_ledoit_wolf`)
- **Problem:** `_subsample` is now a no-op, so `ref = _subsample(phi_fit, self.n_ref)`
  and `f = _subsample(fit_arr, n_fit)` already use the FULL set. **This is what we
  WANT** (more reference points / more covariance samples = more faithful). The
  only issue is the `n_ref=15000` / `n_fit=5000` params are now dead and
  misleading, and `ref_dir` growing is a SPEED (not OOM) cost — `ref_dir` is
  projected to k_pca≈64 dims (~61 MB on GPU) and `_rcf`'s sim matrix is chunked.
- **Fix:** Do NOT cap. Keep full data. Either remove the now-dead `n_ref`/`n_fit`
  params or document them as "ignored outside --fast". (Earlier draft wrongly
  proposed `_cap_subset` here — that would reduce the data the detector learns
  from, which contradicts the full-data goal.)

### 3. kNN reference uploaded whole-to-GPU *before* chunking — STREAM it, don't cap
- **File:** `analysis/vmem_scorers.py:52` (`fit_t`) and
  `analysis/evaluate_detectors.py` `score_knn` (`ref_t`)
- **Problem:** The full clean reference (~2 GB) is moved to the GPU at
  scorer-construction time, *outside* `chunked_apply`, so the OOM-halving never
  covers it. This is an **allocation** problem, not a data-quantity problem — we
  still want the full reference.
- **Failure scenario:** On a small/loaded GPU `to(device)` of the full ~240k×2112
  reference may OOM before any chunk is scored. (On the cluster's big GPU, 2 GB is
  fine — this mainly bites small GPUs.)
- **Fix:** Keep the full reference; tile the allocation — keep the reference on
  CPU and stream reference blocks into the cdist (double-chunk over query AND
  reference), or fall back to CPU if the whole reference doesn't fit. Do NOT cap
  the reference size.

### 4. `fit_ae` uploads the full X to the GPU at once — STREAM it, don't cap
- **File:** `analysis/fit_detectors.py:49`
- **Problem:** `X_tensor = torch.tensor(X).to(device)` puts the entire ~240k×2112
  (~2 GB) on VRAM resident for all epochs. Again an allocation bug — we want to
  train on ALL of X.
- **Failure scenario:** OOMs a small GPU during Stage 6 fitting. Training only
  needs 256-row batches on device.
- **Fix:** Keep the dataset on CPU and move each batch to the device inside the
  loop (mirror `train_ae_model` in `vmem_models.py`). Still trains on full X.

### 5. Non-tileable HOST-side fits on full data — size the cluster job, don't cap
- **File:** `analysis/fit_detectors.py:110,116,124` (LedoitWolf / PCA /
  NearestNeighbors on full `X_train`)
- **Problem:** These sklearn fits run on the full set in **host RAM / CPU**;
  `chunked_apply` can't help (it only tiles GPU loops). This is EXPECTED with
  full-data training — it's a resourcing fact, not a bug to cap away.
- **Failure scenario:** On a small single workstation, host-RAM blowup + slow
  brute-force kNN. On the cluster, provision enough host RAM + CPU.
- **Fix:** Keep full data. Document the host-RAM / CPU-time requirement for the
  cluster job (full φ per run ≈ 2.9 GB + sklearn copies). Optionally use a
  faster ANN backend (e.g. faiss) for the brute-force kNN query — that speeds it
  up WITHOUT reducing the reference set.

---

## P2 — Correctness / test robustness

### 6. `chunked_apply` returns 1-D `empty((0,))` for empty input → 2-D consumers crash
- **File:** `analysis/vmem_utils.py:421`
- **Problem:** `res = torch.cat(out, dim=0) if out else torch.empty((0,))` always
  returns shape `(0,)`, even when `fn` produces a 2-D `(N, k)` result.
- **Failure scenario:** `MDD._rcf` does `out = chunked_apply(fn, u, ...)` where
  `fn` returns `(chunk, 2)`, then `out[:, 1]`. If `u` has 0 rows (a sequence fully
  dropped by `spatial_dropout`, an empty eval split, a degenerate `--fast` run),
  `out` is `(0,)` and `out[:, 1]` raises IndexError. Same risk for the
  PCA-Mahalanobis projection path.
- **Fix:** On the empty path, run `fn` on a zero-row probe slice to learn the
  trailing dims, or return `Xt[:0]`-shaped output; simplest is
  `return np.empty((0,) + r.shape[1:])` using the first chunk's rank — or special-
  case `N == 0` to return an empty array with the correct number of columns.

### 7. ViM/ReAct/DICE download ImageNet weights at fit time; unit tests no longer offline-safe
- **File:** `analysis/evaluate_ann_baselines.py:32-36` (`_resnet_head` calls
  `_resnet_head_weights()` *before* the `feat_dim` check); affects
  `tests/test_ann_baselines.py::test_all_detectors_finite_and_shaped`
- **Problem:** `_resnet_head_weights()` (downloads resnet18 ImageNet, ~45 MB) runs
  before checking `W.shape[1] == feat_dim`, so even 16-dim synthetic test features
  trigger a download to discover the dim mismatch. The detector tests don't
  monkeypatch `resnet18` (unlike the extraction test).
- **Failure scenario:** With network: one cached download → slow CI. Offline:
  `lru_cache` does not cache the *raised* exception, so each of ViM/ReAct/DICE
  re-attempts and fails the download before falling back → flaky/slow tests with a
  newly-introduced network dependency.
- **Fix:** Monkeypatch `resnet18` in the detector test (as the extraction test
  does), or short-circuit `_resnet_head` for obviously-non-512 dims before
  touching the weights, or cache the "unavailable" sentinel.

### 8. n_ref under-estimates peak memory for GMM and Flow scorers
- **File:** `analysis/vmem_scorers.py` (gmm `n_ref=D` ~line 105; flow
  `n_ref=x_proj.shape[1]=50`)
- **Problem:** `query_chunk_rows` is given `n_ref` that doesn't reflect true peak:
  GMM allocates `chunk×D` per component across `nc` components (stacked logsumexp),
  ~`nc`× the estimate; Flow's RealNVP coupling nets expand to a hidden width ≫ 50.
- **Failure scenario:** First chunk is oversized; on a small GPU each score call
  incurs repeated OOM→empty_cache→halve cycles (slow, noisy logs). If the viable
  chunk is forced below `min_chunk=128` it raises instead of completing.
- **Fix:** Pass an effective `n_ref` that reflects peak (e.g. `D*nc` for GMM,
  coupling-hidden width for Flow), or expose `init_chunk`/`frac` so heavy fns
  start smaller.

### 9. Stale GMM test: reference `max_iter=300` vs scorer's new `1000`
- **File:** `tests/test_scorers_models.py:50` (and stale comment ~line 35-36)
- **Problem:** The test builds its reference `GaussianMixture(max_iter=300)`, but
  `gmm_scorer` raised the ceiling to `max_iter=1000`. The comment still cites the
  old `_subsample`-at-n≤3000-identity rationale.
- **Failure scenario:** If EM on the 400×6 fixture hasn't converged by iter 300,
  scorer (1000) and reference (300) diverge beyond rtol/atol=1e-3 → test fails.
  Passes only incidentally when EM converges early.
- **Fix:** Match `max_iter=1000` in the reference and refresh the comment.

---

## P3 — Cleanup / altitude

### 10. Dead imports and duplicated logic left by the refactor
- **File:** `analysis/vmem_scorers.py:6` (`tqdm`), `:16` (`MAX_FIT_SAMPLES`);
  duplicated Energy-fallback across `DetectorReAct`/`DetectorDICE`; duplicated
  cdist→topk→mean kNN closure between `evaluate_detectors.score_knn` and
  `vmem_scorers.knn_scorer`.
- **Problem:** `tqdm` and `MAX_FIT_SAMPLES` are now unused in `vmem_scorers`. The
  `if self.head is None: return -logsumexp(...)` fallback is copy-pasted; three
  divergent kNN semantics now coexist (mean-of-k in scorers, k-th-NN + L2-norm in
  the ANN baseline `DetectorKNN`).
- **Failure scenario:** Lint noise + misleading dependency signal; the duplicated
  fallback and kNN closures must be kept in sync by hand and can silently drift.
- **Fix:** Remove the dead imports; extract a shared `_energy(logits)` helper and
  a single `knn_score(ref, k, X, device)` helper; keep the (intentionally
  different) ANN-baseline k-th-NN semantics documented as such.

---

## Suggested fix order
1. **#1** — delete the legacy CSV block (crash).
2. **#2** — `_cap_subset` for the two MDD `_subsample` calls (half-applied fix).
3. **#3 + #4** — stop full-data-to-GPU uploads before chunking (`fit_ae`, kNN ref).
4. **#6** — `chunked_apply` empty-input trailing-dim preservation.
5. **#7, #9** — test robustness (offline-safe, max_iter parity).
6. **#5, #8, #10** — host-RAM bounding, n_ref estimates, cleanup.
