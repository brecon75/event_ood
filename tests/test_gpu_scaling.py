"""OOM-safety / GPU-scaling stress tests.

These guard the mechanisms that let the analysis stages run on ANY GPU (or CPU)
without OOM: the VRAM-aware `chunked_apply` (auto-halving on OOM), the streaming
`knn_score` (reference never resident whole), and the lazy, memory-bounded
feature loader. They run on CPU but exercise the exact control flow that protects
a small/loaded GPU on the cluster.
"""
import numpy as np
import torch
import pytest
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance

from analysis.vmem_utils import chunked_apply, knn_score


# ───────────────────────── chunked_apply: OOM auto-halving ─────────────────────────

def test_chunked_apply_halves_until_it_fits():
    """A fn that 'OOMs' on chunks > 64 rows must drive the chunk size down and
    still produce the correct full-length result."""
    succeeded = []

    def fn(chunk):
        n = chunk.shape[0]
        if n > 64:
            raise RuntimeError("CUDA out of memory")   # simulated OOM
        succeeded.append(n)
        return chunk.sum(dim=1)

    X = np.random.default_rng(0).random((500, 4)).astype(np.float32)
    out = chunked_apply(fn, X, "cpu", init_chunk=512, min_chunk=8)

    assert out.shape == (500,)
    assert np.allclose(out, X.sum(axis=1), atol=1e-5)
    assert succeeded and max(succeeded) <= 64        # every successful call fit the budget


def test_chunked_apply_raises_when_below_min_chunk():
    """If even min_chunk can't fit, the OOM must surface (not hang or corrupt)."""
    def fn(chunk):
        raise RuntimeError("CUDA out of memory")

    X = np.zeros((10, 3), dtype=np.float32)
    with pytest.raises(RuntimeError, match="out of memory"):
        chunked_apply(fn, X, "cpu", init_chunk=8, min_chunk=8)


def test_chunked_apply_non_oom_error_propagates():
    """A genuine bug (not OOM) must not be silently swallowed by the halving loop."""
    def fn(chunk):
        raise RuntimeError("size mismatch")

    with pytest.raises(RuntimeError, match="size mismatch"):
        chunked_apply(fn, np.zeros((4, 2), dtype=np.float32), "cpu")


def test_chunked_apply_empty_input_preserves_trailing_dims():
    """Empty input must keep a 2-D shape so consumers doing out[:, 1] don't crash."""
    def fn(chunk):
        return torch.stack([chunk.sum(1), chunk.mean(1)], dim=1)   # (m, 2)

    out = chunked_apply(fn, np.zeros((0, 4), dtype=np.float32), "cpu")
    assert out.shape == (0, 2)


# ───────────────────────── knn_score: streaming reference ─────────────────────────

@pytest.mark.parametrize("ref_block", [16, 64, 100000])
def test_knn_score_streaming_matches_bruteforce(ref_block):
    """Streaming the reference in small blocks must give the same mean-k-distance
    as a single brute-force pass, for block sizes both smaller and larger than n_ref."""
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(300, 8)).astype(np.float32)
    X = rng.normal(size=(40, 8)).astype(np.float32)

    expected = NearestNeighbors(n_neighbors=5).fit(ref).kneighbors(X)[0].mean(axis=1)
    got = knn_score(ref, 5, X, "cpu", ref_block=ref_block)
    assert np.allclose(got, expected, atol=1e-4)


def test_knn_score_k_larger_than_ref_is_clamped():
    ref = np.random.default_rng(2).normal(size=(3, 5)).astype(np.float32)
    X = np.zeros((4, 5), dtype=np.float32)
    out = knn_score(ref, k=50, X=X, device="cpu", ref_block=2)   # k >> n_ref
    assert out.shape == (4,) and np.isfinite(out).all()


def test_knn_score_empty_query():
    ref = np.random.default_rng(3).normal(size=(20, 4)).astype(np.float32)
    out = knn_score(ref, 5, np.zeros((0, 4), dtype=np.float32), "cpu")
    assert out.shape == (0,)


@pytest.mark.parametrize("ref_block", [8, 50000])
def test_knn_score_kth_reduction_matches_bruteforce(ref_block):
    """reduce='kth' (the deep-kNN / ANN-baseline convention) must equal the
    distance to the k-th nearest neighbour, streamed or not."""
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(250, 6)).astype(np.float32)
    X = rng.normal(size=(30, 6)).astype(np.float32)
    expected = NearestNeighbors(n_neighbors=10).fit(ref).kneighbors(X)[0][:, -1]
    got = knn_score(ref, 10, X, "cpu", ref_block=ref_block, reduce="kth")
    assert np.allclose(got, expected, atol=1e-4)


def test_knn_score_rejects_bad_reduce():
    with pytest.raises(ValueError, match="reduce"):
        knn_score(np.zeros((5, 3), np.float32), 2, np.zeros((2, 3), np.float32),
                  "cpu", reduce="median")


# ───────────────────────── lazy, memory-bounded feature loader ─────────────────────────

def test_lazy_features_bounded_and_correct():
    from vmem_benchmark import benchmark_config as cfg
    from analysis.representation_ablation import load_all_features

    if not (cfg.PHI_DIR / "clean.pt").exists():
        pytest.skip("no extracted phi available")

    feats = load_all_features(cache_size=1, verbose=False)
    assert "clean" in feats and "definitely_missing_run" not in feats

    # content matches a direct load
    direct = torch.load(cfg.PHI_DIR / "clean.pt", weights_only=True)["phi"].numpy()
    assert np.array_equal(feats["clean"]["phi"], direct)

    # touching many runs must NOT accumulate them all in RAM (the whole point)
    other = [r for r in feats if r != "clean"]
    for r in other:
        _ = feats[r]["phi"]
    assert len(feats._cache) <= 1                  # LRU bound honoured
    assert feats["clean"] is feats["clean"]        # clean pinned (single resident copy)


# ───────────────────────── GPU Mahalanobis equivalence ─────────────────────────

def test_repr_mahalanobis_gpu_matches_einsum():
    from analysis.representation_ablation import fit_mahalanobis

    rng = np.random.default_rng(0)
    train = rng.normal(size=(400, 10)).astype(np.float32)
    test = rng.normal(size=(50, 10)).astype(np.float32)

    got = fit_mahalanobis(train)(test)
    cov = EmpiricalCovariance().fit(train)
    diff = test - cov.location_
    ref = np.einsum("ni,ij,nj->n", diff, cov.precision_, diff)
    assert np.allclose(got, ref, rtol=1e-3, atol=1e-3)
