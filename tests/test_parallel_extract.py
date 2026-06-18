"""run_parallel_extract.plan_extraction — worker/corruption scheduling policy."""
import pytest

from run_parallel_extract import plan_extraction


def test_gpu_workers_and_vram_default():
    workers, chunks, vram = plan_extraction([0, 1], 2, ["a", "b", "c", "d"])
    assert len(workers) == 4
    assert all(w["device"] == "cuda" for w in workers)
    assert [w["gpu_id"] for w in workers] == [0, 0, 1, 1]
    assert vram == pytest.approx(0.475)          # 0.95 / 2


def test_cpu_fallback_no_vram():
    workers, chunks, vram = plan_extraction([], 3, ["a", "b"])
    assert len(workers) == 3
    assert all(w["device"] == "cpu" and w["gpu_id"] is None for w in workers)
    assert vram is None


def test_round_robin_partition_is_even_and_complete():
    corr = ["a", "b", "c", "d", "e", "f"]
    _, chunks, _ = plan_extraction([0, 1], 2, corr)   # 4 workers
    # every corruption assigned exactly once, sizes differ by <= 1
    flat = [c for ch in chunks for c in ch]
    assert sorted(flat) == sorted(corr)
    assert max(len(c) for c in chunks) - min(len(c) for c in chunks) <= 1


def test_more_workers_than_corruptions_leaves_empty_chunks():
    workers, chunks, _ = plan_extraction([0], 4, ["x", "y"])
    assert len(chunks) == 4
    assert sum(len(c) for c in chunks) == 2
    assert chunks.count([]) == 2                  # two idle workers


def test_vram_fraction_override_passthrough():
    _, _, vram = plan_extraction([0], 2, ["a"], vram_fraction=0.3)
    assert vram == 0.3
