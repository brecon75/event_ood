"""Sequence-aware split + per-recording aggregation (leakage-safety core)."""
import numpy as np
import pytest

from analysis.vmem_utils import (
    aggregate_by_seq, seq_lens_after_cut, split_boundary, split_train_eval,
)


def test_aggregate_by_seq_means():
    s = np.array([1.0, 3.0, 10.0, 20.0, 30.0])
    assert np.allclose(aggregate_by_seq(s, [2, 3]), [2.0, 20.0])


def test_aggregate_by_seq_length_mismatch_returns_none():
    s = np.arange(5.0)
    assert aggregate_by_seq(s, [2, 2]) is None
    assert aggregate_by_seq(s, [2, 4]) is None
    assert aggregate_by_seq(s, None) is None
    assert aggregate_by_seq(s, []) is None


def test_aggregate_by_seq_single_sequence():
    assert np.allclose(aggregate_by_seq(np.array([2.0, 4.0, 6.0]), [3]), [4.0])


def test_seq_lens_after_cut_inside_sequence():
    assert seq_lens_after_cut([3, 4, 3], 4) == [3, 3]


def test_seq_lens_after_cut_on_boundary():
    assert seq_lens_after_cut([3, 4, 3], 3) == [4, 3]


def test_seq_lens_after_cut_sums_to_tail_for_all_cuts():
    seq = [10, 7, 13, 5]
    n = sum(seq)
    for cut in range(1, n):
        out = seq_lens_after_cut(seq, cut)
        assert out is not None and sum(out) == n - cut, f"cut={cut}"


def test_seq_lens_after_cut_none_passthrough():
    assert seq_lens_after_cut(None, 5) is None


def test_aggregation_roundtrip_after_cut():
    seq = [4, 6, 5]
    scores = np.arange(float(sum(seq)))
    cut = 5
    agg = aggregate_by_seq(scores[cut:], seq_lens_after_cut(seq, cut))
    assert agg is not None and len(agg) == 2     # cut splits seq[1], leaves 2 tail blocks


def test_split_boundary_plain_ratio():
    assert split_boundary(100, 0.7) == 70
    assert split_boundary(10, 0.5) == 5


def test_split_boundary_degenerate_sizes():
    assert split_boundary(0, 0.7) == 0
    assert split_boundary(1, 0.7) == 1
    assert split_boundary(2, 0.7) == 1


def test_split_boundary_never_empties_a_side():
    assert split_boundary(100, 0.0) == 1
    assert split_boundary(100, 1.0) == 99


def test_split_boundary_single_sequence_uses_ratio():
    assert split_boundary(1192, 0.7, seq_lens=[1192]) == 834


def test_split_boundary_snaps_to_nearest_interior_edge():
    assert split_boundary(100, 0.7, seq_lens=[30, 30, 40]) == 60
    assert split_boundary(100, 0.5, seq_lens=[30, 30, 40]) == 60


def test_split_boundary_ignores_seq_lens_that_dont_sum():
    # seq_lens that don't sum to n must be ignored (fall back to ratio cut).
    assert split_boundary(100, 0.7, seq_lens=[10, 20]) == 70


def test_split_train_eval_disjoint_and_complete():
    arr = np.arange(100).reshape(100, 1)
    tr, ev = split_train_eval(arr, 0.7)
    assert len(tr) + len(ev) == 100
    assert tr[-1, 0] + 1 == ev[0, 0]
