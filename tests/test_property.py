"""Property-based invariants (hypothesis) for metrics and aggregation."""
import numpy as np
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays

from analysis.vmem_utils import auroc_fpr95, aggregate_by_seq, seq_lens_after_cut


@settings(max_examples=60, deadline=None)
@given(
    labels=arrays(np.int32, st.integers(4, 40),
                  elements=st.integers(0, 1)),
    seed=st.integers(0, 1_000_000),
)
def test_auroc_in_unit_interval(labels, seed):
    if len(np.unique(labels)) < 2:
        return
    scores = np.random.default_rng(seed).normal(size=len(labels))
    auroc, fpr = auroc_fpr95(labels, scores)
    assert 0.0 <= auroc <= 1.0
    assert 0.0 <= fpr <= 1.0


@settings(max_examples=60, deadline=None)
@given(n=st.integers(4, 60), seed=st.integers(0, 1_000_000))
def test_auroc_permutation_invariant(n, seed):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    s = rng.normal(size=n)
    base, _ = auroc_fpr95(y, s)
    perm = rng.permutation(n)
    permuted, _ = auroc_fpr95(y[perm], s[perm])   # joint permutation preserves pairing
    assert abs(base - permuted) < 1e-9


@settings(max_examples=60, deadline=None)
@given(data=st.data())
def test_aggregate_by_seq_within_block_bounds(data):
    seq_lens = data.draw(st.lists(st.integers(1, 8), min_size=1, max_size=6))
    n = sum(seq_lens)
    scores = np.array(data.draw(st.lists(
        st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
        min_size=n, max_size=n)))
    out = aggregate_by_seq(scores, seq_lens)
    assert out is not None and len(out) == len(seq_lens)
    start = 0
    for i, L in enumerate(seq_lens):
        block = scores[start:start + L]
        assert block.min() - 1e-6 <= out[i] <= block.max() + 1e-6
        start += L


@settings(max_examples=50, deadline=None)
@given(seq_lens=st.lists(st.integers(1, 10), min_size=1, max_size=6),
       data=st.data())
def test_seq_lens_after_cut_partitions_tail(seq_lens, data):
    n = sum(seq_lens)
    cut = data.draw(st.integers(0, n))
    out = seq_lens_after_cut(seq_lens, cut)
    expected_tail = n - cut
    if expected_tail == 0:
        assert out is None
    else:
        assert out is not None and sum(out) == expected_tail
