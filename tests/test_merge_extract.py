"""extract.py persistence layer: _merge_seq_artifact.

Covers multi-key (phi + phi_spatial) row-alignment, the row-count guard that
prevents silent misalignment when resuming pre-spatial files, sequence ordering,
seq_lens/done_seqs metadata, idempotent resume, and legacy bare-tensor files.
"""
import numpy as np
import pytest
import torch

from vmem_benchmark.extract import _merge_seq_artifact


def _save_seq(tmp_dir, idx, **arrays):
    tmp_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: torch.as_tensor(v) for k, v in arrays.items()},
               tmp_dir / f"seq_{idx:05d}.pt")


def test_merge_multikey_aligned_and_ordered(tmp_path):
    tmp = tmp_path / "_tmp_clean"
    # write out of order (2 before 0) to verify sequence-index ordering
    _save_seq(tmp, 2, phi=torch.randn(4, 2), phi_spatial=torch.randn(4, 5))
    _save_seq(tmp, 0, phi=torch.randn(3, 2), phi_spatial=torch.randn(3, 5))
    final = tmp_path / "clean.pt"

    assert _merge_seq_artifact(tmp, final, ["phi", "phi_spatial"], "clean", log=lambda *a: None)
    d = torch.load(final, weights_only=True)
    assert d["phi"].shape[0] == 7 and d["phi_spatial"].shape[0] == 7
    assert d["done_seqs"] == [0, 2]
    assert d["seq_lens"] == [3, 4]                  # ordered by seq index
    assert not list(tmp.glob("seq_*.pt"))           # tmp cleaned up


def test_merge_rowcount_guard_raises_on_missing_key(tmp_path):
    tmp = tmp_path / "_tmp_clean"
    _save_seq(tmp, 0, phi=torch.randn(3, 2), phi_spatial=torch.randn(3, 5))
    _save_seq(tmp, 1, phi=torch.randn(2, 2))        # phi_spatial missing!
    final = tmp_path / "clean.pt"
    with pytest.raises(RuntimeError, match="mismatched row counts"):
        _merge_seq_artifact(tmp, final, ["phi", "phi_spatial"], "clean", log=lambda *a: None)


def test_merge_idempotent_resume(tmp_path):
    tmp = tmp_path / "_tmp_clean"
    _save_seq(tmp, 0, phi=torch.arange(6.).reshape(3, 2))
    _save_seq(tmp, 1, phi=torch.arange(8.).reshape(4, 2))
    final = tmp_path / "clean.pt"
    _merge_seq_artifact(tmp, final, ["phi"], "clean", log=lambda *a: None)
    first = torch.load(final, weights_only=True)

    # Re-run with empty tmp: must reconstruct identical output from the final file.
    _merge_seq_artifact(tmp, final, ["phi"], "clean", log=lambda *a: None)
    second = torch.load(final, weights_only=True)
    assert torch.equal(first["phi"], second["phi"])
    assert first["seq_lens"] == second["seq_lens"] == [3, 4]


def test_merge_bare_tensor_legacy_files(tmp_path):
    tmp = tmp_path / "_tmp_clean"
    tmp.mkdir(parents=True)
    torch.save(torch.randn(3, 2), tmp / "seq_00000.pt")   # bare tensor, no dict
    final = tmp_path / "clean.pt"
    assert _merge_seq_artifact(tmp, final, ["phi"], "clean", log=lambda *a: None)
    d = torch.load(final, weights_only=True)
    assert d["phi"].shape[0] == 3 and d["seq_lens"] == [3]


def test_merge_empty_returns_false(tmp_path):
    tmp = tmp_path / "_tmp_clean"
    tmp.mkdir(parents=True)
    final = tmp_path / "clean.pt"
    assert _merge_seq_artifact(tmp, final, ["phi"], "clean", log=lambda *a: None) is False
    assert not final.exists()
