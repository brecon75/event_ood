"""Stage-level + end-to-end: reliability detection metric and a full
evaluate_mdd run on synthetic phi files (no model, no real data)."""
import importlib

import numpy as np
import pandas as pd
import pytest
import torch

from analysis.reliability import compute_detection_metric
from analysis.vmem_utils import TOTAL_PHI_DIM


# ───────────────────────── reliability detection metric ─────────────────────────

def test_detection_metric_counts_confident_boxes():
    # (N=2, anchors=3, 7): cols 4=obj_conf, 5:=cls_conf. obj*cls>0.3 => confident.
    det = torch.zeros(2, 3, 7)
    det[0, 0, 4] = 1.0; det[0, 0, 5] = 0.9     # score 0.9 -> confident
    det[0, 1, 4] = 1.0; det[0, 1, 5] = 0.2     # score 0.2 -> not
    det[1, 0, 4] = 0.5; det[1, 0, 6] = 0.8     # score 0.40 -> confident
    counts = compute_detection_metric(det, conf_thresh=0.3)
    assert counts.tolist() == [1, 1]


def test_detection_metric_none_and_low_dim_safe():
    assert compute_detection_metric(None).shape == (1,)
    flat = torch.zeros(4, 7)                    # ndim < 3 -> fallback zeros(N)
    assert compute_detection_metric(flat).shape == (4,)


# ───────────────────────── evaluate_mdd end-to-end ─────────────────────────

def _write_phi(path, arr, seq_lens):
    torch.save({"phi": torch.from_numpy(arr.astype(np.float32)),
                "seq_lens": seq_lens,
                "done_seqs": list(range(len(seq_lens))),
                "run": path.stem}, path)


@pytest.fixture
def synthetic_phi_dir(tmp_path, monkeypatch):
    from vmem_benchmark import benchmark_config as cfg
    phi_dir = tmp_path / "phi"
    phi_dir.mkdir()
    rng = np.random.default_rng(0)
    seq_lens = [80, 80, 80, 80, 80]            # 400 frames, 5 sequences
    n = sum(seq_lens)
    clean = rng.normal(0, 1, (n, TOTAL_PHI_DIM))
    _write_phi(phi_dir / "clean.pt", clean, seq_lens)
    # hot_pixel = a clear dilation (shifted far out) -> fused must separate it
    hot = rng.normal(0, 1, (n, TOTAL_PHI_DIM)) + 6.0
    _write_phi(phi_dir / "hot_pixel_L5.pt", hot, seq_lens)

    monkeypatch.setattr(cfg, "PHI_DIR", phi_dir)
    monkeypatch.setattr(cfg, "OUTPUT_DIR", tmp_path)
    return tmp_path


def test_evaluate_mdd_end_to_end(synthetic_phi_dir):
    import analysis.evaluate_mdd as em
    importlib.reload(em)                        # rebind module globals to patched cfg
    em.main()

    res = synthetic_phi_dir / "results" / "mdd_metrics.csv"
    assert res.exists()
    df = pd.read_csv(res)
    fused = df[(df.branch == "fused") & (df.corruption == "hot_pixel")]
    assert not fused.empty
    # REGRESSION GUARD: fused must strongly separate a planted dilation.
    assert fused.iloc[0]["auroc"] > 0.95
    assert 0.0 <= fused.iloc[0]["fpr95"] <= 1.0

    agg = synthetic_phi_dir / "results" / "mdd_metrics_aggregated.csv"
    assert agg.exists()                         # seq_lens present -> per-sequence written
    adf = pd.read_csv(agg)
    assert (adf["auroc"].between(0, 1)).all()


def test_evaluate_mdd_branch_metrics_in_bounds(synthetic_phi_dir):
    import analysis.evaluate_mdd as em
    importlib.reload(em)
    em.main()
    df = pd.read_csv(synthetic_phi_dir / "results" / "mdd_metrics.csv")
    assert set(df.branch.unique()) >= {"radius", "rcf", "l4", "fused"}
    assert df["auroc"].between(0, 1).all()
    assert df["fpr95"].between(0, 1).all()
