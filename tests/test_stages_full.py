"""Full-loop integration for the analysis stages that were only partially
covered before: severity.py, cross_corruption.py, reliability.py.

Builds a small synthetic outputs/ tree (narrow phi so per-call Ledoit-Wolf stays
fast) and runs each stage's main() end to end, asserting it writes its CSV with
in-range metrics. No model, no real data.
"""
import numpy as np
import pandas as pd
import pytest
import torch

WIDTH = 192          # = 3*64 -> exactly one LAYER_SPEC; keeps covariance fits fast
NFRAMES = 120
SEQ_LENS = [60, 60]


def _write_phi(path, shift, seed):
    rng = np.random.default_rng(seed)
    phi = (rng.normal(0, 1, (NFRAMES, WIDTH)) + shift).astype(np.float32)
    torch.save({"phi": torch.from_numpy(phi), "seq_lens": SEQ_LENS,
                "done_seqs": [0, 1], "run": path.stem}, path)


def _write_det(path, seed):
    rng = np.random.default_rng(seed)
    det = torch.zeros(NFRAMES, 8, 7)
    det[:, :, 4] = torch.from_numpy(rng.uniform(0, 1, (NFRAMES, 8)).astype(np.float32))
    det[:, :, 5:] = torch.from_numpy(rng.uniform(0, 1, (NFRAMES, 8, 2)).astype(np.float32))
    torch.save({"det": det, "run": path.stem}, path)


@pytest.fixture
def synthetic_outputs(tmp_path, monkeypatch):
    from vmem_benchmark import benchmark_config as cfg
    phi_dir = tmp_path / "phi"; phi_dir.mkdir()
    det_dir = tmp_path / "det_outputs"; det_dir.mkdir()

    _write_phi(phi_dir / "clean.pt", shift=0.0, seed=0)
    _write_det(det_dir / "clean.pt", seed=100)
    for ci, (c, base_shift) in enumerate((("hot_pixel", 4.0), ("event_flood", 1.5))):
        for sev in (1, 2, 3, 4, 5):
            _write_phi(phi_dir / f"{c}_L{sev}.pt", shift=base_shift * sev / 5.0,
                       seed=10 + ci * 10 + sev)
            _write_det(det_dir / f"{c}_L{sev}.pt", seed=200 + ci * 10 + sev)

    monkeypatch.setattr(cfg, "PHI_DIR", phi_dir)
    monkeypatch.setattr(cfg, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cfg, "DET_OUT_DIR", det_dir)
    monkeypatch.setattr(cfg, "DETECTOR_DIR", tmp_path / "no_detectors")   # empty -> skip fitted
    monkeypatch.setattr(cfg, "ANN_DIR", tmp_path / "no_ann")
    monkeypatch.setattr(cfg, "SPIKE_DIR", tmp_path / "no_spike")
    return tmp_path


def test_severity_stage_runs_and_writes_csv(synthetic_outputs):
    import analysis.severity as sev
    sev.main()
    csv = synthetic_outputs / "results" / "severity_metrics.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert not df.empty
    assert df["rho"].dropna().between(-1.0, 1.0).all()


def test_cross_corruption_stage_runs_and_writes_csv(synthetic_outputs):
    import analysis.cross_corruption as cc
    cc.main()
    csv = synthetic_outputs / "results" / "cross_corruption.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert not df.empty
    assert df["auroc"].between(0.0, 1.0).all()
    assert set(df["eval_corruption"]) <= {"event_flood", "temporal_jitter",
        "polarity_flip", "event_rate_shift", "spatial_dropout"}


def test_reliability_stage_runs_and_writes_csv(synthetic_outputs):
    import analysis.reliability as rel
    rel.main()
    csv = synthetic_outputs / "results" / "reliability_metrics.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    # may legitimately be empty if every run lacked variance, but columns/bounds
    # must hold when rows exist
    if not df.empty:
        assert df["spearman"].dropna().between(-1.0, 1.0).all()
        assert df["r2"].dropna().between(0.0, 1.0).all()
