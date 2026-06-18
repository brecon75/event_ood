"""ANN baseline OOD detectors + extraction helpers (evaluate/extract_ann_baselines)."""
import numpy as np
import pytest
import torch
import torch.nn as nn
from scipy.special import logsumexp
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

import analysis.evaluate_ann_baselines as eab
import analysis.extract_ann_baselines as xab


# ───────────────────────── detector golden / ranking ─────────────────────────

def test_msp_golden_and_ranking():
    det = eab.DetectorMSP()
    peaked = torch.tensor([[10.0, 0.0, 0.0]])     # confident -> msp~1 -> score~-1
    uniform = torch.tensor([[0.0, 0.0, 0.0]])     # msp=1/3 -> score=-1/3
    assert det.score(None, peaked)[0] == pytest.approx(-torch.softmax(peaked, 1).max().item(), abs=1e-6)
    # uniform (more OOD) must score higher than a confident prediction
    assert det.score(None, uniform)[0] > det.score(None, peaked)[0]


def test_energy_matches_neg_logsumexp():
    det = eab.DetectorEnergy(T=1.0)
    logits = torch.randn(5, 4)
    assert np.allclose(det.score(None, logits), -logsumexp(logits.numpy(), axis=1), atol=1e-5)


def test_mahalanobis_detector_matches_reference_and_detects_shift():
    rng = np.random.default_rng(0)
    feats = torch.tensor(rng.normal(0, 1, (300, 8)), dtype=torch.float32)
    det = eab.DetectorMahalanobis(); det.fit(feats, None)
    cov = LedoitWolf().fit(feats.numpy())
    test = torch.tensor(rng.normal(0, 1, (20, 8)), dtype=torch.float32)
    ref = np.einsum("ni,ij,nj->n", test.numpy() - cov.location_, cov.precision_, test.numpy() - cov.location_)
    assert np.allclose(det.score(test, None), ref, rtol=1e-4, atol=1e-4)
    assert det.score(test + 5.0, None).mean() > det.score(test, None).mean()


def test_knn_detector_normalized_kth_distance():
    # Faithful deep-kNN (Sun et al. 2022): L2-normalise features, then score by
    # the distance to the k-th nearest neighbour (not the mean of k).
    rng = np.random.default_rng(1)
    feats = torch.tensor(rng.normal(0, 1, (200, 6)), dtype=torch.float32)
    det = eab.DetectorKNN(k=5); det.fit(feats, None)
    test = torch.tensor(rng.normal(0, 1, (15, 6)), dtype=torch.float32)
    norm = lambda x: x.numpy() / (np.linalg.norm(x.numpy(), axis=1, keepdims=True) + 1e-10)
    ref = NearestNeighbors(n_neighbors=5).fit(norm(feats)).kneighbors(norm(test))[0][:, -1]
    assert np.allclose(det.score(test, None), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["MSP", "Energy", "ODIN", "Mahalanobis", "kNN",
                                  "ReAct", "ViM", "DICE", "GradNorm"])
def test_all_detectors_finite_and_shaped(name):
    rng = np.random.default_rng(2)
    feats = torch.tensor(rng.normal(0, 1, (120, 16)), dtype=torch.float32)
    logits = torch.tensor(rng.normal(0, 1, (120, 10)), dtype=torch.float32)
    detector = {
        "MSP": eab.DetectorMSP, "Energy": eab.DetectorEnergy, "ODIN": eab.DetectorODIN,
        "Mahalanobis": eab.DetectorMahalanobis, "kNN": eab.DetectorKNN,
        "ReAct": eab.DetectorReAct, "ViM": eab.DetectorViM, "DICE": eab.DetectorDICE,
        "GradNorm": eab.DetectorGradNorm,
    }[name]()
    detector.fit(feats[:100], logits[:100])
    s = detector.score(feats[100:], logits[100:])
    assert s.shape == (20,) and np.isfinite(s).all()


def test_ann_calc_fpr95_guards_single_class():
    assert np.isnan(eab.calc_fpr95(np.zeros(5), np.arange(5.0)))


# ───────────────────────── extraction helpers ─────────────────────────

def test_get_resnet_adapts_in_channels(monkeypatch):
    # avoid the ImageNet weight download — capture the real ctor, then patch the
    # name the module looks up so it builds an UNTRAINED backbone (no download).
    orig = xab.models.resnet18
    monkeypatch.setattr(xab.models, "resnet18", lambda weights=None: orig(weights=None))
    model = xab.get_resnet_feature_extractor(in_channels=2)
    assert model.base.conv1.in_channels == 2
    f, l = model(torch.randn(2, 2, 64, 64))
    assert f.shape == (2, 512) and l.shape == (2, 1000)


def test_run_batched_concatenates(monkeypatch):
    class Stub(nn.Module):
        def forward(self, x):
            return x.mean(dim=(2, 3)), x.mean(dim=(2, 3))[:, :2]
    x = torch.randn(10, 3, 8, 8)
    f, l = xab.run_batched(Stub(), x, device="cpu", batch_size=4)
    assert f.shape == (10, 3) and l.shape == (10, 2)


def test_atomic_save_no_residue(tmp_path):
    p = tmp_path / "out.pt"
    xab.atomic_save({"feat": torch.zeros(3, 4)}, p)
    assert p.exists() and not list(tmp_path.glob("_tmp_*"))
    assert torch.load(p, weights_only=True)["feat"].shape == (3, 4)
