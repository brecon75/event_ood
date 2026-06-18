"""OOD scorer factories (vmem_scorers) vs sklearn references, and the neural
models in vmem_models (RealNVP log-det, AE shapes). Includes a documented
odd-dimension RealNVP defect.
"""
import numpy as np
import pytest
import torch
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture

from analysis.vmem_scorers import mahalanobis_scorer, knn_scorer, gmm_scorer
from analysis.vmem_models import RealNVP, CouplingLayer, Autoencoder, TemporalAutoencoder


# ───────────────────────── scorers vs references ─────────────────────────

def test_mahalanobis_matches_reference_and_detects_shift():
    rng = np.random.default_rng(0)
    clean = rng.normal(0, 1, (500, 10)).astype(np.float32)
    score = mahalanobis_scorer(clean)
    cov = LedoitWolf().fit(clean)
    test = rng.normal(0, 1, (50, 10)).astype(np.float32)
    ref = np.einsum("ni,ij,nj->n", test - cov.location_, cov.precision_, test - cov.location_)
    assert np.allclose(score(test), ref, rtol=1e-4, atol=1e-4)
    shifted = test + 8.0
    assert score(shifted).mean() > score(test).mean()


def test_knn_scorer_matches_sklearn():
    rng = np.random.default_rng(1)
    clean = rng.normal(0, 1, (300, 8)).astype(np.float32)
    score = knn_scorer(clean, k=5)
    test = rng.normal(0, 1, (40, 8)).astype(np.float32)
    # reference: mean distance to 5 nearest clean neighbours. The scorer fits on
    # the FULL clean array (_subsample is a no-op outside --fast), so the
    # reference uses the same full array.
    nn = NearestNeighbors(n_neighbors=5).fit(clean)
    ref = nn.kneighbors(test)[0].mean(axis=1)
    assert np.allclose(score(test), ref, rtol=1e-4, atol=1e-4)


def test_gmm_scorer_matches_neg_score_samples():
    rng = np.random.default_rng(2)
    clean = rng.normal(0, 1, (400, 6)).astype(np.float32)
    score = gmm_scorer(clean, n_components=3)
    test = rng.normal(0, 1, (30, 6)).astype(np.float32)
    # rebuild the same GMM (deterministic random_state) for the reference
    nc = min(3, max(1, len(clean) // 20))
    gmm = GaussianMixture(n_components=nc, covariance_type="full",
                          reg_covar=1e-4, random_state=42, max_iter=1000).fit(clean)
    assert np.allclose(score(test), -gmm.score_samples(test), rtol=1e-3, atol=1e-3)


# ───────────────────────── RealNVP correctness ─────────────────────────

def test_coupling_logdet_matches_autograd():
    torch.manual_seed(0)
    layer = CouplingLayer(dim=6)
    x = torch.randn(4, 6, requires_grad=True)
    y, logdet = layer(x)
    # autograd Jacobian determinant for one sample
    jac = torch.autograd.functional.jacobian(lambda v: layer(v.unsqueeze(0))[0].squeeze(0), x[0])
    assert logdet[0].item() == pytest.approx(torch.logdet(jac).item(), abs=1e-4)


def test_realnvp_logprob_finite_even_dim():
    torch.manual_seed(0)
    flow = RealNVP(dim=8)
    lp = flow.log_prob(torch.randn(16, 8))
    assert torch.isfinite(lp).all() and lp.shape == (16,)


@pytest.mark.parametrize("dim", [5, 7, 33])
def test_realnvp_odd_dim_works(dim):
    # Regression for the fixed odd-dim CouplingLayer bug.
    flow = RealNVP(dim=dim)
    lp = flow.log_prob(torch.randn(4, dim))
    assert torch.isfinite(lp).all() and lp.shape == (4,)


def test_coupling_is_invertible():
    # Affine coupling is exactly invertible: recover x from (y, the conditioner).
    torch.manual_seed(0)
    for flip in (False, True):
        layer = CouplingLayer(dim=7, flip=flip)
        x = torch.randn(8, 7)
        y, _ = layer(x)
        sd = layer.split_dim
        if flip:
            cond = y[:, sd:]                       # x1 passes through unchanged
            s, t = layer.s_net(cond), layer.t_net(cond)
            x2 = (y[:, :sd] - t) * torch.exp(-s)
            x_rec = torch.cat([x2, cond], dim=-1)
        else:
            cond = y[:, :sd]
            s, t = layer.s_net(cond), layer.t_net(cond)
            x2 = (y[:, sd:] - t) * torch.exp(-s)
            x_rec = torch.cat([cond, x2], dim=-1)
        assert torch.allclose(x_rec, x, atol=1e-5)


# ───────────────────────── autoencoders ─────────────────────────

def test_autoencoder_reconstruction_shape():
    ae = Autoencoder(dim=32)
    assert ae(torch.randn(5, 32)).shape == (5, 32)


def test_temporal_ae_reconstruction_shape():
    ae = TemporalAutoencoder(dim=12)
    x = torch.randn(4, 10, 12)              # (B, T=10, sum_C)
    assert ae(x).shape == x.shape
