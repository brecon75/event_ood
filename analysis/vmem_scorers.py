import sys
import math
from pathlib import Path
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from analysis.vmem_utils import (
    _subsample, _cap_subset, GMM_FIT_SAMPLES, chunked_apply, knn_score, device_for,
)
from analysis.gpu_fit import ledoit_wolf_precision, GpuPCA
from analysis.vmem_models import (
    train_flow_model, train_ae_model, train_temporal_ae_model,
    prepare_temporal_ae_input
)

def mahalanobis_scorer(clean: np.ndarray):
    fit = _subsample(clean)
    try:
        cov = ledoit_wolf_precision(fit, op="Mahalanobis fit (Ledoit-Wolf)")
        mu, P = cov.location_, cov.precision_
    except Exception as e:
        print(f"  [!] Mahalanobis: LedoitWolf fit failed ({e}); "
              f"falling back to identity precision (plain squared L2).")
        mu = clean.mean(0)
        P  = np.eye(clean.shape[1])

    device = device_for("Mahalanobis scoring", verbose=False)
    mu_t = torch.from_numpy(mu).float().to(device)
    P_t = torch.from_numpy(P).float().to(device)

    def score(x):
        def fn(chunk):
            d = chunk - mu_t
            return torch.einsum("ni,ij,nj->n", d, P_t, d)
        return chunked_apply(fn, np.ascontiguousarray(x, dtype=np.float32),
                             device, n_ref=mu_t.shape[0])
    return score


def knn_scorer(clean: np.ndarray, k: int = 5):
    fit = _subsample(clean)
    k   = max(1, min(k, len(fit) - 1))
    device = device_for("kNN scoring", verbose=False)

    # Keep the full reference on CPU; `knn_score` streams it to the GPU in
    # blocks so the whole ~2 GB clean set is never resident at once.
    def score(x):
        return knn_score(fit, k, x, device)
    return score


def gmm_scorer(clean: np.ndarray, n_components: int = 5):
    fit = _cap_subset(clean, GMM_FIT_SAMPLES)  # capped: full-cov EM is heavy in 2112-D
    nc  = min(n_components, max(1, len(fit) // 20))
    fast_mode = "--fast" in sys.argv
    gmm_iters = 5 if fast_mode else 1000  # ceiling raised for convergence
    try:
        print("  [CPU] GMM fit (sklearn EM, capped; iterative — not GPU-reimplemented)...",
              flush=True)
        gmm = GaussianMixture(n_components=nc, covariance_type="full",
                              reg_covar=1e-4, random_state=42, max_iter=gmm_iters)
        gmm.fit(fit)

        device = device_for("GMM scoring", verbose=False)
        weights_t = torch.from_numpy(gmm.weights_).float().to(device)
        means_t = torch.from_numpy(gmm.means_).float().to(device)
        precisions_cholesky_t = torch.from_numpy(gmm.precisions_cholesky_).float().to(device)
        
        log_det_precision = 2.0 * torch.sum(
            torch.log(torch.diagonal(precisions_cholesky_t, dim1=-2, dim2=-1)), dim=1
        )

        D = fit.shape[1]
        log_2pi = math.log(2.0 * math.pi)

        def score(x):
            def fn(chunk):
                log_probs = []
                for c in range(nc):
                    diff = chunk - means_t[c]
                    proj = diff @ precisions_cholesky_t[c]
                    quad = torch.sum(proj ** 2, dim=-1)
                    lp = torch.log(weights_t[c]) + 0.5 * log_det_precision[c] - 0.5 * D * log_2pi - 0.5 * quad
                    log_probs.append(lp)
                return -torch.logsumexp(torch.stack(log_probs, dim=1), dim=1)
            # Peak buffer is (chunk x D) per component, stacked across nc
            # components for the logsumexp — size the first chunk to D*nc.
            return chunked_apply(fn, np.ascontiguousarray(x, dtype=np.float32),
                                 device, n_ref=D * nc)
        return score
    except Exception as e:
        print(f"  [!] GMM failed ({e}), falling back to Mahalanobis")
        return mahalanobis_scorer(clean)


def pca_mahalanobis_scorer(clean: np.ndarray, n_components: int = 50):
    fit = _subsample(clean)
    nc  = max(1, min(n_components, fit.shape[1], len(fit) - 1))
    pca = GpuPCA.fit(fit, nc, op="PCA-Mahalanobis fit (SVD)")

    fit_proj = pca.transform(fit)
    base_score_fn = mahalanobis_scorer(fit_proj)

    device = device_for("PCA-Mahalanobis projection", verbose=False)
    pca_mean_t = torch.from_numpy(pca.mean_).float().to(device)
    pca_components_t = torch.from_numpy(pca.components_).float().to(device)

    def score(x):
        def fn(chunk):
            return (chunk - pca_mean_t) @ pca_components_t.T
        proj_all = chunked_apply(fn, np.ascontiguousarray(x, dtype=np.float32),
                                 device, n_ref=pca_components_t.shape[1])
        return base_score_fn(proj_all)
    return score


def ocsvm_scorer(clean):
    fit = _subsample(clean)
    print("  [CPU] OCSVM fit (sklearn/libsvm SMO; O(n^2) — not GPU-reimplemented)...",
          flush=True)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(fit)

    # sklearn's libsvm decision_function is single-threaded with no BLAS — the
    # dominant cost at 343k frames/run. The RBF decision is exactly
    # dual_coef · exp(-gamma·‖x−SV‖²) + intercept, so evaluate it chunked on GPU.
    device = device_for("OCSVM scoring", verbose=False)
    gamma = float(svm._gamma)
    sv_t = torch.from_numpy(np.ascontiguousarray(svm.support_vectors_, dtype=np.float32)).to(device)
    dual_t = torch.from_numpy(np.ascontiguousarray(svm.dual_coef_[0], dtype=np.float32)).to(device)
    intercept = float(svm.intercept_[0])

    def score(x):
        def fn(chunk):
            d2 = torch.cdist(chunk, sv_t).pow_(2)
            return -(torch.exp(-gamma * d2) @ dual_t + intercept)
        return chunked_apply(fn, np.ascontiguousarray(x, dtype=np.float32),
                             device, n_ref=sv_t.shape[0])
    return score


def normalizing_flow_scorer(clean, n_components=50):
    fit = _subsample(clean)
    nc = max(1, min(n_components, fit.shape[1], len(fit) - 1))
    pca = GpuPCA.fit(fit, nc, op="Flow PCA fit (SVD)")

    clean_proj = pca.transform(clean)
    device = device_for("Normalizing-flow training", verbose=False)
    flow = train_flow_model(clean_proj, device=device)
    
    def score(x):
        def fn(chunk):
            with torch.no_grad():
                return -flow.log_prob(chunk)
        x_proj = np.ascontiguousarray(pca.transform(x), dtype=np.float32)
        # Peak isn't the 50-D input: each RealNVP coupling net expands to a
        # hidden width across n_layers (≈ hidden_dim*n_layers*2). Size to that.
        return chunked_apply(fn, x_proj, device, n_ref=64 * 4 * 2)
    return score


def autoencoder_scorer(clean):
    device = device_for("Autoencoder training + scoring")
    # Train on the FULL clean set (consistent with fit_detectors.fit_ae and the
    # other detectors). `_subsample` is a no-op outside --fast and caps to 500
    # for smoke tests. (The old `fit_size=20000` looked like a cap but was
    # silently ignored, since _subsample ignores n outside --fast.)
    fit = _subsample(clean)
    ae = train_ae_model(fit, device=device)
    
    def score(x):
        def fn(chunk):
            with torch.no_grad():
                return ((chunk - ae(chunk)) ** 2).mean(dim=-1)
        return chunked_apply(fn, np.ascontiguousarray(x, dtype=np.float32),
                             device, n_ref=x.shape[1])
    return score


def temporal_autoencoder_scorer(clean_trajs):
    device = device_for("Temporal-autoencoder training + scoring")
    ae = train_temporal_ae_model(clean_trajs, device=device)

    def score(trajs, batch_size=4096):
        if isinstance(trajs, torch.Tensor):
            x = trajs
        else:
            x = prepare_temporal_ae_input(trajs)
        # Batch the transfer + forward pass: a full run's GAP trajectories
        # are ~10 GB and would OOM the GPU if moved over in one piece.
        errs = []
        with torch.no_grad():
            for chunk in torch.split(x, batch_size):
                c = chunk.to(device)
                recon = ae(c)
                errs.append(((c - recon) ** 2).mean(dim=(1, 2)).cpu())
        return torch.cat(errs).numpy()
    return score
