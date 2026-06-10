import sys
import math
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from analysis.vmem_utils import _subsample, MAX_FIT_SAMPLES
from analysis.vmem_models import (
    train_flow_model, train_ae_model, train_temporal_ae_model,
    prepare_temporal_ae_input
)

def mahalanobis_scorer(clean: np.ndarray):
    fit = _subsample(clean)
    try:
        cov = LedoitWolf().fit(fit)
        mu, P = cov.location_, cov.precision_
    except Exception as e:
        print(f"  [!] Mahalanobis: LedoitWolf fit failed ({e}); "
              f"falling back to identity precision (plain squared L2).")
        mu = clean.mean(0)
        P  = np.eye(clean.shape[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mu_t = torch.from_numpy(mu).float().to(device)
    P_t = torch.from_numpy(P).float().to(device)

    def score(x):
        x_t = torch.from_numpy(x).float().to(device)
        scores = []
        chunks = torch.split(x_t, 50000)
        pbar = tqdm(chunks, desc="Mahalanobis", leave=False, disable=len(chunks) <= 1)
        for chunk in pbar:
            d = chunk - mu_t
            s = torch.einsum("ni,ij,nj->n", d, P_t, d)
            scores.append(s)
        return torch.cat(scores).cpu().numpy()
    return score


def knn_scorer(clean: np.ndarray, k: int = 5):
    fit = _subsample(clean)
    k   = max(1, min(k, len(fit) - 1))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fit_t = torch.from_numpy(fit).float().to(device)

    def score(x):
        x_t = torch.from_numpy(x).float().to(device)
        scores = []
        chunks = torch.split(x_t, 10000)
        pbar = tqdm(chunks, desc="k-NN Query", leave=False, disable=len(chunks) <= 1)
        for chunk in pbar:
            dist_matrix = torch.cdist(chunk, fit_t, p=2)
            dists, _ = torch.topk(dist_matrix, k, largest=False, dim=1)
            scores.append(dists.mean(dim=1))
        return torch.cat(scores).cpu().numpy()
    return score


def gmm_scorer(clean: np.ndarray, n_components: int = 5):
    fit = _subsample(clean)
    nc  = min(n_components, max(1, len(fit) // 20))
    fast_mode = "--fast" in sys.argv
    gmm_iters = 5 if fast_mode else 300
    try:
        gmm = GaussianMixture(n_components=nc, covariance_type="full",
                              reg_covar=1e-4, random_state=42, max_iter=gmm_iters)
        gmm.fit(fit)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weights_t = torch.from_numpy(gmm.weights_).float().to(device)
        means_t = torch.from_numpy(gmm.means_).float().to(device)
        precisions_cholesky_t = torch.from_numpy(gmm.precisions_cholesky_).float().to(device)
        
        log_det_precision = 2.0 * torch.sum(
            torch.log(torch.diagonal(precisions_cholesky_t, dim1=-2, dim2=-1)), dim=1
        )

        D = fit.shape[1]
        log_2pi = math.log(2.0 * math.pi)

        def score(x):
            x_t = torch.from_numpy(x).float().to(device)
            scores = []
            chunks = torch.split(x_t, 50000)
            pbar = tqdm(chunks, desc="GMM Query", leave=False, disable=len(chunks) <= 1)
            for chunk in pbar:
                log_probs = []
                for c in range(nc):
                    diff = chunk - means_t[c]
                    proj = diff @ precisions_cholesky_t[c]
                    quad = torch.sum(proj ** 2, dim=-1)
                    
                    lp = torch.log(weights_t[c]) + 0.5 * log_det_precision[c] - 0.5 * D * log_2pi - 0.5 * quad
                    log_probs.append(lp)
                
                log_probs_stack = torch.stack(log_probs, dim=1)
                log_prob_sample = torch.logsumexp(log_probs_stack, dim=1)
                scores.append(-log_prob_sample)
            return torch.cat(scores).cpu().numpy()
        return score
    except Exception as e:
        print(f"  [!] GMM failed ({e}), falling back to Mahalanobis")
        return mahalanobis_scorer(clean)


def pca_mahalanobis_scorer(clean: np.ndarray, n_components: int = 50):
    fit = _subsample(clean)
    nc  = max(1, min(n_components, fit.shape[1], len(fit) - 1))
    pca = PCA(n_components=nc, random_state=42)
    pca.fit(fit)
    
    fit_proj = pca.transform(fit)
    base_score_fn = mahalanobis_scorer(fit_proj)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pca_mean_t = torch.from_numpy(pca.mean_).float().to(device)
    pca_components_t = torch.from_numpy(pca.components_).float().to(device)

    def score(x):
        x_t = torch.from_numpy(x).float().to(device)
        scores = []
        chunks = torch.split(x_t, 50000)
        pbar = tqdm(chunks, desc="PCA Project", leave=False, disable=len(chunks) <= 1)
        for chunk in pbar:
            chunk_proj = (chunk - pca_mean_t) @ pca_components_t.T
            scores.append(chunk_proj.cpu().numpy())
        proj_all = np.concatenate(scores, axis=0)
        return base_score_fn(proj_all)
    return score


def ocsvm_scorer(clean):
    fit = _subsample(clean)
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    svm.fit(fit)
    def score(x):
        return -svm.decision_function(x)
    return score


def normalizing_flow_scorer(clean, n_components=50):
    fit = _subsample(clean)
    nc = max(1, min(n_components, fit.shape[1], len(fit) - 1))
    pca = PCA(n_components=nc, random_state=42)
    pca.fit(fit)
    
    clean_proj = pca.transform(clean)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow = train_flow_model(clean_proj, device=device)
    
    def score(x):
        x_proj = pca.transform(x)
        x_t = torch.from_numpy(x_proj).float().to(device)
        scores = []
        with torch.no_grad():
            for chunk in torch.split(x_t, 20000):
                scores.append(-flow.log_prob(chunk))
        return torch.cat(scores).cpu().numpy()
    return score


def autoencoder_scorer(clean):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fast_mode = "--fast" in sys.argv
    fit_size = 2000 if fast_mode else 20000
    fit = _subsample(clean, n=fit_size)
    ae = train_ae_model(fit, device=device)
    
    def score(x):
        x_t = torch.from_numpy(x).float().to(device)
        scores = []
        with torch.no_grad():
            for chunk in torch.split(x_t, 20000):
                recon = ae(chunk)
                err = ((chunk - recon) ** 2).mean(dim=-1)
                scores.append(err)
        return torch.cat(scores).cpu().numpy()
    return score


def temporal_autoencoder_scorer(clean_trajs):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
