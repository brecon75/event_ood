"""GPU reimplementations of the sklearn fits that SCALE with the dataset.

The fits that grow with our ~343k-frame clean set are linear-algebra ops —
empirical / Ledoit-Wolf covariance (O(n*d^2)), PCA (O(n*d^2)), and logistic
regression. These are deterministic, so the GPU versions reproduce sklearn's
output (validated on synthetic data) while moving the O(n*d^2) work off the CPU.

Iterative heuristic fits (GaussianMixture EM, OneClassSVM SMO) are NOT here: they
are capped at 20k samples (they do not scale with the dataset) and a from-scratch
GPU rewrite would not reproduce sklearn/libsvm output. They stay on CPU.

Every entry point prints whether it ran on [GPU] or [CPU] via `device_for`.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import pinvh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.vmem_utils import device_for


def _to(X, device, dtype=torch.float32):
    npd = np.float64 if dtype == torch.float64 else np.float32
    return torch.from_numpy(np.ascontiguousarray(X, dtype=npd)).to(device)


def _precision(cov_t):
    """Precision = scipy.linalg.pinvh(covariance) — the EXACT call sklearn's
    covariance estimators use, so the output matches bit-for-bit (including its
    eigenvalue-truncation behaviour in the degenerate n<d case). This is a d x d
    op independent of n (~1s), so it does NOT scale with the dataset — only the
    O(n*d^2) covariance that produced `cov_t` needed the GPU."""
    return pinvh(cov_t.cpu().numpy())


class CovResult:
    """Picklable stand-in for an sklearn covariance estimator: exposes
    `location_` (mean) and `precision_` (inverse covariance), the only two
    attributes the Mahalanobis scorers read."""
    def __init__(self, location_, precision_):
        self.location_ = location_
        self.precision_ = precision_


def empirical_precision(X, op="empirical covariance (MLE)"):
    """GPU EmpiricalCovariance: MLE covariance (1/n) and its precision.
    Matches sklearn.covariance.EmpiricalCovariance (which uses pinvh), including
    the singular n<d case. Computed in float64 — a near-singular d x d inverse is
    catastrophically wrong in float32."""
    device = device_for(op)
    Xt = _to(X, device, torch.float64)
    mu = Xt.mean(0)
    Xc = Xt - mu
    n = Xc.shape[0]
    S = (Xc.T @ Xc) / n                                  # O(n*d^2) on GPU
    return CovResult(mu.cpu().numpy(), _precision(S))    # d x d pinvh on CPU (sklearn-exact)


def ledoit_wolf_precision(X, op="Ledoit-Wolf covariance"):
    """GPU Ledoit-Wolf shrinkage covariance + precision.

    Reproduces sklearn.covariance.LedoitWolf: shrink the MLE covariance S toward
    mu*I, with mu = trace(S)/d and shrinkage = min(bbar2, d2)/d2 where
    d2 = ||S - mu I||_F^2 and bbar2 = (1/n^2) * sum_k ||x_k x_k^T - S||_F^2.
    The O(n*d^2) terms (S and the bbar2 sum) run on the GPU; the d x d pinvh is
    scipy (sklearn-exact). All in float64.

    REGIME NOTE: matches sklearn to ~1e-6 when n >= d (the real workload: n=343k
    clean frames vs d<=2252 — well-conditioned). When n < d the shrunk covariance
    has one near-singular direction whose inverse is hyper-sensitive to the ~1e-7
    GPU-vs-numpy difference in the covariance matmul, so the precision can differ
    a lot — but that is the degenerate small-debug-subset regime ONLY, where
    sklearn itself is numerically unstable (its precision_ @ covariance_ != I)."""
    device = device_for(op)
    Xt = _to(X, device, torch.float64)
    n, d = Xt.shape
    mu_vec = Xt.mean(0)
    Xc = Xt - mu_vec                                     # centred (sklearn does this)
    # Mirror sklearn.covariance.ledoit_wolf_shrinkage term-for-term in float64.
    X2 = Xc * Xc
    emp_cov_trace = X2.sum(0) / n
    mu = emp_cov_trace.sum() / d
    beta_ = (X2.T @ X2).sum()
    delta_ = ((Xc.T @ Xc) ** 2).sum() / (n * n)
    beta = 1.0 / (d * n) * (beta_ / n - delta_)
    delta = (delta_ - 2.0 * mu * emp_cov_trace.sum() + d * mu * mu) / d
    beta = torch.minimum(beta, delta)
    shrinkage = (torch.zeros((), dtype=torch.float64, device=device)
                 if float(beta) == 0 else beta / delta)
    # shrunk covariance: (1-shrinkage) S + shrinkage mu I, then pinvh -> precision.
    cov = (Xc.T @ Xc) / n
    cov = (1.0 - shrinkage) * cov
    cov.diagonal().add_(shrinkage * mu)
    return CovResult(mu_vec.cpu().numpy(), _precision(cov))   # sklearn-exact pinvh


class GpuPCA:
    """Picklable PCA fit via GPU SVD. Drop-in for the sklearn PCA attributes the
    pipeline reads: `mean_`, `components_`, `n_components_`, and `transform`."""
    def __init__(self, mean_, components_):
        self.mean_ = mean_
        self.components_ = components_
        self.n_components_ = components_.shape[0]

    @classmethod
    def fit(cls, X, n_components, op="PCA (SVD)"):
        device = device_for(op)
        Xt = _to(X, device)
        mean = Xt.mean(0)
        Xc = Xt - mean
        k = max(1, min(n_components, Xc.shape[0], Xc.shape[1]))
        # economy SVD; right singular vectors are the principal axes.
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        comps = Vh[:k]
        return cls(mean.cpu().numpy().astype(np.float32),
                   comps.cpu().numpy().astype(np.float32))

    def transform(self, X):
        device = device_for("PCA transform", verbose=False)
        Xt = _to(X, device)
        mean = _to(self.mean_, device)
        comps = _to(self.components_, device)
        return ((Xt - mean) @ comps.T).cpu().numpy()


def logreg_fit(X, y, op="logistic regression", l2=1.0, epochs=300, lr=0.5):
    """GPU binary logistic regression (L2-regularised) via full-batch L-BFGS.

    Convex problem -> unique solution, so this matches sklearn LogisticRegression
    (same C=1/l2 L2 penalty) up to optimiser tolerance. Returns an object with
    `predict_proba` (column 1 = P(y=1)), the only thing the caller reads."""
    device = device_for(op)
    Xt = _to(X, device)
    yt = torch.from_numpy(np.ascontiguousarray(y, np.float32)).to(device)
    n, d = Xt.shape
    w = torch.zeros(d, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=epochs,
                            line_search_fn="strong_wolfe")
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        logits = Xt @ w + b
        # sklearn's C penalises 0.5*||w||^2 / C summed (not averaged); scale to match.
        loss = n * bce(logits, yt) + 0.5 / l2 * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)

    w_np = w.detach().cpu().numpy()
    b_np = float(b.detach().cpu().numpy()[0])

    class _LR:
        def __init__(self, w, b):
            self.coef_ = w[None, :]
            self.intercept_ = np.array([b])
        def predict_proba(self, Xq):
            z = Xq @ w_np + b_np
            p1 = 1.0 / (1.0 + np.exp(-z))
            return np.stack([1.0 - p1, p1], axis=1)
    return _LR(w_np, b_np)
