"""GPU reimplementations of the sklearn fits that SCALE with the dataset.

The fits that grow with our ~343k-frame clean set are linear-algebra ops —
empirical / Ledoit-Wolf covariance (O(n*d^2)), PCA (O(n*d^2)), and logistic
regression. These are deterministic, so the GPU versions reproduce sklearn's
output (validated on synthetic data) while moving the O(n*d^2) work off the CPU.

ALL of these STREAM the data to the GPU in row-chunks via the same VRAM-aware
sizing the scoring path uses (`query_chunk_rows`): the full n x d matrix is NEVER
resident on the GPU, so they don't OOM a small/shared GPU. The only resident
state is the d x d second-moment (Gram) accumulator. The expensive O(n*d^2)
matmul is done ONCE per fit. The small d x d reductions (pinvh, eigh) run on CPU
(scipy) / are cheap, independent of n.

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
from analysis.vmem_utils import device_for, query_chunk_rows


def _precision(cov_t):
    """Precision = scipy.linalg.pinvh(covariance) — the EXACT call sklearn's
    covariance estimators use, so the output matches bit-for-bit (including its
    eigenvalue-truncation behaviour in the degenerate n<d case). This is a d x d
    op independent of n (~1s), so it does NOT scale with the dataset."""
    return pinvh(cov_t.cpu().numpy())


def _streamed_moments(X, device, want_x4=False):
    """Stream X to the GPU in row-chunks and return the CENTRED second-moment
    (Gram) matrix Gc = sum_k (x_k - mu)(x_k - mu)^T and the mean mu, optionally
    the centred 4th-moment sum sum_k ||x_k - mu||^4 (for Ledoit-Wolf).

    The full matrix is never resident: only one chunk (chunk x d) plus the d x d
    accumulator. Two passes — one for the mean, one for the centred Gram — so the
    centring is exact (no catastrophic cancellation of an uncentred Gram). The
    O(n*d^2) chunk matmuls run in float32 (fast on every GPU) and accumulate into
    a float64 Gram (stable), reproducing sklearn's float64 result for the
    well-conditioned n>=d workload."""
    Xnp = np.ascontiguousarray(X, dtype=np.float32)
    N, d = Xnp.shape
    chunk = max(256, query_chunk_rows(2 * d, device))   # 2*d: chunk + accum headroom

    # Pass 1 — mean (float64 accumulation).
    colsum = torch.zeros(d, dtype=torch.float64, device=device)
    for i in range(0, N, chunk):
        cb = torch.from_numpy(Xnp[i:i + chunk]).to(device)
        colsum += cb.sum(0, dtype=torch.float64)
        del cb
    mu = colsum / N
    mu32 = mu.to(torch.float32)

    # Pass 2 — centred Gram (float32 matmul -> float64 accumulator) [+ 4th moment].
    Gc = torch.zeros(d, d, dtype=torch.float64, device=device)
    sum_x4 = torch.zeros((), dtype=torch.float64, device=device)
    for i in range(0, N, chunk):
        cb = torch.from_numpy(Xnp[i:i + chunk]).to(device)
        cbc = cb - mu32
        Gc += (cbc.T @ cbc).to(torch.float64)
        if want_x4:
            cb64 = cb.to(torch.float64) - mu
            sq = (cb64 * cb64).sum(1)
            sum_x4 += (sq * sq).sum()
        del cb, cbc
    return (mu, Gc, sum_x4) if want_x4 else (mu, Gc)


class CovResult:
    """Picklable stand-in for an sklearn covariance estimator: exposes
    `location_` (mean) and `precision_` (inverse covariance), the only two
    attributes the Mahalanobis scorers read."""
    def __init__(self, location_, precision_):
        self.location_ = location_
        self.precision_ = precision_


def empirical_precision(X, op="empirical covariance (MLE)"):
    """GPU EmpiricalCovariance: MLE covariance (1/n) and its precision (pinvh),
    matching sklearn.covariance.EmpiricalCovariance including the singular n<d
    case. Streamed — never holds the full matrix on the GPU."""
    device = device_for(op)
    mu, Gc = _streamed_moments(X, device)
    n = len(X)
    S = Gc / n                                           # MLE covariance (d, d)
    return CovResult(mu.cpu().numpy(), _precision(S))


def ledoit_wolf_precision(X, op="Ledoit-Wolf covariance"):
    """GPU Ledoit-Wolf shrinkage covariance + precision, streamed.

    Reproduces sklearn.covariance.LedoitWolf term-for-term from the centred Gram
    Gc = X_c^T X_c and the centred 4th-moment sum (= sklearn's beta_ = sum of
    <X2^T, X2>). The O(n*d^2) work streams to the GPU; the d x d pinvh is scipy
    (sklearn-exact).

    REGIME NOTE: matches sklearn to ~1e-6 when n >= d (the real workload: n=343k
    clean frames vs d<=2252 — well-conditioned). When n < d the shrunk covariance
    has one near-singular direction whose inverse is hyper-sensitive to ~1e-7
    float differences, so the precision can differ a lot — but that is the
    degenerate small-debug-subset regime ONLY, where sklearn itself is unstable
    (its precision_ @ covariance_ != I)."""
    device = device_for(op)
    n, d = X.shape
    mu_vec, Gc, beta_ = _streamed_moments(X, device, want_x4=True)
    # sklearn ledoit_wolf_shrinkage, expressed in centred-Gram terms:
    #   emp_cov S = Gc/n ; emp_cov_trace.sum() = trace(Gc)/n
    #   beta_  = sum_k ||x_c_k||^4                 (== sum(<X2^T,X2>))
    #   delta_ = ||X^T X_centred||_F^2 / n^2 = ||Gc||_F^2 / n^2
    trace_S = torch.trace(Gc) / n
    mu = trace_S / d
    delta_ = (Gc * Gc).sum() / (n * n)
    beta = 1.0 / (d * n) * (beta_ / n - delta_)
    delta = (delta_ - 2.0 * mu * trace_S + d * mu * mu) / d
    beta = torch.minimum(beta, delta)
    shrinkage = (torch.zeros((), dtype=torch.float64, device=device)
                 if float(beta) == 0 else beta / delta)
    cov = (1.0 - shrinkage) * (Gc / n)
    cov.diagonal().add_(shrinkage * mu)
    return CovResult(mu_vec.cpu().numpy(), _precision(cov))


class GpuPCA:
    """Picklable PCA fit. Drop-in for the sklearn PCA attributes the pipeline
    reads: `mean_`, `components_`, `n_components_`, and `transform`.

    Components come from the eigendecomposition of the d x d centred Gram (the
    same matrix as PCA's X_c^T X_c), NOT a full SVD — so we never materialise the
    n x min(n,d) left singular vectors. A deterministic sign convention (largest-
    magnitude loading positive, like sklearn's svd_flip) makes the basis
    reproducible across runs/hardware so a saved flow_pca always transforms the
    same way it was fitted."""
    def __init__(self, mean_, components_):
        self.mean_ = mean_
        self.components_ = components_
        self.n_components_ = components_.shape[0]

    @classmethod
    def fit(cls, X, n_components, op="PCA (Gram eigh)"):
        device = device_for(op)
        mu, Gc = _streamed_moments(X, device)
        d = Gc.shape[0]
        k = max(1, min(n_components, len(X), d))
        # eigh: ascending eigenvalues; top-k = last k, reversed to descending.
        _, evecs = torch.linalg.eigh(Gc)
        comps = evecs[:, d - k:].flip(1).T.contiguous()   # (k, d), PC1 first
        comps = comps.cpu().numpy().astype(np.float32)
        # deterministic sign: make the largest-|loading| entry of each PC positive.
        signs = np.sign(comps[np.arange(k), np.argmax(np.abs(comps), axis=1)])
        signs[signs == 0] = 1.0
        comps *= signs[:, None]
        return cls(mu.cpu().numpy().astype(np.float32), comps)

    def transform(self, X):
        """Project X onto the components, streamed in row-chunks (no full upload)."""
        device = device_for("PCA transform", verbose=False)
        mean = torch.from_numpy(self.mean_).to(device)
        comps = torch.from_numpy(self.components_).to(device)
        Xnp = np.ascontiguousarray(X, dtype=np.float32)
        N = Xnp.shape[0]
        chunk = max(256, query_chunk_rows(comps.shape[0], device))
        out = np.empty((N, comps.shape[0]), dtype=np.float32)
        for i in range(0, N, chunk):
            cb = torch.from_numpy(Xnp[i:i + chunk]).to(device)
            out[i:i + chunk] = ((cb - mean) @ comps.T).cpu().numpy()
            del cb
        return out


def logreg_fit(X, y, op="logistic regression", l2=1.0, max_iter=1000):
    """GPU binary logistic regression (L2-regularised), streamed.

    Convex -> unique solution, so it matches sklearn LogisticRegression (same
    C=1/l2 L2 penalty, intercept unpenalised) up to optimiser tolerance. The data
    streams to the GPU in row-chunks inside the L-BFGS closure (full X never
    resident); the loss and gradient are accumulated exactly over chunks. Returns
    an object with `predict_proba` (column 1 = P(y=1))."""
    device = device_for(op)
    Xnp = np.ascontiguousarray(X, dtype=np.float32)
    ynp = np.ascontiguousarray(y, dtype=np.float32)
    N, d = Xnp.shape
    chunk = max(256, query_chunk_rows(2 * d, device))

    w = torch.zeros(d, device=device, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, device=device, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter,
                            tolerance_grad=1e-7, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        with torch.no_grad():
            loss = torch.zeros((), dtype=torch.float64, device=device)
            gw = torch.zeros_like(w)
            gb = torch.zeros_like(b)
            for i in range(0, N, chunk):
                xb = torch.from_numpy(Xnp[i:i + chunk]).to(device).double()
                yb = torch.from_numpy(ynp[i:i + chunk]).to(device).double()
                z = xb @ w + b
                # sum BCE-with-logits, numerically stable: max(z,0) - z*y + log1p(exp(-|z|))
                loss += (torch.clamp(z, min=0) - z * yb
                         + torch.log1p(torch.exp(-torch.abs(z)))).sum()
                p = torch.sigmoid(z)
                gw += xb.T @ (p - yb)
                gb += (p - yb).sum()
                del xb, yb, z, p
            loss += 0.5 / l2 * (w * w).sum()              # L2 on weights only
            w.grad = gw + w / l2
            b.grad = gb.reshape(1)
        return loss
    opt.step(closure)

    w_np = w.detach().cpu().numpy()
    b_np = float(b.detach().cpu().numpy()[0])

    class _LR:
        def __init__(self, w, b):
            self.coef_ = w[None, :]
            self.intercept_ = np.array([b])
        def predict_proba(self, Xq):
            z = np.asarray(Xq, dtype=np.float64) @ w_np + b_np
            p1 = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                          np.exp(z) / (1.0 + np.exp(z)))   # overflow-safe sigmoid
            return np.stack([1.0 - p1, p1], axis=1)
    return _LR(w_np, b_np)
