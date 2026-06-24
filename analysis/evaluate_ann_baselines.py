import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from tqdm import tqdm
from scipy.special import logsumexp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    split_train_eval, load_phi_seq_lens, chunked_apply, knn_score,
    device_for, split_boundary, TRAIN_RATIO, query_chunk_rows,
)
from analysis.gpu_fit import ledoit_wolf_precision
import functools


def _device(op="ANN-baseline scoring", verbose=False):
    # Silent by default: called inside per-run hot loops. Fit-time device use is
    # announced separately (the covariance/eigh fits print their device).
    return device_for(op, verbose=verbose)


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _chunk_feats(fn, feats, device, n_ref):
    """Run a per-row torch `fn` over feature rows, chunked/OOM-safe on the GPU."""
    X = np.ascontiguousarray(_np(feats), dtype=np.float32)
    return chunked_apply(fn, X, device, n_ref=n_ref)


# ResNet-18's fc head has exactly 512 input features.
_RESNET_FC_IN = 512


@functools.lru_cache(maxsize=1)
def _resnet_head_weights():
    """ImageNet ResNet-18 fc weights (W, b) — the exact classifier head
    extract_ann_baselines.py keeps untouched (only conv1 is re-initialised),
    so every stored logit equals W @ feature + b. Cached to avoid reloading."""
    import torchvision.models as models
    fc = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).fc
    return fc.weight.detach().numpy(), fc.bias.detach().numpy()


def _resnet_head(feat_dim):
    """(W, b) when the feature dimensionality matches the ResNet head, else
    None (e.g. the synthetic unit-test features) so callers can fall back to a
    head-free score.

    Short-circuits on any dim != 512 BEFORE loading the weights, so synthetic
    test features never trigger the ImageNet download (which keeps the unit
    tests offline-safe — `lru_cache` does not cache a raised download error, so
    a probe-then-fail would re-attempt the network on every detector)."""
    if feat_dim != _RESNET_FC_IN:
        return None
    try:
        W, b = _resnet_head_weights()
    except Exception:
        return None
    return (W, b) if W.shape[1] == feat_dim else None


def _energy(logits):
    """OOD Energy score = -logsumexp(logits) (higher = more OOD). Accepts a
    torch tensor or a numpy array of (recomputed) logits."""
    lg = logits.numpy() if hasattr(logits, "numpy") else np.asarray(logits)
    return -logsumexp(lg, axis=1)


def calc_fpr95(y_true, y_score):
    if len(np.unique(y_true)) < 2: return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr >= 0.95))
    return fpr[idx]

class DetectorMSP:
    def __init__(self): pass
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        # Higher score = more OOD
        device = _device()
        def fn(c):
            return -torch.softmax(c, dim=1).max(dim=1).values
        L = np.ascontiguousarray(_np(logits), dtype=np.float32)
        return chunked_apply(fn, L, device, n_ref=L.shape[1])

class DetectorEnergy:
    def __init__(self, T=1.0): self.T = T
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        device, T = _device(), self.T
        def fn(c):
            return -T * torch.logsumexp(c / T, dim=1)
        L = np.ascontiguousarray(_np(logits), dtype=np.float32)
        return chunked_apply(fn, L, device, n_ref=L.shape[1])

class DetectorODIN:
    """ODIN (Liang et al., ICLR 2018). Temperature-scaled max softmax with the
    paper's default T=1000. The input-perturbation step
    x~ = x - eps*sign(-grad_x log S_yhat(x;T)) is OMITTED: it requires the
    gradient of the score w.r.t. the raw input, which this offline pipeline
    (cached penultimate features + logits, no retained inputs/model) cannot
    compute. What remains — temperature scaling — is implemented faithfully."""
    def __init__(self, T=1000.0): self.T = T
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        device, T = _device(), self.T
        def fn(c):
            return -torch.softmax(c / T, dim=1).max(dim=1).values
        L = np.ascontiguousarray(_np(logits), dtype=np.float32)
        return chunked_apply(fn, L, device, n_ref=L.shape[1])

class DetectorMahalanobis:
    """Mahalanobis (Lee et al., NeurIPS 2018), single-class core. The
    class-conditional Gaussians of the original collapse to one Gaussian here
    because the corruption-OOD setting has no semantic class labels for the
    clean (ID) frames — ID is simply "clean event data". Covariance is the
    Ledoit-Wolf shrinkage estimate. The input-perturbation and multi-layer
    feature-ensemble of the paper are omitted (both need the model at inference,
    which the offline feature cache does not provide)."""
    def __init__(self): self.mu = None; self.P = None
    def fit(self, feats, logits):
        try:
            cov = ledoit_wolf_precision(_np(feats), op="ANN Mahalanobis fit (Ledoit-Wolf)")
            self.mu = cov.location_
            self.P = cov.precision_
        except Exception:
            self.mu = _np(feats).mean(0)
            self.P = np.eye(feats.shape[1])
    def score(self, feats, logits):
        device = _device()
        mu = torch.from_numpy(np.ascontiguousarray(self.mu, np.float32)).to(device)
        P = torch.from_numpy(np.ascontiguousarray(self.P, np.float32)).to(device)
        def fn(c):
            d = c - mu
            return ((d @ P) * d).sum(dim=1)
        return _chunk_feats(fn, feats, device, P.shape[0])

class DetectorKNN:
    """Deep nearest-neighbour OOD (Sun et al., ICML 2022). Two faithful details
    the earlier version dropped: (1) features are L2-normalised before any
    distance is taken (z = h/||h||_2), and (2) the score is the distance to the
    k-th nearest training neighbour r_k(z) = ||z - z_(k)||_2, NOT the mean of
    the k distances. k=50 follows the paper's CIFAR-10 setting, clamped to the
    available fit set."""
    def __init__(self, k=50): self.k = k; self._ref = None; self._k = k
    @staticmethod
    def _normalize(feats):
        x = _np(feats)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    def fit(self, feats, logits):
        # Keep the L2-normalised reference; the GPU-streaming knn_score handles
        # it without resident-whole uploads. k clamped to available samples.
        self._ref = self._normalize(feats).astype(np.float32)
        self._k = max(1, min(self.k, self._ref.shape[0]))
    def score(self, feats, logits):
        # Distance to the k-th nearest neighbour (Sun et al. 2022), GPU-chunked.
        return knn_score(self._ref, self._k, self._normalize(feats),
                         _device(), reduce="kth")

class DetectorReAct:
    """ReAct (Sun et al., NeurIPS 2021): rectify (clip) the penultimate
    activations at c = the p-th percentile of ID activations (paper default
    p=90), then recompute the logits through the ORIGINAL head,
    f_ReAct = W @ min(h, c) + b, and apply the Energy score. (The earlier
    version clipped features and then ran Mahalanobis — neither the energy score
    nor the logit recomputation of the actual method.) Falls back to plain
    Energy on the stored logits when the ResNet head cannot be matched, e.g. the
    synthetic unit-test features."""
    def __init__(self, p=90):
        self.p = p
        self.c = None
        self.head = None
    def fit(self, feats, logits):
        f = feats.numpy()
        self.c = np.percentile(f, self.p)
        self.head = _resnet_head(f.shape[1])
    def score(self, feats, logits):
        if self.head is None:
            return _energy(logits)  # fallback: Energy on the stored logits
        device = _device()
        W, b = self.head
        Wt = torch.from_numpy(np.ascontiguousarray(W, np.float32)).to(device)
        bt = torch.from_numpy(np.ascontiguousarray(b, np.float32)).to(device)
        c_clip = float(self.c)
        def fn(c):
            rec = torch.clamp(c, max=c_clip) @ Wt.T + bt   # recomputed logits
            return -torch.logsumexp(rec, dim=1)            # Energy; higher = more OOD
        return _chunk_feats(fn, feats, device, Wt.shape[0])

class DetectorViM:
    """Faithful ViM (Wang et al. 2022, "ViM: Out-Of-Distribution with
    Virtual-logit Matching"):

      1. Re-origin features at the classifier station point  o = -pinv(W) @ b.
      2. Residual (minor) subspace = the small-eigenvalue eigenvectors of the
         o-centred second moment; keep all but the top-D principal directions.
      3. Virtual logit = alpha * ||residual||, where alpha scale-matches the
         residual norm to the real logits (mean max-logit / mean residual norm)
         so the two are commensurate before combining with energy.

    The classifier head (W, b) is the ImageNet-pretrained ResNet-18 ``fc`` —
    extract_ann_baselines.py keeps that head untouched (only conv1 is
    re-initialised), so the logits in every .pt were produced by exactly this
    fc. When the feature dimensionality does not match the ResNet head (e.g. the
    synthetic unit tests), we fall back to mean-centring so the detector still
    yields a finite, correctly-shaped score.
    """
    def __init__(self, D=256):
        self.D = D
        self.NS = None
        self.o = None
        self.alpha = 1.0

    def fit(self, feats, logits):
        f = _np(feats)
        lg = _np(logits)
        d = f.shape[1]

        head = _resnet_head(d)
        if head is not None:
            W, b = head
            self.o = (-np.linalg.pinv(W) @ b).astype(np.float32)   # small, CPU
        else:
            self.o = f.mean(axis=0).astype(np.float32)

        # The O(n*d^2) second moment + eigendecomposition run on the GPU.
        device = device_for("ViM fit (covariance + eigh)")
        Xt = torch.from_numpy(np.ascontiguousarray(f - self.o, np.float32)).to(device)
        cov = (Xt.T @ Xt) / Xt.shape[0]                   # about o, not re-centred
        _, eigvecs = torch.linalg.eigh(cov)               # ascending eigenvalues
        D = min(self.D, max(1, d - 1))                    # keep >=1 residual dim
        NS = eigvecs[:, : d - D]                          # minor (residual) subspace
        self.NS = NS.cpu().numpy().astype(np.float32)
        vlogit_train = torch.norm(Xt @ NS, dim=1)
        denom = float(vlogit_train.mean())
        self.alpha = float(lg.max(axis=1).mean() / denom) if denom > 0 else 1.0

    def score(self, feats, logits):
        if self.NS is None:
            return np.zeros(feats.shape[0])
        device = _device()
        NS = torch.from_numpy(np.ascontiguousarray(self.NS, np.float32)).to(device)
        o = torch.from_numpy(np.ascontiguousarray(self.o, np.float32)).to(device)
        a = float(self.alpha)
        def fn(c):
            return a * torch.norm((c - o) @ NS, dim=1)
        vlogit = _chunk_feats(fn, feats, device, NS.shape[1])
        energy = logsumexp(_np(logits), axis=1)        # cheap, logit-width only
        return vlogit - energy  # higher = more OOD

class DetectorDICE:
    """DICE (Sun & Li, ECCV 2022): directed sparsification of the classifier.
    Rank weights by their contribution V = W (.) E_ID[h] (element-wise weight x
    mean ID feature), keep only the top (1 - p) fraction of entries across the
    whole weight matrix, zero the rest, then recompute logits through the
    sparsified head f_DICE = (M (.) W) @ h + b and apply Energy. p=0.7 follows
    the paper's ImageNet setting (the head here is ImageNet-pretrained). (The
    earlier version masked per-sample features by percentile and ran
    Mahalanobis — not the weight sparsification or energy score of the method.)
    The mask is built from TRAINING statistics, not per-sample. Falls back to
    plain Energy when the head cannot be matched."""
    def __init__(self, p=0.7):
        self.p = p
        self.head = None
        self.mask = None
    def fit(self, feats, logits):
        f = feats.numpy()
        self.head = _resnet_head(f.shape[1])
        if self.head is None:
            return
        W, _ = self.head
        contrib = W * f.mean(axis=0)[None, :]        # V = W (.) E[h]
        thresh = np.quantile(contrib, self.p)        # drop the bottom p fraction
        self.mask = (contrib >= thresh).astype(np.float32)
    def score(self, feats, logits):
        if self.head is None or self.mask is None:
            return _energy(logits)  # fallback: Energy on the stored logits
        device = _device()
        W, b = self.head
        MW = torch.from_numpy(np.ascontiguousarray(self.mask * W, np.float32)).to(device)
        bt = torch.from_numpy(np.ascontiguousarray(b, np.float32)).to(device)
        def fn(c):
            rec = c @ MW.T + bt                    # sparsified-head logits
            return -torch.logsumexp(rec, dim=1)    # Energy; higher = more OOD
        return _chunk_feats(fn, feats, device, MW.shape[0])

class DetectorGradNorm:
    """GradNorm (Huang, Geng & Li, NeurIPS 2021). OOD score is the L1 norm of
    the gradient of KL(uniform || softmax(f/T)) w.r.t. the last linear layer's
    weights. The paper shows this factorises in closed form (Eq. 4):

        ||grad||_1 = (1 / (C*T)) * ||h||_1 * sum_j |1 - C * softmax(f/T)_j|

    so it needs only the penultimate feature h and the logits f — no backprop.
    In-distribution inputs produce LARGER gradient norms, so the OOD score is
    the negated norm (higher => more OOD). (The earlier version used the L1 norm
    of the logits, which is unrelated to the gradient.) T=1 is the paper
    default."""
    def __init__(self, T=1.0): self.T = T
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        lg = _np(logits) / self.T
        C = lg.shape[1]
        p = np.exp(lg - logsumexp(lg, axis=1, keepdims=True))   # softmax(f/T)
        out_term = np.abs(1.0 - C * p).sum(axis=1)              # sum_j|1 - C p_j|
        device = _device()
        def fn(c):
            return torch.abs(c).sum(dim=1)                      # ||h||_1
        feat_term = _chunk_feats(fn, feats, device, 1)
        gradnorm = (feat_term * out_term) / (C * self.T)
        return -gradnorm

# ───────────────────── recent SOTA post-hoc detectors (2022–2024) ─────────────
# All implementable from the offline cache (penultimate features + the untouched
# ImageNet ResNet-18 head's logits) with NO retraining, NO input gradients, and
# NO second model pass — only the head recomputation already used by ReAct/DICE.
# Each cites its paper; head-dependent ones fall back to Energy when the head can
# not be matched (e.g. the synthetic unit-test features), exactly like ReAct/DICE.

def _nnguide_guidance(Xq, Zref, conf_ref, k, device):
    """Guidance term of NNGuide (Park et al., ICCV 2023), Eq. for G(x):
    G(x) = (1/k) * sum_{i=1..k} s_(i) * sim(z_(i), z), where the k neighbours are
    the top-k by the CONFIDENCE-WEIGHTED similarity s_i * cos(z_i, z) (not cosine
    alone), s_i is the bank sample's base confidence, and sim is cosine on
    L2-normalised features. The training bank is streamed in blocks (never fully
    resident); query rows are chunked and halve on CUDA OOM — same memory
    discipline as `knn_score`."""
    Xt = torch.from_numpy(np.ascontiguousarray(Xq, np.float32))
    Zc = torch.from_numpy(np.ascontiguousarray(Zref, np.float32))
    cc = torch.from_numpy(np.ascontiguousarray(conf_ref, np.float32))
    n_ref, N = Zc.shape[0], Xt.shape[0]
    k = max(1, min(k, n_ref))
    ref_block = 16384
    qchunk = query_chunk_rows(min(n_ref, ref_block), device)
    out, i = [], 0
    while i < N:
        end = min(i + qchunk, N)
        try:
            q = Xt[i:end].to(device)
            best_val = None
            for j in range(0, n_ref, ref_block):
                rb = Zc[j:j + ref_block].to(device)
                cb = cc[j:j + ref_block].to(device).unsqueeze(0)         # (1,b)
                val = (q @ rb.T) * cb                                    # s_i * cos(z_i,z)
                if best_val is not None:
                    val = torch.cat([best_val, val], dim=1)
                kk = min(k, val.shape[1])
                best_val, _ = torch.topk(val, kk, dim=1)
                del rb
            out.append(best_val.mean(dim=1).cpu())                       # (1/k) sum top-k
            del q, best_val
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if qchunk <= 128:
                raise
            qchunk = max(128, qchunk // 2)
            continue
        i = end
    return torch.cat(out).numpy() if out else np.empty((0,), np.float32)


class DetectorMaxLogit:
    """MaxLogit (Hendrycks et al., ICML 2022, "Scaling Out-of-Distribution
    Detection for Real-World Settings"). Score = -max_j f_j — an unnormalised,
    softmax-free alternative to MSP that avoids the saturation MSP suffers when
    the class count is large. Logit-only, training-free."""
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        device = _device()
        def fn(c):
            return -c.max(dim=1).values
        L = np.ascontiguousarray(_np(logits), np.float32)
        return chunked_apply(fn, L, device, n_ref=L.shape[1])


class DetectorGEN:
    """GEN (Liu et al., CVPR 2023, "GEN: Pushing the Limits of Softmax-Based
    Out-of-Distribution Detection"). Generalised entropy of the softmax over the
    top-M classes: G = sum_{i in topM} p_i^gamma (1 - p_i)^gamma, with gamma=0.1
    and M=100 (clamped to the number of classes). ID predictions are peaked ->
    small G; OOD predictions are diffuse -> large G. Logit-only, training-free."""
    def __init__(self, gamma=0.1, M=100):
        self.gamma = gamma
        self.M = M
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        device, g, M = _device(), self.gamma, self.M
        def fn(c):
            p = torch.softmax(c, dim=1)
            m = min(M, p.shape[1])
            top = torch.topk(p, m, dim=1).values
            return (top.pow(g) * (1.0 - top).pow(g)).sum(dim=1)
        L = np.ascontiguousarray(_np(logits), np.float32)
        return chunked_apply(fn, L, device, n_ref=L.shape[1])


class DetectorASH:
    """ASH (Djurisic et al., ICLR 2023, "Extremely Simple Activation Shaping for
    Out-of-Distribution Detection"), the ASH-S variant. Per sample: prune the
    bottom p% of the (post-ReLU, non-negative) penultimate activations, then
    rescale the survivors by exp(s1/s2) with s1 = sum of all activations and
    s2 = sum of the surviving ones; recompute logits through the original head and
    apply Energy. p=90 is the paper default. Needs the head to recompute logits;
    falls back to plain Energy on the stored logits otherwise."""
    def __init__(self, p=90):
        self.p = p
        self.head = None
    def fit(self, feats, logits):
        self.head = _resnet_head(_np(feats).shape[1])
    def score(self, feats, logits):
        if self.head is None:
            return _energy(logits)
        device, q = _device(), self.p / 100.0
        W, b = self.head
        Wt = torch.from_numpy(np.ascontiguousarray(W, np.float32)).to(device)
        bt = torch.from_numpy(np.ascontiguousarray(b, np.float32)).to(device)
        def fn(c):
            c = torch.clamp(c, min=0.0)
            thr = torch.quantile(c, q, dim=1, keepdim=True)
            pruned = c * (c >= thr)
            s1 = c.sum(1, keepdim=True)
            s2 = pruned.sum(1, keepdim=True).clamp_min(1e-6)
            scaled = pruned * torch.exp(s1 / s2)
            return -torch.logsumexp(scaled @ Wt.T + bt, dim=1)
        return _chunk_feats(fn, feats, device, Wt.shape[0])


class DetectorSCALE:
    """SCALE (Xu et al., ICLR 2024, "Scaling for Training Time and Post-hoc
    Out-of-distribution Detection Enhancement"), post-hoc variant. Like ASH-S but
    it SCALES the whole feature rather than pruning: h' = h * exp(s1/s2) with
    s1 = sum(h) and s2 = sum of the top (100-p)% activations; recompute logits and
    apply Energy. p=85 default. Needs the head; falls back to Energy otherwise."""
    def __init__(self, p=85):
        self.p = p
        self.head = None
    def fit(self, feats, logits):
        self.head = _resnet_head(_np(feats).shape[1])
    def score(self, feats, logits):
        if self.head is None:
            return _energy(logits)
        device, q = _device(), self.p / 100.0
        W, b = self.head
        Wt = torch.from_numpy(np.ascontiguousarray(W, np.float32)).to(device)
        bt = torch.from_numpy(np.ascontiguousarray(b, np.float32)).to(device)
        def fn(c):
            c = torch.clamp(c, min=0.0)
            thr = torch.quantile(c, q, dim=1, keepdim=True)
            s1 = c.sum(1, keepdim=True)
            s2 = (c * (c >= thr)).sum(1, keepdim=True).clamp_min(1e-6)
            scaled = c * torch.exp(s1 / s2)
            return -torch.logsumexp(scaled @ Wt.T + bt, dim=1)
        return _chunk_feats(fn, feats, device, Wt.shape[0])


class DetectorFDBD:
    """fDBD (Liu & Qin, ICML 2024, "Fast Decision Boundary based Out-of-
    Distribution Detector"). Averages the feature's distance to the decision
    boundaries between the predicted class and every other class,
    (1/(C-1)) sum_{j != yhat} |f_yhat - f_j| / ||w_yhat - w_j||, normalised by the
    feature's distance to the training feature mean ||z - mu_train||. ID features
    sit farther from the boundaries relative to their distance from the ID
    centroid, so the OOD score is the negated ratio. Recomputes logits from the
    untouched head inside the kernel; falls back to Energy without a head."""
    def __init__(self):
        self.head = None
        self.mu = None
    def fit(self, feats, logits):
        self.head = _resnet_head(_np(feats).shape[1])
        self.mu = _np(feats).mean(0).astype(np.float32)
    def score(self, feats, logits):
        if self.head is None:
            return _energy(logits)
        device = _device()
        W, b = self.head
        Wt = torch.from_numpy(np.ascontiguousarray(W, np.float32)).to(device)   # (C,D)
        bt = torch.from_numpy(np.ascontiguousarray(b, np.float32)).to(device)
        wn2 = (Wt * Wt).sum(1)                                                   # (C,)
        mu = torch.from_numpy(np.ascontiguousarray(self.mu, np.float32)).to(device)
        C = Wt.shape[0]
        def fn(c):
            lg = c @ Wt.T + bt                                  # (m,C)
            yhat = lg.argmax(1)                                 # (m,)
            ly = lg.gather(1, yhat[:, None])                    # (m,1)
            wy_n2 = wn2[yhat].unsqueeze(1)                      # (m,1)
            cross = Wt[yhat] @ Wt.T                             # (m,C) = w_yhat . w_j
            pdist = torch.sqrt(torch.clamp(wy_n2 + wn2[None, :] - 2 * cross, min=1e-12))
            bd = (ly - lg).abs() / pdist                        # (m,C); yhat term -> 0
            score_id = bd.sum(1) / max(C - 1, 1)
            znorm = torch.norm(c - mu, dim=1).clamp_min(1e-6)
            return -(score_id / znorm)
        return _chunk_feats(fn, feats, device, C)


class DetectorNECO:
    """NECO (Ammar et al., ICLR 2024, "NECO: NEural Collapse Based Out-of-
    distribution detection"). Standardises the penultimate features
    (StandardScaler: subtract mean, divide by std) and fits PCA on the ID
    training set; the score (Eq. 8) is NECO(x) = ||P h(x)|| / ||h(x)||, the
    fraction of the standardised feature's norm captured by the top-d
    neural-collapse subspace. Because the ResNet-18 penultimate width (512) is
    smaller than the number of classes (1000), the paper multiplies by the max
    logit ("class-based information injection"). ID features concentrate in the
    collapse subspace and are confident -> large; OOD -> small, so the OOD score
    is negated. Uses the stored logits' max (no head needed)."""
    def __init__(self, dim=100):
        self.dim = dim
        self.mu = None
        self.sd = None
        self.V = None
    def fit(self, feats, logits):
        f = _np(feats).astype(np.float32)
        self.mu = f.mean(0)
        self.sd = f.std(0) + 1e-6
        fs = (f - self.mu) / self.sd                       # StandardScaler
        device = device_for("NECO fit (standardise + PCA / eigh)")
        Xt = torch.from_numpy(np.ascontiguousarray(fs, np.float32)).to(device)
        cov = (Xt.T @ Xt) / max(Xt.shape[0], 1)
        _, evecs = torch.linalg.eigh(cov)                 # ascending eigenvalues
        d = min(self.dim, max(1, f.shape[1] - 1))
        self.V = evecs[:, -d:].cpu().numpy().astype(np.float32)   # top-d directions
    def score(self, feats, logits):
        device = _device()
        V = torch.from_numpy(np.ascontiguousarray(self.V, np.float32)).to(device)
        mu = torch.from_numpy(np.ascontiguousarray(self.mu, np.float32)).to(device)
        sd = torch.from_numpy(np.ascontiguousarray(self.sd, np.float32)).to(device)
        def fn(c):
            h = (c - mu) / sd                              # standardised feature
            return torch.norm(h @ V, dim=1) / torch.norm(h, dim=1).clamp_min(1e-6)
        ratio = _chunk_feats(fn, feats, device, V.shape[1])
        maxlogit = _np(logits).max(axis=1)
        return -(ratio * maxlogit)


class DetectorNNGuide:
    """NNGuide (Park et al., ICCV 2023, "Nearest Neighbor Guidance for Out-of-
    Distribution Detection"). Multiplies the network's confidence (energy) by a
    guidance term — the confidence-weighted average cosine similarity to the k
    nearest TRAINING features: guidance(x) = (1/k) sum_{i in kNN_cos(x)}
    conf_i cos(x, x_i). ID inputs are both confident and close to confident
    training samples -> large product; OOD -> small, so the OOD score is negated.
    k=10 default. Features-and-logits only (no head)."""
    def __init__(self, k=10):
        self.k = k
        self.Zref = None
        self.conf = None
    @staticmethod
    def _norm(f):
        x = _np(f).astype(np.float32)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    def fit(self, feats, logits):
        self.Zref = self._norm(feats)
        self.conf = logsumexp(_np(logits), axis=1).astype(np.float32)   # train energy-conf
    def score(self, feats, logits):
        k = max(1, min(self.k, self.Zref.shape[0]))
        guidance = _nnguide_guidance(self._norm(feats), self.Zref, self.conf, k, _device())
        conf = logsumexp(_np(logits), axis=1)
        return -(conf * guidance)


def evaluate_representation(rep_name, rep_dir, limit=None):
    detectors = {
        # classic / established post-hoc baselines
        "MSP": DetectorMSP(),
        "Energy": DetectorEnergy(),
        "ODIN": DetectorODIN(),
        "Mahalanobis": DetectorMahalanobis(),
        "kNN": DetectorKNN(),
        "ReAct": DetectorReAct(),
        "ViM": DetectorViM(),
        "DICE": DetectorDICE(),
        "GradNorm": DetectorGradNorm(),
        # recent SOTA post-hoc detectors (2022–2024)
        "MaxLogit": DetectorMaxLogit(),
        "GEN": DetectorGEN(),
        "ASH": DetectorASH(),
        "SCALE": DetectorSCALE(),
        "fDBD": DetectorFDBD(),
        "NECO": DetectorNECO(),
        "NNGuide": DetectorNNGuide(),
    }
    
    clean_path = rep_dir / "clean.pt"
    if not clean_path.exists():
        print(f"Skipping {rep_name}, clean.pt not found.")
        return []
        
    d = torch.load(clean_path, weights_only=True, map_location="cpu")
    c_feats, c_logits = d["feat"], d["logit"]

    if len(c_feats) < 10:
        print(f"Skipping {rep_name}: only {len(c_feats)} clean samples — too "
              f"few to split into train/eval. These look like legacy "
              f"per-sequence features; re-run extract_ann_baselines.py to get "
              f"per-frame features.")
        return []

    # Same sequence-aware 70/30 split used everywhere else: fit on the train
    # portion, use only the HELD-OUT portion as clean negatives so detectors
    # (especially kNN) never score their own fitting data.
    seq_lens = load_phi_seq_lens("clean")
    fit_feats, eval_feats = split_train_eval(c_feats, seq_lens=seq_lens)
    fit_logits, eval_logits = split_train_eval(c_logits, seq_lens=seq_lens)

    device_for(f"scoring {len(detectors)} ANN baseline detectors ({rep_name})")
    for name, det in detectors.items():
        det.fit(fit_feats, fit_logits)

    clean_scores = {name: det.score(eval_feats, eval_logits) for name, det in detectors.items()}
    
    results = []

    run_files = [f for f in sorted(rep_dir.glob("*.pt"))
                 if not f.name.startswith("_tmp_") and f.stem != "clean"]
    if limit is not None:
        run_files = run_files[:limit]

    for f in tqdm(run_files, desc=f"Evaluating {rep_name} runs"):
        run_name = f.stem
        
        d = torch.load(f, weights_only=True, map_location="cpu")
        t_feats, t_logits = d["feat"], d["logit"]

        # Positives = held-out tail of the run only, matched to the clean
        # negatives' held-out sequences. feat/logit are row-aligned, so cut both
        # at the same sequence-aligned boundary.
        run_cut = split_boundary(len(t_feats), TRAIN_RATIO, load_phi_seq_lens(run_name))
        t_feats, t_logits = t_feats[run_cut:], t_logits[run_cut:]

        parts = run_name.rsplit('_L', 1)
        corruption = parts[0]
        severity = int(parts[1]) if len(parts) > 1 else 0
        
        for name, det in detectors.items():
            t_scores = det.score(t_feats, t_logits)
            
            y_true = np.concatenate([np.zeros(len(clean_scores[name])), np.ones(len(t_scores))])
            y_score = np.concatenate([clean_scores[name], t_scores])
            
            # Guard against degenerate case (only 1 class present)
            if len(np.unique(y_true)) < 2:
                continue
            try:
                auroc = roc_auc_score(y_true, y_score)
                aupr = average_precision_score(y_true, y_score)
                fpr95 = calc_fpr95(y_true, y_score)
            except Exception:
                continue
            
            results.append({
                "model": "ResNet18",
                "representation": rep_name,
                "detector": name,
                "corruption": corruption,
                "severity": severity,
                "auroc": auroc,
                "aupr": aupr,
                "fpr95": fpr95
            })
            
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Score ANN ResNet-18 OOD baselines.")
    parser.add_argument(
        "--ann-dir", type=Path, default=cfg.ANN_DIR,
        help="Directory holding the extracted ANN features, with event_image/ "
             "and voxel_grid/ subdirs (default: cfg.ANN_DIR).",
    )
    parser.add_argument(
        "--output", type=Path, default=cfg.OUTPUT_DIR / "results" / "ann_baselines.csv",
        help="Output CSV path (default: <OUTPUT_DIR>/results/ann_baselines.csv).",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Quick smoke mode: evaluate only the first 2 corrupted runs per "
             "representation instead of all of them.",
    )
    args = parser.parse_args()

    base_dir = args.ann_dir
    limit = 2 if args.test else None

    all_results = []
    for rep in ["event_image", "voxel_grid"]:
        rep_dir = base_dir / rep
        if rep_dir.exists():
            print(f"Evaluating {rep}...")
            res = evaluate_representation(rep, rep_dir, limit=limit)
            all_results.extend(res)

    if all_results:
        df = pd.DataFrame(all_results)
        out_path = args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
