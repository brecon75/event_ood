import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.special import logsumexp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import split_train_eval, load_phi_seq_lens
import functools


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
        probs = torch.softmax(logits, dim=1)
        msp = probs.max(dim=1).values
        return -msp.numpy()

class DetectorEnergy:
    def __init__(self, T=1.0): self.T = T
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        energy = self.T * logsumexp(logits.numpy() / self.T, axis=1)
        return -energy

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
        probs = torch.softmax(logits / self.T, dim=1)
        msp = probs.max(dim=1).values
        return -msp.numpy()

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
            cov = LedoitWolf().fit(feats.numpy())
            self.mu = cov.location_
            self.P = cov.precision_
        except Exception:
            self.mu = feats.numpy().mean(0)
            self.P = np.eye(feats.shape[1])
    def score(self, feats, logits):
        d = feats.numpy() - self.mu
        return np.einsum("ni,ij,nj->n", d, self.P, d)

class DetectorKNN:
    """Deep nearest-neighbour OOD (Sun et al., ICML 2022). Two faithful details
    the earlier version dropped: (1) features are L2-normalised before any
    distance is taken (z = h/||h||_2), and (2) the score is the distance to the
    k-th nearest training neighbour r_k(z) = ||z - z_(k)||_2, NOT the mean of
    the k distances. k=50 follows the paper's CIFAR-10 setting, clamped to the
    available fit set."""
    def __init__(self, k=50): self.k = k; self.nn = None
    @staticmethod
    def _normalize(feats):
        x = feats.numpy() if hasattr(feats, "numpy") else np.asarray(feats)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
    def fit(self, feats, logits):
        z = self._normalize(feats)
        k = max(1, min(self.k, z.shape[0]))  # clamp k to available samples
        self.nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(z)
    def score(self, feats, logits):
        dists, _ = self.nn.kneighbors(self._normalize(feats))
        return dists[:, -1]  # distance to the k-th nearest neighbour

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
        W, b = self.head
        h = np.clip(feats.numpy(), a_min=None, a_max=self.c)
        recomputed = h @ W.T + b
        return _energy(recomputed)  # Energy; higher = more OOD

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
        f = feats.numpy()
        lg = logits.numpy()
        d = f.shape[1]

        head = _resnet_head(d)
        if head is not None:
            W, b = head
            self.o = -np.linalg.pinv(W) @ b
        else:
            self.o = f.mean(axis=0)

        X = f - self.o
        # assume_centred second moment about o (not re-centred on X's own mean)
        cov = (X.T @ X) / X.shape[0]
        eigvals, eigvecs = np.linalg.eigh(cov)            # ascending eigenvalues
        D = min(self.D, max(1, d - 1))                    # keep >=1 residual dim
        self.NS = eigvecs[:, : d - D]                     # minor (residual) subspace

        vlogit_train = np.linalg.norm(X @ self.NS, axis=1)
        denom = vlogit_train.mean()
        self.alpha = float(lg.max(axis=1).mean() / denom) if denom > 0 else 1.0

    def score(self, feats, logits):
        if self.NS is None:
            return np.zeros(feats.shape[0])
        f = feats.numpy()
        vlogit = np.linalg.norm((f - self.o) @ self.NS, axis=1) * self.alpha
        energy = logsumexp(logits.numpy(), axis=1)
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
        W, b = self.head
        recomputed = feats.numpy() @ (self.mask * W).T + b
        return _energy(recomputed)  # Energy; higher = more OOD

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
        f = feats.numpy()
        lg = logits.numpy() / self.T
        C = lg.shape[1]
        p = np.exp(lg - logsumexp(lg, axis=1, keepdims=True))   # softmax(f/T)
        out_term = np.abs(1.0 - C * p).sum(axis=1)              # sum_j|1 - C p_j|
        feat_term = np.abs(f).sum(axis=1)                       # ||h||_1
        gradnorm = (feat_term * out_term) / (C * self.T)
        return -gradnorm

def evaluate_representation(rep_name, rep_dir):
    detectors = {
        "MSP": DetectorMSP(),
        "Energy": DetectorEnergy(),
        "ODIN": DetectorODIN(),
        "Mahalanobis": DetectorMahalanobis(),
        "kNN": DetectorKNN(),
        "ReAct": DetectorReAct(),
        "ViM": DetectorViM(),
        "DICE": DetectorDICE(),
        "GradNorm": DetectorGradNorm()
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

    for name, det in detectors.items():
        det.fit(fit_feats, fit_logits)

    clean_scores = {name: det.score(eval_feats, eval_logits) for name, det in detectors.items()}
    
    results = []
    
    for f in tqdm(list(rep_dir.glob("*.pt")), desc=f"Evaluating {rep_name} runs"):
        if f.name.startswith("_tmp_"):
            continue  # leftover partial write from an interrupted extraction
        run_name = f.stem
        if run_name == "clean": continue
        
        d = torch.load(f, weights_only=True, map_location="cpu")
        t_feats, t_logits = d["feat"], d["logit"]
        
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
    base_dir = cfg.ANN_DIR
    
    all_results = []
    for rep in ["event_image", "voxel_grid"]:
        rep_dir = base_dir / rep
        if rep_dir.exists():
            print(f"Evaluating {rep}...")
            res = evaluate_representation(rep, rep_dir)
            all_results.extend(res)
            
    if all_results:
        df = pd.DataFrame(all_results)
        out_dir = cfg.OUTPUT_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / "ann_baselines.csv", index=False)
        print(f"Results saved to {out_dir / 'ann_baselines.csv'}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
