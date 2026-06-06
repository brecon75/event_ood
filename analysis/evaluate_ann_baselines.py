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

def calc_fpr95(y_true, y_score):
    if len(np.unique(y_true)) < 2: return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.searchsorted(tpr, 0.95)
    return fpr[min(idx, len(fpr) - 1)]

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
    # Simplified ODIN without input perturbation since we evaluate offline
    def __init__(self, T=1000.0): self.T = T
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        probs = torch.softmax(logits / self.T, dim=1)
        msp = probs.max(dim=1).values
        return -msp.numpy()

class DetectorMahalanobis:
    def __init__(self): self.mu = None; self.P = None
    def fit(self, feats, logits):
        try:
            cov = LedoitWolf().fit(feats.numpy())
            self.mu = cov.location_
            self.P = cov.precision_
        except:
            self.mu = feats.numpy().mean(0)
            self.P = np.eye(feats.shape[1])
    def score(self, feats, logits):
        d = feats.numpy() - self.mu
        return np.einsum("ni,ij,nj->n", d, self.P, d)

class DetectorKNN:
    def __init__(self, k=5): self.k = k; self.nn = None
    def fit(self, feats, logits):
        k = min(self.k, feats.shape[0])  # clamp k to available samples
        self.nn = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(feats.numpy())
    def score(self, feats, logits):
        dists, _ = self.nn.kneighbors(feats.numpy())
        return dists.mean(axis=1)

class DetectorReAct:
    def __init__(self, p=0.9):
        self.p = p
        self.c = None
    def fit(self, feats, logits):
        self.c = np.percentile(feats.numpy(), self.p * 100)
    def score(self, feats, logits):
        # ReAct truncates features. But we can't easily re-compute logits without the fc layer.
        # So we just evaluate energy on the clipped features using pseudo-weights?
        # Actually, since we only have penultimate, ReAct usually applies clipping before the classification head.
        # We don't have the head weights saved in extract! 
        # We can approximate by just clipping the features and using Mahalanobis.
        f_clip = np.clip(feats.numpy(), a_min=None, a_max=self.c)
        try:
            cov = LedoitWolf().fit(f_clip)
            mu, P = cov.location_, cov.precision_
        except:
            mu, P = f_clip.mean(0), np.eye(f_clip.shape[1])
        d = f_clip - mu
        return np.einsum("ni,ij,nj->n", d, P, d)

class DetectorViM:
    def __init__(self):
        self.vh = None
    def fit(self, feats, logits):
        # Simplified ViM: Principal space of features + logits
        f = feats.numpy() - feats.numpy().mean(axis=0)
        u, s, vh = np.linalg.svd(f, full_matrices=False)
        n_comp = min(50, vh.shape[0])  # clamp to available components
        self.vh = vh[:n_comp]
    def score(self, feats, logits):
        if self.vh is None:
            return np.zeros(feats.shape[0])
        f = feats.numpy()
        proj = f @ self.vh.T @ self.vh
        res = f - proj
        norm = np.linalg.norm(res, axis=1)
        # Combine with Energy
        energy = logsumexp(logits.numpy(), axis=1)
        return norm - energy

class DetectorDICE:
    def __init__(self): pass
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        # DICE sparsifies weights. We don't have weights here. We will just use sparsified features + Mahalanobis.
        f = feats.numpy()
        mask = f > np.percentile(f, 90, axis=1, keepdims=True)
        f_sp = f * mask
        try:
            cov = LedoitWolf().fit(f_sp)
            mu, P = cov.location_, cov.precision_
        except:
            mu, P = f_sp.mean(0), np.eye(f_sp.shape[1])
        d = f_sp - mu
        return np.einsum("ni,ij,nj->n", d, P, d)

class DetectorGradNorm:
    def __init__(self): pass
    def fit(self, feats, logits): pass
    def score(self, feats, logits):
        # Approximation of GradNorm using L1 norm of logits
        return -np.linalg.norm(logits.numpy(), ord=1, axis=1)

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
    
    for name, det in detectors.items():
        det.fit(c_feats, c_logits)
        
    clean_scores = {name: det.score(c_feats, c_logits) for name, det in detectors.items()}
    
    results = []
    
    for f in tqdm(list(rep_dir.glob("*.pt")), desc=f"Evaluating {rep_name} runs"):
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
