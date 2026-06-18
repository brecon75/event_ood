import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from collections import OrderedDict
from collections.abc import Mapping
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    slice_phi_stat, split_train_eval, load_phi_seq_lens, chunked_apply, device_for,
)
from analysis.gpu_fit import empirical_precision

def calc_fpr95(y_true, y_score):
    # Guard single-class input the way vmem_utils.auroc_fpr95 does: roc_curve
    # raises ("Only one class present") otherwise, and callers that wrap this
    # in a bare `except: continue` would silently drop the metric.
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr >= 0.95))
    return float(fpr[idx])

def fit_mahalanobis(train_feat):
    """Fit mean + precision once and return a score(test_feat) closure.

    The covariance fit/inversion is O(d^3); callers that score many runs
    against the same train split must fit once and reuse the closure. Scoring
    runs chunked on the GPU (the closure captures only mu/P, not train_feat), so
    Stages 9/10/11 — which score every run on the full-dimensional split — are
    no longer single-core numpy einsum bound.
    """
    try:
        cov = empirical_precision(train_feat, op="repr-ablation Mahalanobis fit")
        mu = cov.location_
        P = cov.precision_
    except Exception as e:
        print(f"Warning: Covariance fit failed ({e}). Using simple L2.")
        mu = train_feat.mean(axis=0)
        P = np.eye(len(mu))

    device = device_for("repr-ablation Mahalanobis scoring", verbose=False)
    mu_t = torch.from_numpy(np.ascontiguousarray(mu, dtype=np.float32)).to(device)
    P_t = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float32)).to(device)

    def score(test_feat):
        def fn(chunk):
            diff = chunk - mu_t
            return ((diff @ P_t) * diff).sum(dim=1)
        return chunked_apply(fn, np.ascontiguousarray(test_feat, dtype=np.float32),
                             device, n_ref=P_t.shape[0])
    return score

def get_mahalanobis_scores(train_feat, test_feat):
    return fit_mahalanobis(train_feat)(test_feat)

class LazyFeatures(Mapping):
    """Lazy, memory-bounded drop-in for the old eager feature dict.

    The eager version loaded EVERY run's phi (+ann/spike/fused) into RAM at
    once: at full scale that is ~31 runs x ~3 GB ≈ 95 GB resident, which OOMs a
    cluster node. This loads each run's features from disk ON DEMAND and keeps
    only a few resident: 'clean' is pinned (every stage reuses it as the
    reference) and other runs ride a small LRU, so a single-pass stage holds
    ~1-2 runs at a time.

    Implements the full Mapping interface, so it is a drop-in for code using
    `'x' in feats`, `feats['x']`, and `for k, v in feats.items()`. NOTE: do NOT
    wrap `.items()` in `list(...)` — that re-materialises every run and defeats
    the bound (callers iterate the view directly)."""

    def __init__(self, cache_size: int = 2, verbose: bool = True):
        self._phi_files = {f.stem: f for f in cfg.PHI_DIR.glob("*.pt")}
        self._ann_files = ({f.stem: f for f in cfg.ANN_DIR.glob("*.pt")}
                           if cfg.ANN_DIR.exists() else {})
        self._spike_files = ({f.stem: f for f in cfg.SPIKE_DIR.glob("*.pt")}
                             if cfg.SPIKE_DIR.exists() else {})
        fused_dir = cfg.OUTPUT_DIR / "features/fused"
        self._fused_files = ({f.stem: f for f in fused_dir.glob("*.pt")}
                             if fused_dir.exists() else {})
        self._cache_size = max(1, cache_size)
        self._verbose = verbose
        self._cache = OrderedDict()   # run -> feats (LRU; 'clean' pinned apart)
        self._clean = None

        # One-time missing-feature warning (parity with the eager version).
        runs = set(self._phi_files)
        for label, files in (("ANN", self._ann_files), ("spike", self._spike_files)):
            if files:
                missing = sorted(runs - set(files))
                if missing:
                    print(f"Warning: {len(missing)} run(s) have phi but no {label} "
                          f"features ({', '.join(missing[:5])}"
                          f"{', ...' if len(missing) > 5 else ''}); "
                          f"{label}-based representations will skip them.")

    def _build(self, run):
        if self._verbose:
            print(f"  [load] {run}", flush=True)
        feats = {
            'phi': torch.load(self._phi_files[run], weights_only=True)['phi'].numpy(),
            'ann': {}, 'spike': {},
        }
        if run in self._ann_files:
            feats['ann'] = {k: v.numpy()
                            for k, v in torch.load(self._ann_files[run], weights_only=True).items()
                            if isinstance(v, torch.Tensor)}
        if run in self._spike_files:
            d = torch.load(self._spike_files[run], weights_only=True)
            sp = {k: v.numpy() for k, v in d.items() if isinstance(v, torch.Tensor)}
            # Legacy spike files contain NaN entropy where p was exactly 0 or 1;
            # the binary-entropy limit there is 0.
            if "spike_entropy" in sp:
                sp["spike_entropy"] = np.nan_to_num(sp["spike_entropy"], nan=0.0)
            feats['spike'] = sp
        if run in self._fused_files:
            d = torch.load(self._fused_files[run], weights_only=True, map_location="cpu")
            feats['fused'] = {k: (v.numpy() if isinstance(v, torch.Tensor) else v)
                              for k, v in d.items() if v is not None}
        return feats

    def __getitem__(self, run):
        if run not in self._phi_files:
            raise KeyError(run)
        if run == 'clean':
            if self._clean is None:
                self._clean = self._build('clean')
            return self._clean
        if run in self._cache:
            self._cache.move_to_end(run)
            return self._cache[run]
        feats = self._build(run)
        self._cache[run] = feats
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)   # evict least-recently-used
        return feats

    def __iter__(self):
        return iter(self._phi_files)

    def __len__(self):
        return len(self._phi_files)

    def __contains__(self, run):
        return run in self._phi_files


def load_all_features(cache_size: int = 2, verbose: bool = True):
    """Return a lazy, memory-bounded feature mapping (see LazyFeatures).

    Was an eager dict that held all runs in RAM (~95 GB at full scale); now
    loads on demand so the fit/eval stages stay within a few runs of memory."""
    return LazyFeatures(cache_size=cache_size, verbose=verbose)

def extract_representation(feats, rep_name):
    """
    Given a dict of {phi, ann, spike} features for a run, return the specific representation as a 2D numpy array.
    """
    if rep_name == "full_membrane":
        return feats['phi']
    elif rep_name == "membrane_mean":
        return slice_phi_stat(feats['phi'], 'mu')
    elif rep_name == "membrane_var":
        return slice_phi_stat(feats['phi'], 'var')
    elif rep_name == "membrane_kurtosis":
        return slice_phi_stat(feats['phi'], 'kurtosis')
    elif rep_name == "ANN":
        # Let's use last_ann_gap
        return feats['ann'].get('last_ann_gap', feats['ann'].get('asab_gap'))
    elif rep_name == "logits":
        # Let's use head_cls_L0_gap
        return feats['ann'].get('head_cls_L0_gap')
    elif rep_name == "spike":
        # Let's use spike_rate
        return feats.get('spike', {}).get('spike_rate')
    elif rep_name == "spike_entropy":
        return feats.get('spike', {}).get('spike_entropy')
    elif rep_name == "membrane_fused" and 'fused' in feats:
        return feats['fused'].get('membrane_fused')
    
    return None

def main():
    print("Running representation ablation...")
    all_feats = load_all_features()
    if 'clean' not in all_feats:
        print("Error: 'clean' run not found. Run extract.py first.")
        return
        
    reps = [
        "logits", "ANN", "spike", "spike_entropy", 
        "membrane_mean", "membrane_var", "membrane_kurtosis", "full_membrane",
        "membrane_fused"
    ]
    
    results = []
    clean_seq_lens = load_phi_seq_lens("clean")

    # Fit every representation's scorer on clean FIRST (clean is loaded once and
    # pinned), then make a SINGLE pass over the corruption runs scoring all reps
    # per run. Loading each run once instead of once-per-representation cuts the
    # phi disk reads by ~9x without holding more than ~1-2 runs in RAM.
    print(f"Fitting {len(reps)} representation scorers on clean...")
    fitted = {}   # rep -> (scorer, clean_scores, train_width)
    for rep in reps:
        train_feat = extract_representation(all_feats['clean'], rep)
        if train_feat is None:
            print(f"Skipping {rep} (not found)")
            continue
        # Sequence-aware 70/30 split: fit on the train portion, score the
        # held-out clean frames as negatives. A random frame-level split would
        # leak near-identical neighboring frames between fit and eval.
        train_feat_fit, clean_test_feat = split_train_eval(
            train_feat, seq_lens=clean_seq_lens)
        scorer = fit_mahalanobis(train_feat_fit)
        fitted[rep] = (scorer, scorer(clean_test_feat), train_feat_fit.shape[1])

    run_names = sorted(name for name in all_feats if name != 'clean')
    for run_name in tqdm(run_names, desc="Representation Ablation"):
        feats = all_feats[run_name]
        parts = run_name.rsplit('_L', 1)
        corruption = parts[0]
        severity = int(parts[1]) if len(parts) > 1 else 0

        for rep, (scorer, clean_scores, train_width) in fitted.items():
            test_feat = extract_representation(feats, rep)
            if test_feat is None:
                continue
            if test_feat.shape[1] != train_width:
                continue   # representation width differs for this run — skip

            corr_scores = scorer(test_feat)

            y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(corr_scores))])
            y_score = np.concatenate([clean_scores, corr_scores])

            if len(np.unique(y_true)) < 2:
                continue
            try:
                auroc = roc_auc_score(y_true, y_score)
                aupr = average_precision_score(y_true, y_score)
                fpr95 = calc_fpr95(y_true, y_score)
            except Exception:
                continue

            results.append({
                "model": "hybrid",
                "representation": rep,
                "detector": "mahalanobis",
                "corruption": corruption,
                "severity": severity,
                "auroc": auroc,
                "aupr": aupr,
                "fpr95": fpr95
            })
            
    df = pd.DataFrame(results)
    out_dir = cfg.OUTPUT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "representation_metrics.csv", index=False)
    
    # Generate Heatmap
    if not df.empty:
        plt.figure(figsize=(10, 8))
        pivot_df = df.pivot_table(index="representation", columns="severity", values="auroc", aggfunc='mean')
        sns.heatmap(pivot_df, annot=True, cmap="YlOrRd", fmt=".3f")
        plt.title("AUROC by Representation and Severity (Averaged across Corruptions)")
        plt.tight_layout()
        
        fig_dir = cfg.OUTPUT_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_dir / "representation_heatmap.pdf")
        plt.close()
        
    print(f"Representation ablation complete. Results saved to {out_dir / 'representation_metrics.csv'} and figures/representation_heatmap.pdf")

if __name__ == "__main__":
    main()
