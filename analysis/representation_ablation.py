import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import slice_phi_stat, split_train_eval, load_phi_seq_lens

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
    against the same train split must fit once and reuse the closure.
    """
    try:
        cov = EmpiricalCovariance().fit(train_feat)
        mu = cov.location_
        P = cov.precision_
    except Exception as e:
        print(f"Warning: Covariance fit failed ({e}). Using simple L2.")
        mu = train_feat.mean(axis=0)
        P = np.eye(len(mu))

    def score(test_feat):
        diff = test_feat - mu
        return np.einsum('ni,ij,nj->n', diff, P, diff)
    return score

def get_mahalanobis_scores(train_feat, test_feat):
    return fit_mahalanobis(train_feat)(test_feat)

def load_all_features():
    phi = {f.stem: torch.load(f, weights_only=True)['phi'].numpy() for f in cfg.PHI_DIR.glob("*.pt")}
    
    ann = {}
    if cfg.ANN_DIR.exists():
        for f in cfg.ANN_DIR.glob("*.pt"):
            ann[f.stem] = {k: v.numpy()
                           for k, v in torch.load(f, weights_only=True).items()
                           if isinstance(v, torch.Tensor)}
        
    spike = {}
    if cfg.SPIKE_DIR.exists():
        for f in cfg.SPIKE_DIR.glob("*.pt"):
            d = torch.load(f, weights_only=True)
            spike[f.stem] = {k: v.numpy() for k, v in d.items()
                             if isinstance(v, torch.Tensor)}
            # Legacy spike files (extracted before monitor.py clamped the
            # spike rate) contain NaN entropy where p was exactly 0 or 1;
            # the binary-entropy limit there is 0.
            if "spike_entropy" in spike[f.stem]:
                spike[f.stem]["spike_entropy"] = np.nan_to_num(
                    spike[f.stem]["spike_entropy"], nan=0.0)
        
    # We don't necessarily have all representations for all runs if a run
    # failed. Keep every run that has phi: extract_representation() returns
    # None for missing parts, and callers skip None — dropping the whole run
    # here would also exclude it from membrane-only analyses that never need
    # ANN/spike features.
    valid_runs = set(phi.keys())
    for name, d in (("ANN", ann), ("spike", spike)):
        if d:
            missing = sorted(valid_runs - set(d.keys()))
            if missing:
                print(f"Warning: {len(missing)} run(s) have phi but no {name} "
                      f"features ({', '.join(missing[:5])}"
                      f"{', ...' if len(missing) > 5 else ''}); "
                      f"{name}-based representations will skip them.")
    
    fused_dir = cfg.OUTPUT_DIR / "features/fused"
    fused = {}
    if fused_dir.exists():
        for f in fused_dir.glob("*.pt"):
            d = torch.load(f, weights_only=True, map_location="cpu")
            fused[f.stem] = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in d.items() if v is not None}
            
    res = {}
    for run in valid_runs:
        res[run] = {
            'phi': phi[run],
            'ann': ann.get(run, {}),
            'spike': spike.get(run, {})
        }
        if run in fused:
            res[run]['fused'] = fused[run]
            
    return res

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

    for rep in tqdm(reps, desc="Representation Ablation"):
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
        clean_scores = scorer(clean_test_feat)

        for run_name, feats in all_feats.items():
            if run_name == 'clean': continue

            test_feat = extract_representation(feats, rep)
            if test_feat is None: continue

            try:
                corr_scores = scorer(test_feat)
            except ValueError:
                # Shape mismatch
                continue
                
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
            
            parts = run_name.rsplit('_L', 1)
            corruption = parts[0]
            severity = int(parts[1]) if len(parts) > 1 else 0
            
            res_dict = {
                "model": "hybrid",
                "representation": rep,
                "detector": "mahalanobis",
                "corruption": corruption,
                "severity": severity,
                "auroc": auroc,
                "aupr": aupr,
                "fpr95": fpr95
            }
            results.append(res_dict)
            
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
