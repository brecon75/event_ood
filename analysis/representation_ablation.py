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

def calc_fpr95(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx]

def get_mahalanobis_scores(train_feat, test_feat):
    try:
        cov = EmpiricalCovariance().fit(train_feat)
        mu = cov.location_
        P = cov.precision_
    except Exception as e:
        print(f"Warning: Covariance fit failed ({e}). Using simple L2.")
        mu = train_feat.mean(axis=0)
        P = np.eye(len(mu))
        
    diff = test_feat - mu
    scores = np.einsum('ni,ij,nj->n', diff, P, diff)
    return scores

def load_all_features():
    phi = {f.stem: torch.load(f, weights_only=True)['phi'].numpy() for f in cfg.PHI_DIR.glob("*.pt")}
    
    ann = {}
    if cfg.ANN_DIR.exists():
        for f in cfg.ANN_DIR.glob("*.pt"):
            ann[f.stem] = {k: v.numpy() for k, v in torch.load(f, weights_only=True).items()}
        
    spike = {}
    if cfg.SPIKE_DIR.exists():
        for f in cfg.SPIKE_DIR.glob("*.pt"):
            spike[f.stem] = {k: v.numpy() for k, v in torch.load(f, weights_only=True).items()}
        
    # we don't necessarily have all representations for all runs if a run failed
    valid_runs = set(phi.keys())
    if ann:
        valid_runs = valid_runs.intersection(ann.keys())
    if spike:
        valid_runs = valid_runs.intersection(spike.keys())
    
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
        # phi is [mean, var, kurtosis] for each layer, concatenated.
        # It's (B, 3*C). We need every 1st out of 3 blocks.
        # Actually in monitor.py it's cat([mu_gap, var_gap, kurt_gap], dim=-1) per layer.
        # We can just reshape or slice.
        c = feats['phi'].shape[1] // 3
        return feats['phi'][:, :c]
    elif rep_name == "membrane_var":
        c = feats['phi'].shape[1] // 3
        return feats['phi'][:, c:2*c]
    elif rep_name == "membrane_kurtosis":
        c = feats['phi'].shape[1] // 3
        return feats['phi'][:, 2*c:]
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
    
    for rep in tqdm(reps, desc="Representation Ablation"):
        train_feat = extract_representation(all_feats['clean'], rep)
        if train_feat is None:
            print(f"Skipping {rep} (not found)")
            continue
            
        # Fit Mahalanobis on clean
        clean_scores = get_mahalanobis_scores(train_feat, train_feat)
        
        for run_name, feats in all_feats.items():
            if run_name == 'clean': continue
            
            test_feat = extract_representation(feats, rep)
            if test_feat is None: continue
            
            try:
                corr_scores = get_mahalanobis_scores(train_feat, test_feat)
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
