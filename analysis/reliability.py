import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, auc

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.representation_ablation import load_all_features, get_mahalanobis_scores, extract_representation

def compute_detection_metric(det_outputs, conf_thresh=0.3):
    """
    Given (N, anchors, 85) detector outputs, compute a reliability metric per sample.
    Here we compute the number of confident detections per frame.
    Returns array of shape (N,)
    """
    if det_outputs is None or det_outputs.ndim < 3:
        # Fallback if outputs aren't saved properly
        return np.zeros(det_outputs.shape[0] if det_outputs is not None else 1)
        
    # obj_conf = det_outputs[..., 4]
    # cls_conf = det_outputs[..., 5:]
    # scores = obj_conf.unsqueeze(-1) * cls_conf
    
    scores = det_outputs[:, :, 4:5] * det_outputs[:, :, 5:]
    max_scores, _ = scores.max(dim=-1)
    
    # Count of confident boxes
    confident_counts = (max_scores > conf_thresh).sum(dim=1).numpy()
    return confident_counts

def calc_aurc(risks, coverages):
    # Sort coverages descending, risks accordingly
    sort_idx = np.argsort(coverages)[::-1]
    sorted_cov = coverages[sort_idx]
    sorted_risk = risks[sort_idx]
    return auc(sorted_cov, sorted_risk)

def main():
    print("Running reliability prediction analysis...")
    all_feats = load_all_features()
    
    if 'clean' not in all_feats:
        print("Error: 'clean' run not found.")
        return
        
    # Load detector outputs
    det_outputs = {}
    for f in cfg.DETECTOR_DIR.glob("*.pt"):
        det_outputs[f.stem] = torch.load(f, weights_only=True)
        
    if 'clean' not in det_outputs:
        print("Error: 'clean' detector outputs not found.")
        return
        
    clean_det_metric = compute_detection_metric(det_outputs['clean'])
    rep = 'membrane_fused'
    train_feat = extract_representation(all_feats['clean'], rep)
    if train_feat is None:
        rep = 'full_membrane'
        train_feat = extract_representation(all_feats['clean'], rep)
    
    results = []
    
    for run_name, feats in all_feats.items():
        if run_name == 'clean': continue
        if run_name not in det_outputs: continue
        
        test_feat = extract_representation(feats, rep)
        corr_det_metric = compute_detection_metric(det_outputs[run_name])
        
        if len(test_feat) != len(corr_det_metric):
            print(f"Skipping {run_name} due to shape mismatch.")
            continue
            
        ood_scores = get_mahalanobis_scores(train_feat, test_feat)
        
        # Degradation: clean metric - corrupt metric
        # Ensure we can pair them (assume sequence alignment if sizes match)
        if len(clean_det_metric) == len(corr_det_metric):
            degradation = clean_det_metric - corr_det_metric
        else:
            # If sequence dropping occurred differently, we cannot do point-to-point.
            # We'll just correlate OOD score with the metric directly instead of degradation.
            degradation = -corr_det_metric # higher degradation = fewer boxes
            
        # Guard: correlation functions require at least 2 samples and non-constant arrays
        if len(ood_scores) < 2 or np.std(ood_scores) == 0 or np.std(degradation) == 0:
            print(f"  Skipping {run_name}: insufficient variance for correlation.")
            continue

        try:
            spearman_rho, _ = spearmanr(ood_scores, degradation)
            pearson_r, _ = pearsonr(ood_scores, degradation)
            r2 = r2_score(degradation, ood_scores)
        except Exception as e:
            print(f"  Skipping {run_name}: correlation failed ({e})")
            continue
        
        # AURC (Area Under the Risk-Coverage Curve)
        # Sort by OOD score (uncertainty). Reject highest OOD scores first.
        sorted_idx = np.argsort(ood_scores) # Ascending OOD score -> low uncertainty first
        
        coverages = []
        risks = []
        n = len(sorted_idx)
        
        # Risk = average degradation of ACCEPTED samples
        for i in range(1, n + 1):
            accepted_idx = sorted_idx[:i]
            coverage = i / n
            risk = degradation[accepted_idx].mean() if degradation[accepted_idx].mean() > 0 else 0
            
            coverages.append(coverage)
            risks.append(risk)
            
        aurc_val = auc(coverages, risks)
        
        parts = run_name.rsplit('_L', 1)
        corruption = parts[0]
        severity = int(parts[1]) if len(parts) > 1 else 0
        
        results.append({
            "corruption": corruption,
            "severity": severity,
            "spearman": spearman_rho,
            "pearson": pearson_r,
            "r2": r2,
            "aurc": aurc_val
        })
        
    df = pd.DataFrame(results)
    out_dir = cfg.OUTPUT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "reliability_metrics.csv", index=False)
    
    # Plot Reliability Curve
    if not df.empty:
        fig_dir = cfg.OUTPUT_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="severity", y="spearman")
        plt.title("Spearman Correlation (OOD Score vs Degradation) by Severity")
        plt.tight_layout()
        plt.savefig(fig_dir / "reliability_curve.pdf")
        plt.close()
        
        # Plot Risk-Coverage for a sample run
        if len(risks) > 0 and len(coverages) > 0:
            plt.figure(figsize=(8, 6))
            plt.plot(coverages, risks, marker='o', markersize=3)
            plt.xlabel("Coverage")
            plt.ylabel("Risk (Mean Degradation)")
            plt.title("Risk-Coverage Curve (Sample)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fig_dir / "risk_coverage.pdf")
            plt.close()
            
    print(f"Reliability complete. Results saved to {out_dir / 'reliability_metrics.csv'}")

if __name__ == "__main__":
    main()
