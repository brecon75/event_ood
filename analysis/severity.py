import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.representation_ablation import load_all_features, extract_representation, get_mahalanobis_scores
from analysis.evaluate_detectors import score_mahalanobis, score_knn, score_gmm, score_ocsvm, score_pca, score_ae

def main():
    print("Running severity monotonicity analysis...")
    all_feats = load_all_features()
    
    if 'clean' not in all_feats:
        print("Error: 'clean' run not found.")
        return
        
    out_dir = Path("outputs/detectors")
    
    detectors = {}
    if out_dir.exists():
        for f in out_dir.glob("*.joblib"):
            detectors[f.stem] = joblib.load(f)
            
    reps = [
        "logits", "ANN", "spike", 
        "membrane_mean", "membrane_var", "membrane_kurtosis", "full_membrane",
        "membrane_fused"
    ]
    
    results = []
    
    # We will test severity monotonicity for:
    # 1. Full membrane with all available fitted detectors
    # 2. All representations with Mahalanobis
    
    for c_name in cfg.CORRUPTIONS:
        print(f"Analyzing corruption: {c_name}")
        # Build lists of scores and severities
        
        # --- 1. All representations with Mahalanobis ---
        for rep in reps:
            train_feat = extract_representation(all_feats['clean'], rep)
            if train_feat is None: continue
            
            clean_scores = get_mahalanobis_scores(train_feat, train_feat)
            
            all_scores = list(clean_scores)
            all_severities = [0] * len(clean_scores)
            
            for sev in cfg.SEVERITIES:
                run_name = f"{c_name}_L{sev}"
                if run_name not in all_feats: continue
                
                test_feat = extract_representation(all_feats[run_name], rep)
                if test_feat is None: continue
                
                try:
                    scores = get_mahalanobis_scores(train_feat, test_feat)
                    all_scores.extend(scores)
                    all_severities.extend([sev] * len(scores))
                except:
                    continue
                    
            if len(set(all_severities)) > 1:
                rho, _ = spearmanr(all_scores, all_severities)
                results.append({
                    "model": "hybrid",
                    "representation": rep,
                    "detector": "mahalanobis",
                    "corruption": c_name,
                    "rho": rho
                })
                
        # --- 2. Membrane Fused with all fitted detectors ---
        rep = 'membrane_fused'
        train_feat = extract_representation(all_feats['clean'], rep)
        if train_feat is None:
            rep = 'full_membrane'
            train_feat = extract_representation(all_feats['clean'], rep)
        for d_name, d_model in detectors.items():
            if d_name == 'mahalanobis': continue # Already done
            
            def get_scores(model, X):
                if d_name == 'knn': return score_knn(model, X)
                if d_name == 'gmm': return score_gmm(model, X)
                if d_name == 'ocsvm': return score_ocsvm(model, X)
                if d_name == 'pca': return score_pca(model, X)
                if d_name == 'ae': return score_ae(model, X)
                return score_mahalanobis(model, X)
                
            clean_scores = get_scores(d_model, train_feat)
            all_scores = list(clean_scores)
            all_severities = [0] * len(clean_scores)
            
            for sev in cfg.SEVERITIES:
                run_name = f"{c_name}_L{sev}"
                if run_name not in all_feats: continue
                test_feat = extract_representation(all_feats[run_name], rep)
                if test_feat is None: continue
                
                scores = get_scores(d_model, test_feat)
                all_scores.extend(scores)
                all_severities.extend([sev] * len(scores))
                
            if len(set(all_severities)) > 1:
                rho, _ = spearmanr(all_scores, all_severities)
                results.append({
                    "model": "hybrid",
                    "representation": rep,
                    "detector": d_name,
                    "corruption": c_name,
                    "rho": rho
                })
                
    df = pd.DataFrame(results)
    res_dir = Path("results")
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "severity_metrics.csv", index=False)
    
    # Plotting
    if not df.empty:
        fig_dir = Path("figures")
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df[df['detector'] == 'mahalanobis'], x="representation", y="rho", hue="corruption")
        plt.title("Severity Monotonicity (Spearman ρ) by Representation (Detector: Mahalanobis)")
        plt.xticks(rotation=45)
        plt.ylabel("Spearman ρ (Score vs Severity)")
        plt.tight_layout()
        plt.savefig(fig_dir / "severity_curves.pdf")
        plt.close()
        
    print(f"Severity monotonicity complete. Results saved to {res_dir / 'severity_metrics.csv'}")

if __name__ == "__main__":
    main()
