import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from scipy.stats import spearmanr
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.representation_ablation import load_all_features, extract_representation, get_mahalanobis_scores
from analysis.evaluate_detectors import SCORERS
from analysis.vmem_utils import split_train_eval, load_phi_seq_lens

def main():
    print("Running severity monotonicity analysis...")
    all_feats = load_all_features()
    
    if 'clean' not in all_feats:
        print("Error: 'clean' run not found.")
        return
        
    out_dir = cfg.DETECTOR_DIR
    
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
    
    clean_seq_lens = load_phi_seq_lens("clean")

    for c_name in tqdm(cfg.CORRUPTIONS, desc="Severity analysis"):
        # Build lists of scores and severities

        # --- 1. All representations with Mahalanobis ---
        for rep in reps:
            train_feat = extract_representation(all_feats['clean'], rep)
            if train_feat is None: continue

            # Sequence-aware split: fit on train, clean baseline = held-out eval
            train_fit, clean_eval = split_train_eval(train_feat, seq_lens=clean_seq_lens)
            clean_scores = get_mahalanobis_scores(train_fit, clean_eval)

            all_scores = list(clean_scores)
            all_severities = [0] * len(clean_scores)

            for sev in cfg.SEVERITIES:
                run_name = f"{c_name}_L{sev}"
                if run_name not in all_feats: continue

                test_feat = extract_representation(all_feats[run_name], rep)
                if test_feat is None: continue

                try:
                    scores = get_mahalanobis_scores(train_fit, test_feat)
                    all_scores.extend(scores)
                    all_severities.extend([sev] * len(scores))
                except Exception as e:
                    print(f"  [!] {run_name}/{rep}: scoring failed ({e})")
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
        if train_feat is None:
            continue
        # The fitted detectors were trained on the clean-train portion
        # (fit_detectors.py); only score the held-out eval portion as the
        # severity-0 baseline so it is not the detectors' own training data.
        _, clean_eval = split_train_eval(train_feat, seq_lens=clean_seq_lens)
        for d_name, d_model in detectors.items():
            if d_name == 'mahalanobis': continue # Already done
            score_fn = SCORERS.get(d_name)
            if score_fn is None:
                print(f"  [!] Unknown detector '{d_name}' — skipping.")
                continue

            clean_scores = score_fn(d_model, clean_eval)
            all_scores = list(clean_scores)
            all_severities = [0] * len(clean_scores)

            for sev in cfg.SEVERITIES:
                run_name = f"{c_name}_L{sev}"
                if run_name not in all_feats: continue
                test_feat = extract_representation(all_feats[run_name], rep)
                if test_feat is None: continue

                scores = score_fn(d_model, test_feat)
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
    res_dir = cfg.OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "severity_metrics.csv", index=False)
    
    # Plotting
    if not df.empty:
        fig_dir = cfg.OUTPUT_DIR / "figures"
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
