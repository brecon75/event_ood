import json
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
            if f.stem == "flow_pca":
                continue  # PCA half of the 'flow' detector, not a detector
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

    # Fitted detectors must be scored on the representation they were fitted
    # on (recorded in split.json by fit_detectors.py) — substituting another
    # representation would feed them wrong-dimensional features.
    det_rep = 'membrane_fused'
    split_path = out_dir / "split.json"
    if split_path.exists():
        try:
            with open(split_path) as fh:
                det_rep = json.load(fh).get("representation", det_rep)
        except Exception as e:
            print(f"  [!] Could not read split.json ({e}); assuming '{det_rep}'.")

    # Pre-compute each detector's clean-baseline scores ONCE. The fitted
    # detectors were trained on the clean-train portion (fit_detectors.py);
    # only the held-out eval portion is scored as the severity-0 baseline.
    # A failing detector is skipped instead of aborting the whole analysis.
    det_clean_scores = {}
    if detectors:
        det_train_feat = extract_representation(all_feats['clean'], det_rep)
        if det_train_feat is None:
            print(f"  [!] Clean '{det_rep}' representation missing — "
                  f"skipping fitted-detector severity analysis.")
        else:
            _, det_clean_eval = split_train_eval(det_train_feat,
                                                 seq_lens=clean_seq_lens)
            for d_name, d_model in detectors.items():
                if d_name == 'mahalanobis':
                    continue  # covered by the per-representation loop below
                score_fn = SCORERS.get(d_name)
                if score_fn is None:
                    print(f"  [!] Unknown detector '{d_name}' — skipping.")
                    continue
                try:
                    det_clean_scores[d_name] = score_fn(d_model, det_clean_eval)
                except Exception as e:
                    print(f"  [!] {d_name}: clean scoring failed ({e}) — skipping.")

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
                
        # --- 2. Fitted-detector representation with all fitted detectors ---
        for d_name, clean_scores in det_clean_scores.items():
            d_model = detectors[d_name]
            score_fn = SCORERS[d_name]
            all_scores = list(clean_scores)
            all_severities = [0] * len(clean_scores)

            for sev in cfg.SEVERITIES:
                run_name = f"{c_name}_L{sev}"
                if run_name not in all_feats: continue
                test_feat = extract_representation(all_feats[run_name], det_rep)
                if test_feat is None: continue

                try:
                    scores = score_fn(d_model, test_feat)
                except Exception as e:
                    print(f"  [!] {run_name}/{d_name}: scoring failed ({e})")
                    continue
                all_scores.extend(scores)
                all_severities.extend([sev] * len(scores))

            if len(set(all_severities)) > 1:
                rho, _ = spearmanr(all_scores, all_severities)
                results.append({
                    "model": "hybrid",
                    "representation": det_rep,
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
