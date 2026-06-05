import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.representation_ablation import load_all_features, extract_representation

def main():
    print("Running cross-corruption generalization analysis...")
    all_feats = load_all_features()
    
    if 'clean' not in all_feats:
        print("Error: 'clean' run not found.")
        return
        
    rep = 'membrane_fused'
    X_clean = extract_representation(all_feats['clean'], rep)
    if X_clean is None:
        rep = 'full_membrane'
        X_clean = extract_representation(all_feats['clean'], rep)
    
    if X_clean is None:
        print(f"Error: Representation '{rep}' not found in clean data.")
        return
        
    # We aggregate all severities of hot_pixel to train the detector
    X_train_corr = []
    for sev in cfg.SEVERITIES:
        run_name = f"hot_pixel_L{sev}"
        if run_name in all_feats:
            feats = extract_representation(all_feats[run_name], rep)
            if feats is not None:
                X_train_corr.append(feats)
                
    if not X_train_corr:
        print("Error: 'hot_pixel' runs not found for training.")
        return
        
    X_train_corr = np.concatenate(X_train_corr, axis=0)
    
    # Train binary classifier
    X_train = np.concatenate([X_clean, X_train_corr], axis=0)
    y_train = np.concatenate([np.zeros(len(X_clean)), np.ones(len(X_train_corr))])
    
    print("Training binary detector on clean vs hot_pixel...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate generalization on other corruptions
    eval_corruptions = [
        "event_flood", "temporal_jitter", 
        "polarity_flip", "event_rate_shift", "spatial_dropout"
    ]
    
    results = []
    
    for c_name in eval_corruptions:
        print(f"Evaluating on {c_name}...")
        for sev in cfg.SEVERITIES:
            run_name = f"{c_name}_L{sev}"
            if run_name not in all_feats: continue
            
            X_test_corr = extract_representation(all_feats[run_name], rep)
            if X_test_corr is None: continue
            
            X_test = np.concatenate([X_clean, X_test_corr], axis=0)
            y_test = np.concatenate([np.zeros(len(X_clean)), np.ones(len(X_test_corr))])
            
            y_score = clf.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, y_score)
            
            results.append({
                "train_corruption": "hot_pixel",
                "eval_corruption": c_name,
                "severity": sev,
                "auroc": auroc
            })
            
    df = pd.DataFrame(results)
    res_dir = Path("results")
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "cross_corruption.csv", index=False)
    
    print(f"Cross-corruption generalization complete. Results saved to {res_dir / 'cross_corruption.csv'}")

if __name__ == "__main__":
    main()
