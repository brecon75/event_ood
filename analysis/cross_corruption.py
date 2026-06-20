import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.representation_ablation import load_all_features, extract_representation
from analysis.vmem_utils import split_train_eval, held_out_eval, load_phi_seq_lens
from analysis.gpu_fit import logreg_fit

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
    # Positive TRAINING data = the TRAIN portion of each hot_pixel run only, so
    # the corrupted-train frames stay disjoint from the held-out tails used for
    # evaluation below (val-only evaluation on both clean and corrupted sides).
    X_train_corr = []
    for sev in cfg.SEVERITIES:
        run_name = f"hot_pixel_L{sev}"
        if run_name in all_feats:
            feats = extract_representation(all_feats[run_name], rep)
            if feats is not None:
                feats_tr, _ = split_train_eval(feats, seq_lens=load_phi_seq_lens(run_name))
                X_train_corr.append(feats_tr)
                
    if not X_train_corr:
        print("Error: 'hot_pixel' runs not found for training.")
        return
        
    X_train_corr = np.concatenate(X_train_corr, axis=0)

    # Split clean data so training and evaluation use disjoint subsets.
    # Sequence-aware contiguous split — random frame splits leak temporally
    # correlated neighboring frames between train and eval.
    X_clean_train, X_clean_eval = split_train_eval(
        X_clean, seq_lens=load_phi_seq_lens("clean"))

    # Train binary classifier
    X_train = np.concatenate([X_clean_train, X_train_corr], axis=0)
    y_train = np.concatenate([np.zeros(len(X_clean_train)), np.ones(len(X_train_corr))])
    
    print("Training binary detector on clean vs hot_pixel...")
    clf = logreg_fit(X_train, y_train, op="cross-corruption logistic regression")
    
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
            # Positives = held-out tail of the eval-corruption run only, matched
            # to the clean negatives' held-out sequences (X_clean_eval).
            X_test_corr = held_out_eval(X_test_corr, seq_lens=load_phi_seq_lens(run_name))

            X_test = np.concatenate([X_clean_eval, X_test_corr], axis=0)
            y_test = np.concatenate([np.zeros(len(X_clean_eval)), np.ones(len(X_test_corr))])
            
            y_score = clf.predict_proba(X_test)[:, 1]
            auroc = roc_auc_score(y_test, y_score)
            
            results.append({
                "train_corruption": "hot_pixel",
                "eval_corruption": c_name,
                "severity": sev,
                "auroc": auroc
            })
            
    df = pd.DataFrame(results)
    res_dir = cfg.OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "cross_corruption.csv", index=False)
    
    print(f"Cross-corruption generalization complete. Results saved to {res_dir / 'cross_corruption.csv'}")

if __name__ == "__main__":
    main()
