import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import joblib

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_scorers import mahalanobis_scorer
from analysis.vmem_utils import slice_phi_layer


def load_all_features():
    # Load all representations
    feats = {}
    for f in cfg.PHI_DIR.glob("*.pt"):
        run_name = f.stem
        feats[run_name] = {}
        
        # phi
        d = torch.load(f, weights_only=True, map_location="cpu")
        feats[run_name]["membrane_stats"] = d["phi"].numpy()
        
        # margin hist
        mh_path = Path("outputs/features/margin_hist") / f"{run_name}.pt"
        if mh_path.exists():
            feats[run_name]["membrane_margin_hist"] = torch.load(mh_path, weights_only=True, map_location="cpu")["margin_hist"].numpy()
            
        # trajectory latent
        tl_path = Path("outputs/features/trajectory_latent") / f"{run_name}.pt"
        if tl_path.exists():
            feats[run_name]["membrane_temporal_latent"] = torch.load(tl_path, weights_only=True, map_location="cpu")["trajectory_latent"].numpy()
            
        # load temporal features from analysis.legacy_utils
        from analysis.vmem_utils import load_traj_as_temporal_phi
        tf = load_traj_as_temporal_phi(run_name)
        if tf is not None:
            feats[run_name]["membrane_temporal"] = tf
            
    return feats

def align_and_concat(feat_dict, keys):
    """
    Concatenate representations. If shapes (samples) mismatch,
    slice to the minimum length (e.g. 50 samples for traj-based features).
    """
    arrays = []
    min_len = min(feat_dict[k].shape[0] for k in keys if k in feat_dict)
    
    for k in keys:
        if k in feat_dict:
            arrays.append(feat_dict[k][:min_len])
    
    return np.concatenate(arrays, axis=1) if arrays else None


def main():
    print("Running Trainable Layer Fusion and Unified Representation Extraction...")
    
    out_dir = Path("outputs/features/fused")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    feats = load_all_features()
    
    if "clean" not in feats:
        print("Error: clean features not found.")
        return
        
    # Task C: Layer Attention Fusion
    # We will fit weights on clean vs all severities of 'hot_pixel' as a proxy for OOD,
    # or just use uniform weights if strictly "train-clean only".
    # Wait, the prompt says "Fit alpha_i using train-clean only".
    # If we only have train-clean (no OOD), how do we learn alpha_i?
    # Maybe unsupervised? e.g. inverse variance or PCA component?
    # Let's just use uniform weights for now as a fallback, or fit a 1-class SVM and use dual coefficients?
    # The simplest is to just average them if we can't do supervised, or fit PCA and take 1st component weights.
    
    # Actually, the user says "using train-clean only". I will compute the inverse variance of each layer's clean features to weight them, normalized to sum to 1.
    print("Computing Layer Fusion weights (Task C)...")
    clean_phi = feats["clean"]["membrane_stats"]
    
    alpha = []
    for i in range(4):
        layer_feat = slice_phi_layer(clean_phi, i)
        if layer_feat.shape[1] > 0:
            var = layer_feat.var(axis=0).sum()
            alpha.append(1.0 / (var + 1e-8))
        else:
            alpha.append(0)
            
    alpha = np.array(alpha)
    alpha = alpha / alpha.sum()
    print(f"Learned Layer Attention Weights: {alpha}")
    
    # Task E: Per-Layer Mahalanobis Aggregation
    # Learn weights S = sum(w_i * score_i) on train-clean.
    print("Computing Per-Layer Mahalanobis weights (Task E)...")
    layer_scorers = []
    clean_layer_scores = []
    
    for i in range(4):
        layer_feat = slice_phi_layer(clean_phi, i)
        if layer_feat.shape[1] > 0:
            scorer = mahalanobis_scorer(layer_feat)
            layer_scorers.append(scorer)
            score_i = scorer(layer_feat)
            clean_layer_scores.append(score_i)
            
    if clean_layer_scores:
        cls_matrix = np.stack(clean_layer_scores, axis=1) # (N, L)
        # Weight by inverse variance of clean scores
        score_var = cls_matrix.var(axis=0)
        w = 1.0 / (score_var + 1e-8)
        w = w / w.sum()
        print(f"Learned Mahalanobis Score Weights: {w}")
    
    # Process all runs
    for run_name, run_feats in feats.items():
        phi = run_feats["membrane_stats"]
        
        # Apply layer fusion to phi (Task C)
        layer_fused_stats = []
        for i in range(4):
            layer_feat = slice_phi_layer(phi, i)
            if layer_feat.shape[1] > 0:
                layer_fused_stats.append(layer_feat * alpha[i])
        
        if layer_fused_stats:
            layer_fused_stats = np.concatenate(layer_fused_stats, axis=1)
            run_feats["layer_fused_stats"] = layer_fused_stats
            
        # Apply per-layer mahalanobis (Task E)
        run_layer_scores = []
        for i, scorer in enumerate(layer_scorers):
            layer_feat = slice_phi_layer(phi, i)
            if layer_feat.shape[1] > 0:
                run_layer_scores.append(scorer(layer_feat))
        
        if run_layer_scores:
            rls_matrix = np.stack(run_layer_scores, axis=1) # (N, L)
            layer_score_fusion = (rls_matrix * w).sum(axis=1)
            run_feats["layer_score_fusion"] = layer_score_fusion
            
        # Task D: Unified Membrane Representation (membrane_fused)
        keys_to_fuse = ["membrane_stats", "membrane_margin_hist", "membrane_temporal_latent"]
        # In the original vmem, threshold features are part of temporal? "threshold features" -> load_traj_as_temporal_phi gives threshold margin features.
        # So membrane_temporal contains threshold features.
        if "membrane_temporal" in run_feats:
            keys_to_fuse.append("membrane_temporal")
            
        membrane_fused = align_and_concat(run_feats, keys_to_fuse)
        if membrane_fused is not None:
            run_feats["membrane_fused"] = membrane_fused
            
        # Save fused features
        torch.save(
            {
                "layer_fused_stats": run_feats.get("layer_fused_stats"),
                "layer_score_fusion": run_feats.get("layer_score_fusion"),
                "membrane_fused": run_feats.get("membrane_fused")
            },
            out_dir / f"{run_name}.pt"
        )
        
    print("Fusion complete.")

if __name__ == "__main__":
    main()
