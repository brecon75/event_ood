import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_scorers import mahalanobis_scorer
from analysis.vmem_utils import slice_phi_layer, split_train_eval, load_phi_seq_lens


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
        mh_path = cfg.OUTPUT_DIR / "features/margin_hist" / f"{run_name}.pt"
        if mh_path.exists():
            feats[run_name]["membrane_margin_hist"] = torch.load(mh_path, weights_only=True, map_location="cpu")["margin_hist"].numpy()
            
        # trajectory latent
        tl_path = cfg.OUTPUT_DIR / "features/trajectory_latent" / f"{run_name}.pt"
        if tl_path.exists():
            feats[run_name]["membrane_temporal_latent"] = torch.load(tl_path, weights_only=True, map_location="cpu")["trajectory_latent"].numpy()
            
        # load temporal features
        tphi_path = cfg.TEMPORAL_PHI_DIR / f"{run_name}.pt"
        tf = None
        if tphi_path.exists():
            try:
                tf = torch.load(tphi_path, weights_only=True, map_location="cpu")["temporal_phi"].float().numpy()
            except Exception:
                pass
        if tf is None:
            from analysis.vmem_utils import load_traj_as_temporal_phi
            tf = load_traj_as_temporal_phi(run_name)
            
        if tf is not None:
            feats[run_name]["membrane_temporal"] = tf
            
    return feats

def align_and_concat(feat_dict, keys, run_name=""):
    """
    Concatenate representations. If row counts mismatch, slice to the minimum
    length. All representations are saved in the same frame order, so the
    first min_len rows of each refer to the same frames.
    """
    present = [k for k in keys if k in feat_dict]
    if not present:
        return None
    lens = {k: feat_dict[k].shape[0] for k in present}
    min_len, max_len = min(lens.values()), max(lens.values())
    if min_len < max_len:
        print(f"  [!] {run_name}: fusing representations with mismatched row "
              f"counts {lens}; truncating to {min_len} frames. If this is much "
              f"smaller than the full run, a temporal feature file is likely "
              f"missing (re-run extract_offline_features.py).")
    return np.concatenate([feat_dict[k][:min_len] for k in present], axis=1)


def main():
    print("Running Trainable Layer Fusion and Unified Representation Extraction...")
    
    out_dir = cfg.OUTPUT_DIR / "features/fused"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    feats = load_all_features()
    
    if "clean" not in feats:
        print("Error: clean features not found.")
        return
        
    # Task C: Layer Attention Fusion — unsupervised weights from clean data
    # only: each layer is weighted by the inverse of its clean feature
    # variance (stable layers get more weight), normalized to sum to 1.
    # Weights are estimated on the clean TRAIN split so held-out clean frames
    # stay untouched for downstream evaluation.
    print("Computing Layer Fusion weights (Task C)...")
    clean_phi = feats["clean"]["membrane_stats"]
    clean_phi_train, _ = split_train_eval(clean_phi, seq_lens=load_phi_seq_lens("clean"))

    alpha = []
    for i in range(4):
        layer_feat = slice_phi_layer(clean_phi_train, i)
        if layer_feat.shape[1] > 0:
            var = layer_feat.var(axis=0).sum()
            alpha.append(1.0 / (var + 1e-8))
        else:
            alpha.append(0)

    alpha = np.array(alpha)
    alpha = alpha / alpha.sum()
    print(f"Learned Layer Attention Weights: {alpha}")

    # Task E: Per-Layer Mahalanobis Aggregation
    # Learn weights S = sum(w_i * score_i) on the clean train split.
    print("Computing Per-Layer Mahalanobis weights (Task E)...")
    layer_scorers = []
    clean_layer_scores = []

    for i in range(4):
        layer_feat = slice_phi_layer(clean_phi_train, i)
        if layer_feat.shape[1] > 0:
            scorer = mahalanobis_scorer(layer_feat)
            layer_scorers.append(scorer)
            score_i = scorer(layer_feat)
            clean_layer_scores.append(score_i)

    w = None
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
        
        if run_layer_scores and w is not None:
            rls_matrix = np.stack(run_layer_scores, axis=1) # (N, L)
            layer_score_fusion = (rls_matrix * w).sum(axis=1)
            run_feats["layer_score_fusion"] = layer_score_fusion

        # Task D: Unified Membrane Representation (membrane_fused).
        # membrane_temporal (handcrafted temporal phi) already includes the
        # threshold-margin features.
        keys_to_fuse = ["membrane_stats", "membrane_margin_hist", "membrane_temporal_latent"]
        if "membrane_temporal" in run_feats:
            keys_to_fuse.append("membrane_temporal")

        membrane_fused = align_and_concat(run_feats, keys_to_fuse, run_name=run_name)
        if membrane_fused is not None:
            run_feats["membrane_fused"] = membrane_fused
            
        # Save fused features
        torch.save(
            {
                "layer_fused_stats": torch.from_numpy(run_feats.get("layer_fused_stats")) if run_feats.get("layer_fused_stats") is not None else None,
                "layer_score_fusion": torch.from_numpy(run_feats.get("layer_score_fusion")) if run_feats.get("layer_score_fusion") is not None else None,
                "membrane_fused": torch.from_numpy(run_feats.get("membrane_fused")) if run_feats.get("membrane_fused") is not None else None
            },
            out_dir / f"{run_name}.pt"
        )
        
    print("Fusion complete.")

if __name__ == "__main__":
    main()
