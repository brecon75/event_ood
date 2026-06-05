import sys
import csv
from pathlib import Path
import numpy as np
import torch

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from vmem_benchmark import benchmark_config as cfg

from analysis.vmem_utils import (
    TABLE_DIR, load_all_temporal_phi, load_traj_as_temporal_phi,
    auroc_fpr95, _get_present
)
from analysis.vmem_models import prepare_temporal_ae_input
from analysis.vmem_scorers import mahalanobis_scorer, temporal_autoencoder_scorer
from analysis.analyse_plots import plot_temporal_comparison

def run_temporal_analysis(all_phi):
    print("\n======================================================")
    print(" LEVEL 4 - Temporal & Sequence Learning (from trajs)")
    print("======================================================")
    
    present = _get_present(all_phi)
    if not present:
        print("  No corrupted runs found.")
        return

    all_tphi = load_all_temporal_phi()
    
    # Fallback to computing on-the-fly from trajs if temporal_phi is missing
    if "clean" not in all_tphi:
        print("  No pre-computed temporal phi found. Computing on-the-fly from traj files...")
        clean_tphi = load_traj_as_temporal_phi("clean")
        if clean_tphi is None:
            print("  No clean.pt found in trajs/. Skipping temporal analysis.")
            return
        all_tphi["clean"] = clean_tphi
        
        for c_name in present:
            for sev in cfg.SEVERITIES:
                rn = f"{c_name}_L{sev}"
                tp = load_traj_as_temporal_phi(rn)
                if tp is not None:
                    all_tphi[rn] = tp

    n_clean = all_tphi["clean"].shape[0]
    n_runs  = len(all_tphi)
    print(f"  Temporal phi loaded: {n_runs} runs, {n_clean} samples per run.")

    if n_runs <= 1:
        print("  Only clean available. Skipping OOD scoring.")
        return

    clean_tp = all_tphi["clean"]
    hc_scorer = mahalanobis_scorer(clean_tp)
    hc_clean_scores = hc_scorer(clean_tp)

    clean_traj_path = cfg.TRAJ_DIR / "clean.pt"
    ta_scorer = None
    ta_clean_scores = None
    all_prepared_trajs = {}

    if clean_traj_path.exists():
        print("  Loading and preparing trajectories for sequence learning...")
        import gc
        try:
            clean_data = torch.load(clean_traj_path, map_location="cpu", weights_only=True)
            clean_trajs = clean_data["trajs"]
            
            print("  Training Temporal Autoencoder on clean trajectories...")
            ta_scorer = temporal_autoencoder_scorer(clean_trajs)
            ta_clean_scores = ta_scorer(clean_trajs)
            
            all_prepared_trajs["clean"] = prepare_temporal_ae_input(clean_trajs)
            
            del clean_data, clean_trajs
            gc.collect()
            
            for c_name in present:
                for sev in cfg.SEVERITIES:
                    rn = f"{c_name}_L{sev}"
                    tp_path = cfg.TRAJ_DIR / f"{rn}.pt"
                    if tp_path.exists():
                        try:
                            print(f"  Preparing trajectory {tp_path.name} for sequence learning...")
                            c_data = torch.load(tp_path, map_location="cpu", weights_only=True)
                            all_prepared_trajs[rn] = prepare_temporal_ae_input(c_data["trajs"])
                            del c_data
                            gc.collect()
                        except Exception as e:
                            print(f"    Failed to load/prepare {tp_path.name}: {e}")
        except Exception as e:
            print(f"  [!] Failed to initialize sequence learning: {e}")
            ta_scorer = None
    else:
        print("  No clean.pt found in trajs/. Skipping Temporal Autoencoder sequence learning.")

    print(f"\n  {'Corruption':<25}  Handcrafted AUROC  Temporal AE AUROC")
    print("  " + "-" * 60)
    
    comparison_results = []
    for c_name in present:
        hc_aurocs = []
        ta_aurocs = []
        for sev in cfg.SEVERITIES:
            rn = f"{c_name}_L{sev}"
            
            if rn in all_tphi:
                tp = all_tphi[rn]
                yt_hc = np.concatenate([np.zeros(len(hc_clean_scores)), np.ones(len(tp))])
                ys_hc = np.concatenate([hc_clean_scores, hc_scorer(tp)])
                a_hc, _ = auroc_fpr95(yt_hc, ys_hc)
                hc_aurocs.append(a_hc)
            else:
                hc_aurocs.append(float("nan"))

            if ta_scorer is not None and rn in all_prepared_trajs:
                prep_traj = all_prepared_trajs[rn]
                ta_scores = ta_scorer(prep_traj)
                yt_ta = np.concatenate([np.zeros(len(ta_clean_scores)), np.ones(len(ta_scores))])
                ys_ta = np.concatenate([ta_clean_scores, ta_scores])
                a_ta, _ = auroc_fpr95(yt_ta, ys_ta)
                ta_aurocs.append(a_ta)
            else:
                ta_aurocs.append(float("nan"))

        if 5 in cfg.SEVERITIES:
            idx_5 = cfg.SEVERITIES.index(5)
            a_hc_5 = hc_aurocs[idx_5] if idx_5 < len(hc_aurocs) else float("nan")
            a_ta_5 = ta_aurocs[idx_5] if idx_5 < len(ta_aurocs) else float("nan")
            print(f"  {c_name:<25}  {a_hc_5:.4f}             {a_ta_5:.4f}")
            comparison_results.append({
                "Corruption": c_name,
                "Handcrafted": a_hc_5,
                "TemporalAE": a_ta_5
            })

    if comparison_results:
        plot_temporal_comparison(comparison_results)
        
        with open(TABLE_DIR / "temporal_comparison.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Corruption", "Handcrafted_AUROC_L5", "TemporalAE_AUROC_L5"])
            w.writeheader()
            for r in comparison_results:
                w.writerow({
                    "Corruption": r["Corruption"],
                    "Handcrafted_AUROC_L5": round(r["Handcrafted"], 4) if not np.isnan(r["Handcrafted"]) else "",
                    "TemporalAE_AUROC_L5": round(r["TemporalAE"], 4) if not np.isnan(r["TemporalAE"]) else ""
                })
            print("  Saved temporal_comparison.csv")
