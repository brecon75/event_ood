"""
analyse.py — Analysis and visualization for the Vmem robustness benchmark.

Generates:
- Trajectory plots (V over time)
- KDE distribution plots
- Sensitivity Heatmap
- PCA Embedding
- OOD detection ROC curves
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve

import benchmark_config as cfg

def load_all_phi():
    """Load all run results into a dict."""
    out = {}
    for f in sorted(cfg.PHI_DIR.glob("*.pt")):
        d = torch.load(f, weights_only=True)
        out[d['run']] = d['phi'].numpy()
    return out

def plot_trajectories(layer_idx=0, n_samples=8):
    print(f"Plotting trajectories for layer {layer_idx}...")
    fig, axes = plt.subplots(len(cfg.CORRUPTIONS), 1,
                             figsize=(12, 3 * len(cfg.CORRUPTIONS)), sharex=True)
    axes = np.atleast_1d(axes)  # guard: subplots returns bare Axes when n=1

    clean_data = torch.load(cfg.TRAJ_DIR / "clean.pt", weights_only=True)
    clean_traj = clean_data['trajs'][layer_idx].numpy() # (T, N, D)
    clean_mean = clean_traj[:, :n_samples, :].mean(axis=(1, 2))
    
    colors = cm.plasma(np.linspace(0, 1, len(cfg.SEVERITIES)))
    
    for ax, c_name in zip(axes, cfg.CORRUPTIONS):
        ax.plot(clean_mean, color='black', linewidth=2, label='clean', linestyle='--')
        
        for sev in cfg.SEVERITIES:
            run_name = f"{c_name}_L{sev}"
            path = cfg.TRAJ_DIR / f"{run_name}.pt"
            if not path.exists(): continue
            
            traj = torch.load(path, weights_only=True)['trajs'][layer_idx].numpy()
            m = traj[:, :n_samples, :].mean(axis=(1, 2))
            ax.plot(m, color=colors[sev-1], label=f"L{sev}", alpha=0.8)
            
        ax.set_title(f"Corruption: {c_name}")
        ax.set_ylabel("Mean V(t)")
        ax.legend(loc='upper right', ncol=2)
        ax.grid(alpha=0.3)
        
    axes[-1].set_xlabel("Timestep t")
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / f"trajectories_L{layer_idx}.pdf")
    plt.close()

def plot_sensitivity_heatmap(all_phi):
    print("Plotting sensitivity heatmap...")
    clean_mean = all_phi["clean"].mean(axis=0, keepdims=True)
    
    matrix = np.zeros((len(cfg.CORRUPTIONS), len(cfg.SEVERITIES)))
    
    for i, c_name in enumerate(cfg.CORRUPTIONS):
        for j, sev in enumerate(cfg.SEVERITIES):
            run_name = f"{c_name}_L{sev}"
            if run_name in all_phi:
                # Mean L2 distance from clean center
                dist = np.linalg.norm(all_phi[run_name] - clean_mean, axis=1).mean()
                matrix[i, j] = dist
                
    plt.figure(figsize=(10, 6))
    im = plt.imshow(matrix, cmap="YlOrRd")
    plt.colorbar(im, label="L2 Distance from Clean Mean")
    plt.xticks(range(len(cfg.SEVERITIES)), [f"L{s}" for s in cfg.SEVERITIES])
    plt.yticks(range(len(cfg.CORRUPTIONS)), cfg.CORRUPTIONS)
    plt.title("Vmem Sensitivity Heatmap (Shift in Phi Space)")
    
    # Annotate values
    for i in range(len(cfg.CORRUPTIONS)):
        for j in range(len(cfg.SEVERITIES)):
            plt.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center")
            
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "sensitivity_heatmap.pdf")
    plt.close()

def run_ood_detection(all_phi):
    print("Running OOD detection analysis...")
    clean_phi = all_phi["clean"]
    
    # Fit Mahalanobis on clean
    try:
        cov = EmpiricalCovariance().fit(clean_phi)
        mu = cov.location_
        P = cov.precision_
    except:
        print("Warning: Covariance fit failed (possibly singular). Using simple L2.")
        mu = clean_phi.mean(0)
        P = np.eye(len(mu))

    def get_score(x):
        diff = x - mu
        return np.einsum('ni,ij,nj->n', diff, P, diff)

    clean_scores = get_score(clean_phi)
    
    results = {}
    plt.figure(figsize=(12, 8))
    
    for c_name in cfg.CORRUPTIONS:
        aurocs = []
        for sev in cfg.SEVERITIES:
            run_name = f"{c_name}_L{sev}"
            if run_name not in all_phi: continue
            
            corr_scores = get_score(all_phi[run_name])
            y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(corr_scores))])
            y_score = np.concatenate([clean_scores, corr_scores])
            auroc = roc_auc_score(y_true, y_score)
            aurocs.append(auroc)
            
        plt.plot(cfg.SEVERITIES[:len(aurocs)], aurocs, marker='o', label=c_name)
        
    plt.axhline(0.5, color='gray', linestyle='--')
    plt.xlabel("Severity Level")
    plt.ylabel("OOD Detection AUROC")
    plt.title("OOD Detectability vs Corruption Severity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(cfg.PLOT_DIR / "auroc_vs_severity.pdf")
    plt.close()

if __name__ == "__main__":
    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    all_phi = load_all_phi()
    
    if not all_phi:
        print("No results found in outputs/phi/. Run extract.py first.")
    else:
        plot_sensitivity_heatmap(all_phi)
        run_ood_detection(all_phi)
        # Assuming at least one trajectory was saved
        if (cfg.TRAJ_DIR / "clean.pt").exists():
            plot_trajectories(layer_idx=0)
        
        print(f"Analysis complete. Plots saved to {cfg.PLOT_DIR}")
