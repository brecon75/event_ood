import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
import sys
from pathlib import Path

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import LAYER_SPECS, _get_present, mahalanobis_scorer, auroc_fpr95
from sklearn.decomposition import PCA

def plot_sensitivity_heatmap(all_phi):
    print("Plotting sensitivity heatmap...")
    present = _get_present(all_phi)
    if not present:
        print("  No corrupted runs found, skipping.")
        return

    clean_mean = all_phi["clean"].mean(axis=0, keepdims=True)
    matrix = np.zeros((len(present), len(cfg.SEVERITIES)))
    for i, c in enumerate(present):
        for j, s in enumerate(cfg.SEVERITIES):
            rn = f"{c}_L{s}"
            if rn in all_phi:
                matrix[i, j] = np.linalg.norm(all_phi[rn] - clean_mean, axis=1).mean()

    fig, ax = plt.subplots(figsize=(10, max(3, len(present))))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean L2 distance from clean φ")
    ax.set_xticks(range(len(cfg.SEVERITIES)))
    ax.set_xticklabels([f"L{s}" for s in cfg.SEVERITIES])
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present)
    ax.set_title("Vmem Sensitivity Heatmap (Shift in φ Space)")
    for i in range(len(present)):
        for j in range(len(cfg.SEVERITIES)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "sensitivity_heatmap.pdf")
    plt.close()
    print("  Saved sensitivity_heatmap.pdf")


def plot_all_trajectories(n_samples=8):
    import gc
    traj_clean = cfg.TRAJ_DIR / "clean.pt"
    if not traj_clean.exists():
        return
    try:
        clean_data = torch.load(traj_clean, map_location="cpu", weights_only=True)
    except Exception:
        return

    print("Plotting trajectories for SNN layers...")
    clean_means = {}
    for li in range(len(LAYER_SPECS)):
        if li in clean_data["trajs"]:
            clean_traj = clean_data["trajs"][li].numpy()  # (T, N, D)
            clean_means[li] = clean_traj[:, :n_samples, :].mean(axis=(1, 2))
    
    del clean_data
    gc.collect()

    present = [c for c in cfg.CORRUPTIONS
               if any((cfg.TRAJ_DIR / f"{c}_L{s}.pt").exists()
                      for s in cfg.SEVERITIES)]
    if not present:
        present = []

    corr_means = {li: {c: {} for c in present} for li in range(len(LAYER_SPECS))}
    for c_name in present:
        for sev in cfg.SEVERITIES:
            p = cfg.TRAJ_DIR / f"{c_name}_L{sev}.pt"
            if not p.exists():
                continue
            try:
                print(f"  Loading trajectory {p.name} to extract mean curves...")
                td = torch.load(p, map_location="cpu", weights_only=True)
                for li in range(len(LAYER_SPECS)):
                    if li in td["trajs"]:
                        m = td["trajs"][li].numpy()[:, :n_samples, :].mean(axis=(1, 2))
                        corr_means[li][c_name][sev] = m
                del td
                gc.collect()
            except Exception as e:
                print(f"  [!] Failed to extract curves from {p.name}: {e}")

    colors = cm.plasma(np.linspace(0, 1, len(cfg.SEVERITIES)))
    for li in range(len(LAYER_SPECS)):
        if li not in clean_means:
            continue
        n_rows = max(1, len(present))
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
        axes = np.atleast_1d(axes)

        for ax, c_name in zip(axes, present):
            ax.plot(clean_means[li], color="black", lw=2, ls="--", label="clean")
            for sev in cfg.SEVERITIES:
                if sev in corr_means[li][c_name]:
                    ax.plot(corr_means[li][c_name][sev], color=colors[sev - 1], label=f"L{sev}", alpha=0.8)
            ax.set_title(f"Layer {li} - Corruption: {c_name}")
            ax.set_ylabel("Mean V(t)")
            ax.legend(loc="upper right", ncol=3, fontsize=7)
            ax.grid(alpha=0.3)

        if len(axes):
            axes[-1].set_xlabel("SNN Timestep t")
        plt.tight_layout()
        out = cfg.PLOT_DIR / f"trajectories_L{li}.pdf"
        plt.savefig(out)
        plt.close()
        print(f"  Saved trajectories_L{li}.pdf")


def plot_auroc_vs_severity(all_phi):
    print("Plotting AUROC vs severity (Mahalanobis)...")
    present = _get_present(all_phi)
    if not present:
        print("  No corrupted runs found, skipping.")
        return

    clean = all_phi["clean"]
    scorer = mahalanobis_scorer(clean)
    cs = scorer(clean)

    plt.figure(figsize=(11, 5))
    for c_name in present:
        aurocs, sevs = [], []
        for sev in cfg.SEVERITIES:
            rn = f"{c_name}_L{sev}"
            if rn not in all_phi:
                continue
            yt = np.concatenate([np.zeros(len(cs)), np.ones(len(all_phi[rn]))])
            ys = np.concatenate([cs, scorer(all_phi[rn])])
            a, _ = auroc_fpr95(yt, ys)
            aurocs.append(a)
            sevs.append(sev)
        if aurocs:
            plt.plot(sevs, aurocs, marker="o", label=c_name)

    plt.axhline(0.5, color="gray", ls="--", label="random")
    plt.xlabel("Severity Level")
    plt.ylabel("OOD Detection AUROC")
    plt.title("Mahalanobis-φ AUROC vs Corruption Severity")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "auroc_vs_severity.pdf")
    plt.close()
    print("  Saved auroc_vs_severity.pdf")


def _plot_per_layer_heatmap(rows, present):
    matrix = np.array([[r.get(c, float("nan")) for c in present] for r in rows])
    fig, ax = plt.subplots(figsize=(max(8, len(present)), max(3, len(rows))))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Avg AUROC across severities")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=25, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["Layer"] for r in rows])
    ax.set_title("Per-Layer AUROC: which SNN block carries OOD signal?")
    for i, row in enumerate(rows):
        for j, c in enumerate(present):
            v = row.get(c, float("nan"))
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < 0.6 or v > 0.9 else "black")
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "per_layer_auroc_heatmap.pdf")
    plt.close()
    print("  Saved per_layer_auroc_heatmap.pdf")


def plot_pca_subspaces(all_phi):
    print("\n======================================================")
    print(" LEVEL 6 - PCA Subspace Scatter Plots (Idea 7)")
    print("======================================================")
    
    clean_phi = all_phi["clean"]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, c_name in enumerate(cfg.CORRUPTIONS):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        corrupt_list = []
        for sev in cfg.SEVERITIES:
            rn = f"{c_name}_L{sev}"
            if rn in all_phi:
                corrupt_list.append(all_phi[rn])
                
        if not corrupt_list:
            ax.text(0.5, 0.5, f"No data for {c_name}", ha='center', va='center')
            ax.set_title(c_name)
            continue
            
        corrupt_phi = np.concatenate(corrupt_list, axis=0)
        
        sub_clean = clean_phi
        if len(clean_phi) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(clean_phi), 5000, replace=False)
            sub_clean = clean_phi[indices]
            
        sub_corrupt = corrupt_phi
        if len(corrupt_phi) > 5000:
            np.random.seed(42)
            indices = np.random.choice(len(corrupt_phi), 5000, replace=False)
            sub_corrupt = corrupt_phi[indices]
            
        X = np.concatenate([sub_clean, sub_corrupt], axis=0)
        y = np.concatenate([np.zeros(len(sub_clean)), np.ones(len(sub_corrupt))])
        
        try:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c='#4C72B0', alpha=0.5, label='Clean', s=10)
            ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c='#C44E52', alpha=0.5, label='Corrupt', s=10)
            
            ax.set_title(f"{c_name}")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            if idx == 0:
                ax.legend()
        except Exception as e:
            ax.text(0.5, 0.5, f"Error fitting PCA:\n{e}", ha='center', va='center')
            ax.set_title(c_name)
            
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "pca_subspaces.pdf")
    plt.close()
    print(f"  Saved pca_subspaces.pdf")


def plot_statwise_ablation(results, present):
    stat_labels = list(results.keys())
    x    = np.arange(len(present))
    w    = 0.2
    cols = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    fig, ax = plt.subplots(figsize=(max(8, len(present) * 1.8), 5))
    for i, (label, c_aurocs) in enumerate(results.items()):
        vals = [c_aurocs.get(c, float("nan")) for c in present]
        ax.bar(x + i * w, vals, w, label=label, color=cols[i], alpha=0.85)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(present, rotation=20, ha="right")
    ax.set_ylabel("Avg AUROC (across severities)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Stat-wise Ablation: Which Moment Drives OOD Signal?")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "statwise_ablation.pdf")
    plt.close()
    print("  Saved statwise_ablation.pdf")


def plot_detector_comparison(summary, per_corr, det_names, present):
    auroc_vals = [summary[d]["auroc"] for d in det_names]
    fpr95_vals = [summary[d]["fpr95"] for d in det_names]
    cols       = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(det_names))
    ax1.bar(x, auroc_vals, color=cols[:len(det_names)], alpha=0.85)
    ax1.axhline(0.5, color="gray", ls="--")
    ax1.set_xticks(x);  ax1.set_xticklabels(det_names, rotation=25, ha="right")
    ax1.set_ylabel("Avg AUROC");  ax1.set_ylim(0, 1.05)
    ax1.set_title("Average AUROC by Detector")
    ax1.grid(axis="y", alpha=0.3)

    ax2.bar(x, fpr95_vals, color=cols[:len(det_names)], alpha=0.85)
    ax2.axhline(0.05, color="red", ls="--", label="5% target")
    ax2.set_xticks(x);  ax2.set_xticklabels(det_names, rotation=25, ha="right")
    ax2.set_ylabel("Avg FPR@95TPR");  ax2.set_ylim(0, 1.05)
    ax2.set_title("Average FPR@95TPR  (-> better)")
    ax2.legend();  ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "detector_comparison.pdf")
    plt.close()
    print("  Saved detector_comparison.pdf")


def plot_temporal_comparison(comparison_results):
    corrs = [r["Corruption"] for r in comparison_results]
    hc_vals = [r["Handcrafted"] for r in comparison_results]
    ta_vals = [r["TemporalAE"] for r in comparison_results]
    
    x = np.arange(len(corrs))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, hc_vals, width, label='Handcrafted (Mahalanobis)', color='#4C72B0')
    ax.bar(x + width/2, ta_vals, width, label='Temporal AE (Sequence Learning)', color='#55A868')
    
    ax.set_ylabel('AUROC at Severity 5')
    ax.set_title('Temporal OOD Detection: Handcrafted vs Sequence Learning')
    ax.set_xticks(x)
    ax.set_xticklabels(corrs, rotation=20, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "temporal_features.pdf")
    plt.close()
    print("  Saved temporal_features.pdf (comparison chart)")


def plot_corruption_confusion_matrix(y_true, y_pred, target_names):
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=target_names, cmap='Blues', ax=ax, xticks_rotation=45
    )
    plt.tight_layout()
    plt.savefig(cfg.PLOT_DIR / "corruption_confusion_matrix.pdf")
    plt.close()
    print("  Saved corruption_confusion_matrix.pdf")

