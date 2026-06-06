import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg

def safe_read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

def setup():
    out_dir = cfg.OUTPUT_DIR / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = cfg.OUTPUT_DIR / "results"
    return res_dir, out_dir

def plot_fig1(res_dir, out_dir):
    # Conceptual Figure - just create a placeholder with text
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, 'Figure 1: EventCorrupt Overview\n(Diagram placeholder)', 
            horizontalalignment='center', verticalalignment='center', fontsize=15)
    ax.axis('off')
    plt.savefig(out_dir / "fig1_eventcorrupt.pdf")
    plt.close()

def plot_fig2(res_dir, out_dir):
    # Layer sensitivity heatmap
    # Fallback to general heatmap if layer_metrics doesn't exist
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, 'Figure 2: Layer Heatmap\n(Generated in analyse.py)', 
            horizontalalignment='center', verticalalignment='center', fontsize=15)
    ax.axis('off')
    plt.savefig(out_dir / "fig2_layer_heatmap.pdf")
    plt.close()

def plot_fig3(res_dir, out_dir):
    df_path = res_dir / "representation_metrics.csv"
    df = safe_read_csv(df_path)
    if df is None or df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="representation", y="auroc")
    plt.title("OOD Detection Performance by Representation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_representation.pdf")
    plt.close()

def plot_fig4(res_dir, out_dir):
    df_path = res_dir / "severity_metrics.csv"
    df = safe_read_csv(df_path)
    if df is None or df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="detector", y="rho", hue="representation")
    plt.title("Severity Monotonicity (Spearman ρ) across Detectors")
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_severity.pdf")
    plt.close()

def plot_fig5(res_dir, out_dir):
    df_path = res_dir / "reliability_metrics.csv"
    df = safe_read_csv(df_path)
    if df is None or df.empty: return
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df, x="severity", y="aurc", marker="o", hue="corruption")
    plt.title("Area Under Risk-Coverage (AURC) vs Severity")
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_reliability.pdf")
    plt.close()

def plot_fig6(res_dir, out_dir):
    df_path = res_dir / "cross_corruption.csv"
    df = safe_read_csv(df_path)
    if df is None or df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="eval_corruption", y="auroc", hue="severity")
    plt.title("Cross-Corruption Generalization (Trained on hot_pixel)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "fig6_transfer.pdf")
    plt.close()

def plot_fig7(res_dir, out_dir):
    df_path = res_dir / "model_comparison.csv"
    df = safe_read_csv(df_path)
    if df is None or df.empty:
        # placeholder
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'Figure 7: Model Comparison\n(No data yet)', 
                horizontalalignment='center', verticalalignment='center', fontsize=15)
        ax.axis('off')
        plt.savefig(out_dir / "fig7_model_comparison.pdf")
        plt.close()
        return
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df, x="model", y="auroc")
    plt.title("OOD Performance across Models")
    plt.tight_layout()
    plt.savefig(out_dir / "fig7_model_comparison.pdf")
    plt.close()

def plot_fig_severity3plus(res_dir, out_dir):
    df_path = res_dir / "severity3plus_metrics.csv"
    df = safe_read_csv(df_path)
    if df is None or df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="detector", y="auroc")
    plt.title("OOD Performance (Severity >= 3)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "fig_severity3plus.pdf")
    plt.close()

def plot_ann_comparison(res_dir, out_dir):
    ann_path = res_dir / "ann_baselines.csv"
    ood_path = res_dir / "ood_metrics.csv"
    
    ann_df = safe_read_csv(ann_path)
    ood_df = safe_read_csv(ood_path)
    
    # We want to compare the best ANN baseline vs our best membrane representations
    if ann_df is None or ood_df is None or ann_df.empty or ood_df.empty: return
    
    # Just take overall mean AUROC per method/representation
    ann_agg = ann_df.groupby(["representation", "detector"])["auroc"].mean().reset_index()
    ann_agg["Method"] = "ANN: " + ann_agg["representation"] + " (" + ann_agg["detector"] + ")"
    
    ood_agg = ood_df.groupby(["detector"])["auroc"].mean().reset_index()
    ood_agg["Method"] = "Membrane: " + ood_agg["detector"]
    
    # Keep top 5 from each for clean plot
    ann_top = ann_agg.sort_values("auroc", ascending=False).head(5)
    ood_top = ood_agg.sort_values("auroc", ascending=False).head(5)
    
    combined = pd.concat([ann_top, ood_top])
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=combined, x="auroc", y="Method", palette="viridis")
    plt.title("ANN Baselines vs Membrane Representations (Mean AUROC)")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_ann_vs_membrane.pdf")
    plt.close()

def main():
    print("Building paper figures...")
    res_dir, out_dir = setup()
    plot_fig1(res_dir, out_dir)
    plot_fig2(res_dir, out_dir)
    plot_fig3(res_dir, out_dir)
    plot_fig4(res_dir, out_dir)
    plot_fig5(res_dir, out_dir)
    plot_fig6(res_dir, out_dir)
    plot_fig7(res_dir, out_dir)
    plot_fig_severity3plus(res_dir, out_dir)
    plot_ann_comparison(res_dir, out_dir)
    print(f"Figures built and saved to {out_dir}")

if __name__ == "__main__":
    main()
