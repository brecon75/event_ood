"""Generate two summary figures from the banked result CSVs (no phi needed):

  25_detectability_summary_L5.png  -- per-corruption fused AUROC at three decision
       granularities (per-frame W=1, windowed W=64, full sequence), severity 5.
       One figure that tells the whole result story: what is solved per frame,
       what a short window buys, and the two residuals.
  24_sensitivity_L5.png            -- fused AUROC vs each MDD hyperparameter
       (k_pca, k_nn, n_ref) at W=64, L5: the visual robustness result.

Sources: outputs/results/final_results.csv, outputs/results/mdd_sensitivity.csv.
Writes PNGs into outputs/graphs/. Run: python analysis/plot_new_graphs.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg

RES = cfg.OUTPUT_DIR / "results"
GRAPHS = cfg.OUTPUT_DIR / "graphs"
GRAPHS.mkdir(parents=True, exist_ok=True)

# corruptions ordered solved -> residual, with display labels
ORDER = ["hot_pixel", "event_rate_shift", "temporal_jitter", "event_flood",
         "polarity_flip", "spatial_dropout"]
LBL = {c: c.replace("_", "\n") for c in ORDER}


def detectability_summary(sev=5):
    df = pd.read_csv(RES / "final_results.csv")
    d = df[df.severity == sev].set_index("corruption")
    cols = ["improved_frame", "improved_w64", "improved_seq"]
    names = ["per-frame (W=1)", "windowed (W=64)", "full sequence"]
    colors = ["#9ecae1", "#3182bd", "#08519c"]
    x = np.arange(len(ORDER)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    for i, (col, nm, cl) in enumerate(zip(cols, names, colors)):
        vals = [d.loc[c, col] if c in d.index else np.nan for c in ORDER]
        ax.bar(x + (i - 1) * w, vals, w, label=nm, color=cl, edgecolor="white")
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.text(len(ORDER) - 0.5, 0.505, "chance", fontsize=7, color="grey", va="bottom", ha="right")
    ax.axhline(0.85, ls=":", lw=0.8, color="darkgreen")
    ax.text(len(ORDER) - 0.5, 0.855, "solved (0.85)", fontsize=7, color="darkgreen", va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([LBL[c] for c in ORDER], fontsize=8)
    ax.set_ylim(0.3, 1.02); ax.set_ylabel("fused AUROC (L%d)" % sev)
    ax.set_title("Detection by decision granularity: four solved, two residuals", fontsize=10)
    ax.legend(fontsize=8, ncol=3, loc="lower center", frameon=False)
    fig.tight_layout()
    out = GRAPHS / "25_detectability_summary_L5.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print("wrote", out)


def sensitivity(sev=5, window=64):
    df = pd.read_csv(RES / "mdd_sensitivity.csv")
    df = df[(df.severity == sev) & (df.window == window) & (df.branch == "fused")]
    params = [("k_pca", "PCA dim $k_{pca}$"), ("k_nn", "RCF neighbours $k_{nn}$"),
              ("n_ref", "clean reference $n_{ref}$")]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0), sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, (p, title) in zip(axes, params):
        sub = df[df.param == p]
        for j, c in enumerate(ORDER):
            cc = sub[sub.corruption == c].sort_values("value")
            if cc.empty:
                continue
            ax.plot(cc.value, cc.auroc, "-o", ms=3, lw=1.3, color=cmap(j % 10),
                    label=c.replace("_", " "))
        ax.set_xscale("log"); ax.set_title(title, fontsize=9)
        ax.axhline(0.5, ls="--", lw=0.7, color="grey")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("fused AUROC (W=%d, L%d)" % (window, sev))
    axes[0].set_ylim(0.45, 1.02)
    axes[-1].legend(fontsize=6.5, loc="center right", frameon=False)
    fig.suptitle("MDD is insensitive to its hyperparameters", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = GRAPHS / "24_sensitivity_L5.png"
    fig.savefig(out, dpi=160); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    detectability_summary()
    if (RES / "mdd_sensitivity.csv").exists():
        sensitivity()
    else:
        print("skip sensitivity: mdd_sensitivity.csv not found (run mdd_sensitivity.py).")
