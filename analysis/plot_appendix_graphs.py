"""Appendix figures from the banked metric CSVs (no phi needed), with large,
legible fonts so they stay readable when scaled into the paper.

  26_representation_ablation.png -- representation x corruption Mahalanobis AUROC
        (L5): full phi vs each moment (mu/sigma^2/kappa) vs spike vs ANN vs logits.
  27_detector_comparison.png     -- the 7 classical OOD detectors, mean AUROC over
        all corruptions/severities (one-sided; contractions pull the mean toward/
        below chance).

Sources: outputs/results/representation_metrics.csv, ood_metrics.csv.
Run: python analysis/plot_appendix_graphs.py
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

# large fonts everywhere
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 13, "legend.fontsize": 12,
})
RES = cfg.OUTPUT_DIR / "results"
GRAPHS = cfg.OUTPUT_DIR / "graphs"; GRAPHS.mkdir(parents=True, exist_ok=True)

CORR = ["hot_pixel", "event_rate_shift", "temporal_jitter",
        "event_flood", "polarity_flip", "spatial_dropout"]
CLBL = [c.replace("_", "\n") for c in CORR]


def representation(sev=5):
    df = pd.read_csv(RES / "representation_metrics.csv")
    df = df[df.severity == sev]
    order = ["full_membrane", "membrane_mean", "membrane_var", "membrane_kurtosis",
             "spike", "spike_entropy", "ANN", "logits"]
    disp = {"full_membrane": r"full $\varphi$", "membrane_mean": r"$\mu$",
            "membrane_var": r"$\sigma^2$", "membrane_kurtosis": r"$\kappa$",
            "spike": "spike rate", "spike_entropy": "spike entropy",
            "ANN": "ANN feat.", "logits": "logits"}
    piv = df.pivot_table(index="representation", columns="corruption", values="auroc")
    piv = piv.reindex(index=[o for o in order if o in piv.index], columns=CORR)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(CORR))); ax.set_xticklabels(CLBL)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([disp.get(r, r) for r in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if abs(v - 0.5) > 0.30 else "black")
    ax.set_title("Membrane representation ablation (Mahalanobis AUROC, L5)")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02); cb.set_label("AUROC")
    ax.text(1.0, -1.05, "blue $>0.5$ detectable   red $<0.5$ inverted",
            transform=ax.transData, fontsize=11, color="#444")
    fig.tight_layout()
    out = GRAPHS / "26_representation_ablation.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def detectors():
    df = pd.read_csv(RES / "ood_metrics.csv")
    m = df.groupby("detector").auroc.mean().sort_values(ascending=False)
    names = {"pca": "PCA-Mahal", "mahalanobis": "Mahalanobis", "gmm": "GMM",
             "ae": "MLP-AE", "flow": "RealNVP", "ocsvm": "OCSVM", "knn": "kNN"}
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(m))
    cols = ["#3182bd" if v >= 0.5 else "#d6604d" for v in m.values]
    ax.bar(x, m.values, color=cols, edgecolor="white")
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.text(len(m) - 0.4, 0.505, "chance", fontsize=11, color="grey", va="bottom", ha="right")
    for i, v in enumerate(m.values):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([names.get(d, d) for d in m.index], rotation=20)
    ax.set_ylim(0.40, 0.62); ax.set_ylabel("mean AUROC")
    ax.set_title("Classical OOD detectors on static $\\varphi$ (mean over 6$\\times$5)")
    fig.tight_layout()
    out = GRAPHS / "27_detector_comparison.png"
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    representation()
    detectors()
