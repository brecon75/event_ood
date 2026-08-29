"""Plot fused MDD AUROC vs. aggregation window size, L5, ONE individual
subplot per corruption (2x3 grid) -- reads the already-computed
`mdd_window_sweep.csv` (canonical 85/15 split, new phi_spatial, plain-max
fusion), no refit needed.

Output: vmem_benchmark/outputs/plots/auroc_vs_window_l5.png
"""
import sys
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg

SEV = 5
WINDOWS = [1, 8, 16, 32, 64, 128, 256, "full"]
XPOS = list(range(len(WINDOWS)))  # evenly spaced categorical x-axis (log data, incl. "full")

SRC = Path(__file__).resolve().parent.parent / "unified_numbers" / "mdd_window_sweep.csv"


def main():
    with open(SRC) as f:
        rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes = axes.ravel()

    for i, corruption in enumerate(cfg.CORRUPTIONS):
        ax = axes[i]
        ys = []
        for w in WINDOWS:
            match = [r for r in rows if r["corruption"] == corruption
                     and r["severity"] == str(SEV) and r["branch"] == "fused"
                     and r["window"] == str(w)]
            ys.append(float(match[0]["auroc"]) if match else float("nan"))
        ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6)
        ax.plot(XPOS, ys, marker="o", lw=2, color="C0")
        ax.set_xticks(XPOS)
        ax.set_xticklabels([str(w) for w in WINDOWS], rotation=45)
        ax.set_xlabel("Aggregation window (frames)")
        ax.set_ylabel("AUROC")
        ax.set_title(f"{corruption}  (AUROC={ys[-1]:.3f} @ full)")
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.3)

    fig.suptitle(f"MDD fused AUROC vs. window, per corruption, L{SEV}")
    fig.tight_layout()

    out_dir = cfg.OUTPUT_DIR / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "auroc_vs_window_l5.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
