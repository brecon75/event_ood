"""Check whether the event-camera corruptions actually degrade the detector's mAP.

This is the analysis-side CHECK + visualization for the motivation experiment. The
*model run* is `HybridDetection/validation_corrupt.py` (it feeds clean vs corrupted
event input through the pretrained Hybrid detector and logs Prophesee COCO mAP to
`results/neftci_map_degradation.csv`); this script consumes that CSV and answers:

  1. Does each corruption reduce mAP versus the clean baseline, and by how much?
  2. Is the drop monotone in severity (a well-behaved corruption)?
  3. Does mAP damage line up with OOD-detectability? — i.e. are the corruptions that
     hurt the model the ones the φ detector (MDD) catches? (the paper's core link)

It writes a verdict table + a tidy summary CSV and two figures into outputs/graphs/:
  24_map_degradation.png          mAP vs severity per corruption (clean baseline line)
  25_map_drop_vs_detectability.png  mAP-drop vs MDD AUROC per corruption (motivation)

If the mAP CSV does not exist yet it prints exactly how to generate it and exits 0.

Run:  python analysis/check_map_degradation.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
MAP_CSV = REPO / "results" / "neftci_map_degradation.csv"
SUMMARY = REPO / "results" / "neftci_map_degradation_summary.csv"
AUROC_CSV = REPO / "vmem_benchmark" / "outputs" / "results" / "final_results.csv"
GRAPH_DIR = REPO / "vmem_benchmark" / "outputs" / "graphs"

CORRS = ["hot_pixel", "event_flood", "temporal_jitter",
         "event_rate_shift", "polarity_flip", "spatial_dropout"]
REL_DROP_SIGNIF = 5.0   # >5% relative mAP loss counts as "degrades"


def _how_to_generate():
    print(f"[check_map_degradation] {MAP_CSV} not found — the detector sweep hasn't run yet.\n")
    print("Generate it (runs the pretrained detector on clean + the 6x5 grid; needs the")
    print("HybridDetection env + GPU + Gen1 test split):\n")
    print("    cd HybridDetection")
    print("    GPU_ID=0 USE_TEST_SET=true sh run_map_degradation.sh        # full sweep")
    print("    SMOKE=1 sh run_map_degradation.sh                           # quick pipeline check\n")
    print("Then re-run:  python analysis/summarize_map_degradation.py  &&  python analysis/check_map_degradation.py")


def load_map():
    df = pd.read_csv(MAP_CSV)
    df = df.drop_duplicates(subset=["corruption", "severity"], keep="last")
    clean = df[df.corruption.isin(["clean", "none"])]
    if clean.empty:
        raise SystemExit("no clean baseline row in the CSV — run the `+corruption=clean` case.")
    clean_ap = float(clean.AP.iloc[0]); clean_ap50 = float(clean.AP_50.iloc[0])
    rows = df[~df.corruption.isin(["clean", "none"])].copy()
    rows["mAP_drop"] = clean_ap - rows.AP
    rows["rel_drop_pct"] = 100.0 * rows.mAP_drop / clean_ap if clean_ap else np.nan
    rows["AP50_drop"] = clean_ap50 - rows.AP_50
    rows["degraded"] = rows.rel_drop_pct > REL_DROP_SIGNIF
    return rows, clean_ap, clean_ap50


def verdict(rows, clean_ap):
    present = [c for c in CORRS if c in set(rows.corruption)]
    print(f"\nClean baseline mAP = {clean_ap:.4f}\n")
    hdr = f"  {'corruption':<18} {'sev->mAP (drop%)':<46} {'monotone?':<10} {'degrades?'}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    n_deg = 0
    for c in present:
        d = rows[rows.corruption == c].sort_values("severity")
        cells = "  ".join(f"L{int(s)}:{ap:.3f}(-{rd:.0f}%)" for s, ap, rd in
                          zip(d.severity, d.AP, d.rel_drop_pct))
        # monotone decreasing in severity (Spearman ρ < 0, |ρ| high)
        mono = "yes" if (len(d) >= 3 and np.corrcoef(d.severity, d.AP)[0, 1] < -0.5) else "—"
        deg = bool(d.degraded.any())
        n_deg += deg
        print(f"  {c:<18} {cells:<46} {mono:<10} {'YES' if deg else 'no'}")
    print(f"\n  => {n_deg}/{len(present)} corruptions significantly reduce mAP "
          f"(>{REL_DROP_SIGNIF:.0f}% relative at some severity).")
    worst = rows.loc[rows.mAP_drop.idxmax()]
    print(f"  => largest damage: {worst.corruption} L{int(worst.severity)} "
          f"mAP {worst.AP:.3f} (-{worst.mAP_drop:.3f}, -{worst.rel_drop_pct:.0f}%)")


def fig_degradation(rows, clean_ap):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    present = [c for c in CORRS if c in set(rows.corruption)]
    COLORS = dict(zip(CORRS, plt.cm.tab10(np.linspace(0, 1, len(CORRS)))))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(clean_ap, ls="--", c="black", lw=1.2, label=f"clean ({clean_ap:.3f})")
    for c in present:
        d = rows[rows.corruption == c].sort_values("severity")
        ax.plot(d.severity, d.AP, "o-", color=COLORS[c], lw=2, label=c)
    ax.set_xticks(sorted(rows.severity.unique())); ax.set_ylim(bottom=0)
    ax.set_xlabel("severity"); ax.set_ylabel("detector mAP (COCO AP@[.5:.95])")
    ax.set_title("Detector mAP under corruption (pretrained, no retraining)")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.25)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(GRAPH_DIR / "24_map_degradation.png", dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"wrote {GRAPH_DIR/'24_map_degradation.png'}")


def fig_drop_vs_detectability(rows):
    """Cross-plot mAP-drop vs MDD OOD-AUROC per corruption at the top severity —
    the motivation: corruptions that damage the model should be detectable."""
    if not AUROC_CSV.exists():
        print(f"[skip] {AUROC_CSV} missing — run analysis/plot_corruption_graphs.py for the scatter.")
        return
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    auroc = pd.read_csv(AUROC_CSV)
    top = int(rows.severity.max())
    a = auroc[auroc.severity == top].set_index("corruption")["improved_seq"]
    m = rows[rows.severity == top].set_index("corruption")
    present = [c for c in CORRS if c in m.index and c in a.index]
    if not present:
        print("[skip] no overlapping corruptions/severity between mAP and AUROC CSVs."); return
    COLORS = dict(zip(CORRS, plt.cm.tab10(np.linspace(0, 1, len(CORRS)))))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in present:
        ax.scatter(m.loc[c, "mAP_drop"], a[c], s=90, color=COLORS[c], zorder=3)
        ax.annotate(c, (m.loc[c, "mAP_drop"], a[c]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.set_xlabel(f"mAP drop vs clean (L{top})"); ax.set_ylabel(f"MDD OOD AUROC, per-seq (L{top})")
    ax.set_title("Does mAP damage align with detectability?"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(GRAPH_DIR / "25_map_drop_vs_detectability.png", dpi=130, bbox_inches="tight")
    plt.close(fig); print(f"wrote {GRAPH_DIR/'25_map_drop_vs_detectability.png'}")


def main():
    if not MAP_CSV.exists():
        _how_to_generate(); return
    rows, clean_ap, clean_ap50 = load_map()
    verdict(rows, clean_ap)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    cols = ["corruption", "severity", "AP", "AP_50", "mAP_drop", "rel_drop_pct", "AP50_drop", "degraded"]
    rows.sort_values(["corruption", "severity"])[cols].to_csv(SUMMARY, index=False)
    print(f"\nwrote {SUMMARY}")
    try:
        fig_degradation(rows, clean_ap)
        fig_drop_vs_detectability(rows)
    except Exception as e:
        print(f"[!] figure step failed: {e}")


if __name__ == "__main__":
    main()
