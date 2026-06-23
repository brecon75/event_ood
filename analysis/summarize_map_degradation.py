"""
Summarize the mAP-degradation motivation experiment.

Reads results/neftci_map_degradation.csv (written by HybridDetection/validation_corrupt.py),
computes the absolute and relative mAP drop of each (corruption, severity) versus the clean
baseline, and emits:
  * results/neftci_map_degradation_summary.csv  -- tidy table with drop columns
  * results/neftci_map_degradation_table.tex    -- a LaTeX table for the paper

This is the empirical motivation: a SOTA pretrained detector loses substantial mAP under
event-camera corruptions, so detecting those corruptions (the rest of the paper) matters.

Usage:
    python analysis/summarize_map_degradation.py
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "results" / "neftci_map_degradation.csv"
SUMMARY = REPO_ROOT / "results" / "neftci_map_degradation_summary.csv"
TABLE_TEX = REPO_ROOT / "results" / "neftci_map_degradation_table.tex"

CORRUPTION_ORDER = [
    "hot_pixel", "event_flood", "temporal_jitter",
    "event_rate_shift", "polarity_flip", "spatial_dropout",
]


def load() -> pd.DataFrame:
    if not RAW.exists():
        raise SystemExit(f"missing {RAW} -- run HybridDetection/run_map_degradation.sh first")
    df = pd.read_csv(RAW)
    # keep the last run for any duplicate (corruption, severity) -- reruns overwrite
    df = df.drop_duplicates(subset=["corruption", "severity"], keep="last")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["corruption"].isin(["clean", "none"])]
    if clean.empty:
        raise SystemExit("no clean baseline row found -- run the `clean` case")
    clean_ap = float(clean["AP"].iloc[0])
    clean_ap50 = float(clean["AP_50"].iloc[0])

    rows = df[~df["corruption"].isin(["clean", "none"])].copy()
    rows["clean_AP"] = clean_ap
    rows["mAP_drop"] = clean_ap - rows["AP"]
    rows["rel_drop_pct"] = 100.0 * rows["mAP_drop"] / clean_ap if clean_ap else float("nan")
    rows["clean_AP_50"] = clean_ap50
    rows["AP_50_drop"] = clean_ap50 - rows["AP_50"]

    rows["corruption"] = pd.Categorical(rows["corruption"], CORRUPTION_ORDER, ordered=True)
    rows = rows.sort_values(["corruption", "severity"]).reset_index(drop=True)
    return rows, clean_ap, clean_ap50


def to_latex(rows: pd.DataFrame, clean_ap: float, clean_ap50: float) -> str:
    """Pivot to corruption x severity mAP (with the clean baseline in the caption)."""
    piv = rows.pivot_table(index="corruption", columns="severity", values="AP",
                           observed=True)
    lines = [
        r"\begin{table}[t]", r"  \centering", r"  \small",
        r"  \begin{tabular}{l" + "c" * len(piv.columns) + "}",
        r"    \toprule",
        "    Corruption & " + " & ".join(f"Sev.~{int(s)}" for s in piv.columns) + r" \\",
        r"    \midrule",
    ]
    for corr, srow in piv.iterrows():
        cells = " & ".join("---" if pd.isna(v) else f"{v:.3f}" for v in srow.values)
        lines.append(f"    \\texttt{{{str(corr).replace('_', chr(92)+'_')}}} & {cells} \\\\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        (r"  \caption{mAP (COCO AP@[.5:.95]) of the pretrained hybrid detector under each "
         rf"corruption and severity. Clean baseline mAP $= {clean_ap:.3f}$ "
         rf"(AP$_{{50}}={clean_ap50:.3f}$); every corruption is run on the same checkpoint "
         r"with no retraining. Larger drops motivate event-OOD detection.}"),
        r"  \label{tab:map-degradation}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Summarize the mAP-degradation experiment.")
    ap.add_argument("--input", default=str(RAW),
                    help="raw mAP CSV from validation_corrupt.py "
                         "(default: results/neftci_map_degradation.csv)")
    ap.add_argument("--output-dir", default=None,
                    help="dir for the summary CSV + LaTeX table (default: results/)")
    return ap.parse_args()


def main():
    global RAW, SUMMARY, TABLE_TEX
    args = _parse_args()
    RAW = Path(args.input)
    if args.output_dir:
        out = Path(args.output_dir)
        SUMMARY = out / "neftci_map_degradation_summary.csv"
        TABLE_TEX = out / "neftci_map_degradation_table.tex"
    df = load()
    rows, clean_ap, clean_ap50 = summarize(df)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(SUMMARY, index=False)
    TABLE_TEX.write_text(to_latex(rows, clean_ap, clean_ap50), encoding="utf-8")

    print(f"clean baseline: mAP={clean_ap:.4f}  AP_50={clean_ap50:.4f}")
    worst = rows.loc[rows["mAP_drop"].idxmax()]
    print(f"largest drop:   {worst['corruption']} sev{int(worst['severity'])}  "
          f"mAP {worst['AP']:.4f}  (-{worst['mAP_drop']:.4f}, "
          f"-{worst['rel_drop_pct']:.1f}%)")
    print(f"wrote {SUMMARY}")
    print(f"wrote {TABLE_TEX}")


if __name__ == "__main__":
    main()
