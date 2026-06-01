"""
Robustness metrics for event-based object detection.

mPC (Mean Performance under Corruption):
    Average mAP across all corruption types and severity levels.

RR (Relative Robustness):
    mPC expressed as a percentage of the clean baseline mAP.
"""
import numpy as np


def compute_mPC(results: dict) -> float:
    """
    Compute Mean Performance under Corruption.

    Parameters
    ----------
    results : dict mapping
        {corruption_name: {severity (int): mAP (float)}}

    Returns
    -------
    float — mean mAP across all (corruption, severity) pairs
    """
    scores = [
        mAP
        for sev_dict in results.values()
        for mAP in sev_dict.values()
    ]
    if not scores:
        raise ValueError("results dict is empty")
    return float(np.mean(scores))


def compute_RR(mPC: float, clean_mAP: float) -> float:
    """
    Relative Robustness as a percentage of the clean baseline.

    Returns 0.0 if clean_mAP == 0 to avoid division by zero.
    """
    if clean_mAP == 0.0:
        return 0.0
    return (mPC / clean_mAP) * 100.0


def print_results_table(results: dict, clean_mAP: float) -> None:
    """
    Pretty-print per-corruption, per-severity mAP table plus mPC and RR.

    Parameters
    ----------
    results   : {corruption_name: {severity: mAP}}
    clean_mAP : mAP on uncorrupted data
    """
    mPC = compute_mPC(results)
    RR  = compute_RR(mPC, clean_mAP)

    header = f"\n{'Corruption':<25} {'s=1':>6} {'s=2':>6} {'s=3':>6} {'s=4':>6} {'s=5':>6} {'Mean':>7}"
    print(header)
    print("-" * 65)

    for corruption, sev_dict in results.items():
        vals = [sev_dict.get(s, float("nan")) for s in [1, 2, 3, 4, 5]]
        mean = float(np.nanmean(vals))
        cells = "  ".join(
            f"{v:.3f}" if not np.isnan(v) else "  --- " for v in vals
        )
        print(f"{corruption:<25} {cells}  {mean:.3f}")

    print("-" * 65)
    print(f"{'Clean mAP':<25} {clean_mAP:.3f}")
    print(f"{'mPC':<25} {mPC:.3f}")
    print(f"{'RR (%)':<25} {RR:.1f}%")
