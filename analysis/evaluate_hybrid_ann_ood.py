"""
evaluate_hybrid_ann_ood.py — Run the post-hoc OOD detector suite on the
HYBRID model's OWN ANN-side representation: its penultimate ANN feature
(``last_ann_gap``) paired with its detection-head logits (``head_cls_L0_gap``).

WHY THIS EXISTS (separate from representation_ablation.py)
----------------------------------------------------------
``representation_ablation.py`` scores every representation (membrane phi, ANN,
logits, spikes) but only with a single detector (Mahalanobis). This script
applies the FULL SOTA post-hoc suite — the same detectors used for the external
ResNet-18 baseline in ``evaluate_ann_baselines.py`` — to the hybrid detector's
own free (feature, logit) pair. It answers the reviewer question head-on:

    "You have a detection head emitting logits for free — does the best post-hoc
     OOD score on it beat the membrane-potential phi signal?"

The output CSV is schema-compatible with ``representation_metrics.csv``
(model, representation, detector, corruption, severity, auroc, aupr, fpr95), so
the two can be concatenated for a single phi-vs-ANN-suite comparison.

FAITHFUL-DETECTOR SCOPE (important)
-----------------------------------
Ten of the sixteen detectors need only (features, logits) and are applied as-is:

    MSP, Energy, ODIN, MaxLogit, GEN   — logit-only
    GradNorm                           — closed form from (||h||_1, softmax)
    Mahalanobis, kNN                   — penultimate-feature density / distance
    NECO, NNGuide                      — feature subspace / kNN, guided by logits

The other six (ReAct, ViM, DICE, ASH, SCALE, fDBD) recompute logits as W@h+b
through a SINGLE LINEAR classifier head. The hybrid head is convolutional and
multi-stage (stem -> cls_convs -> cls_preds) and the stored penultimate
(last_ann_gap) is not its linear input, so those six cannot be reproduced
faithfully offline for this model (they would silently collapse to Energy). They
are intentionally OMITTED rather than reported as misleading Energy duplicates.

NOTE on logit width: the hybrid head emits 2 class logits (car/pedestrian). The
logit-based detectors are still well-defined at C=2 (Energy/MaxLogit are the
natural choices; GEN/MSP/ODIN's "probability mass spreads over many classes"
intuition is weaker but the scores remain valid).

USAGE (the prof runs this against the full extractions in ANN_DIR)
------------------------------------------------------------------
    python analysis/evaluate_hybrid_ann_ood.py            # all runs, last_ann_gap
    python analysis/evaluate_hybrid_ann_ood.py --feat-keys last_ann_gap asab_gap
    python analysis/evaluate_hybrid_ann_ood.py --test     # first 2 corrupted runs
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    split_train_eval, split_boundary, load_phi_seq_lens, TRAIN_RATIO, auroc_aupr_fpr95,
)
from analysis.evaluate_ann_baselines import (
    DetectorMSP, DetectorEnergy, DetectorODIN, DetectorMaxLogit, DetectorGEN,
    DetectorGradNorm, DetectorMahalanobis, DetectorKNN, DetectorNECO,
    DetectorNNGuide, DetectorMahalanobisPP, DetectorLogitGap,
)

# Logit produced by the hybrid detection head (GAP of cls_preds[0] output).
LOGIT_KEY = "head_cls_L0_gap"


def _build_detectors():
    """The ten detectors that need only (features, logits) — see module docstring
    for why the six head-recomputation detectors are excluded for this model."""
    return {
        # logit-only
        "MSP": DetectorMSP(),
        "Energy": DetectorEnergy(),
        "ODIN": DetectorODIN(),
        "MaxLogit": DetectorMaxLogit(),
        "GEN": DetectorGEN(),
        # closed-form gradient norm (features + logits)
        "GradNorm": DetectorGradNorm(),
        # penultimate-feature detectors
        "Mahalanobis": DetectorMahalanobis(),
        "kNN": DetectorKNN(),
        "NECO": DetectorNECO(),
        "NNGuide": DetectorNNGuide(),
        # 2025 additions (feature-/logit-only, so faithful for this head too;
        # LogitGap degenerates to the logit margin at the head's K=2)
        "Mahalanobis++": DetectorMahalanobisPP(),
        "LogitGap": DetectorLogitGap(),
    }


def _load_pair(ann_path, feat_key):
    """Return (feat, logit) as contiguous float32 numpy arrays from a hybrid ANN
    .pt file, or (None, None) if either key is missing."""
    d = torch.load(ann_path, weights_only=True, map_location="cpu")
    if feat_key not in d or LOGIT_KEY not in d:
        return None, None
    feat = np.ascontiguousarray(d[feat_key].float().numpy(), dtype=np.float32)
    logit = np.ascontiguousarray(d[LOGIT_KEY].float().numpy(), dtype=np.float32)
    return feat, logit


def evaluate_feat_key(ann_dir, feat_key, limit=None):
    """Fit the suite on clean (feat_key, logits) and score every corruption run.

    Mirrors evaluate_ann_baselines.evaluate_representation but sources the
    (feature, logit) pair from the hybrid model's ANN extractions instead of the
    ResNet baseline files, and uses the faithful ten-detector subset.
    """
    clean_path = ann_dir / "clean.pt"
    if not clean_path.exists():
        print(f"[skip {feat_key}] clean.pt not found in {ann_dir}.")
        return []

    c_feat, c_logit = _load_pair(clean_path, feat_key)
    if c_feat is None:
        print(f"[skip {feat_key}] clean.pt lacks '{feat_key}' or '{LOGIT_KEY}'.")
        return []
    if len(c_feat) < 10:
        print(f"[skip {feat_key}] only {len(c_feat)} clean frames — too few to "
              f"split. (Per-sequence legacy file? Re-extract per-frame ANN feats.)")
        return []

    # Sequence-aware 70/30 split (same convention as every other stage): fit on
    # the train portion of clean, use only the held-out tail as ID negatives so
    # no detector — kNN especially — scores its own fitting data.
    clean_seq_lens = load_phi_seq_lens("clean", artifact_dir=ann_dir)
    fit_feat, eval_feat = split_train_eval(c_feat, seq_lens=clean_seq_lens)
    fit_logit, eval_logit = split_train_eval(c_logit, seq_lens=clean_seq_lens)

    detectors = _build_detectors()
    print(f"[{feat_key}] fitting {len(detectors)} detectors on "
          f"{len(fit_feat)} clean frames (feat dim={c_feat.shape[1]}, "
          f"logit dim={c_logit.shape[1]}) ...")
    for det in detectors.values():
        det.fit(fit_feat, fit_logit)
    clean_scores = {name: det.score(eval_feat, eval_logit)
                    for name, det in detectors.items()}

    run_files = [f for f in sorted(ann_dir.glob("*.pt"))
                 if not f.name.startswith("_tmp_") and f.stem != "clean"]
    if limit is not None:
        run_files = run_files[:limit]

    results = []
    for f in tqdm(run_files, desc=f"hybrid ANN suite [{feat_key}]"):
        run_name = f.stem
        t_feat, t_logit = _load_pair(f, feat_key)
        if t_feat is None:
            continue
        if t_feat.shape[1] != c_feat.shape[1]:
            continue  # representation width differs for this run — skip

        # Positives = held-out tail only, matched to the clean negatives'
        # held-out sequences. feat/logit are row-aligned: cut both identically.
        run_cut = split_boundary(len(t_feat), TRAIN_RATIO,
                                 load_phi_seq_lens(run_name, artifact_dir=ann_dir))
        t_feat, t_logit = t_feat[run_cut:], t_logit[run_cut:]

        parts = run_name.rsplit("_L", 1)
        corruption = parts[0]
        severity = int(parts[1]) if len(parts) > 1 else 0

        for name, det in detectors.items():
            t_scores = det.score(t_feat, t_logit)
            metrics = auroc_aupr_fpr95(clean_scores[name], t_scores)
            if metrics is None:
                continue
            auroc, aupr, fpr95 = metrics
            results.append({
                "model": "hybrid",
                "representation": feat_key,
                "detector": name,
                "corruption": corruption,
                "severity": severity,
                "auroc": auroc,
                "aupr": aupr,
                "fpr95": fpr95,
            })
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run the post-hoc OOD detector suite on the hybrid model's "
                    "own ANN penultimate features + detection-head logits.")
    parser.add_argument(
        "--ann-dir", type=Path, default=cfg.ANN_DIR,
        help="Directory of hybrid ANN extractions (<run>.pt with last_ann_gap / "
             "asab_gap / head_cls_L0_gap). Default: cfg.ANN_DIR.")
    parser.add_argument(
        "--feat-keys", nargs="+", default=["last_ann_gap"],
        choices=["last_ann_gap", "asab_gap", "head_cls_L0_gap"],
        help="Which penultimate feature(s) to pair with the head logits. "
             "Default: last_ann_gap (the feature feeding the detection head).")
    parser.add_argument(
        "--output", type=Path,
        default=cfg.OUTPUT_DIR / "results" / "hybrid_ann_ood_metrics.csv",
        help="Output CSV path.")
    parser.add_argument(
        "--test", action="store_true",
        help="Smoke mode: only the first 2 corrupted runs per feature key.")
    args = parser.parse_args()

    if not args.ann_dir.exists():
        print(f"ANN dir {args.ann_dir} does not exist. Run extract.py WITHOUT "
              f"--no-ann/--phi-only so ANN features are saved, then retry.")
        return

    limit = 2 if args.test else None
    all_results = []
    for feat_key in args.feat_keys:
        all_results.extend(evaluate_feat_key(args.ann_dir, feat_key, limit=limit))

    if not all_results:
        print("No results generated (no ANN extractions found?).")
        return

    df = pd.DataFrame(all_results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nSaved {len(df)} rows -> {args.output}")

    # Quick at-a-glance: best detector per corruption at the max severity present.
    if not df.empty:
        sev = df["severity"].max()
        top = (df[df["severity"] == sev]
               .sort_values("auroc", ascending=False)
               .groupby("corruption")
               .first()[["representation", "detector", "auroc"]])
        print(f"\nBest hybrid-ANN detector per corruption at severity {sev}:")
        print(top.to_string())


if __name__ == "__main__":
    main()
