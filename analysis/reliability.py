import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.representation_ablation import load_all_features, get_mahalanobis_scores, extract_representation
from analysis.vmem_utils import split_train_eval, load_phi_seq_lens

def compute_detection_metric(det_outputs, conf_thresh=0.3):
    """
    Given (N, anchors, 7) padded YOLOX outputs saved by extract.py
    (columns: x, y, w, h, obj_conf, cls_conf_0, cls_conf_1), compute a
    reliability metric per frame: the number of confident detections.
    Returns array of shape (N,).
    """
    if det_outputs is None or det_outputs.ndim < 3:
        # Fallback if outputs aren't saved properly
        return np.zeros(det_outputs.shape[0] if det_outputs is not None else 1)

    scores = det_outputs[:, :, 4:5] * det_outputs[:, :, 5:]
    max_scores, _ = scores.max(dim=-1)

    # Count of confident boxes
    confident_counts = (max_scores > conf_thresh).sum(dim=1).numpy()
    return confident_counts


def load_det_outputs():
    """Load per-run detection outputs.

    New extract.py saves them as dicts under outputs/det_outputs/; legacy runs
    stored raw tensors in outputs/detectors/ next to the fitted detector
    models, so fall back to run-shaped files found there.
    """
    out = {}
    primary = getattr(cfg, "DET_OUT_DIR", cfg.OUTPUT_DIR / "det_outputs")
    for d in (primary, cfg.DETECTOR_DIR):
        if not d.exists():
            continue
        for f in d.glob("*.pt"):
            if f.stem in out:
                continue
            try:
                loaded = torch.load(f, weights_only=True, map_location="cpu")
            except Exception:
                continue
            data = loaded.get("det") if isinstance(loaded, dict) else loaded
            # Only accept (N, anchors, 7)-shaped tensors — the detectors dir
            # also contains fitted-model files (ae.pt, flow.pt, ...).
            if isinstance(data, torch.Tensor) and data.ndim == 3:
                out[f.stem] = data
    return out


def main():
    print("Running reliability prediction analysis...")
    all_feats = load_all_features()

    if 'clean' not in all_feats:
        print("Error: 'clean' run not found.")
        return

    det_outputs = load_det_outputs()
    if 'clean' not in det_outputs:
        print("Error: 'clean' detector outputs not found.")
        return

    clean_det_metric = compute_detection_metric(det_outputs['clean'])
    rep = 'membrane_fused'
    train_feat = extract_representation(all_feats['clean'], rep)
    if train_feat is None:
        rep = 'full_membrane'
        train_feat = extract_representation(all_feats['clean'], rep)
    if train_feat is None:
        print("Error: no usable clean representation found.")
        return

    # Fit the OOD scorer on the shared clean-train split (the corrupted runs
    # being scored are disjoint from it by construction).
    train_fit, _ = split_train_eval(train_feat, seq_lens=load_phi_seq_lens("clean"))

    results = []
    sample_curve = None  # (coverages, risks) of the last processed run, for plotting

    for run_name, feats in all_feats.items():
        if run_name == 'clean': continue
        if run_name not in det_outputs: continue

        test_feat = extract_representation(feats, rep)
        if test_feat is None:
            continue
        corr_det_metric = compute_detection_metric(det_outputs[run_name])

        if len(test_feat) != len(corr_det_metric):
            print(f"Skipping {run_name} due to shape mismatch.")
            continue

        ood_scores = get_mahalanobis_scores(train_fit, test_feat)

        # Degradation: clean metric - corrupt metric
        # Ensure we can pair them (assume sequence alignment if sizes match)
        if len(clean_det_metric) == len(corr_det_metric):
            degradation = clean_det_metric - corr_det_metric
        else:
            # If sequence dropping occurred differently, we cannot do point-to-point.
            # We'll just correlate OOD score with the metric directly instead of degradation.
            degradation = -corr_det_metric # higher degradation = fewer boxes

        # Guard: correlation functions require at least 2 samples and non-constant arrays
        if len(ood_scores) < 2 or np.std(ood_scores) == 0 or np.std(degradation) == 0:
            print(f"  Skipping {run_name}: insufficient variance for correlation.")
            continue

        try:
            spearman_rho, _ = spearmanr(ood_scores, degradation)
            pearson_r, _ = pearsonr(ood_scores, degradation)
            # R^2 of the best linear predictor of degradation from the OOD
            # score; for simple linear regression this equals pearson_r^2.
            # (Comparing the raw score to degradation with r2_score would be
            # meaningless — they are in different units.)
            r2 = pearson_r ** 2
        except Exception as e:
            print(f"  Skipping {run_name}: correlation failed ({e})")
            continue

        # AURC (Area Under the Risk-Coverage Curve)
        # Accept lowest-OOD-score frames first; risk = mean degradation among
        # accepted frames. Computed with a cumulative sum (O(n log n)).
        sorted_idx = np.argsort(ood_scores)
        n = len(sorted_idx)
        counts = np.arange(1, n + 1)
        run_risks = np.cumsum(degradation[sorted_idx]) / counts
        run_coverages = counts / n
        aurc_val = float(np.trapz(run_risks, run_coverages))
        sample_curve = (run_coverages, run_risks)

        parts = run_name.rsplit('_L', 1)
        corruption = parts[0]
        severity = int(parts[1]) if len(parts) > 1 else 0

        results.append({
            "corruption": corruption,
            "severity": severity,
            "spearman": spearman_rho,
            "pearson": pearson_r,
            "r2": r2,
            "aurc": aurc_val
        })

    df = pd.DataFrame(results)
    out_dir = cfg.OUTPUT_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "reliability_metrics.csv", index=False)

    # Plot Reliability Curve
    if not df.empty:
        fig_dir = cfg.OUTPUT_DIR / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="severity", y="spearman")
        plt.title("Spearman Correlation (OOD Score vs Degradation) by Severity")
        plt.tight_layout()
        plt.savefig(fig_dir / "reliability_curve.pdf")
        plt.close()

        # Plot Risk-Coverage for the last processed run as a sample
        if sample_curve is not None:
            cov, risk = sample_curve
            step = max(1, len(cov) // 500)  # keep the PDF lightweight
            cov, risk = cov[::step], risk[::step]
            plt.figure(figsize=(8, 6))
            plt.plot(cov, risk, marker='o', markersize=3)
            plt.xlabel("Coverage")
            plt.ylabel("Risk (Mean Degradation)")
            plt.title("Risk-Coverage Curve (Sample)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(fig_dir / "risk_coverage.pdf")
            plt.close()

    print(f"Reliability complete. Results saved to {out_dir / 'reliability_metrics.csv'}")

if __name__ == "__main__":
    main()
