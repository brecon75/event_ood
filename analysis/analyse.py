import sys
from pathlib import Path
import torch

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from vmem_benchmark import benchmark_config as cfg

# Import components
from analysis.vmem_utils import LazyPhiDict, TABLE_DIR
from analysis.analyse_plots import (
    plot_sensitivity_heatmap, plot_auroc_vs_severity,
    plot_all_trajectories, plot_pca_subspaces
)
from analysis.analyse_comparisons import (
    run_per_layer_auroc_table, run_statwise_ablation,
    run_detector_comparison, run_spearman_severity,
    save_full_results_table, run_severity_regression,
    run_corruption_classification, run_conformal_prediction,
    _build_detectors, split_clean,
)
from analysis.analyse_temporal import run_temporal_analysis

def main():
    cfg.PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n============================================================")
    print(f"CUDA Available in PyTorch: {torch.cuda.is_available()}")
    print(f"Target Device for OOD Scorers: {device.upper()}")
    print(f"============================================================\n")

    print("Initializing lazy phi loader...")
    all_phi = LazyPhiDict()

    if not all_phi:
        print(f"No results found in {cfg.PHI_DIR}. Run extract.py first.")
        sys.exit(1)
    if "clean" not in all_phi:
        print("ERROR: clean.pt missing from phi directory.")
        sys.exit(1)

    n_corrupted = sum(1 for k in all_phi if k != "clean")
    print(f"Loaded {len(all_phi)} phi files  ({n_corrupted} corrupted runs).")
    print(f"Clean phi shape: {all_phi['clean'].shape}  "
          f"(~{all_phi['clean'].shape[0]} frames from {all_phi['clean'].shape[0]//30:.0f} sequences)")

    # ── Level 1: original plots ──────────────────────────────────────────────
    plot_sensitivity_heatmap(all_phi)
    plot_auroc_vs_severity(all_phi)
    plot_all_trajectories()

    # ── Level 2: per-layer + stat-wise ──────────────────────────────────────
    run_per_layer_auroc_table(all_phi)
    if n_corrupted > 0:
        run_statwise_ablation(all_phi)

    # ── Level 3: detector comparison ────────────────────────────────────────
    if n_corrupted > 0:
        # Build all 7 detectors once — Flow and AE training is expensive
        clean_train, _ = split_clean(all_phi["clean"])
        detectors = _build_detectors(clean_train)
        run_detector_comparison(all_phi, detectors=detectors)

    # ── Level 4: temporal features from trajs ───────────────────────────────
    run_temporal_analysis(all_phi)

    # ── Level 5: Spearman + full table ──────────────────────────────────────
    if n_corrupted > 0:
        run_spearman_severity(all_phi)
        save_full_results_table(all_phi, detectors=detectors)

    # ── Ideas 3, 5, 6, 7: PCA, Severity Regression, Classification, Conformal ──
    if n_corrupted > 0:
        plot_pca_subspaces(all_phi)
        run_severity_regression(all_phi)
        run_corruption_classification(all_phi)
        run_conformal_prediction(all_phi)

    print(f"\nAnalysis complete.")
    print(f"  Plots  -> {cfg.PLOT_DIR}")
    print(f"  Tables -> {TABLE_DIR}")

if __name__ == "__main__":
    main()
