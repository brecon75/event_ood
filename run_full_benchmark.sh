#!/bin/bash
# run_full_benchmark.sh
# Runs the full benchmark pipeline end-to-end on the complete dataset.

set -e

PYTHON="./vmem_benchmark/.venv/Scripts/python"

echo "======================================================================="
echo "   RUNNING FULL SNN-ANN VMEM OOD BENCHMARK PIPELINE"
echo "======================================================================="

echo "Stage 1: Running full feature extraction (31 runs, all sequences)..."
$PYTHON vmem_benchmark/extract.py

echo "Stage 2: Extracting offline features (Temporal AE + margin histograms)..."
$PYTHON analysis/extract_offline_features.py

echo "Stage 3: Running feature fusion and Logistic Regression meta-classifier..."
$PYTHON analysis/fusion_features.py

echo "Stage 4: Extracting ResNet-18 ANN baselines (event image & voxel grid)..."
$PYTHON analysis/extract_ann_baselines.py

echo "Stage 5: Evaluating ResNet-18 ANN baselines..."
$PYTHON analysis/evaluate_ann_baselines.py

echo "Stage 6: Fitting OOD detectors on clean SNN fused representation..."
$PYTHON analysis/fit_detectors.py

echo "Stage 7: Evaluating fitted OOD detectors on all corrupted runs..."
$PYTHON analysis/evaluate_detectors.py

echo "Stage 8: Running representation ablation (Mahalanobis comparison)..."
$PYTHON analysis/representation_ablation.py

echo "Stage 9: Running severity monotonicity analysis (Spearman rho)..."
$PYTHON analysis/severity.py

echo "Stage 10: Running downstream task reliability prediction..."
$PYTHON analysis/reliability.py

echo "Stage 11: Running cross-corruption zero-shot generalization..."
$PYTHON analysis/cross_corruption.py

echo "Stage 12: Running Free Rider validity ablation..."
$PYTHON analysis/free_rider_ablation.py

echo "Stage 13: Running analysis and main plotting script..."
$PYTHON analysis/analyse.py

echo "Stage 14: Building final paper LaTeX tables..."
$PYTHON reporting/build_paper_tables.py

echo "Stage 15: Building final paper figures..."
$PYTHON reporting/build_paper_figures.py

echo "======================================================================="
echo "   FULL PIPELINE EXECUTION SUCCESSFUL!"
echo "======================================================================="
