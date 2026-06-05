#!/bin/bash
# run_full_benchmark.sh
# One-command execution to build the paper-ready robustness benchmark.

set -e

echo "============================================================"
echo " Starting SNN OOD Benchmark Pipeline "
echo "============================================================"

# Ensure output directories exist
mkdir -p results paper_figures paper_tables outputs/detectors outputs/phi outputs/trajs

#echo ">>> [1/13] Running Extraction (extract.py) ..."
#python vmem_benchmark/extract.py

echo ">>> [2/13] Offline Feature Extraction (extract_offline_features.py) ..."
python analysis/extract_offline_features.py

echo ">>> [3/13] Feature Fusion (fusion_features.py) ..."
python analysis/fusion_features.py

echo ">>> [4/13] Extract ANN Baselines (extract_ann_baselines.py) ..."
python analysis/extract_ann_baselines.py

echo ">>> [5/13] Evaluate ANN Baselines (evaluate_ann_baselines.py) ..."
python analysis/evaluate_ann_baselines.py

echo ">>> [6/13] Fitting Detectors (fit_detectors.py) ..."
python analysis/fit_detectors.py

echo ">>> [7/13] Evaluating Detectors (evaluate_detectors.py) ..."
python analysis/evaluate_detectors.py

echo ">>> [8/13] Representation Ablation (representation_ablation.py) ..."
python analysis/representation_ablation.py

echo ">>> [9/13] Severity Monotonicity (severity.py) ..."
python analysis/severity.py

echo ">>> [10/13] Reliability Prediction (reliability.py) ..."
python analysis/reliability.py

echo ">>> [11/13] Cross-Corruption Generalization (cross_corruption.py) ..."
python analysis/cross_corruption.py

echo ">>> [12/13] Building Paper Tables (build_paper_tables.py) ..."
python reporting/build_paper_tables.py

echo ">>> [13/13] Building Paper Figures (build_paper_figures.py) ..."
python reporting/build_paper_figures.py

echo "============================================================"
echo " Benchmark Complete! "
echo " Check 'results/', 'paper_figures/', and 'paper_tables/' for the final outputs."
echo "============================================================"
