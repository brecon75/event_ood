#!/bin/bash
# run_full_benchmark.sh
# Runs the full benchmark pipeline end-to-end on the complete dataset.

set -e

PYTHON="./vmem_benchmark/.venv/Scripts/python"

# ── CUDA Device Check ─────────────────────────────────────────────────────
CUDA_AVAILABLE=false
CUDA_STATUS=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "${CUDA_STATUS}" = "True" ]; then
    CUDA_AVAILABLE=true
fi

echo "======================================================================="
echo "   RUNNING FULL SNN-ANN VMEM OOD BENCHMARK PIPELINE"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo -e "   \033[0;32m[CUDA Status: ACTIVE / AVAILABLE (Using GPU)]\033[0m"
else
    echo -e "   \033[0;33m[CUDA Status: INACTIVE / CPU ONLY (Running on CPU)]\033[0m"
fi
echo "======================================================================="

echo "Stage 1: Running parallel feature extraction (31 runs, all sequences)..."
if [ "$CUDA_AVAILABLE" = true ]; then echo -e "  --> Running run_parallel_extract.py on GPU (CUDA)..."; fi
$PYTHON vmem_benchmark/run_parallel_extract.py "$@"

echo "Stage 2: Extracting offline features (Temporal AE + margin histograms)..."
if [ "$CUDA_AVAILABLE" = true ]; then echo -e "  --> Running extract_offline_features.py (Temporal AE training) on GPU (CUDA)..."; fi
$PYTHON analysis/extract_offline_features.py

echo "Stage 3: Running feature fusion and Logistic Regression meta-classifier..."
$PYTHON analysis/fusion_features.py

echo "Stage 4: Extracting ResNet-18 ANN baselines (event image & voxel grid)..."
if [ "$CUDA_AVAILABLE" = true ]; then echo -e "  --> Running extract_ann_baselines.py on GPU (CUDA)..."; fi
$PYTHON analysis/extract_ann_baselines.py

echo "Stage 5: Evaluating ResNet-18 ANN baselines..."
if [ "$CUDA_AVAILABLE" = true ]; then echo -e "  --> Running evaluate_ann_baselines.py on GPU (CUDA)..."; fi
$PYTHON analysis/evaluate_ann_baselines.py

echo "Stage 6: Fitting OOD detectors on clean SNN fused representation..."
if [ "$CUDA_AVAILABLE" = true ]; then echo -e "  --> Running fit_detectors.py (SimpleAE training) on GPU (CUDA)..."; fi
$PYTHON analysis/fit_detectors.py

echo "Stage 7: Evaluating fitted OOD detectors on all corrupted runs..."
if [ "$CUDA_AVAILABLE" = true ]; then echo -e "  --> Running evaluate_detectors.py (AE evaluation) on GPU (CUDA)..."; fi
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
if [ "$CUDA_AVAILABLE" = true ]; then echo -e "  --> Running free_rider_ablation.py on GPU (CUDA)..."; fi
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
