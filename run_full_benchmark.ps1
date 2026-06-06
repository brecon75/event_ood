# run_full_benchmark.ps1
# Runs the full benchmark pipeline end-to-end on the complete dataset in PowerShell.

$ErrorActionPreference = "Stop"

$PYTHON = ".\vmem_benchmark\.venv\Scripts\python.exe"

# ── CUDA Device Check ─────────────────────────────────────────────────────
$CUDA_AVAILABLE = $false
try {
    $CUDA_STATUS = & $PYTHON -c "import torch; print(torch.cuda.is_available())"
    if ($CUDA_STATUS.Trim() -eq "True") {
        $CUDA_AVAILABLE = $true
    }
} catch {
    Write-Host "[CUDA Check] Failed to query PyTorch CUDA availability." -ForegroundColor Red
}

Write-Host "======================================================================="
Write-Host "   RUNNING FULL SNN-ANN VMEM OOD BENCHMARK PIPELINE"
if ($CUDA_AVAILABLE) {
    Write-Host "   [CUDA Status: ACTIVE / AVAILABLE (Using GPU)]" -ForegroundColor Green
} else {
    Write-Host "   [CUDA Status: INACTIVE / CPU ONLY (Running on CPU)]" -ForegroundColor Yellow
}
Write-Host "======================================================================="

Write-Host "Stage 1: Running parallel feature extraction (31 runs, all sequences)..."
if ($CUDA_AVAILABLE) { Write-Host "  --> Running run_parallel_extract.py on GPU (CUDA)..." -ForegroundColor Cyan }
& $PYTHON vmem_benchmark/run_parallel_extract.py $args

Write-Host "Stage 2: Extracting offline features (Temporal AE + margin histograms)..."
if ($CUDA_AVAILABLE) { Write-Host "  --> Running extract_offline_features.py (Temporal AE training) on GPU (CUDA)..." -ForegroundColor Cyan }
& $PYTHON analysis/extract_offline_features.py

Write-Host "Stage 3: Running feature fusion and Logistic Regression meta-classifier..."
& $PYTHON analysis/fusion_features.py

Write-Host "Stage 4: Extracting ResNet-18 ANN baselines (event image & voxel grid)..."
if ($CUDA_AVAILABLE) { Write-Host "  --> Running extract_ann_baselines.py on GPU (CUDA)..." -ForegroundColor Cyan }
& $PYTHON analysis/extract_ann_baselines.py

Write-Host "Stage 5: Evaluating ResNet-18 ANN baselines..."
if ($CUDA_AVAILABLE) { Write-Host "  --> Running evaluate_ann_baselines.py on GPU (CUDA)..." -ForegroundColor Cyan }
& $PYTHON analysis/evaluate_ann_baselines.py

Write-Host "Stage 6: Fitting OOD detectors on clean SNN fused representation..."
if ($CUDA_AVAILABLE) { Write-Host "  --> Running fit_detectors.py (SimpleAE training) on GPU (CUDA)..." -ForegroundColor Cyan }
& $PYTHON analysis/fit_detectors.py

Write-Host "Stage 7: Evaluating fitted OOD detectors on all corrupted runs..."
if ($CUDA_AVAILABLE) { Write-Host "  --> Running evaluate_detectors.py (AE evaluation) on GPU (CUDA)..." -ForegroundColor Cyan }
& $PYTHON analysis/evaluate_detectors.py

Write-Host "Stage 8: Running representation ablation (Mahalanobis comparison)..."
& $PYTHON analysis/representation_ablation.py

Write-Host "Stage 9: Running severity monotonicity analysis (Spearman rho)..."
& $PYTHON analysis/severity.py

Write-Host "Stage 10: Running downstream task reliability prediction..."
& $PYTHON analysis/reliability.py

Write-Host "Stage 11: Running cross-corruption zero-shot generalization..."
& $PYTHON analysis/cross_corruption.py

Write-Host "Stage 12: Running Free Rider validity ablation..."
if ($CUDA_AVAILABLE) { Write-Host "  --> Running free_rider_ablation.py on GPU (CUDA)..." -ForegroundColor Cyan }
& $PYTHON analysis/free_rider_ablation.py

Write-Host "Stage 13: Running analysis and main plotting script..."
& $PYTHON analysis/analyse.py

Write-Host "Stage 14: Building final paper LaTeX tables..."
& $PYTHON reporting/build_paper_tables.py

Write-Host "Stage 15: Building final paper figures..."
& $PYTHON reporting/build_paper_figures.py

Write-Host "======================================================================="
Write-Host "   FULL PIPELINE EXECUTION SUCCESSFUL!"
Write-Host "======================================================================="
