# run_full_benchmark.ps1
# Runs the full benchmark pipeline end-to-end on the complete dataset in PowerShell.

$ErrorActionPreference = "Stop"

$PYTHON = ".\vmem_benchmark\.venv\Scripts\python.exe"

# ── Stage runner ──────────────────────────────────────────────────────────
# Windows PowerShell 5.1 does NOT turn a native command's non-zero exit code
# into a terminating error (even with $ErrorActionPreference = "Stop").
# Without this guard, a crashed stage would be silently skipped and the script
# would still print "SUCCESSFUL". Invoke-Stage checks $LASTEXITCODE after every
# stage and aborts immediately with a clear message naming the failed stage.
function Invoke-Stage {
    param(
        [int]$Number,
        [string]$Description,
        [string]$Script,
        [string[]]$ScriptArgs = @()
    )
    Write-Host "Stage ${Number}: $Description"
    & $PYTHON $Script @ScriptArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "=======================================================================" -ForegroundColor Red
        Write-Host "   PIPELINE FAILED at Stage ${Number} ($Script), exit code $LASTEXITCODE" -ForegroundColor Red
        Write-Host "   Fix the error above and re-run. Stages already completed are cached;" -ForegroundColor Red
        Write-Host "   re-running will resume rather than redo finished work." -ForegroundColor Red
        Write-Host "=======================================================================" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

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

if ($CUDA_AVAILABLE) { Write-Host "  --> Stage 1 (extraction) and GPU stages will use CUDA." -ForegroundColor Cyan }

Invoke-Stage 1  "Running parallel feature extraction (31 runs, all sequences)..."   "vmem_benchmark/run_parallel_extract.py" $args
Invoke-Stage 2  "Extracting offline features (Temporal AE + margin histograms)..."   "analysis/extract_offline_features.py"
Invoke-Stage 3  "Running feature fusion and Logistic Regression meta-classifier..."  "analysis/fusion_features.py"
Invoke-Stage 4  "Extracting ResNet-18 ANN baselines (event image & voxel grid)..."   "analysis/extract_ann_baselines.py"
Invoke-Stage 5  "Evaluating ResNet-18 ANN baselines..."                              "analysis/evaluate_ann_baselines.py"
Invoke-Stage 6  "Fitting OOD detectors on clean SNN fused representation..."          "analysis/fit_detectors.py"
Invoke-Stage 7  "Evaluating fitted OOD detectors on all corrupted runs..."           "analysis/evaluate_detectors.py"
Invoke-Stage 8  "Evaluating MDD (per-frame + per-sequence, all branches)..."          "analysis/evaluate_mdd.py"
Invoke-Stage 9  "Running representation ablation (Mahalanobis comparison)..."        "analysis/representation_ablation.py"
Invoke-Stage 10 "Running severity monotonicity analysis (Spearman rho)..."           "analysis/severity.py"
Invoke-Stage 11 "Running downstream task reliability prediction..."                  "analysis/reliability.py"
Invoke-Stage 12 "Running cross-corruption zero-shot generalization..."              "analysis/cross_corruption.py"
Invoke-Stage 13 "Running Free Rider validity ablation..."                            "analysis/free_rider_ablation.py"
Invoke-Stage 14 "Running analysis and main plotting script..."                       "analysis/analyse.py"
Invoke-Stage 15 "Building final paper LaTeX tables..."                               "reporting/build_paper_tables.py"
Invoke-Stage 16 "Building final paper figures..."                                    "reporting/build_paper_figures.py"

Write-Host "======================================================================="
Write-Host "   FULL PIPELINE EXECUTION SUCCESSFUL!" -ForegroundColor Green
Write-Host "======================================================================="
