# run_test_pipeline.ps1
# Runs a fast validation test of the full SNN-ANN benchmark pipeline end-to-end (2 sequences).

$ErrorActionPreference = "Stop"

$CONFIG_FILE = "vmem_benchmark/benchmark_config.py"
$BACKUP_FILE = "vmem_benchmark/benchmark_config.py.bak"

Write-Host "======================================================================="
Write-Host "   PREPARING FAST PIPELINE TEST (Capping sequences to 2)"
Write-Host "======================================================================="

# 1. Back up original config
if (Test-Path $BACKUP_FILE) { Remove-Item $BACKUP_FILE }
Copy-Item $CONFIG_FILE -Destination $BACKUP_FILE

try {
    # 2. Append temporary test overrides to config file (will override previous declarations)
    $Overrides = @"

# --- TEMPORARY TEST OVERRIDES ---
MAX_SEQUENCES = 1
CORRUPTIONS = ["hot_pixel", "event_flood"]
SEVERITIES = [5]
"@
    Add-Content $CONFIG_FILE $Overrides

    # 3. Execute full benchmark forwarding any arguments (like GPU configuration)
    .\run_full_benchmark.ps1 @args
}
finally {
    # 4. Restore original config
    if (Test-Path $BACKUP_FILE) {
        Copy-Item $BACKUP_FILE -Destination $CONFIG_FILE -Force
        Remove-Item $BACKUP_FILE
        Write-Host "--> Restored original configuration." -ForegroundColor Green
    }
}
