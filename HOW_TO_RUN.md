# HOW TO RUN THE SNN Robustness Benchmark

This guide outlines how to install dependencies, configure the environment, and execute the robustness evaluation pipeline.

---

## 1. Installation

Execute the following commands to initialize the Python virtual environment and install dependencies:

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Upgrade pip and install compatible setuptools/PyTorch
python -m pip install --upgrade pip wheel
pip install "setuptools==81.0.0"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install benchmark dependencies
pip install -r requirements.txt
```

---

## 2. Running the Full Pipeline

The benchmark can be executed end-to-end using the native runner scripts in the root directory. They run Stage 1 in parallel to speed up extraction, then run the downstream analysis and reporting scripts.

### On Windows (PowerShell):
```powershell
# Run with default settings (automatically uses parallel extraction on visible GPUs)
.\run_full_benchmark.ps1

# Run with custom paths and settings forwarded to the extraction engine
.\run_full_benchmark.ps1 --gpus 0 --workers-per-gpu 2 --gen1-root "C:/path/to/gen1" --output-dir "C:/path/to/outputs"
```

### On Linux / Bash:
```bash
# Run with default settings
./run_full_benchmark.sh

# Run with custom paths and settings
./run_full_benchmark.sh --gpus 0 1 2 3 --max-seq 200
```

---

## 3. Running Parallel Extraction Manually

If you only want to run the parallel extraction stage (`Stage 1`) to generate feature files:

```powershell
# Run across 4 GPUs on a cluster node (one worker per GPU)
python vmem_benchmark/run_parallel_extract.py --gpus 0 1 2 3

# Run on a single GPU with 3 parallel processes (maximizes GPU utilization)
python vmem_benchmark/run_parallel_extract.py --gpus 0 --workers-per-gpu 3

# Override dataset paths and split
python vmem_benchmark/run_parallel_extract.py --gpus 0 --gen1-root "/path/to/gen1" --split test
```

*Logs for individual parallel worker processes are saved under `vmem_benchmark/outputs/logs/worker_<id>.log` to prevent interleaved console prints.*
