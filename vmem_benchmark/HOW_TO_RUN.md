# HOW TO RUN

## 1. Installation

> [!IMPORTANT]
> **System Requirement:** This benchmark requires **Python >= 3.11** (recommended: Python 3.11 or 3.12) due to library compatibility restrictions (e.g. SciPy v1.16+).

Run these commands in your terminal inside the `vmem_benchmark` directory to set up the environment and install all dependencies:

```powershell
# Create virtual environment with Python 3.11+ using the Windows Python launcher
py -3.11 -m venv .venv
# (Or if you prefer Python 3.12: py -3.12 -m venv .venv)

# Activate virtual environment
.venv\Scripts\activate

# Upgrade pip and install compatible setuptools/PyTorch
python -m pip install --upgrade pip wheel
pip install "setuptools==81.0.0"
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install benchmark dependencies
pip install -r requirements.txt
```

---

## 2. Run the Benchmark

Run the feature extraction command by explicitly passing the paths to your local directories:

### High-GPU (16GB+ VRAM):
```powershell
python extract.py --gen1-root "C:/path/to/gen1" --ckpt "C:/path/to/HybridDetection/gen1_mAP36.ckpt" --hybrid-dir "C:/path/to/HybridDetection" --corruption-dir "C:/path/to/event_corruption" --vram-fraction 1.0
```

### Low-GPU (8GB VRAM or less):
```powershell
python extract.py --gen1-root "C:/path/to/gen1" --ckpt "C:/path/to/HybridDetection/gen1_mAP36.ckpt" --hybrid-dir "C:/path/to/HybridDetection" --corruption-dir "C:/path/to/event_corruption" --vram-fraction 0.7
```

---

## 3. Run Downstream Analysis

Once features have been extracted (the `outputs/` folder is populated with `.pt` files in `outputs/phi/` and `outputs/trajs/`), execute the analysis script to train density estimators, evaluate sequence learning, generate comparisons, and save tables/plots:

### Standard Run
```powershell
python analyse.py
```

### Command Line Arguments for `analyse.py`
* `--fast`: Runs in fast test mode. Subsamples dataset features to a maximum of 2,000 frames and caps model training (2 epochs for RealNVP/MLP-AE, 5 epochs for Temporal-AE) to verify the pipeline within ~30 seconds.
* `--output-dir <path>`: Overrides the path to the directory containing `phi/` and `trajs/` subdirectories. By default, looks for a local directory named `outputs/`.

**Example with custom paths:**
```powershell
python analyse.py --output-dir "D:/my_custom_run_outputs" --fast
```
