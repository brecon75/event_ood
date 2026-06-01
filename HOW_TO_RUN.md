# HOW TO RUN

## 1. Installation

Run these commands in your terminal inside the `vmem_benchmark` directory to set up the environment and install all dependencies:

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
