# Vmem Robustness Benchmark & Inference Runner

This directory contains the **Membrane Potential ($v_{\text{mem}}$) Robustness and Out-of-Distribution (OOD) Benchmark** runner for the Hybrid SNN-ANN Object Detection model. It hooks and monitors internal spiking neuron activations (PLIF layers) under clean and corrupted event streams to construct expressive "cognitive state vectors" ($\phi$).

This guide walks you through setting up and running the benchmark from absolute scratch (starting from downloading code and data) to running the pipeline on any machine.

---

## 1. Directory & Workspace Structure

To make installation and execution completely seamless, the pipeline sibling auto-detection works out-of-the-box if you organize your workspace as shown below. The parent folder can be named anything (e.g. `C:\Users\researcher\vmem_project`):

```text
<ANY_PARENT_WORKSPACE_DIR>/          # Your root parent folder (name does not matter!)
│
├── HybridDetection/                 # Backbone Model Repository
│   ├── config/                      # YAML files for datasets/models
│   ├── modules/                     # SNN-ANN architecture implementations
│   └── gen1_mAP36.ckpt              # Trained network checkpoint (weights)
│
├── event_corruption/                # Event stream corruption simulation library
│   ├── pipeline/                    # Python loader and saver utilities
│   └── corrupt/                     # GPU/CPU event-corruption mathematical operators
│
├── vmem_benchmark/                  # THIS Repository (Membrane Potential Benchmark)
│   ├── extract.py                   # Main inference runner and feature extractor
│   ├── analyse.py                   # Plotting and OOD score generation utility
│   ├── requirements.txt             # Unified python package dependencies list
│   ├── monitor.py                   # LIF hooking and pooling engine
│   ├── model_loader.py              # Configuration assembler and checkpoint loader
│   ├── test_pipeline.py             # Integrity check and validator script
│   └── outputs/                     # (Auto-created) Feature/trajectory tensors & plots
│
└── gen1/                            # Preprocessed Gen1 Dataset
    └── test/                        # Target evaluation split containing event subfolders
```

---

## 2. Configuring Paths Manually (CRITICAL)

If you are running the benchmark in a custom directory structure, or if the auto-detection is not aligned with your environment, you can configure the paths manually using **two highly flexible methods**:

### Method A: Editing the Central Configuration File (Persistent Setup)
Open the file `vmem_benchmark/benchmark_config.py` in any text editor. Locate lines **19 to 21** and modify the three main paths to match your local folders:

```python
# ---------------------------------------------------------------------------
# !! EDIT THESE THREE PATHS TO MATCH YOUR LOCAL SYSTEM !!
# ---------------------------------------------------------------------------
GEN1_ROOT  = Path("C:/your/custom/path/to/gen1")                        # Path containing train/val/test splits
CKPT_PATH  = Path("C:/your/custom/path/to/HybridDetection/gen1_mAP36.ckpt")  # Path to the model weights checkpoint
HYBRID_DIR = Path("C:/your/custom/path/to/HybridDetection")             # Path to the HybridDetection repository
```
*Note: You can use standard absolute paths (e.g., `"D:/Datasets/gen1"`) or relative paths (e.g., `Path("../gen1")`) depending on your preference.*

### Method B: Passing Path Overrides via Command Line CLI (Dynamic Setup)
You can leave `benchmark_config.py` unmodified and explicitly supply paths at execution time. The script accepts these standard command-line flags:

* `--gen1-root`: Direct path to your `gen1` dataset root folder.
* `--ckpt`: Direct path to your model checkpoint `.ckpt` weights file.
* `--hybrid-dir`: Direct path to your `HybridDetection` repository root.
* `--corruption-dir`: Direct path to your `event_corruption` repository root.
* `--output-dir`: Direct path to the folder where you want to write benchmark output files.

**Example Command:**
```bash
python extract.py --gen1-root "E:/Data/gen1" --ckpt "F:/Models/gen1_mAP36.ckpt" --hybrid-dir "F:/Repositories/HybridDetection" --corruption-dir "F:/event_corruption" --output-dir "./custom_outputs"
```

---

## 3. Step-by-Step Installation Guide

Follow these steps in your terminal (PowerShell on Windows, or Bash on Linux) to set up the entire environment:

### Step 3.1: Navigate to Workspace and Activate Environment
Ensure you are in the `vmem_benchmark` directory:
```bash
cd vmem_benchmark

# Create virtual environment named '.venv'
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

### Step 3.2: Upgrade Package Installer
```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 3.3: Install PyTorch with CUDA 12.1
Install PyTorch version `2.1.2` which is highly optimized for the spiking convolutional layers in SpikingJelly:
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchdata==0.7.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3.4: Install Remaining Dependencies
Install all required secondary packages (SpikingJelly, PyTables, Omegaconf, etc.) directly using the updated `requirements.txt`:
```bash
pip install -r requirements.txt
```

> [!NOTE]  
> **PyTables Installation Note:** The `tables` package handles rapid HDF5 loads without the standard `h5py` chunk regression. It is automatically installed via `requirements.txt`.
> 
> **CuPy Installation Note:** `requirements.txt` installs `cupy-cuda12x` by default for CUDA 12.x GPUs. If you are using a CUDA 11.x GPU, uninstall it and install the matching version:
> ```bash
> pip uninstall -y cupy-cuda12x
> pip install cupy-cuda11x
> ```

---

## 4. High-GPU vs. Low-GPU Resource Profiles

To run the benchmark effectively on both laptops/desktop workstations (8 GB VRAM) and heavy server rigs (16 GB - 80 GB VRAM), choose the appropriate command parameters below:

| GPU Category | GPU Examples | Recommended Commands & Parameters | VRAM Allocation | Details / Target |
| :--- | :--- | :--- | :---: | :--- |
| **Low-VRAM GPU** | Laptop GPUs, RTX 3060, RTX 4060 (8 GB or less) | `python extract.py --vram-fraction 0.7` | Capped to **~5.5 GB** | Limits PyTorch memory allocations to 70%, preventing Windows system lag, UI freezes, or fragmented page faults. |
| **High-VRAM GPU** | RTX 3090, RTX 4090, A100, H100 (16 GB to 80 GB) | `python extract.py --vram-fraction 1.0` | **Unlimited** (Default) | Operates without memory restrictions, utilizing full hardware capacity. |

---

## 5. How to Execute the Benchmark

The benchmark performs **31 full evaluation passes** (1 Clean + 6 Corruptions × 5 Severity Levels) across your sequence files, saving SNN potential trajectories and pooling state vectors ($\phi$).

### 5.1 Check Setup Integrity (Dry Run)
Before launching a full, multi-hour run, execute the comprehensive test suite to verify that the weights, SNN hooks, and event-corruption engines are working seamlessly:
```bash
python test_pipeline.py
```
If you see `SUCCESS: ALL ENCOMPASSING TESTS PASSED FLAWLESSLY!`, your environment is fully operational!

### 5.2 Run the Robustness Feature Extraction
Execute the extraction script with the desired GPU VRAM profile (combining path overrides if using **Method B**):

* **For Low-VRAM (8 GB GPUs):**
  ```bash
  python extract.py --vram-fraction 0.7
  ```
  *(To speed up testing, you can cap the number of sequences evaluated using `--max-seq 20` or hook a subset of SNN layers e.g. `--plif-layers 2 3`).*

* **For High-VRAM (Large Workstations/Servers):**
  ```bash
  python extract.py --vram-fraction 1.0
  ```

### 5.3 Downstream Analysis & Plotting
Once the feature extraction finishes, run the downstream analysis to fit the Mahalanobis OOD detector and generate publication-quality diagnostic figures:
```bash
python analyse.py
```
This will automatically generate a new `outputs/plots/` folder containing:
1. `auroc_vs_severity.pdf` (Robustness Out-of-Distribution alarm curves)
2. `sensitivity_heatmap.pdf` (L2 activation shift in state space)
3. `trajectories_L0.pdf` (Continuous membrane potentials $v_{\text{mem}}(t)$ over SNN time steps)

---

## 6. Built-in Enterprise-Grade Guardrails

1. **Mid-Run Crash Recovery & Resume:** If your machine loses power, crashes, or is aborted mid-way, **simply run the exact same command again**. The script will automatically scan the output directory, load previously processed sequence chunks (`seq_*.pt`), output a `[resume]` message, and pick up right where it was interrupted. Completed passes are automatically skipped (`[skip]`).
2. **Aggressive VRAM Cleanup:** Membrane potentials are detatched (`.detach()`) and immediately migrated to system CPU memory (`.cpu()`) and garbage-collected at sequence boundaries. This prevents VRAM accumulation and memory leaks over long multi-hour runs.

---

## 7. CRITICAL WARNING: The Strict $B = 1$ Mathematical Constraint

> [!CAUTION]
> **NEVER set `--batch-size > 1` when running the benchmark.**
> 
> The CLI script `extract.py` contains a strict guard validation that will throw a fatal `ValueError` if a batch size greater than 1 is passed.
> 
> **Why B = 1 is mathematically required:**
> - SpikingJelly's convolutional SNN backbone unrolls the spiking process over its **first dimension (dimension 0)** as the time axis.
> - The original model's SNN backbone implementation (`spike_model.py`) formats the temporal inputs as `(B, T, C, H, W)` where `B` is the batch size and `T = 10` are the micro-timesteps. It feeds this tensor directly into SpikingJelly without permuting the axes.
> - As a result, the SNN is forced to treat the batch dimension `B` as the temporal time axis, and treats the 10 timesteps `T` as parallel batch items!
> - If `B = 1`, the SNN runs for exactly 1 step (treating the 10 timesteps as parallel batches). While mathematically unique, the trained model weights are completely adapted to this shape contract, and batch independence is preserved (since there is only one sample).
> - If `B > 1` (e.g. `B = 2`), the SNN unrolls for 2 time steps. The membrane potentials from batch item 1 **leak directly** into batch item 2 as its initial charging state. This introduces severe cross-sample state leakage, causing completely incorrect and scientifically invalid features.
> - Therefore, to produce mathematically valid outputs and match the official validation/visualization outputs, **inference must be strictly executed with BATCH_SIZE = 1**.
