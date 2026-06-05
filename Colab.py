import zipfile
from tqdm import tqdm
import os
import sys

zip_path = "/content/drive/MyDrive/Colab_Vmem/test.zip"
extract_path = "/content/gen1"

# Create the folder if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zf:
    file_list = zf.namelist()
    print(f"Found {len(file_list)} files in the dataset.")

    # Extract each file while updating the tqdm bar
    for file in tqdm(file_list, desc="Unzipping to SSD", unit="file"):
        zf.extract(file, path=extract_path)

print("Unzipping complete!")


# 3. Create a clean slate for the local workspace
!rm -rf /content/workspace
!mkdir -p /content/workspace

# Copy your code over
!cp -r "/content/drive/MyDrive/Colab_Vmem/vmem_benchmark" /content/workspace/
!cp -r "/content/drive/MyDrive/Colab_Vmem/event_corruption" /content/workspace/
!cp -r "/content/drive/MyDrive/Colab_Vmem/HybridDetection" /content/workspace/

# 4. Move your checkpoint into the HybridDetection folder
!cp "/content/drive/MyDrive/Colab_Vmem/gen1_mAP36.ckpt" /content/workspace/HybridDetection/

# 5. Point the output directory directly back to Google Drive for safe saving
# We remove the local output folder first just in case it was copied from Drive
!rm -rf /content/workspace/vmem_benchmark/outputs
!mkdir -p "/content/drive/MyDrive/Colab_Vmem/outputs"
!ln -s "/content/drive/MyDrive/Colab_Vmem/outputs" /content/workspace/vmem_benchmark/outputs

# 6. Patch benchmark_config.py with Colab-correct paths
#    (the file has hardcoded d:/Perdue/... paths from local dev)
config_path = "/content/workspace/vmem_benchmark/benchmark_config.py"
with open(config_path, "r") as f:
    cfg_text = f.read()

cfg_text = cfg_text.replace(
    'GEN1_ROOT  = Path("d:/Perdue/gen1")',
    'GEN1_ROOT  = Path("/content/gen1")'
)
cfg_text = cfg_text.replace(
    'CKPT_PATH  = Path("d:/Perdue/HybridDetection/gen1_mAP36.ckpt")',
    'CKPT_PATH  = Path("/content/workspace/HybridDetection/gen1_mAP36.ckpt")'
)
cfg_text = cfg_text.replace(
    'HYBRID_DIR = Path("d:/Perdue/HybridDetection")',
    'HYBRID_DIR = Path("/content/workspace/HybridDetection")'
)
cfg_text = cfg_text.replace(
    'OUTPUT_DIR = Path("d:/Perdue/vmem_benchmark/outputs")',
    'OUTPUT_DIR = Path("/content/workspace/vmem_benchmark/outputs")'
)

with open(config_path, "w") as f:
    f.write(cfg_text)

print("benchmark_config.py patched for Colab paths.")

# 7. Install all required dependencies
!pip install -q spikingjelly>=0.0.0.0.14 pytorch-lightning==1.8.6 tables>=3.9 hdf5plugin scikit-learn einops>=0.8.2 torchdata==0.7.1

# 8. Run — HybridDetection must be on PYTHONPATH so its internal imports resolve
%cd /content/workspace/vmem_benchmark
!PYTHONPATH=/content/workspace/HybridDetection:/content/workspace/event_corruption:$PYTHONPATH python extract.py
