"""
export_weights.py — Export the HybridDetection checkpoint to a standard PyTorch state_dict.

This script:
1. Loads the Lightning .ckpt file.
2. Extracts the 'state_dict'.
3. Applies legacy key remapping (LSTM and ANN feature renames).
4. Saves as a clean .pt file.
"""
import torch
import sys
from pathlib import Path

# Setup paths to reuse the remapping logic
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import benchmark_config as cfg
from model_loader import _remap_legacy_backbone_keys

def export():
    input_path = cfg.CKPT_PATH
    output_path = input_path.with_suffix(".pt")

    print(f"Loading checkpoint from: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")
    
    # Lightning checkpoints have 'state_dict' key
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        print("Found 'state_dict' in checkpoint.")
    else:
        state_dict = ckpt
        print("No 'state_dict' wrapper found, using raw checkpoint.")

    # Apply legacy remapping
    print("Applying legacy key remapping...")
    remapped_sd = _remap_legacy_backbone_keys(state_dict)
    
    # Strip the 'mdl.' prefix if present to make it a bare model state_dict
    # (Optional, but often cleaner for pure-PyTorch usage)
    clean_sd = {}
    for k, v in remapped_sd.items():
        if k.startswith("mdl."):
            clean_sd[k[4:]] = v
        else:
            clean_sd[k] = v

    print(f"Exporting {len(clean_sd)} keys to: {output_path}")
    torch.save(clean_sd, output_path)
    print("Export complete.")

if __name__ == "__main__":
    export()
