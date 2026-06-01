"""
corruption_wrap.py — Bridge between event_corruption/corrupt/ and torch histograms.

This module provides a unified interface to apply any of the 6 corruptions
to a PyTorch histogram tensor of shape (N, 20, H, W).
"""
import torch
import numpy as np
import sys
from pathlib import Path

# Add the event_corruption root to sys.path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "event_corruption"))

from corrupt.registry import apply_corruption
import benchmark_config as cfg

def apply_corruption_to_tensor(
    tensor: torch.Tensor,
    name: str,
    severity: int,
    seed: int = 42
) -> torch.Tensor:
    """
    Apply a named corruption to a torch.Tensor (N, 20, H, W).
    
    Parameters
    ----------
    tensor   : (N, 20, H, W) float32 or uint8 torch.Tensor
    name     : corruption name (e.g., 'hot_pixel')
    severity : 1-5
    seed     : for deterministic results
    
    Returns
    -------
    Corrupted (N, 20, H, W) torch.Tensor (uint8)
    """
    # 1. Prepare data
    # Corruptions expect uint8 numpy arrays
    device = tensor.device
    if tensor.is_floating_point():
        # Histograms are usually stored as counts; if they are floats, 
        # we assume they are already in the 0-255 range.
        arr = tensor.detach().cpu().numpy().astype(np.uint8)
    else:
        arr = tensor.detach().cpu().numpy()

    # 2. Setup RNG
    rng = np.random.default_rng(seed)

    # 3. Apply via registry (using timestamps=None since most don't use it for histograms)
    # The registry uses (histogram, timestamps, name, severity, rng)
    corrupted_arr = apply_corruption(arr, None, name, severity, rng)

    # 4. Return as tensor on original device
    return torch.from_numpy(corrupted_arr).to(device)

def get_corruption_names():
    return cfg.CORRUPTIONS

if __name__ == "__main__":
    # Quick sanity check
    test_tensor = torch.zeros((10, 20, 240, 304), dtype=torch.uint8)
    corrupted = apply_corruption_to_tensor(test_tensor, "hot_pixel", 5)
    print(f"Original sum: {test_tensor.sum()}")
    print(f"Corrupted sum (hot_pixel L5): {corrupted.sum()}")
    assert corrupted.sum() > 0
    print("Sanity check passed.")
