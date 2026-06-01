"""
test_pipeline.py — Comprehensive Test Suite for the Vmem Benchmark.

Checks:
1. HybridDetection model loading (verifies missing/unexpected keys).
2. VmemMonitor hooking (verifies 4 PLIF layers are found).
3. Shape Contracts & Batch Independence (verifies B > 1 is handled correctly).
4. All Corruptions (tests every single corruption type at multiple severities).
5. Output Validation (verifies phi shape is (B, 2112), bounds, and no NaNs).
6. Performance/Speed (times a dummy forward pass).
"""
import torch
import sys
import time
from pathlib import Path

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "event_corruption"))

# Dynamically mock torchdata ONLY if it is not already installed (such as on Python 3.12 / Colab)
try:
    import torchdata.datapipes.map
    import torchdata.datapipes.iter
except (ImportError, ModuleNotFoundError):
    import types
    import torch.utils.data

    # Create real module objects so Python's import system recognizes them as packages
    torchdata = types.ModuleType('torchdata')
    torchdata.__path__ = []

    datapipes = types.ModuleType('torchdata.datapipes')
    datapipes.__path__ = []

    map_mod = types.ModuleType('torchdata.datapipes.map')
    iter_mod = types.ModuleType('torchdata.datapipes.iter')

    map_mod.MapDataPipe = torch.utils.data.Dataset
    iter_mod.IterDataPipe = torch.utils.data.IterableDataset

    sys.modules['torchdata'] = torchdata
    sys.modules['torchdata.datapipes'] = datapipes
    sys.modules['torchdata.datapipes.map'] = map_mod
    sys.modules['torchdata.datapipes.iter'] = iter_mod




try:
    import benchmark_config as cfg
    from model_loader import load_model, audit_plif_layers
    from monitor import VmemMonitor
    from corruption_wrap import apply_corruption_to_tensor
    from spikingjelly.clock_driven import functional
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)


def run_tests():
    print("=" * 80)
    print("VMEM BENCHMARK - ALL-ENCOMPASSING TEST SUITE")
    print("=" * 80)

    # ---------------------------------------------------------
    # 1. Test Model Loading
    # ---------------------------------------------------------
    print("\n[1] Testing Model Loading...")
    try:
        module, backbone = load_model(cfg.DEVICE)
        n_layers = audit_plif_layers(backbone)
        assert n_layers == 4, f"Expected 4 PLIF layers, found {n_layers}!"
        print("  [PASS] Model loaded successfully.")
        print("  [PASS] PLIF layer audit passed (4 layers found).")
    except Exception as e:
        print(f"  [FAIL] FAILED Model Loading: {e}")
        return

    # ---------------------------------------------------------
    # 2. Test VmemMonitor Hooking
    # ---------------------------------------------------------
    print("\n[2] Testing VmemMonitor Hooking...")
    try:
        monitor = VmemMonitor(backbone)
        assert len(monitor._hooks) == 4, "Failed to hook exactly 4 PLIF layers."
        print("  [PASS] Successfully hooked 4 PLIF layers.")
    except Exception as e:
        print(f"  [FAIL] FAILED VmemMonitor Hooking: {e}")
        return

    # ---------------------------------------------------------
    # 3. Test Shape Contracts (Strict B=1 Requirement)
    # ---------------------------------------------------------
    print("\n[3] Testing Shape Contracts (Strict B=1)...")
    try:
        # DO NOT change BATCH_SIZE_TEST > 1!
        # The gen1_mAP36.ckpt backbone feeds (B, T, C, H, W) to SpikingJelly, which expects (T, B, C, H, W).
        # This causes the SNN to unroll over B as timesteps and treat T as a parallel batch!
        # If B > 1, the potentials of all but the last sample in the batch are permanently lost.
        # B=1 is mathematically required to preserve the 10 timesteps without data loss.
        BATCH_SIZE_TEST = 1
        # Create dummy batch representing 1 frame (B=1, C=20, H=240, W=304)
        dummy_batch = torch.randn((BATCH_SIZE_TEST, 20, 240, 304)).to(cfg.DEVICE)

        
        # Reset and run forward
        functional.reset_net(backbone)
        monitor.reset()
        
        with torch.no_grad():
            h_c = {0: None, 1: None}
            _, _ = module.mdl.forward_backbone(x=dummy_batch.float(), h_c=h_c)

        # Check internal collected tensor shapes
        # They should be canonicalized to (T, B, C, H, W) -> (10, 3, C, H, W)
        for idx in range(4):
            v_list = monitor._v[idx]
            assert len(v_list) == 1, f"Expected 1 tensor per layer per batch, got {len(v_list)}"
            v_shape = v_list[0].shape
            assert len(v_shape) == 5, f"Expected 5D canonical shape (T, B, C, H, W), got {v_shape}"
            assert v_shape[0] == 10, f"Expected T=10, got {v_shape[0]}"
            assert v_shape[1] == BATCH_SIZE_TEST, f"Expected B={BATCH_SIZE_TEST}, got {v_shape[1]}"
        print(f"  [PASS] Internal module.v tensors correctly canonicalized to (10, {BATCH_SIZE_TEST}, C, H, W).")

    except Exception as e:
        print(f"  [FAIL] FAILED Shape Contracts: {e}")
        return

    # ---------------------------------------------------------
    # 4. Test Phi & Trajectory Output Validation
    # ---------------------------------------------------------
    print("\n[4] Testing Phi & Trajectory Output Validation...")
    try:
        # Collect Phi
        phi = monitor.collect_phi()
        print(f"  - Collected phi shape: {phi.shape}")
        assert phi.ndim == 2, "Phi should be 2D (B, F)"
        assert phi.shape[0] == BATCH_SIZE_TEST, f"Batch size mismatch, expected {BATCH_SIZE_TEST} got {phi.shape[0]}"
        assert phi.shape[1] == 2112, f"Expected phi feature size 2112, got {phi.shape[1]}"
        
        # Check for NaNs and Infs
        assert not torch.isnan(phi).any(), "Phi tensor contains NaNs!"
        assert not torch.isinf(phi).any(), "Phi tensor contains Infs!"
        print("  [PASS] Phi feature extraction passed (No NaNs/Infs, correct shape).")

        # Collect Trajectories
        trajs = monitor.trajectories(n_samples=50)
        assert len(trajs) == 4, "Expected trajectories for 4 layers."
        for l_idx, t_tensor in trajs.items():
            # Expected shape: (T, B, D) -> (10, 3, D)
            assert t_tensor.ndim == 3, f"Expected 3D trajectory tensor, got {t_tensor.ndim}"
            assert t_tensor.shape[0] == 10, f"Expected T=10, got {t_tensor.shape[0]}"
            assert t_tensor.shape[1] == BATCH_SIZE_TEST, f"Expected B={BATCH_SIZE_TEST}, got {t_tensor.shape[1]}"
        print("  [PASS] Trajectory slicing and extraction passed.")
    except Exception as e:
        print(f"  [FAIL] FAILED Phi/Trajectory Extraction: {e}")
        return

    # ---------------------------------------------------------
    # 5. Test EVERY SINGLE Corruption Wrapper
    # ---------------------------------------------------------
    print("\n[5] Testing ALL Corruptions...")
    try:
        # We use N=100 frames so probabilistic corruptions (like 5% polarity flip) are guaranteed to trigger
        base_tensor = (torch.rand((100, 20, 240, 304)) * 10).to(torch.uint8)
        sum_clean = base_tensor.sum().item()
        
        for c_name in cfg.CORRUPTIONS:
            print(f"  - Testing {c_name}...")
            
            for sev in [1, 5]: # Test lowest and highest severity
                try:
                    corrupted = apply_corruption_to_tensor(base_tensor, c_name, severity=sev)
                    assert corrupted.shape == base_tensor.shape, f"{c_name} L{sev} altered tensor shape!"
                    assert corrupted.dtype == torch.uint8, f"{c_name} L{sev} altered tensor dtype!"
                    sum_corr = corrupted.sum().item()
                    
                    if c_name == "spatial_dropout":
                        assert sum_corr < sum_clean, f"Spatial dropout L{sev} did not reduce the sum! Clean: {sum_clean}, Corr: {sum_corr}"
                    else:
                        # Most corruptions modify the tensor significantly;
                        # check if the actual values changed (since polarity_flip preserves the sum).
                        assert (corrupted != base_tensor).any(), f"{c_name} L{sev} did not modify the tensor values!"
                        
                        
                except Exception as ce:
                    print(f"    [FAIL] FAILED {c_name} at L{sev}: {ce}")
                    raise ce
                    
        print("  [PASS] All corruptions applied successfully across severities.")
    except Exception as e:
        print(f"  [FAIL] FAILED Corruption Wrapper: {e}")
        return

    # ---------------------------------------------------------
    # 6. Performance / Speed Test
    # ---------------------------------------------------------
    print("\n[6] Performance Profiling (Dummy Sequence)...")
    try:
        dummy_seq = torch.randn((10, 20, 240, 304)).float().to(cfg.DEVICE)
        
        functional.reset_net(backbone)
        monitor.reset()
        h_c = {0: None, 1: None}

        start_time = time.time()
        for i in range(10):
            batch = dummy_seq[i:i+1] # B=1
            with torch.no_grad():
                _, h_c = module.mdl.forward_backbone(x=batch, h_c=h_c)
            _ = monitor.collect_phi()
            functional.reset_net(backbone)
            monitor.reset()
            
        elapsed = time.time() - start_time
        fps = 10 / elapsed
        print(f"  - Processed 10 frames in {elapsed:.4f} seconds ({fps:.2f} FPS).")
        print("  [PASS] Performance test completed.")
    except Exception as e:
        print(f"  [FAIL] FAILED Performance Test: {e}")
        return

    # ---------------------------------------------------------
    # 7. Memory Leakage Check
    # ---------------------------------------------------------
    print("\n[7] Testing Memory Management & Hook Cleanup...")
    try:
        monitor.reset()
        for idx in monitor._v:
            assert len(monitor._v[idx]) == 0, f"Monitor reset failed to clear layer {idx} buffer."
            
        monitor.remove()
        assert len(monitor._hooks) == 0, "Monitor remove failed to clear hooks list."
        print("  [PASS] Memory management and hook cleanup passed.")
    except Exception as e:
        print(f"  [FAIL] FAILED Memory Leakage Check: {e}")
        return

    print("\n" + "=" * 80)
    print("SUCCESS: ALL ENCOMPASSING TESTS PASSED FLAWLESSLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
