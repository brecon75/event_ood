import sys
import gc
import csv
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "vmem_benchmark"))

# Sibling path resolution for HybridDetection and event_corruption
sibling_hybrid = _ROOT / "HybridDetection"
sibling_corruption = _ROOT / "event_corruption"
for path in (sibling_hybrid, sibling_corruption):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from vmem_benchmark import benchmark_config as cfg
from vmem_benchmark.model_loader import load_model
from vmem_benchmark.monitor import VmemMonitor
from event_corruption.pipeline.loader import load_histogram
from vmem_benchmark.corruption_wrap import apply_corruption_to_tensor
from spikingjelly.clock_driven import functional

from analysis.vmem_scorers import mahalanobis_scorer
from analysis.vmem_utils import auroc_fpr95, TABLE_DIR
from analysis.analyse_plots import plot_free_rider_ablation

def extract_snn_phi(module, backbone, monitor, hist_torch, device, desc="Extracting Vmem", batch_size=1):
    functional.reset_net(backbone)
    monitor.reset()
    seq_phi = []
    n_frames = hist_torch.shape[0]
    h_c = {0: None, 1: None}
    
    pbar = tqdm(range(0, n_frames, batch_size), desc=desc, leave=False)
    for j in pbar:
        batch_end = min(j + batch_size, n_frames)
        batch = hist_torch[j:batch_end].to(device).float()
        functional.reset_net(backbone)
        with torch.no_grad():
            _, h_c = module.mdl.forward_backbone(x=batch, h_c=h_c)
        phi_batch = monitor.collect_phi()
        monitor.reset()
        if phi_batch.numel() > 0:
            seq_phi.append(phi_batch.cpu())
            
    if not seq_phi:
        return torch.empty((0, 2112))
    return torch.cat(seq_phi, dim=0)


def extract_raw_input_stats(hist_torch):
    if hist_torch.ndim == 5:
        N_frames, T, C, H, W = hist_torch.shape
        x_flat = hist_torch.view(N_frames, T * C, H * W).float()
    elif hist_torch.ndim == 4:
        N_frames, TC, H, W = hist_torch.shape
        x_flat = hist_torch.view(N_frames, TC, H * W).float()
    else:
        raise ValueError(f"Unexpected shape for hist_torch: {hist_torch.shape}")
    
    mu = x_flat.mean(dim=-1)  # (N_frames, T*C)
    diff = x_flat - mu.unsqueeze(-1)
    var = (diff ** 2).mean(dim=-1)  # (N_frames, T*C)
    
    std = torch.sqrt(var).clamp(min=1e-8)
    kurt = (diff ** 4).mean(dim=-1) / (std ** 4) - 3.0  # (N_frames, T*C)
    
    raw_phi = torch.cat([mu, var, kurt], dim=-1)
    return raw_phi


def randomize_weights(backbone):
    for m in backbone.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv1d, torch.nn.Linear)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0.0)


def main():
    print("\n======================================================")
    print(" LEVEL 10 - Free Rider Ablation (Idea 9)")
    print("======================================================")
    
    device = cfg.DEVICE
    cuda_avail = torch.cuda.is_available()
    print(f"[CUDA Status] PyTorch CUDA available: {cuda_avail}")
    print(f"[CUDA Status] Configured device: {device}")
    if device == "cuda" and cuda_avail:
        print("[CUDA Status] CUDA is active and will be used for SNN Vmem extraction.")
    else:
        print("[CUDA Status] WARNING: Running on CPU (CUDA not active/available).")
    print("======================================================\n")

    input_dir = cfg.GEN1_ROOT / "val"
    if not input_dir.exists():
        print(f"Error: val directory not found in {cfg.GEN1_ROOT}")
        return
        
    label_files = sorted(input_dir.glob("*/labels_v2/labels.npz"))
    max_seq = 5
    if getattr(cfg, "MAX_SEQUENCES", None) == 1 or "--test" in sys.argv or "--fast" in sys.argv:
        max_seq = 1
    seq_dirs = [p.parent.parent for p in label_files][:max_seq]
    if not seq_dirs:
        print(f"No validation sequences found in {input_dir}")
        return
        
    print(f"Running Free Rider Ablation on {len(seq_dirs)} validation sequences...")

    # Pre-load all sequence inputs (Clean, Hot Pixel L5, Event Flood L5)
    clean_inputs = []
    hot_inputs = []
    flood_inputs = []
    
    for i, seq_dir in enumerate(tqdm(seq_dirs, desc="Loading validation sequences")):
        hist_np, _ = load_histogram(seq_dir)
        hist_clean = torch.from_numpy(hist_np)
        clean_inputs.append(hist_clean)
        
        hist_hot = apply_corruption_to_tensor(hist_clean, "hot_pixel", 5)
        hot_inputs.append(hist_hot)
        
        hist_flood = apply_corruption_to_tensor(hist_clean, "event_flood", 5)
        flood_inputs.append(hist_flood)
        
    print("\n[Condition C] Extracting Raw Input Stats...")
    clean_raw = torch.cat([extract_raw_input_stats(h) for h in clean_inputs], dim=0).numpy()
    hot_raw = torch.cat([extract_raw_input_stats(h) for h in hot_inputs], dim=0).numpy()
    flood_raw = torch.cat([extract_raw_input_stats(h) for h in flood_inputs], dim=0).numpy()

    device = cfg.DEVICE
    print(f"\n[Condition A] Loading Trained SNN Model...")
    module, backbone = load_model(device)
    monitor = VmemMonitor(backbone, selected=cfg.PLIF_LAYERS)
    print(f"[Model Status] Backbone parameters loaded on: {next(backbone.parameters()).device}")
    
    print("Extracting Trained SNN Vmem Stats...")
    clean_trained = torch.cat([extract_snn_phi(module, backbone, monitor, h, device, desc=f"Trained SNN (Clean) seq {i}") for i, h in enumerate(clean_inputs)], dim=0).numpy()
    hot_trained = torch.cat([extract_snn_phi(module, backbone, monitor, h, device, desc=f"Trained SNN (Hot Pixel) seq {i}") for i, h in enumerate(hot_inputs)], dim=0).numpy()
    flood_trained = torch.cat([extract_snn_phi(module, backbone, monitor, h, device, desc=f"Trained SNN (Event Flood) seq {i}") for i, h in enumerate(flood_inputs)], dim=0).numpy()

    print(f"\n[Condition B] Randomizing SNN Backbone Weights...")
    randomize_weights(backbone)
    
    print("Extracting Random SNN Vmem Stats...")
    clean_random = torch.cat([extract_snn_phi(module, backbone, monitor, h, device, desc=f"Random SNN (Clean) seq {i}") for i, h in enumerate(clean_inputs)], dim=0).numpy()
    hot_random = torch.cat([extract_snn_phi(module, backbone, monitor, h, device, desc=f"Random SNN (Hot Pixel) seq {i}") for i, h in enumerate(hot_inputs)], dim=0).numpy()
    flood_random = torch.cat([extract_snn_phi(module, backbone, monitor, h, device, desc=f"Random SNN (Event Flood) seq {i}") for i, h in enumerate(flood_inputs)], dim=0).numpy()
    
    monitor.remove()
    del module, backbone
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Fit Mahalanobis scorer and compute AUROC for each condition
    results = {}
    
    # 1. Condition A (Trained SNN)
    scorer_trained = mahalanobis_scorer(clean_trained)
    scores_trained_clean = scorer_trained(clean_trained)
    
    yt_hot = np.concatenate([np.zeros(len(scores_trained_clean)), np.ones(len(hot_trained))])
    ys_hot_trained = np.concatenate([scores_trained_clean, scorer_trained(hot_trained)])
    a_hot_trained, _ = auroc_fpr95(yt_hot, ys_hot_trained)
    
    yt_flood = np.concatenate([np.zeros(len(scores_trained_clean)), np.ones(len(flood_trained))])
    ys_flood_trained = np.concatenate([scores_trained_clean, scorer_trained(flood_trained)])
    a_flood_trained, _ = auroc_fpr95(yt_flood, ys_flood_trained)
    
    results['Trained SNN'] = {'hot_pixel': a_hot_trained, 'event_flood': a_flood_trained}

    # 2. Condition B (Random SNN)
    scorer_random = mahalanobis_scorer(clean_random)
    scores_random_clean = scorer_random(clean_random)

    yt_hot_random = np.concatenate([np.zeros(len(scores_random_clean)), np.ones(len(hot_random))])
    ys_hot_random = np.concatenate([scores_random_clean, scorer_random(hot_random)])
    a_hot_random, _ = auroc_fpr95(yt_hot_random, ys_hot_random)

    yt_flood_random = np.concatenate([np.zeros(len(scores_random_clean)), np.ones(len(flood_random))])
    ys_flood_random = np.concatenate([scores_random_clean, scorer_random(flood_random)])
    a_flood_random, _ = auroc_fpr95(yt_flood_random, ys_flood_random)

    results['Random SNN'] = {'hot_pixel': a_hot_random, 'event_flood': a_flood_random}

    # 3. Condition C (Raw Input Stats)
    scorer_raw = mahalanobis_scorer(clean_raw)
    scores_raw_clean = scorer_raw(clean_raw)

    yt_hot_raw = np.concatenate([np.zeros(len(scores_raw_clean)), np.ones(len(hot_raw))])
    ys_hot_raw = np.concatenate([scores_raw_clean, scorer_raw(hot_raw)])
    a_hot_raw, _ = auroc_fpr95(yt_hot_raw, ys_hot_raw)

    yt_flood_raw = np.concatenate([np.zeros(len(scores_raw_clean)), np.ones(len(flood_raw))])
    ys_flood_raw = np.concatenate([scores_raw_clean, scorer_raw(flood_raw)])
    a_flood_raw, _ = auroc_fpr95(yt_flood_raw, ys_flood_raw)
    
    results['Raw Input Stats'] = {'hot_pixel': a_hot_raw, 'event_flood': a_flood_raw}

    # Print comparative results
    print("\n=======================================================")
    print(" FREE RIDER ABLATION RESULTS (AUROC)")
    print("=======================================================")
    print(f"  {'Condition':<18} | {'hot_pixel_L5':<12} | {'event_flood_L5':<12}")
    print("  " + "-" * 50)
    for cond in ['Trained SNN', 'Random SNN', 'Raw Input Stats']:
        print(f"  {cond:<18} | {results[cond]['hot_pixel']:.4f}       | {results[cond]['event_flood']:.4f}")
    print("=======================================================\n")

    # Plot results
    plot_free_rider_ablation(results)

    # Save to CSV
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / "free_rider_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Condition", "hot_pixel_AUROC_L5", "event_flood_AUROC_L5"])
        w.writeheader()
        for cond in ['Trained SNN', 'Random SNN', 'Raw Input Stats']:
            w.writerow({
                "Condition": cond,
                "hot_pixel_AUROC_L5": round(results[cond]['hot_pixel'], 4),
                "event_flood_AUROC_L5": round(results[cond]['event_flood'], 4)
            })
    print("  Saved free_rider_ablation.csv")

if __name__ == "__main__":
    main()
