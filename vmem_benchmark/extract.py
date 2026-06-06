# %%
"""
extract.py — Inference runner for the Vmem robustness benchmark.

Performs 31 inference passes (1 clean + 6 types x 5 severities).
Collects phi features and trajectory subsets.

Optimized with:
- concurrent.futures prefetching (hides I/O and corruption latency)
- Deferred GPU-CPU syncing (eliminates per-batch blocking)
- cudnn.benchmark = True
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys
import gc
import concurrent.futures
import argparse

# Setup paths and resolve CLI overrides before other imports
_HERE = Path(__file__).resolve().parent

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




def resolve_defaults_and_args():
    parser = argparse.ArgumentParser(
        description="Inference runner for the Vmem robustness benchmark. Supports flexible paths and configuration overrides.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gen1-root", type=str, help="Path to Gen1 dataset root directory")
    parser.add_argument("--ckpt", type=str, help="Path to model checkpoint (.ckpt) file")
    parser.add_argument("--hybrid-dir", type=str, help="Path to HybridDetection repository root")
    parser.add_argument("--corruption-dir", type=str, help="Path to event_corruption repository root")
    parser.add_argument("--output-dir", type=str, help="Path to save benchmark outputs")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], help="Compute device ('cuda' or 'cpu')")
    parser.add_argument("--batch-size", type=int, help="Batch size for inference")
    parser.add_argument("--max-seq", type=int, default=-2, help="Max sequences to process per run (set -1 or 0 for no cap/full split)")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--corruptions", type=str, nargs="+", help="Specific corruption types to run (space-separated)")
    parser.add_argument("--severities", type=int, nargs="+", help="Specific severity levels to run (space-separated, e.g. 1 2 3 4 5)")
    parser.add_argument("--save-every", type=int, help="Save/flush phi interval (number of sequences)")
    parser.add_argument("--plif-layers", type=int, nargs="+", help="PLIF layers to monitor (e.g. 0 1 2 3). Omit to hook all.")
    parser.add_argument("--input-dir", type=str, help="Direct path to the directory containing sequence folders (bypasses --gen1-root and --split)")
    parser.add_argument("--vram-fraction", type=float, default=1.0, help="Fraction of GPU VRAM PyTorch is allowed to allocate (0.0 to 1.0). Set to 1.0 for unlimited.")
    
    
    # We parse known args so it doesn't break if run inside environment frameworks with extra args
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[*] Ignoring unknown arguments: {unknown}")

    # Import configuration module to modify it in-place
    import benchmark_config as cfg

    # Sibling auto-detection paths
    sibling_hybrid = _HERE.parent / "HybridDetection"
    sibling_gen1 = _HERE.parent / "gen1"
    sibling_corruption = _HERE.parent / "event_corruption"
    
    print("=" * 60)
    print("  VMEM BENCHMARK: RESOLVING PATHS AND CONFIGURATION")
    print("=" * 60)

    # 1. Resolve HybridDetection Directory
    resolved_hybrid = None
    if args.hybrid_dir:
        resolved_hybrid = Path(args.hybrid_dir)
    elif cfg.HYBRID_DIR.exists():
        resolved_hybrid = cfg.HYBRID_DIR
    elif sibling_hybrid.exists():
        resolved_hybrid = sibling_hybrid
        print(f"[*] Auto-detected HybridDetection sibling directory: {resolved_hybrid}")
    else:
        resolved_hybrid = cfg.HYBRID_DIR
        print(f"[!] WARNING: HybridDetection directory not found at default {cfg.HYBRID_DIR} or sibling {sibling_hybrid}")
    
    # 2. Resolve event_corruption Directory
    resolved_corruption = None
    if args.corruption_dir:
        resolved_corruption = Path(args.corruption_dir)
    elif sibling_corruption.exists():
        resolved_corruption = sibling_corruption
        print(f"[*] Auto-detected event_corruption sibling directory: {resolved_corruption}")
    else:
        resolved_corruption = sibling_corruption
        print(f"[!] WARNING: event_corruption directory not found at default sibling {sibling_corruption}")

    # 3. Resolve Gen1 Dataset Root
    resolved_gen1 = None
    if args.gen1_root:
        resolved_gen1 = Path(args.gen1_root)
    elif cfg.GEN1_ROOT.exists():
        resolved_gen1 = cfg.GEN1_ROOT
    elif sibling_gen1.exists():
        resolved_gen1 = sibling_gen1
        print(f"[*] Auto-detected Gen1 dataset root sibling directory: {resolved_gen1}")
    else:
        resolved_gen1 = cfg.GEN1_ROOT
        print(f"[!] WARNING: Gen1 dataset root not found at default {cfg.GEN1_ROOT} or sibling {sibling_gen1}")

    # 4. Resolve Checkpoint Path
    resolved_ckpt = None
    if args.ckpt:
        resolved_ckpt = Path(args.ckpt)
    elif cfg.CKPT_PATH.exists():
        resolved_ckpt = cfg.CKPT_PATH
    else:
        # Try to find *.ckpt files in resolved hybrid_dir, _HERE.parent, or _HERE
        candidates = []
        if resolved_hybrid.exists():
            candidates.extend(list(resolved_hybrid.glob("*.ckpt")))
        candidates.extend(list(_HERE.parent.glob("*.ckpt")))
        candidates.extend(list(_HERE.glob("*.ckpt")))
        
        if candidates:
            resolved_ckpt = candidates[0]
            print(f"[*] Auto-detected checkpoint file: {resolved_ckpt}")
        else:
            resolved_ckpt = cfg.CKPT_PATH
            print(f"[!] WARNING: Checkpoint file not found. Falling back to default path: {resolved_ckpt}")

    # 5. Resolve Output Directory
    resolved_output = None
    if args.output_dir:
        resolved_output = Path(args.output_dir)
    elif hasattr(cfg, "OUTPUT_DIR"):
        resolved_output = cfg.OUTPUT_DIR
    else:
        resolved_output = _HERE / "outputs"
        print(f"[*] Using local outputs directory: {resolved_output}")

    # Apply paths to configuration module
    cfg.HYBRID_DIR = resolved_hybrid.resolve()
    cfg.GEN1_ROOT = resolved_gen1.resolve()
    cfg.CKPT_PATH = resolved_ckpt.resolve()
    cfg.OUTPUT_DIR = resolved_output.resolve()
    cfg.PHI_DIR = cfg.OUTPUT_DIR / "phi"
    cfg.TRAJ_DIR = cfg.OUTPUT_DIR / "trajs"
    cfg.PLOT_DIR = cfg.OUTPUT_DIR / "plots"
    cfg.TEMPORAL_PHI_DIR = cfg.OUTPUT_DIR / "temporal_phi"
    
    # Direct Input Directory Bypass (None if not explicitly passed)
    if args.input_dir:
        cfg.INPUT_DIR = Path(args.input_dir).resolve()
    else:
        cfg.INPUT_DIR = None

    # 6. Overrides for other configuration parameters
    if args.device:
        cfg.DEVICE = args.device
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    
    if cfg.BATCH_SIZE > 1:
        raise ValueError(
            f"\n[FATAL ERROR] BATCH_SIZE is currently configured to {cfg.BATCH_SIZE}.\n"
            f"BATCH_SIZE must be exactly 1! Setting BATCH_SIZE > 1 violates the strict shape contract\n"
            f"of the pre-trained SNN-ANN backbone. The backbone folds batch and sequence dimensions\n"
            f"such that if B > 1, the SNN unrolls over B as the time axis and treats T as a parallel\n"
            f"batch dimension. This leads to complete state loss for all but the last sample in the\n"
            f"batch, producing scientifically incorrect and invalid outputs.\n"
            f"Please keep BATCH_SIZE = 1 for all robustness benchmark evaluations."
        )

    
    # Handle capping overrides (-2 means keep default, -1/0 means no cap, >0 means cap)
    if args.max_seq > 0:
        cfg.MAX_SEQUENCES = args.max_seq
    elif args.max_seq in (-1, 0):
        cfg.MAX_SEQUENCES = None

    if args.split:
        cfg.SPLIT = args.split
    if args.corruptions:
        cfg.CORRUPTIONS = args.corruptions
    if args.severities:
        cfg.SEVERITIES = args.severities
    if args.save_every is not None:
        cfg.PHI_SAVE_EVERY = args.save_every
    if args.plif_layers is not None:
        cfg.PLIF_LAYERS = args.plif_layers
    cfg.VRAM_FRACTION = args.vram_fraction


    # Dynamically inject resolved HybridDetection and event_corruption to sys.path
    resolved_corruption_path = resolved_corruption.resolve() if resolved_corruption.exists() else resolved_corruption
    for path in (cfg.HYBRID_DIR, resolved_corruption_path):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # Print a clear, clean summary of active settings
    print("-" * 60)
    print(f"  GEN1 Dataset Root  : {cfg.GEN1_ROOT}")
    print(f"  Model Checkpoint   : {cfg.CKPT_PATH}")
    print(f"  HybridDetection Root: {cfg.HYBRID_DIR}")
    print(f"  event_corruption   : {resolved_corruption_path}")
    print(f"  Output Directory   : {cfg.OUTPUT_DIR}")
    print(f"  Target Device      : {cfg.DEVICE}")
    print(f"  Batch Size         : {cfg.BATCH_SIZE}")
    print(f"  Max Sequences Capping: {cfg.MAX_SEQUENCES}")
    print(f"  Dataset Split      : {cfg.SPLIT}")
    print(f"  Corruptions        : {cfg.CORRUPTIONS}")
    print(f"  Severities         : {cfg.SEVERITIES}")
    print(f"  Save Every N Seqs  : {cfg.PHI_SAVE_EVERY}")
    plif_display = cfg.PLIF_LAYERS if cfg.PLIF_LAYERS is not None else "All (0, 1, 2, 3)"
    print(f"  PLIF Layers Hooked : {plif_display}")
    input_dir_display = cfg.INPUT_DIR if cfg.INPUT_DIR is not None else "Default (derived from gen1-root/split)"
    print(f"  Direct Input Dir   : {input_dir_display}")
    vram_display = f"{cfg.VRAM_FRACTION * 100}%" if cfg.VRAM_FRACTION < 1.0 else "Unlimited"
    print(f"  VRAM Allocation Cap: {vram_display}")

    print("=" * 60)

# Execute the path/config resolution immediately on import/run
resolve_defaults_and_args()

import benchmark_config as cfg
from model_loader import load_model
from monitor import VmemMonitor
from corruption_wrap import apply_corruption_to_tensor
from pipeline.loader import load_histogram

# Import spikingjelly reset
from spikingjelly.clock_driven import functional


def _cpu_loader_worker(seq_dir, c_name, severity):
    """
    Worker function to load and corrupt a single sequence on the CPU.
    Returns the numpy array ready for GPU transfer.
    """
    try:
        hist_np, _ = load_histogram(seq_dir)
        if c_name is not None:
            # Apply corruption on CPU numpy array
            hist_np = apply_corruption_to_tensor(
                torch.from_numpy(hist_np), c_name, severity
            ).numpy()
        return hist_np
    except Exception as e:
        print(f"\nError loading {seq_dir}: {e}")
        return None


def run_benchmark():
    # 0. PyTorch Optimizations & Resource Limiting
    torch.backends.cudnn.benchmark = True  # Accelerates convolutions for fixed 240x304 shape
    
    # Cap VRAM usage if requested (fraction < 1.0) to prevent OS lag on local GPUs
    if cfg.DEVICE == "cuda" and getattr(cfg, "VRAM_FRACTION", 1.0) < 1.0:
        torch.cuda.set_per_process_memory_fraction(cfg.VRAM_FRACTION, 0)

        
    # 1. Setup Directories
    cfg.PHI_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TRAJ_DIR.mkdir(parents=True, exist_ok=True)
    cfg.ANN_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Load Model & Setup Monitor
    module, backbone = load_model(cfg.DEVICE)
    monitor = VmemMonitor(backbone, selected=cfg.PLIF_LAYERS)

    # 3. Discover Sequences
    input_dir = getattr(cfg, "INPUT_DIR", None)
    if input_dir is None:
        default_dir = cfg.GEN1_ROOT / cfg.SPLIT
        label_files = sorted(default_dir.glob("*/labels_v2/labels.npz"))
        if len(label_files) > 0:
            input_dir = default_dir
            print_desc = f"in {cfg.SPLIT} split"
        else:
            # Fallback: check if the gen1-root directory itself contains the sequences directly
            direct_files = sorted(cfg.GEN1_ROOT.glob("*/labels_v2/labels.npz"))
            if len(direct_files) > 0:
                input_dir = cfg.GEN1_ROOT
                label_files = direct_files
                print_desc = "directly in Gen1 root directory"
            else:
                input_dir = default_dir
                print_desc = f"in {cfg.SPLIT} split"
    else:
        label_files = sorted(input_dir.glob("*/labels_v2/labels.npz"))
        print_desc = "in custom input directory"
    
    seq_dirs = [p.parent.parent for p in label_files]
    
    # We process all sequences in the split
    if not seq_dirs:
        print(f"Error: No sequences found in {input_dir}")
        return

    print(f"\nFound {len(seq_dirs)} sequences {print_desc}.")

    # --- Sequence cap ---
    max_seq = getattr(cfg, "MAX_SEQUENCES", None)
    if max_seq is not None:
        seq_dirs = seq_dirs[:max_seq]
        print(f"[cap] Capped to {len(seq_dirs)} sequences (MAX_SEQUENCES={max_seq}).")

    # 4. Define Runs (Clean + Corruptions)
    runs = [("clean", None, 0)]
    for c_name in cfg.CORRUPTIONS:
        for sev in cfg.SEVERITIES:
            runs.append((f"{c_name}_L{sev}", c_name, sev))

    print(f"Starting benchmark with {len(runs)} total runs...")

    # 5. Main Loop
    overall_pbar = tqdm(runs, desc="Overall Progress", position=0)
    for run_name, c_name, severity in overall_pbar:
        phi_path = cfg.PHI_DIR / f"{run_name}.pt"
        traj_path = cfg.TRAJ_DIR / f"{run_name}.pt"

        # --- Smart resume: check if existing phi covers all requested sequences ---
        historical_phi = None    # phi tensor loaded from a previous completed run
        historical_ann = None    # ann dict loaded from a previous completed run
        done_seqs = set()        # sequence indices already processed

        tgap_path = cfg.OUTPUT_DIR / "temporal_gap" / f"{run_name}.pt"
        ann_path = cfg.ANN_DIR / f"{run_name}.pt"
        if phi_path.exists() and tgap_path.exists() and ann_path.exists():
            existing = torch.load(phi_path, weights_only=False)
            done_seqs_saved = set(existing.get('done_seqs', []))
            max_seq = getattr(cfg, 'MAX_SEQUENCES', None)
            needed = set(range(max_seq)) if max_seq is not None else set(range(len(seq_dirs)))

            if needed.issubset(done_seqs_saved):
                # All requested sequences are already in the file — skip entirely
                overall_pbar.write(f"[skip] {run_name} — all {len(done_seqs_saved)} seqs already done.")
                continue

            # Partial: load existing phi so we can extend it
            historical_phi = existing['phi']   # CPU tensor, kept for final cat

            # Load existing ANN features to extend
            existing_ann = torch.load(ann_path, weights_only=False)
            historical_ann = {
                'asab_gap': existing_ann['asab_gap'],
                'last_ann_gap': existing_ann['last_ann_gap'],
                'head_cls_L0_gap': existing_ann['head_cls_L0_gap']
            }

            done_seqs = done_seqs_saved
            overall_pbar.write(
                f"\n[RUN] {run_name}  (extending: {len(done_seqs)} seqs done, "
                f"{len(needed - done_seqs)} new)"
            )
        else:
            overall_pbar.write(f"\n[RUN] {run_name}")

        # Also pick up any seq_*.pt files left from a mid-run crash
        phi_tmp_dir = cfg.PHI_DIR / f"_tmp_{run_name}"
        phi_tmp_dir.mkdir(parents=True, exist_ok=True)
        crash_seqs = {int(f.stem.split('_')[1]) for f in phi_tmp_dir.glob("seq_*.pt")}
        done_seqs |= crash_seqs
        if crash_seqs:
            overall_pbar.write(f"  [resume] {len(crash_seqs)} sequences recovered from tmp dir.")

        traj_bank = {}
        n_trajs_saved = 0

        # We process sequences one by one
        pbar = tqdm(seq_dirs, desc=run_name, unit="seq", position=1, leave=False)
        for i, seq_dir in enumerate(pbar):
            # --- Resume check: skip sequences whose phi was already saved ---
            if i in done_seqs:
                pbar.set_postfix_str(f"seq {i} skipped (done)")
                continue

            # Load and corrupt synchronously (uses 50% less CPU RAM)
            hist_np = _cpu_loader_worker(seq_dir, c_name, severity)

            if hist_np is None:
                continue

            # --- 3. Sequence Initialization (CPU) ---
            # Keep the sequence as uint8 on CPU to prevent massive VRAM allocations.
            # Delete hist_np immediately after to avoid keeping two copies alive.
            hist_torch_cpu = torch.from_numpy(hist_np)
            del hist_np  # free the numpy array — only the torch view survives
            n_frames = hist_torch_cpu.shape[0]
            
            # Reset LSTM states for new sequence
            h_c = {0: None, 1: None}
            
            # These lists hold GPU tensors temporarily for this sequence to avoid blocking syncs
            seq_phi_gpu = []
            seq_temporal_phi_cpu = []
            seq_temporal_gap_cpu = []
            seq_asab_gap_cpu = []
            seq_last_ann_gap_cpu = []
            seq_head_cls_L0_gap_cpu = []
            seq_traj_cpu = {l: [] for l in range(4)}  # Max 4 layers

            for j in range(0, n_frames, cfg.BATCH_SIZE):
                batch_end = min(j + cfg.BATCH_SIZE, n_frames)
                # Slice on CPU as uint8, transfer to GPU (fast 1x bandwidth), then cast to float
                batch = hist_torch_cpu[j:batch_end].to(cfg.DEVICE, non_blocking=True).float()

                # Reset SNN neuron states and hook buffers before each batch
                functional.reset_net(backbone)
                monitor.reset()

                with torch.no_grad():
                    # Pad batch using the model's input padder to ensure dimensions match up in FPN
                    padded_batch = module.input_padder.pad_tensor_ev_repr(batch)
                    backbone_features, h_c = module.mdl.forward_backbone(x=padded_batch, h_c=h_c)
                    
                    # Extract ANN/ASAB features
                    x_1 = backbone_features[2]
                    asab_gap = x_1.mean(dim=(2, 3)).cpu()
                    seq_asab_gap_cpu.append(asab_gap)
                    
                    x_3 = backbone_features[4]
                    last_ann_gap = x_3.mean(dim=(2, 3)).cpu()
                    seq_last_ann_gap_cpu.append(last_ann_gap)
                    
                    # Run FPN and classification head for level 0 (stride 8)
                    fpn_features = module.mdl.fpn(backbone_features)
                    x_stem = module.mdl.yolox_head.stems[0](fpn_features[0])
                    cls_feat = module.mdl.yolox_head.cls_convs[0](x_stem)
                    cls_output = module.mdl.yolox_head.cls_preds[0](cls_feat)
                    head_cls_L0_gap = cls_output.mean(dim=(2, 3)).cpu()
                    seq_head_cls_L0_gap_cpu.append(head_cls_L0_gap)

                # Collect phi for this batch (KEEP ON GPU to prevent blocking sync)
                phi_batch = monitor.collect_phi()
                if phi_batch.numel() > 0:
                    seq_phi_gpu.append(phi_batch)

                # Collect temporal phi online
                tphi_batch = monitor.collect_temporal_phi()
                if tphi_batch.numel() > 0:
                    seq_temporal_phi_cpu.append(tphi_batch)

                # Collect temporal GAP online (breaks 50-sequence bottleneck)
                tgap_batch = monitor.collect_temporal_gap()
                if tgap_batch.numel() > 0:
                    seq_temporal_gap_cpu.append(tgap_batch)

                # Collect trajectory snapshots
                if n_trajs_saved < cfg.TRAJ_SAVE_N:
                    remaining = cfg.TRAJ_SAVE_N - n_trajs_saved
                    trajs = monitor.trajectories(n_samples=remaining)
                    for l_idx, t_tensor in trajs.items():
                        # Move to CPU immediately to prevent massive VRAM accumulation (each is ~93MB)
                        seq_traj_cpu[l_idx].append(t_tensor.cpu())
                    
                    if trajs:
                        collected = next(iter(trajs.values())).shape[1]
                        n_trajs_saved += collected

            # --- 4. Save this sequence's phi immediately to disk ---
            if seq_phi_gpu:
                phi_seq = torch.cat(seq_phi_gpu, dim=0).cpu()
                seq_pt = phi_tmp_dir / f"seq_{i:05d}.pt"
                torch.save(phi_seq, seq_pt)
                del phi_seq
                gc.collect()

            # Save sequence temporal phi online
            if seq_temporal_phi_cpu:
                tphi_seq = torch.cat(seq_temporal_phi_cpu, dim=0)
                tphi_tmp_dir = cfg.TEMPORAL_PHI_DIR / f"_tmp_{run_name}"
                tphi_tmp_dir.mkdir(parents=True, exist_ok=True)
                torch.save(tphi_seq, tphi_tmp_dir / f"seq_{i:05d}.pt")
                del tphi_seq
                gc.collect()

            # Save sequence temporal gap online
            if seq_temporal_gap_cpu:
                tgap_seq = torch.cat(seq_temporal_gap_cpu, dim=0)
                tgap_tmp_dir = cfg.OUTPUT_DIR / "temporal_gap" / f"_tmp_{run_name}"
                tgap_tmp_dir.mkdir(parents=True, exist_ok=True)
                torch.save(tgap_seq, tgap_tmp_dir / f"seq_{i:05d}.pt")
                del tgap_seq
                gc.collect()

            # Save sequence ANN features online
            if seq_asab_gap_cpu:
                asab_gap_seq = torch.cat(seq_asab_gap_cpu, dim=0)
                last_ann_gap_seq = torch.cat(seq_last_ann_gap_cpu, dim=0)
                head_cls_L0_gap_seq = torch.cat(seq_head_cls_L0_gap_cpu, dim=0)
                
                ann_tmp_dir = cfg.ANN_DIR / f"_tmp_{run_name}"
                ann_tmp_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'asab_gap': asab_gap_seq,
                    'last_ann_gap': last_ann_gap_seq,
                    'head_cls_L0_gap': head_cls_L0_gap_seq
                }, ann_tmp_dir / f"seq_{i:05d}.pt")
                del asab_gap_seq, last_ann_gap_seq, head_cls_L0_gap_seq
                gc.collect()

                
            for l_idx, tensor_list in seq_traj_cpu.items():
                if tensor_list:
                    if l_idx not in traj_bank:
                        traj_bank[l_idx] = []
                    # Concatenate on CPU (tensors are already on CPU)
                    traj_bank[l_idx].append(torch.cat(tensor_list, dim=1))

            # --- 5. Aggressive Memory Cleanup ---
            del hist_torch_cpu, seq_phi_gpu, seq_traj_cpu, seq_asab_gap_cpu, seq_last_ann_gap_cpu, seq_head_cls_L0_gap_cpu
            gc.collect()
            if cfg.DEVICE == "cuda":
                torch.cuda.empty_cache()

        # --- Merge all per-sequence phi files into the final output ---
        seq_files = sorted(phi_tmp_dir.glob("seq_*.pt"))
        new_parts = [torch.load(f, weights_only=True) for f in seq_files]

        if new_parts or historical_phi is not None:
            parts_to_cat = ([historical_phi] if historical_phi is not None else []) + new_parts
            phi_final = torch.cat(parts_to_cat, dim=0)

            # Track every sequence index that is now in the file
            all_done = done_seqs | {int(f.stem.split('_')[1]) for f in seq_files}
            torch.save({'phi': phi_final, 'run': run_name, 'done_seqs': sorted(all_done)}, phi_path)
            overall_pbar.write(
                f"  Saved phi: {phi_final.shape[0]} rows from {len(all_done)} sequences "
                f"-> {phi_path.name}"
            )
            del phi_final, historical_phi

            # Clean up temp directory
            for f in seq_files:
                f.unlink()
            if not any(phi_tmp_dir.iterdir()):
                phi_tmp_dir.rmdir()

            if traj_bank:
                traj_final = {l: torch.cat(v, dim=1) for l, v in traj_bank.items()}
                torch.save({'trajs': traj_final, 'run': run_name}, traj_path)

            # --- Merge temporal phi tmp files ---
            tphi_tmp_dir = cfg.TEMPORAL_PHI_DIR / f"_tmp_{run_name}"
            if tphi_tmp_dir.exists():
                tphi_files = sorted(tphi_tmp_dir.glob("seq_*.pt"))
                if tphi_files:
                    tphi_final = torch.cat(
                        [torch.load(f, weights_only=True) for f in tphi_files], dim=0
                    )
                    cfg.TEMPORAL_PHI_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save({'temporal_phi': tphi_final, 'run': run_name},
                               cfg.TEMPORAL_PHI_DIR / f"{run_name}.pt")
                    overall_pbar.write(
                        f"  Saved temporal phi: {tphi_final.shape} -> {run_name}.pt"
                    )
                    del tphi_final
                    for f in tphi_files:
                        f.unlink()
                    if not any(tphi_tmp_dir.iterdir()):
                        tphi_tmp_dir.rmdir()

            # --- Merge temporal gap trajectories tmp files ---
            tgap_tmp_dir = cfg.OUTPUT_DIR / "temporal_gap" / f"_tmp_{run_name}"
            if tgap_tmp_dir.exists():
                tgap_files = sorted(tgap_tmp_dir.glob("seq_*.pt"))
                if tgap_files:
                    tgap_final = torch.cat(
                        [torch.load(f, weights_only=True) for f in tgap_files], dim=0
                    )
                    tgap_dir = cfg.OUTPUT_DIR / "temporal_gap"
                    tgap_dir.mkdir(parents=True, exist_ok=True)
                    torch.save({'temporal_gap': tgap_final, 'run': run_name},
                               tgap_dir / f"{run_name}.pt")
                    overall_pbar.write(
                        f"  Saved temporal gap: {tgap_final.shape} -> {run_name}.pt"
                    )
                    del tgap_final
                    for f in tgap_files:
                        f.unlink()
                    if not any(tgap_tmp_dir.iterdir()):
                        tgap_tmp_dir.rmdir()

            # --- Merge ANN features tmp files ---
            ann_tmp_dir = cfg.ANN_DIR / f"_tmp_{run_name}"
            if ann_tmp_dir.exists():
                ann_files = sorted(ann_tmp_dir.glob("seq_*.pt"))
                if ann_files:
                    ann_loaded = [torch.load(f, weights_only=True) for f in ann_files]
                    
                    asab_parts = ([historical_ann['asab_gap']] if historical_ann is not None else []) + [d['asab_gap'] for d in ann_loaded]
                    last_ann_parts = ([historical_ann['last_ann_gap']] if historical_ann is not None else []) + [d['last_ann_gap'] for d in ann_loaded]
                    head_cls_parts = ([historical_ann['head_cls_L0_gap']] if historical_ann is not None else []) + [d['head_cls_L0_gap'] for d in ann_loaded]
                    
                    asab_gap_final = torch.cat(asab_parts, dim=0)
                    last_ann_gap_final = torch.cat(last_ann_parts, dim=0)
                    head_cls_L0_gap_final = torch.cat(head_cls_parts, dim=0)
                    
                    cfg.ANN_DIR.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'asab_gap': asab_gap_final,
                        'last_ann_gap': last_ann_gap_final,
                        'head_cls_L0_gap': head_cls_L0_gap_final
                    }, cfg.ANN_DIR / f"{run_name}.pt")
                    overall_pbar.write(f"  Saved ANN features: {asab_gap_final.shape} -> {run_name}.pt")
                    
                    del asab_gap_final, last_ann_gap_final, head_cls_L0_gap_final, ann_loaded
                    for f in ann_files:
                        f.unlink()
                    if not any(ann_tmp_dir.iterdir()):
                        ann_tmp_dir.rmdir()

        else:
            overall_pbar.write(f"  Warning: No data collected for {run_name}")
            if not any(phi_tmp_dir.iterdir()):
                phi_tmp_dir.rmdir()
    monitor.remove()
    print("\nBenchmark extraction complete.")


if __name__ == "__main__":
    run_benchmark()
