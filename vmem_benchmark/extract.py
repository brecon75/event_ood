# %%
"""
extract.py — Inference runner for the Vmem robustness benchmark.

Performs 31 inference passes (1 clean + 6 types x 5 severities).
Collects phi features and trajectory subsets.

Optimized with:
- Deferred GPU-CPU syncing (eliminates per-batch blocking)
- cudnn.benchmark = True

All per-run artifacts (phi, temporal phi, temporal GAP, ANN features, spike
stats, detection outputs) are written per sequence to a _tmp_<run> directory
and merged into the final <run>.pt in sequence-index order, together with
'done_seqs' and 'seq_lens' metadata. This makes crash/resume safe for every
artifact type and lets downstream analyses split frames by sequence.
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time
import sys
import gc
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
    parser.add_argument("--skip-clean", action="store_true", help="Skip running the clean/baseline run")
    parser.add_argument("--clean-only", action="store_true", help="Run only the clean/baseline pass (no corruptions)")
    
    
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
    # Rebase ALL output subdirectories onto the resolved OUTPUT_DIR so that a
    # --output-dir override moves every artifact, not just phi/trajs/plots.
    cfg.PHI_DIR = cfg.OUTPUT_DIR / "phi"
    cfg.TRAJ_DIR = cfg.OUTPUT_DIR / "trajs"
    cfg.PLOT_DIR = cfg.OUTPUT_DIR / "plots"
    cfg.TEMPORAL_PHI_DIR = cfg.OUTPUT_DIR / "temporal_phi"
    cfg.ANN_DIR = cfg.OUTPUT_DIR / "ann_features"
    cfg.SPIKE_DIR = cfg.OUTPUT_DIR / "spike"
    cfg.DETECTOR_DIR = cfg.OUTPUT_DIR / "detectors"
    cfg.DET_OUT_DIR = cfg.OUTPUT_DIR / "det_outputs"

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
    if args.clean_only:
        cfg.CORRUPTIONS = []
    elif args.corruptions:
        cfg.CORRUPTIONS = args.corruptions
    if args.severities:
        cfg.SEVERITIES = args.severities
    if args.save_every is not None:
        cfg.PHI_SAVE_EVERY = args.save_every
    if args.plif_layers is not None:
        cfg.PLIF_LAYERS = args.plif_layers
    cfg.VRAM_FRACTION = args.vram_fraction
    cfg.SKIP_CLEAN = args.skip_clean


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


def _cpu_loader_worker(seq_dir, c_name, severity, seq_idx):
    """
    Load and corrupt a single sequence on the CPU.
    Returns the numpy array ready for GPU transfer.

    The corruption seed is derived from the sequence index so every sequence
    gets an independent noise realization (same base seed keeps the benchmark
    deterministic and pairs realizations across severities).
    """
    try:
        hist_np, _ = load_histogram(seq_dir)
        if c_name is not None:
            hist_np = apply_corruption_to_tensor(
                torch.from_numpy(hist_np), c_name, severity, seed=[42, seq_idx]
            ).numpy()
        return hist_np
    except Exception as e:
        print(f"\nError loading {seq_dir}: {e}")
        return None


def _merge_seq_artifact(tmp_dir, final_path, data_keys, run_name, log=print):
    """
    Merge per-sequence tmp files (seq_<idx>.pt) with any existing final file
    into a single artifact ordered by sequence index.

    Each tmp file holds either a raw tensor (single-key artifacts) or a dict
    of tensors. The final file stores, for every key in `data_keys`, the rows
    of all sequences concatenated in ascending sequence-index order, plus:
      'run'       — run name
      'done_seqs' — sorted list of sequence indices contained in the file
      'seq_lens'  — frame count per sequence (aligned with done_seqs), or
                    absent when a legacy file without metadata was merged.

    Returns True if a final file was written.
    """
    tmp_files = sorted(tmp_dir.glob("seq_*.pt")) if tmp_dir.exists() else []
    per_seq = {}
    for f in tmp_files:
        idx = int(f.stem.split("_")[1])
        loaded = torch.load(f, weights_only=True, map_location="cpu")
        per_seq[idx] = loaded if isinstance(loaded, dict) else {data_keys[0]: loaded}

    # Load and, where possible, decompose the existing final file so resumed
    # runs keep their historical sequences in the right place.
    legacy_block = None  # (data dict, done set) when old rows can't be split
    if final_path.exists():
        old = torch.load(final_path, weights_only=True, map_location="cpu")
        if not isinstance(old, dict):
            old = {data_keys[0]: old}
        old_done = list(old.get("done_seqs", []))
        old_lens = list(old.get("seq_lens", []))
        old_data = {k: old[k] for k in data_keys if k in old}
        if old_data:
            n_rows = int(next(iter(old_data.values())).shape[0])
            if old_done and len(old_done) == len(old_lens) and sum(old_lens) == n_rows:
                off = 0
                for s_idx, s_len in zip(old_done, old_lens):
                    # Freshly extracted tmp data wins over historical rows.
                    if s_idx not in per_seq:
                        per_seq[s_idx] = {k: v[off:off + s_len] for k, v in old_data.items()}
                    off += s_len
            else:
                # Legacy file without per-sequence metadata: keep it as one
                # opaque block and drop seqs it already covers from tmp data.
                legacy_done = set(old_done)
                per_seq = {i: d for i, d in per_seq.items() if i not in legacy_done}
                legacy_block = (old_data, legacy_done)
                log(f"  [merge] {final_path.name}: legacy file without seq metadata; "
                    f"appending new sequences after the historical block.")

    if not per_seq and legacy_block is None:
        if tmp_dir.exists() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
        return False

    parts = {k: [] for k in data_keys}
    done_all = []
    seq_lens = []
    if legacy_block is not None:
        old_data, legacy_done = legacy_block
        for k in data_keys:
            if k in old_data:
                parts[k].append(old_data[k])
        done_all.extend(sorted(legacy_done))
        seq_lens = None  # row→sequence mapping unknown for the legacy block

    for s_idx in sorted(per_seq.keys()):
        chunk = per_seq[s_idx]
        for k in data_keys:
            if k in chunk:
                parts[k].append(chunk[k])
        done_all.append(s_idx)
        if seq_lens is not None:
            seq_lens.append(int(next(iter(chunk.values())).shape[0]))

    out = {k: torch.cat(v, dim=0) for k, v in parts.items() if v}

    # All saved data keys must stay row-aligned (they share done_seqs/seq_lens).
    # A mismatch means we merged a historical file missing one key (e.g. a
    # pre-spatial phi file lacking 'phi_spatial'): refuse rather than silently
    # corrupt. Re-extract such runs into a fresh output dir.
    row_counts = {k: int(v.shape[0]) for k, v in out.items()}
    if len(set(row_counts.values())) > 1:
        raise RuntimeError(
            f"{final_path.name}: data keys have mismatched row counts {row_counts}. "
            f"This usually means resuming on top of a final file that lacks one of "
            f"{data_keys} (e.g. legacy phi without phi_spatial). Extract this run "
            f"into a clean output directory instead of resuming."
        )

    out["run"] = run_name
    out["done_seqs"] = sorted(set(done_all))
    if seq_lens is not None:
        out["seq_lens"] = seq_lens

    final_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, final_path)

    for f in tmp_files:
        f.unlink()
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    return True


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
    runs = []
    if not getattr(cfg, "SKIP_CLEAN", False):
        runs.append(("clean", None, 0))
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
        done_seqs = set()        # sequence indices already processed

        tgap_path = cfg.OUTPUT_DIR / "temporal_gap" / f"{run_name}.pt"
        ann_path = cfg.ANN_DIR / f"{run_name}.pt"
        if phi_path.exists() and tgap_path.exists() and ann_path.exists():
            existing = torch.load(phi_path, weights_only=True, map_location="cpu")
            done_seqs_saved = set(existing.get('done_seqs', []))
            del existing
            needed = set(range(len(seq_dirs)))

            if needed.issubset(done_seqs_saved):
                # All requested sequences are already in the file — skip entirely
                overall_pbar.write(f"[skip] {run_name} — all {len(done_seqs_saved)} seqs already done.")
                continue

            # Partial: the merge step prepends historical data per artifact,
            # so here we only need to know which sequences to skip.
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
            hist_np = _cpu_loader_worker(seq_dir, c_name, severity, i)

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
            seq_phi_spatial_gpu = []
            seq_temporal_phi_cpu = []
            seq_temporal_gap_cpu = []
            seq_asab_gap_cpu = []
            seq_last_ann_gap_cpu = []
            seq_head_cls_L0_gap_cpu = []
            seq_spike_rate_cpu = []
            seq_spike_entropy_cpu = []
            seq_det_outputs_cpu = []
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

                    # Run downstream YOLOX detection head to get predictions (B, anchors, 7)
                    predictions, _ = module.mdl.forward_detect(backbone_features=backbone_features)
                    # Extract for batch size 1
                    obj_conf = predictions[0, :, 4]
                    cls_conf, _ = predictions[0, :, 5:].max(dim=-1)
                    scores = obj_conf * cls_conf  # (anchors,)
                    
                    # Filter anchors with score > 0.05
                    mask = scores > 0.05
                    filtered_pred = predictions[0, mask]  # (K, 7)
                    
                    # Sort and keep top 100 anchors
                    if len(filtered_pred) > 0:
                        f_scores = scores[mask]
                        sort_idx = torch.argsort(f_scores, descending=True)
                        filtered_pred = filtered_pred[sort_idx[:100]]
                    
                    # Pad to fixed size (100, 7)
                    K = len(filtered_pred)
                    padded_pred = torch.zeros((100, 7), device=predictions.device)
                    if K > 0:
                        padded_pred[:K] = filtered_pred
                        
                    seq_det_outputs_cpu.append(padded_pred.unsqueeze(0).cpu())

                # Collect phi for this batch (KEEP ON GPU to prevent blocking sync)
                phi_batch = monitor.collect_phi()
                if phi_batch.numel() > 0:
                    seq_phi_gpu.append(phi_batch)

                # Collect spatial-dispersion phi (the signal GAP discards)
                phi_spatial_batch = monitor.collect_phi_spatial()
                if phi_spatial_batch.numel() > 0:
                    seq_phi_spatial_gpu.append(phi_spatial_batch)

                # Collect temporal phi online
                tphi_batch = monitor.collect_temporal_phi()
                if tphi_batch.numel() > 0:
                    seq_temporal_phi_cpu.append(tphi_batch)

                # Collect temporal GAP online (breaks 50-sequence bottleneck)
                tgap_batch = monitor.collect_temporal_gap()
                if tgap_batch.numel() > 0:
                    seq_temporal_gap_cpu.append(tgap_batch)

                # Collect spike stats online
                sp_stats = monitor.collect_spikes()
                if sp_stats['spike_rate'].numel() > 0:
                    seq_spike_rate_cpu.append(sp_stats['spike_rate'])
                    seq_spike_entropy_cpu.append(sp_stats['spike_entropy'])

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
            # phi and phi_spatial share one tmp file so the merge keeps them
            # row-aligned and under the same seq_lens / done_seqs metadata.
            if seq_phi_gpu:
                phi_seq = torch.cat(seq_phi_gpu, dim=0).cpu()
                seq_payload = {"phi": phi_seq}
                if seq_phi_spatial_gpu:
                    # Stored float32 (~+60 GB across 31 runs). float16 was tried
                    # but high-activity corruptions (event_flood) push spatial_var
                    # past the float16 max (65504) -> +inf -> NaN scores; float32
                    # has the headroom, so keep it simple and lossless.
                    seq_payload["phi_spatial"] = torch.cat(seq_phi_spatial_gpu, dim=0).cpu()
                seq_pt = phi_tmp_dir / f"seq_{i:05d}.pt"
                torch.save(seq_payload, seq_pt)
                del phi_seq, seq_payload
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

                
            # Save sequence spike stats online
            if seq_spike_rate_cpu:
                sp_rate_seq = torch.cat(seq_spike_rate_cpu, dim=0)
                sp_ent_seq = torch.cat(seq_spike_entropy_cpu, dim=0)
                spike_tmp_dir = cfg.SPIKE_DIR / f"_tmp_{run_name}"
                spike_tmp_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'spike_rate': sp_rate_seq,
                    'spike_entropy': sp_ent_seq
                }, spike_tmp_dir / f"seq_{i:05d}.pt")
                del sp_rate_seq, sp_ent_seq
                gc.collect()

            # Save sequence detector outputs online
            if seq_det_outputs_cpu:
                det_seq = torch.cat(seq_det_outputs_cpu, dim=0)
                det_tmp_dir = cfg.DET_OUT_DIR / f"_tmp_{run_name}"
                det_tmp_dir.mkdir(parents=True, exist_ok=True)
                torch.save(det_seq, det_tmp_dir / f"seq_{i:05d}.pt")
                del det_seq
                gc.collect()
                
            for l_idx, tensor_list in seq_traj_cpu.items():
                if tensor_list:
                    if l_idx not in traj_bank:
                        traj_bank[l_idx] = []
                    # Concatenate on CPU (tensors are already on CPU)
                    traj_bank[l_idx].append(torch.cat(tensor_list, dim=1))

            # --- 5. Aggressive Memory Cleanup ---
            del hist_torch_cpu, seq_phi_gpu, seq_phi_spatial_gpu, seq_traj_cpu, seq_temporal_phi_cpu, seq_temporal_gap_cpu, seq_asab_gap_cpu, seq_last_ann_gap_cpu, seq_head_cls_L0_gap_cpu, seq_spike_rate_cpu, seq_spike_entropy_cpu, seq_det_outputs_cpu
            gc.collect()
            if cfg.DEVICE == "cuda":
                torch.cuda.empty_cache()

        # --- Merge all per-sequence tmp files (plus any historical final
        #     file) into sequence-index-ordered final outputs ---
        wrote_phi = _merge_seq_artifact(
            phi_tmp_dir, phi_path, ["phi", "phi_spatial"], run_name, log=overall_pbar.write)

        if wrote_phi:
            saved = torch.load(phi_path, weights_only=True, map_location="cpu")
            spatial_note = (f", phi_spatial {tuple(saved['phi_spatial'].shape)}"
                            if "phi_spatial" in saved else "")
            overall_pbar.write(
                f"  Saved phi: {saved['phi'].shape[0]} rows from "
                f"{len(saved['done_seqs'])} sequences{spatial_note} -> {phi_path.name}"
            )
            del saved
        else:
            overall_pbar.write(f"  Warning: No data collected for {run_name}")

        if traj_bank:
            traj_final = {l: torch.cat(v, dim=1) for l, v in traj_bank.items()}
            torch.save({'trajs': traj_final, 'run': run_name}, traj_path)

        if _merge_seq_artifact(
                cfg.TEMPORAL_PHI_DIR / f"_tmp_{run_name}",
                cfg.TEMPORAL_PHI_DIR / f"{run_name}.pt",
                ["temporal_phi"], run_name, log=overall_pbar.write):
            overall_pbar.write(f"  Saved temporal phi -> {run_name}.pt")

        if _merge_seq_artifact(
                cfg.OUTPUT_DIR / "temporal_gap" / f"_tmp_{run_name}",
                cfg.OUTPUT_DIR / "temporal_gap" / f"{run_name}.pt",
                ["temporal_gap"], run_name, log=overall_pbar.write):
            overall_pbar.write(f"  Saved temporal gap -> {run_name}.pt")

        if _merge_seq_artifact(
                cfg.ANN_DIR / f"_tmp_{run_name}",
                cfg.ANN_DIR / f"{run_name}.pt",
                ["asab_gap", "last_ann_gap", "head_cls_L0_gap"],
                run_name, log=overall_pbar.write):
            overall_pbar.write(f"  Saved ANN features -> {run_name}.pt")

        if _merge_seq_artifact(
                cfg.SPIKE_DIR / f"_tmp_{run_name}",
                cfg.SPIKE_DIR / f"{run_name}.pt",
                ["spike_rate", "spike_entropy"], run_name, log=overall_pbar.write):
            overall_pbar.write(f"  Saved spike stats -> {run_name}.pt")

        if _merge_seq_artifact(
                cfg.DET_OUT_DIR / f"_tmp_{run_name}",
                cfg.DET_OUT_DIR / f"{run_name}.pt",
                ["det"], run_name, log=overall_pbar.write):
            overall_pbar.write(f"  Saved detection outputs -> {run_name}.pt")
    monitor.remove()
    print("\nBenchmark extraction complete.")


if __name__ == "__main__":
    run_benchmark()
