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
from analysis.vmem_utils import auroc_fpr95, split_train_eval, TABLE_DIR
from analysis.analyse_plots import plot_free_rider_ablation

# ---------------------------------------------------------------------------
# Number of sequences to run. Trained phi is loaded from cache (no recompute),
# raw stats are computed live, and the Random SNN is forwarded + cached. Keep
# this small while iterating; raise it (or pass --max-seq N) once the full
# Stage-1 extraction exists so the trained cache covers enough sequences.
# ---------------------------------------------------------------------------
MAX_SEQ = 50


def resolve_max_seq():
    m = MAX_SEQ
    for i, a in enumerate(sys.argv):
        if a == "--max-seq" and i + 1 < len(sys.argv):
            m = int(sys.argv[i + 1])
    if "--test" in sys.argv or "--fast" in sys.argv or getattr(cfg, "MAX_SEQUENCES", None) == 1:
        m = 1
    return max(1, m)


def atomic_save(obj, path):
    tmp = path.with_name(f"_tmp_{path.name}")
    torch.save(obj, tmp)
    tmp.replace(path)


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
    import math
    from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
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
            # Reset the trained running statistics too — otherwise the
            # "Random SNN" condition still carries learned normalization.
            if m.running_mean is not None:
                torch.nn.init.constant_(m.running_mean, 0.0)
            if m.running_var is not None:
                torch.nn.init.constant_(m.running_var, 1.0)
        elif isinstance(m, MultiStepParametricLIFNode):
            # Reset the learned membrane time constant to its init (tau=2).
            if hasattr(m, "w") and isinstance(m.w, torch.nn.Parameter):
                torch.nn.init.constant_(m.w, -math.log(2.0 - 1.0))


# ---------------------------------------------------------------------------
# Loaders / extractors per condition
# ---------------------------------------------------------------------------

def _slice_first_seqs(phi, seq_lens, max_seq):
    """Keep the rows belonging to the first `max_seq` sequences.

    phi rows are concatenated in sequence-processing order, so the first
    `max_seq` sequences are the first sum(seq_lens[:max_seq]) rows. Returns
    (sliced_phi, n_seqs_used)."""
    if seq_lens:
        k = min(max_seq, len(seq_lens))
        rows = int(sum(seq_lens[:k]))
        return phi[:rows], k
    return phi, None


def load_trained_phi(run_name, max_seq):
    """Trained-SNN phi from the Stage-1 cache (never recomputed)."""
    f = cfg.PHI_DIR / f"{run_name}.pt"
    if not f.exists():
        return None, 0
    d = torch.load(f, map_location="cpu", weights_only=True)
    phi = d["phi"].float()
    sliced, k = _slice_first_seqs(phi, d.get("seq_lens"), max_seq)
    return sliced.numpy(), (k if k is not None else "all")


def compute_raw_run(seq_dirs, c_name, severity):
    """Raw-input moment stats over the first len(seq_dirs) sequences (live)."""
    parts = []
    for i, seq_dir in enumerate(seq_dirs):
        try:
            hist_np, _ = load_histogram(seq_dir)
        except Exception as e:
            print(f"  [raw] skipping {seq_dir.name}: {e}")
            continue
        h = torch.from_numpy(hist_np)
        if c_name is not None:
            h = apply_corruption_to_tensor(h, c_name, severity, seed=[42, i])
        parts.append(extract_raw_input_stats(h))
    if not parts:
        return np.empty((0, 0))
    return torch.cat(parts, dim=0).numpy()


def extract_random_run(module, backbone, monitor, run_name, c_name, severity, seq_dirs, device):
    """Random-SNN phi for a run, cached + resumable.

    Writes phi/seq_lens/done_seqs to outputs/phi/random_<run>.pt. Per-sequence
    work is checkpointed to a tmp dir so a crash mid-run resumes instead of
    restarting. Returns phi sliced to the requested sequences."""
    out_path = cfg.PHI_DIR / f"random_{run_name}.pt"
    tmp_dir = cfg.PHI_DIR / f"_tmp_random_{run_name}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    needed = set(range(len(seq_dirs)))

    # Fully cached already?
    if out_path.exists():
        ex = torch.load(out_path, map_location="cpu", weights_only=True)
        if needed.issubset(set(ex.get("done_seqs", []))):
            sliced, _ = _slice_first_seqs(ex["phi"].float(), ex.get("seq_lens"), len(seq_dirs))
            return sliced.numpy()

    # Recover any per-seq checkpoints from a previous crashed/partial run.
    done = {int(f.stem.split("_")[1]) for f in tmp_dir.glob("seq_*.pt")}

    for i, seq_dir in enumerate(tqdm(seq_dirs, desc=f"Random {run_name}", leave=False)):
        if i in done:
            continue
        try:
            hist_np, _ = load_histogram(seq_dir)
        except Exception as e:
            print(f"  [random] skipping {seq_dir.name}: {e}")
            continue
        h = torch.from_numpy(hist_np)
        if c_name is not None:
            h = apply_corruption_to_tensor(h, c_name, severity, seed=[42, i])
        phi = extract_snn_phi(module, backbone, monitor, h, device,
                              desc=f"Random {run_name} seq {i}")
        atomic_save({"phi": phi, "idx": i}, tmp_dir / f"seq_{i}.pt")

    # Merge per-seq checkpoints in sequence-index order.
    seq_files = sorted(tmp_dir.glob("seq_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    parts, seq_lens, done_seqs = [], [], []
    for f in seq_files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        parts.append(d["phi"])
        seq_lens.append(int(d["phi"].shape[0]))
        done_seqs.append(int(d["idx"]))
    phi_all = torch.cat(parts, dim=0) if parts else torch.empty((0, 2112))
    atomic_save({"phi": phi_all, "seq_lens": seq_lens, "done_seqs": done_seqs}, out_path)

    # Clean up the tmp checkpoints now that the merged file is written.
    for f in seq_files:
        f.unlink()
    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    sliced, _ = _slice_first_seqs(phi_all.float(), seq_lens, len(seq_dirs))
    return sliced.numpy()


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def make_scorer(clean_X):
    """Fit Mahalanobis on the clean train split; return (scorer, held-out clean scores).

    Only the held-out clean portion is scored as negatives so the AUROC is not
    biased by scoring the scorer's own training data."""
    clean_fit, clean_eval = split_train_eval(clean_X)
    scorer = mahalanobis_scorer(clean_fit)
    return scorer, scorer(clean_eval)


def auroc_vs_clean(scorer, clean_scores, X):
    yt = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(X))])
    ys = np.concatenate([clean_scores, scorer(X)])
    a, _ = auroc_fpr95(yt, ys)
    return a


def main():
    max_seq = resolve_max_seq()
    max_sev = max(cfg.SEVERITIES)

    print("\n======================================================")
    print(" LEVEL 10 - Free Rider Ablation (Idea 9)")
    print("======================================================")
    device = cfg.DEVICE
    cuda_avail = torch.cuda.is_available()
    print(f"[CUDA Status] PyTorch CUDA available: {cuda_avail}")
    print(f"[CUDA Status] Configured device: {device}")
    if device == "cuda" and cuda_avail:
        print("[CUDA Status] CUDA active — used for the Random SNN forward pass.")
    else:
        print("[CUDA Status] WARNING: Running on CPU (CUDA not active/available).")
    print(f"[Config] split={cfg.SPLIT}  max_seq={max_seq}  max_severity=L{max_sev}")
    print("======================================================\n")

    input_dir = cfg.GEN1_ROOT / cfg.SPLIT
    if not input_dir.exists():
        print(f"Error: '{cfg.SPLIT}' split not found in {cfg.GEN1_ROOT}")
        return

    label_files = sorted(input_dir.glob("*/labels_v2/labels.npz"))
    seq_dirs = [p.parent.parent for p in label_files][:max_seq]
    if not seq_dirs:
        print(f"No sequences found in {input_dir}")
        return
    print(f"Running on {len(seq_dirs)} '{cfg.SPLIT}' sequence(s).")

    # Which corruptions have a trained-phi cache at max severity? We can only
    # run a corruption end-to-end if its trained run exists (Stage 1 output).
    corruptions = []
    missing = []
    for c in cfg.CORRUPTIONS:
        if (cfg.PHI_DIR / f"{c}_L{max_sev}.pt").exists():
            corruptions.append(c)
        else:
            missing.append(c)
    if missing:
        print(f"[!] No trained-phi cache for: {', '.join(f'{c}_L{max_sev}' for c in missing)}")
        print("    Run the full Stage-1 extraction to include these. Skipping them.\n")
    if not (cfg.PHI_DIR / "clean.pt").exists():
        print("Error: clean trained phi cache missing — run Stage 1 (extract.py) first.")
        return
    if not corruptions:
        print("Error: no corruption has a trained-phi cache at max severity. Run Stage 1.")
        return

    # ------------------------------------------------------------------
    # CLEAN reference for each condition (fit scorers once, reuse per corruption)
    # ------------------------------------------------------------------
    print("[Clean] Loading trained clean phi (cache) + computing raw clean stats...")
    clean_trained, n_tr = load_trained_phi("clean", max_seq)
    clean_raw = compute_raw_run(seq_dirs, None, 0)
    if n_tr != "all" and n_tr < len(seq_dirs):
        print(f"  [note] trained cache only covers {n_tr} seq(s); "
              f"trained condition runs on fewer sequences than raw/random.")

    # Random SNN: load the model once, randomize, keep it for every run.
    print("[Random] Loading model + randomizing backbone weights...")
    module, backbone = load_model(device)
    monitor = VmemMonitor(backbone, selected=cfg.PLIF_LAYERS)
    randomize_weights(backbone)
    print("[Random] Extracting random clean phi...")
    clean_random = extract_random_run(module, backbone, monitor, "clean", None, 0, seq_dirs, device)

    scorers = {
        "Trained SNN":     make_scorer(clean_trained),
        "Random SNN":      make_scorer(clean_random),
        "Raw Input Stats": make_scorer(clean_raw),
    }

    # ------------------------------------------------------------------
    # Per-corruption scoring (memory-bounded: one corruption resident at a time)
    # ------------------------------------------------------------------
    results = {cond: {} for cond in scorers}
    for c in corruptions:
        run_name = f"{c}_L{max_sev}"
        print(f"\n[Corruption] {run_name}")

        tr, _ = load_trained_phi(run_name, max_seq)
        rd = extract_random_run(module, backbone, monitor, run_name, c, max_sev, seq_dirs, device)
        rw = compute_raw_run(seq_dirs, c, max_sev)

        cond_X = {"Trained SNN": tr, "Random SNN": rd, "Raw Input Stats": rw}
        for cond, X in cond_X.items():
            scorer, clean_scores = scorers[cond]
            results[cond][c] = auroc_vs_clean(scorer, clean_scores, X)
        del tr, rd, rw, cond_X
        gc.collect()

    # Done with the model.
    monitor.remove()
    del module, backbone
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n=======================================================")
    print(f" FREE RIDER ABLATION RESULTS (AUROC, L{max_sev})")
    print("=======================================================")
    header = f"  {'Condition':<18} | " + " | ".join(f"{c:<16}" for c in corruptions)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cond in ["Trained SNN", "Random SNN", "Raw Input Stats"]:
        row = " | ".join(f"{results[cond].get(c, float('nan')):<16.4f}" for c in corruptions)
        print(f"  {cond:<18} | {row}")
    print("=======================================================\n")

    plot_free_rider_ablation(results)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Condition"] + [f"{c}_AUROC_L{max_sev}" for c in corruptions]
    with open(TABLE_DIR / "free_rider_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cond in ["Trained SNN", "Random SNN", "Raw Input Stats"]:
            row = {"Condition": cond}
            for c in corruptions:
                row[f"{c}_AUROC_L{max_sev}"] = round(results[cond].get(c, float("nan")), 4)
            w.writerow(row)
    print("  Saved free_rider_ablation.csv")


if __name__ == "__main__":
    main()
