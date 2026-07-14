import sys
import gc
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
from analysis.vmem_utils import auroc_fpr95, split_train_eval, held_out_eval, load_pt, materialize_f32
from analysis.analyse_plots import plot_free_rider_ablation

# ---------------------------------------------------------------------------
# Number of sequences to run. Trained phi is loaded from cache (no recompute),
# raw stats are computed live, and the Random SNN is forwarded + cached. Keep
# this small while iterating; raise it (or pass --max-seq N) once the full
# Stage-1 extraction exists so the trained cache covers enough sequences.
# ---------------------------------------------------------------------------
MAX_SEQ = 50  # control ablation: 50 sequences is enough to show the gap; --max-seq to override


def resolve_max_seq():
    m = MAX_SEQ
    for i, a in enumerate(sys.argv):
        if a == "--max-seq" and i + 1 < len(sys.argv):
            m = int(sys.argv[i + 1])
    if "--test" in sys.argv or "--fast" in sys.argv or getattr(cfg, "MAX_SEQUENCES", None) == 1:
        m = 1
    if m is None:
        return None  # no cap — process every sequence
    return max(1, m)


def atomic_save(obj, path):
    tmp = path.with_name(f"_tmp_{path.name}")
    torch.save(obj, tmp)
    tmp.replace(path)


def _run_snn_stem(backbone, padded):
    """Run only the SNN stem (features_01 + features_23); ~1.8x faster, phi bit-identical."""
    x = torch.cat(
        (padded[:, 0:10, :, :].unsqueeze(2), padded[:, 10:, :, :].unsqueeze(2)),
        dim=2,
    )
    x = backbone.features_01(x)
    backbone.features_23(x)


def extract_snn_phi(module, backbone, monitor, hist_torch, device, desc="Extracting Vmem", batch_size=1):
    functional.reset_net(backbone)
    monitor.reset()
    seq_phi = []
    n_frames = hist_torch.shape[0]

    pbar = tqdm(range(0, n_frames, batch_size), desc=desc, leave=False)
    for j in pbar:
        batch_end = min(j + batch_size, n_frames)
        batch = hist_torch[j:batch_end].to(device, non_blocking=True).float()
        padded = module.input_padder.pad_tensor_ev_repr(batch)
        functional.reset_net(backbone)
        monitor.reset()
        with torch.no_grad():
            _run_snn_stem(backbone, padded)
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
    if max_seq is None:  # no cap — keep every sequence
        return phi, (len(seq_lens) if seq_lens else None)
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
    # mmap: slicing the first sequences faults only those phi rows (and never
    # the sibling phi_spatial); materialize the slice into a writable f32 array.
    d = load_pt(f)
    sliced, k = _slice_first_seqs(d["phi"], d.get("seq_lens"), max_seq)
    return materialize_f32(sliced), (k if k is not None else "all")


def compute_raw_run(seq_dirs, c_name, severity, run_name):
    """Raw-input moment stats over the first len(seq_dirs) sequences.

    Cached + resumable, mirroring extract_random_run: writes
    phi/seq_lens/done_seqs to outputs/phi/raw_<run>.pt with per-sequence
    checkpoints under a tmp dir, so a crash mid-run resumes instead of
    restarting and reruns are free. The corruption seed is deterministic
    (seed=[42, i]) so the cache is valid across invocations."""
    out_path = cfg.PHI_DIR / f"raw_{run_name}.pt"
    tmp_dir = cfg.PHI_DIR / f"_tmp_raw_{run_name}"
    needed = set(range(len(seq_dirs)))

    # Fully cached already?
    if out_path.exists():
        ex = torch.load(out_path, map_location="cpu", weights_only=True)
        if needed.issubset(set(ex.get("done_seqs", []))):
            sliced, _ = _slice_first_seqs(ex["phi"].float(), ex.get("seq_lens"), len(seq_dirs))
            return sliced.numpy()

    tmp_dir.mkdir(parents=True, exist_ok=True)
    # Recover any per-seq checkpoints from a previous crashed/partial run.
    done = {int(f.stem.split("_")[1]) for f in tmp_dir.glob("seq_*.pt")}

    for i, seq_dir in enumerate(tqdm(seq_dirs, desc=f"Raw {run_name}", leave=False)):
        if i in done:
            continue
        try:
            hist_np, _ = load_histogram(seq_dir)
        except Exception as e:
            print(f"  [raw] skipping {seq_dir.name}: {e}")
            continue
        h = torch.from_numpy(hist_np)
        if c_name is not None:
            h = apply_corruption_to_tensor(h, c_name, severity, seed=[42, i])
        phi = extract_raw_input_stats(h)
        atomic_save({"phi": phi, "idx": i}, tmp_dir / f"seq_{i}.pt")

    # Merge per-seq checkpoints in sequence-index order.
    seq_files = sorted(tmp_dir.glob("seq_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    parts, seq_lens, done_seqs = [], [], []
    for f in seq_files:
        d = torch.load(f, map_location="cpu", weights_only=True)
        parts.append(d["phi"])
        seq_lens.append(int(d["phi"].shape[0]))
        done_seqs.append(int(d["idx"]))
    if not parts:
        return np.empty((0, 0))
    phi_all = torch.cat(parts, dim=0)
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
    # Positives = held-out tail of the run only, cut with the same plain ratio
    # as the clean negatives in make_scorer (no seq_lens here), so both classes
    # come from the matched held-out portion.
    X = held_out_eval(X)
    yt = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(X))])
    ys = np.concatenate([clean_scores, scorer(X)])
    a, _ = auroc_fpr95(yt, ys)
    return a


def main():
    # --gen1-root: test-data root (test split = <root>/test). --output-dir: where
    # free_rider_ablation.csv + the plot go. (parse_known_args so --max-seq/--test
    # still flow to resolve_max_seq.) Both default to benchmark_config.
    import argparse
    _ap = argparse.ArgumentParser(add_help=False)
    _ap.add_argument("--gen1-root", default=None,
                     help="Gen1 dataset root; the test split is <root>/test")
    _ap.add_argument("--output-dir", default=None,
                     help="dir for free_rider_ablation.csv and the plot (default: outputs/results + outputs/plots)")
    _a, _ = _ap.parse_known_args()
    if _a.gen1_root:
        cfg.GEN1_ROOT = Path(_a.gen1_root)
    results_out = Path(_a.output_dir) if _a.output_dir else (cfg.OUTPUT_DIR / "results")
    if _a.output_dir:
        cfg.PLOT_DIR = Path(_a.output_dir)

    max_seq = resolve_max_seq()
    # Control ablation: max severity only (clearest Trained/Random/Raw gap).
    # Use list(cfg.SEVERITIES) here to sweep all severities instead.
    severities = [max(cfg.SEVERITIES)]

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
    seq_label = "all" if max_seq is None else max_seq
    print(f"[Config] split={cfg.SPLIT}  max_seq={seq_label}  "
          f"severities={severities}")
    print("======================================================\n")

    torch.backends.cudnn.benchmark = True  # fixed 240x304 input shape — amortises kernel selection

    input_dir = cfg.GEN1_ROOT / cfg.SPLIT
    if not input_dir.exists():
        # Allow --gen1-root to point directly AT the test split (test-only data):
        # either it's named like the split, or it already holds the sequences.
        if cfg.GEN1_ROOT.name == cfg.SPLIT or list(cfg.GEN1_ROOT.glob("*/labels_v2/labels.npz")):
            input_dir = cfg.GEN1_ROOT
        else:
            print(f"Error: '{cfg.SPLIT}' split not found in {cfg.GEN1_ROOT} "
                  f"(pass a dir containing {cfg.SPLIT}/, or the {cfg.SPLIT} dir itself)")
            return

    label_files = sorted(input_dir.glob("*/labels_v2/labels.npz"))
    seq_dirs = [p.parent.parent for p in label_files][:max_seq]
    if not seq_dirs:
        print(f"No sequences found in {input_dir}")
        return
    print(f"Running on {len(seq_dirs)} '{cfg.SPLIT}' sequence(s).")

    # Which (corruption, severity) runs have a trained-phi cache? We can only
    # run a run end-to-end if its trained run exists (Stage 1 output). All
    # severities are now included (the L5-only cap is removed).
    runs = []          # (corruption, severity, run_name)
    missing = []
    for c in cfg.CORRUPTIONS:
        for sev in severities:
            run_name = f"{c}_L{sev}"
            if (cfg.PHI_DIR / f"{run_name}.pt").exists():
                runs.append((c, sev, run_name))
            else:
                missing.append(run_name)
    if missing:
        print(f"[!] No trained-phi cache for: {', '.join(missing)}")
        print("    Run the full Stage-1 extraction to include these. Skipping them.\n")
    if not (cfg.PHI_DIR / "clean.pt").exists():
        print("Error: clean trained phi cache missing — run Stage 1 (extract.py) first.")
        return
    if not runs:
        print("Error: no (corruption, severity) run has a trained-phi cache. Run Stage 1.")
        return

    # ------------------------------------------------------------------
    # CLEAN reference for each condition (fit scorers once, reuse per corruption)
    # ------------------------------------------------------------------
    print("[Clean] Loading trained clean phi (cache) + computing raw clean stats...")
    clean_trained, n_tr = load_trained_phi("clean", max_seq)
    clean_raw = compute_raw_run(seq_dirs, None, 0, "clean")
    if n_tr != "all" and n_tr < len(seq_dirs):
        print(f"  [note] trained cache only covers {n_tr} seq(s); "
              f"trained condition runs on fewer sequences than raw/random.")

    # Random SNN: load the model once, randomize, keep it for every run.
    print("[Random] Loading model + randomizing backbone weights...")
    module, backbone = load_model(device)
    monitor = VmemMonitor(backbone, selected=cfg.PLIF_LAYERS)
    # Seed BEFORE randomizing so the random weights are reproducible across
    # invocations. extract_random_run caches per-run phi and resumes; without a
    # fixed seed a crash+resume would re-randomize to DIFFERENT weights, mixing
    # phi from two different random networks (clean vs corrupt) and corrupting
    # the Random-SNN ablation.
    torch.manual_seed(0)
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
    for c, sev, run_name in runs:
        print(f"\n[Corruption] {run_name}")

        tr, _ = load_trained_phi(run_name, max_seq)
        rd = extract_random_run(module, backbone, monitor, run_name, c, sev, seq_dirs, device)
        rw = compute_raw_run(seq_dirs, c, sev, run_name)

        cond_X = {"Trained SNN": tr, "Random SNN": rd, "Raw Input Stats": rw}
        for cond, X in cond_X.items():
            scorer, clean_scores = scorers[cond]
            results[cond][run_name] = auroc_vs_clean(scorer, clean_scores, X)
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
    run_names = [rn for _, _, rn in runs]
    print("\n=======================================================")
    print(" FREE RIDER ABLATION RESULTS (AUROC, all severities)")
    print("=======================================================")
    header = f"  {'Condition':<18} | " + " | ".join(f"{rn:<18}" for rn in run_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for cond in ["Trained SNN", "Random SNN", "Raw Input Stats"]:
        row = " | ".join(f"{results[cond].get(rn, float('nan')):<18.4f}" for rn in run_names)
        print(f"  {cond:<18} | {row}")
    print("=======================================================\n")

    # Wide grid (3 conditions x up to 30 runs) — persist a tidy CSV too.
    try:
        import pandas as pd
        rows = []
        for c, sev, rn in runs:
            for cond in results:
                rows.append({"condition": cond, "corruption": c, "severity": sev,
                             "auroc": results[cond].get(rn, float("nan"))})
        out_dir = results_out
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "free_rider_ablation.csv", index=False)
        print(f"  Saved {out_dir / 'free_rider_ablation.csv'}")
    except Exception as e:
        print(f"  [!] Could not write free_rider CSV: {e}")

    plot_free_rider_ablation(results)


if __name__ == "__main__":
    main()
