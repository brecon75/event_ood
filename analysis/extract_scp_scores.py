"""SCP as published (Martinez-Seras et al. 2023) — faithful per-NEURON extraction.

WHY THIS SCRIPT EXISTS
----------------------
`analysis/scp_baseline.py` implements SCP's *rule* (cluster -> median archetype ->
min-L1) but has no faithful input to feed it. The existing extraction's
`monitor.collect_spikes()` returns

    p = S.mean(dim=(0, 3, 4))        # mean over (T, H, W)  ->  (B, C)

which averages over T *and over space*. SCP's Eq. (2) is

    q(n) = sum_t s_{n,t}             # sum over T only, PER NEURON n

The mean-vs-sum over T is a uniform 1/T scale factor and is harmless under L1
(it rescales every distance identically, leaving AUROC unchanged). The spatial
mean is NOT harmless: it collapses C*H*W neurons to C channel means and destroys
the per-neuron *pattern* that SCP's clustering operates on. Feeding that to SCP
is not SCP.

This script produces the real thing.

THE STORAGE PROBLEM, AND WHY THIS IS TWO PASSES
------------------------------------------------
At SNN Block 4 the map is 256 x H' x W'. For Gen1 (240x304, /16) that is
256 x 15 x 19 = 72,960 neurons per frame. Stored even as uint8:

    72,960 B/frame x 343,099 frames = 25.0 GB per run  x  31 runs = ~776 GB

That is not a feature bank anyone wants to keep. But SCP never needs the counts
kept — it needs (a) P=1000 in-distribution count vectors to fit archetypes, and
(b) one scalar score per frame at test time. So:

    stage 1 (fit)    clean run only, reservoir-sample P frames of per-neuron
                     counts (P x 72,960 floats ~ 280 MB), fit the archetype
                     bank, save the bank (a few MB).
    stage 2 (score)  every run, compute q(n) on the fly, score it against the
                     bank, keep ONLY the scalar. Output ~1.4 MB per run.

Total persistent storage: well under 100 MB, versus 776 GB. Nothing is
approximated to get there — every frame is scored on its true per-neuron count
vector.

FAITHFULNESS TO THE PAPER (arXiv 2210.00894, Alg. 1 + Eqs. 2-4, Sec. 4)
------------------------------------------------------------------------
  Eq. 2  q(n) = sum_t s_{n,t}, last layer, per neuron        EXACT (this script)
  Eq. 3  pairwise L1 / Manhattan distance                    EXACT (SCP class)
  step   agglomerative hierarchical clustering               EXACT (SCP class)
  step   archetype = MEDIAN of each cluster (f_agg)          EXACT (SCP class)
  Eq. 4  score = min_m L1(q_x, archetype_m)                  EXACT (SCP class)
  Sec.4  P = 1000 characterization instances                 EXACT (--p)
  Sec.4  standardization                                     NOT applied: spike
         counts are homogeneous units, so the published rule takes raw L1. (The
         z-scoring in scp_baseline.py exists only for the phi-fed rule-ablation
         cell, where the moment blocks are not commensurate.)

  D1  CLASS CONDITIONING DROPPED (C = 1). Their host is a classifier with one
      predicted yhat per sample; archetypes and thresholds are conditioned on it.
      Ours is YOLOX: 0..N boxes over 2 classes, no per-frame label. We fit one
      unconditional archetype bank. Clustering, median archetypes and min-L1 are
      preserved exactly; only the conditioning drops. Defensible because their
      class assignment already uses model *predictions*, not ground truth.
      THIS MUST BE STATED IN THE PAPER, not just here.

  Padding is cropped out before counting (`monitor.valid_frac`): the input padder
  adds right/bottom columns that are constant-zero neurons in every frame. They
  contribute an identical 0 to every L1 distance, so they cannot change the
  ranking — but they inflate the neuron count and make the "per-neuron" claim
  false. Cropped for correctness, not for the score.

  AUROC CONVENTION: the paper treats in-distribution as POSITIVE, which is
  inverted relative to this repo. This script emits raw scores only (higher =
  more OOD, the repo convention). Transcribe with care — three corruptions in
  this benchmark already sit below chance, so a silent flip is invisible.

USAGE (on the cluster, where the extractor and Gen1 live)
----------------------------------------------------------
    # stage 1 — fit the archetype bank on clean (one pass over clean)
    python analysis/extract_scp_scores.py --stage fit

    # stage 2 — score every run (clean + 6 corruptions x 5 severities)
    python analysis/extract_scp_scores.py --stage score

    # then, off-cluster, turn the scores into AUROC/AUPR/FPR95:
    python analysis/extract_scp_scores.py --stage report

Resume is per-run: stage 2 skips any run whose score file already exists.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg

LAST_LAYER = 3          # SNN Block 4 — SCP's "last layer L"
P_CHARACTERIZE = 1000   # their |D^c_tr|
N_CLUSTERS = 8          # M^c


# ── per-neuron count extraction ──────────────────────────────────────────────
def counts_from_monitor(monitor, layer=LAST_LAYER):
    """SCP Eq. 2 on the monitor's raw spike buffer: q(n) = sum_t s_{n,t}.

    monitor._spikes[layer] is a list of (T, B, C, H, W) tensors (one per forward
    call since the last reset). Returns (B, C*H'*W') float32 on CPU, where H'/W'
    are the valid (un-padded) extent. Summing over dim 0 is the ONLY reduction —
    no spatial pooling, which is the whole point of this script.
    """
    sp_list = monitor._spikes.get(layer)
    if not sp_list:
        return None
    S = torch.cat(sp_list, dim=1)            # (T, B, C, H, W)
    q = S.sum(dim=0)                         # (B, C, H, W)  <- Eq. 2, sum over t

    vf = getattr(monitor, "valid_frac", None)
    if vf is not None:                       # drop padder-added rows/cols
        H, W = q.shape[-2:]
        q = q[..., : max(1, round(H * vf[0])), : max(1, round(W * vf[1]))]

    return q.flatten(1).to(torch.float32).cpu().numpy()


# ── the forward pass, shared by both stages ──────────────────────────────────
def iter_run_counts(run_name, corruption, severity, module, backbone, monitor,
                    seq_dirs, on_frame):
    """Run one (corruption, severity) over every sequence, handing each frame's
    per-neuron count vector to `on_frame(seq_idx, counts)`.

    Mirrors extract.py's per-sequence loop exactly: same loader, same seeded
    corruption ([42, seq_idx]), same padder, same per-batch net/monitor reset,
    same BATCH_SIZE=1 (SpikingJelly treats the batch axis as time; B>1 leaks
    membrane state across frames and corrupts the counts).
    """
    from spikingjelly.clock_driven import functional
    from pipeline.loader import load_histogram
    from corruption_wrap import apply_corruption_to_tensor

    for seq_idx, seq_dir in enumerate(seq_dirs):
        hist_np, _ = load_histogram(seq_dir)
        if corruption is not None:
            hist_np = apply_corruption_to_tensor(
                torch.from_numpy(hist_np), corruption, severity, seed=[42, seq_idx]
            ).numpy()
        hist = torch.from_numpy(hist_np)
        del hist_np

        monitor.new_sequence()
        for j in range(0, hist.shape[0], cfg.BATCH_SIZE):
            batch = hist[j:j + cfg.BATCH_SIZE].to(cfg.DEVICE).float()
            functional.reset_net(backbone)
            monitor.reset()
            with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.float16,
                    enabled=(cfg.DEVICE == "cuda")):
                padded = module.input_padder.pad_tensor_ev_repr(batch)
                if monitor.valid_frac is None:
                    monitor.set_valid_frac((batch.shape[-2] / padded.shape[-2],
                                            batch.shape[-1] / padded.shape[-1]))
                # SNN stem only: all 4 PLIF layers fire inside these two
                # Sequentials, and nothing downstream feeds back into them.
                x = torch.cat((padded[:, 0:10].unsqueeze(2),
                               padded[:, 10:].unsqueeze(2)), dim=2)
                x = backbone.features_01(x)
                backbone.features_23(x)

            q = counts_from_monitor(monitor)
            if q is not None:
                on_frame(seq_idx, q)


def _discover_sequences():
    """Same discovery extract.py uses (extract.py:479-501), so sequence ORDER —
    and therefore the per-sequence corruption seed [42, seq_idx] and the
    seq_lens alignment with the banked phi — is identical."""
    input_dir = getattr(cfg, "INPUT_DIR", None)
    if input_dir is None:
        label_files = sorted((cfg.GEN1_ROOT / cfg.SPLIT).glob("*/labels_v2/labels.npz"))
        if not label_files:
            label_files = sorted(cfg.GEN1_ROOT.glob("*/labels_v2/labels.npz"))
    else:
        label_files = sorted(Path(input_dir).glob("*/labels_v2/labels.npz"))
    seq_dirs = [p.parent.parent for p in label_files]
    if not seq_dirs:
        raise SystemExit(f"No sequences found under {cfg.GEN1_ROOT}/{cfg.SPLIT}")
    return seq_dirs


def _load_everything():
    sys.path.insert(0, str(cfg.REPO_ROOT / "vmem_benchmark"))
    from model_loader import load_model
    from monitor import VmemMonitor
    module, backbone = load_model(cfg.DEVICE)
    monitor = VmemMonitor(backbone, selected=cfg.PLIF_LAYERS)
    return module, backbone, monitor, _discover_sequences()


# ── stage 1: fit the archetype bank on clean ─────────────────────────────────
def stage_fit(args, out_dir):
    from analysis.scp_baseline import SCP

    module, backbone, monitor, seq_dirs = _load_everything()
    rng = np.random.default_rng(args.seed)

    # Reservoir sample P frames uniformly over the whole clean run without
    # knowing the frame count in advance (Vitter's Algorithm R).
    reservoir, seen = [], 0

    def on_frame(seq_idx, q):
        nonlocal seen
        for row in q:
            seen += 1
            if len(reservoir) < args.p:
                reservoir.append(row.copy())
            else:
                j = rng.integers(0, seen)
                if j < args.p:
                    reservoir[j] = row.copy()

    print(f"[fit] clean pass over {len(seq_dirs)} sequences, reservoir P={args.p} ...")
    iter_run_counts("clean", None, 0, module, backbone, monitor, seq_dirs, on_frame)

    counts = np.stack(reservoir)
    print(f"[fit] sampled {len(counts)} of {seen} clean frames; "
          f"{counts.shape[1]} neurons at SNN Block {LAST_LAYER + 1}")

    # standardize=False: published SCP takes RAW spike counts (homogeneous units).
    scp = SCP(n_clusters=args.clusters, p=args.p, seed=args.seed, standardize=False)
    scp.fit(counts)

    dest = out_dir / "scp_archetypes.npz"
    np.savez_compressed(dest, archetypes=scp.archetypes, n_neurons=counts.shape[1],
                        p=args.p, clusters=args.clusters, seen=seen)
    print(f"[fit] wrote {dest} — {scp.archetypes.shape[0]} archetypes "
          f"x {scp.archetypes.shape[1]} neurons")


# ── stage 2: score every run ─────────────────────────────────────────────────
def stage_score(args, out_dir):
    from analysis.scp_baseline import SCP
    from scipy.spatial.distance import cdist

    bank = np.load(out_dir / "scp_archetypes.npz")
    archetypes = bank["archetypes"]
    scp = SCP(standardize=False)
    scp.archetypes = archetypes

    module, backbone, monitor, seq_dirs = _load_everything()

    runs = [("clean", None, 0)] + [
        (f"{c}_L{s}", c, s) for c in cfg.CORRUPTIONS for s in cfg.SEVERITIES]

    for run_name, corruption, severity in runs:
        dest = out_dir / f"{run_name}.npz"
        if dest.exists():
            print(f"[score] {run_name}: exists, skipping")
            continue

        scores, seq_lens, cur = [], [], [0, None]

        def on_frame(seq_idx, q):
            if cur[1] is None:
                cur[1] = seq_idx
            if seq_idx != cur[1]:
                seq_lens.append(cur[0]); cur[0] = 0; cur[1] = seq_idx
            # Eq. 4 inline — never materialize the count matrix for a whole run.
            scores.append(cdist(q, archetypes, metric="cityblock").min(axis=1))
            cur[0] += len(q)

        print(f"[score] {run_name} ...")
        iter_run_counts(run_name, corruption, severity, module, backbone,
                        monitor, seq_dirs, on_frame)
        seq_lens.append(cur[0])

        np.savez_compressed(dest,
                            scores=np.concatenate(scores).astype(np.float32),
                            seq_lens=np.array(seq_lens, dtype=np.int64))
        print(f"[score] {run_name}: {sum(seq_lens)} frames -> {dest}")


# ── stage 3: scores -> metrics ───────────────────────────────────────────────
def stage_report(args, out_dir):
    import pandas as pd
    from analysis.vmem_utils import (split_boundary, seq_lens_after_cut,
                                     aggregate_by_seq, auroc_aupr_fpr95,
                                     TRAIN_RATIO)

    clean = np.load(out_dir / "clean.npz")
    cs, csl = clean["scores"], list(clean["seq_lens"])
    cut = split_boundary(len(cs), TRAIN_RATIO, csl)
    clean_eval, clean_eval_lens = cs[cut:], seq_lens_after_cut(csl, cut)

    rows = []
    for c in cfg.CORRUPTIONS:
        for s in cfg.SEVERITIES:
            f = out_dir / f"{c}_L{s}.npz"
            if not f.exists():
                continue
            d = np.load(f)
            rs, rsl = d["scores"], list(d["seq_lens"])
            rcut = split_boundary(len(rs), TRAIN_RATIO, rsl)
            corr_eval, corr_lens = rs[rcut:], seq_lens_after_cut(rsl, rcut)

            for gran, a, b in (
                ("frame", clean_eval, corr_eval),
                ("sequence", aggregate_by_seq(clean_eval, clean_eval_lens),
                             aggregate_by_seq(corr_eval, corr_lens)),
            ):
                m = auroc_aupr_fpr95(a, b)
                if m is None:
                    continue
                rows.append({"model": "hybrid", "representation": "spike_counts_perneuron",
                             "detector": "SCP", "corruption": c, "severity": s,
                             "granularity": gran, "auroc": m[0], "aupr": m[1],
                             "fpr95": m[2]})

    dest = cfg.OUTPUT_DIR / "results" / "scp_published.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"Wrote {dest} ({len(rows)} rows). AUROC convention: corrupt = positive "
          f"(INVERTED vs the SCP paper's ID-as-positive).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["fit", "score", "report"], required=True)
    ap.add_argument("--p", type=int, default=P_CHARACTERIZE,
                    help="Characterization instances (paper: 1000).")
    ap.add_argument("--clusters", type=int, default=N_CLUSTERS,
                    help="M^c, archetypes per bank (paper lets clustering decide).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=cfg.OUTPUT_DIR / "scp")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    {"fit": stage_fit, "score": stage_score, "report": stage_report}[args.stage](
        args, args.out_dir)


if __name__ == "__main__":
    main()
