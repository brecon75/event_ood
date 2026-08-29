"""95% bootstrap confidence intervals for the headline MDD AUROCs.

The independent unit is a RECORDING (sequence): per-frame scores within a recording
are correlated, so we resample recordings with replacement (cluster bootstrap),
pool the windows of the resampled recordings, and recompute AUROC. Reports the
per-sequence (full-recording) and W=64 fused AUROC with a 95% percentile CI for
every corruption at L5.

Fused = SHIPPED studentized max (max - alpha*median) over branches, matching
MDD.score_branches / evaluate_mdd / tab:window, so the CI point estimates equal
the deployed-detector tables. alpha=0 restores the legacy plain max.
Input: cached per-branch calibrated scores outputs/results/mdd_branch_scores_L5.pkl
(generator scratchpad/fusion_explore.py). CPU only.
"""
import pickle, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg

SEV = 5
B = 2000
SEED = 0
ALPHA = 0.5   # MDD.fusion_alpha default (studentized max - alpha*median)
CACHE = cfg.OUTPUT_DIR / "results" / f"mdd_branch_scores_L{SEV}.pkl"


def fused(zdict, keys, alpha=ALPHA):
    stack = np.stack([zdict[k] for k in keys], axis=1)
    mx = np.max(stack, axis=1)
    return mx - alpha * np.median(stack, axis=1) if alpha else mx


def per_recording_windows(scores, seq_lens, W):
    """List (one entry per recording) of arrays of window-mean scores."""
    out, i = [], 0
    for L in seq_lens:
        seq = scores[i:i + L]; i += L
        if W is None:
            out.append(np.array([seq.mean()]))
        else:
            w = [seq[j:j + W].mean() for j in range(0, L, W)
                 if len(seq[j:j + W]) >= max(1, W // 2)]
            out.append(np.array(w))
    return out


def auroc(neg, pos):
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
    return roc_auc_score(y, np.r_[neg, pos])


def boot_ci(clean_recs, corr_recs, rng, B):
    """Cluster bootstrap: resample recordings, pool their windows, AUROC."""
    nc, nt = len(clean_recs), len(corr_recs)
    stats = np.empty(B)
    for b in range(B):
        ci = rng.integers(0, nc, nc)
        ti = rng.integers(0, nt, nt)
        neg = np.concatenate([clean_recs[k] for k in ci])
        pos = np.concatenate([corr_recs[k] for k in ti])
        stats[b] = auroc(neg, pos)
    return np.percentile(stats, [2.5, 97.5])


def main():
    D = pickle.load(open(CACHE, "rb"))
    keys, csl = D["keys"], D["eval_sl"]
    clean = fused(D["clean_z"], keys)
    rng = np.random.default_rng(SEED)

    ORDER = ["hot_pixel", "event_rate_shift", "event_flood",
             "temporal_jitter", "spatial_dropout", "polarity_flip"]
    for W, tag in [(64, "W=64"), (None, "full")]:
        clean_recs = per_recording_windows(clean, csl, W)
        clean_pool = np.concatenate(clean_recs)
        print(f"\n=== fused AUROC + 95% bootstrap CI ({tag}, L5, {B} resamples) ===")
        print(f"{'corruption':<17} | {'AUROC':>6} | {'95% CI':>16}")
        print("-" * 46)
        for c in ORDER:
            corr = fused(D["corr"][c]["z"], keys)
            corr_recs = per_recording_windows(corr, D["corr"][c]["sl"], W)
            pt = auroc(clean_pool, np.concatenate(corr_recs))
            lo, hi = boot_ci(clean_recs, corr_recs, rng, B)
            print(f"{c:<17} | {pt:6.3f} | [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
