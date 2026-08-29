"""Stage: reproduce the paper's "MDD Hyperparameter Sensitivity" appendix
(Docs/paper/paper_iclr.tex, \\label{app:sensitivity}) on the current
fit/calib/sensitivity/final split, instead of the old split the paper's
cached numbers came from.

Sweeps the three free knobs one at a time, refitting MDD and re-scoring,
FUSED (studentized max - 0.5*median, the shipped default -- this experiment
is about hyperparameter sensitivity of the shipped detector, not the
combiner-choice question the other mdd_*_sweep.py scripts investigate),
window=64, L5, all 6 corruptions -- matching the paper's own description
exactly ("full L5 grid, studentized fusion", Figure 24_sensitivity_L5.png):

  k_pca in [16, 32, 64, 128, 256]   (shipped default 64)
  k_nn  in [8, 16, 32, 64, 128]     (shipped default 64)
  n_ref in [1000, 5000, 15000, 50000, full]   REAL cap: the shipped `n_ref`
        constructor arg is a no-op (RCF always scores against the full
        reference -- see mdd.py's `_subsample`), so this knob is swept by
        fitting ONCE then randomly subsampling `mdd.ref_dir`/`mdd.ref_r`
        post-fit, matching the paper's "real cap, random subsample" method.

Dev-pool exploration (like every other mdd_*_sweep.py here): runs on the
SENSITIVITY pool by default; --pool final for the frozen one-shot check.

Output: outputs/results/mdd_hparam_sensitivity_<pool>.csv
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import LazyPhiDict, split_boundaries, auroc_fpr95
from analysis.mdd import MDD
from analysis.mdd_split_sensitivity import POOL_NAMES, POOL_FRACS, _seq_ids
from analysis.evaluate_mdd_windows import aggregate_by_windows

K_PCA_VALUES = [16, 32, 64, 128, 256]
K_NN_VALUES = [8, 16, 32, 64, 128]
N_REF_VALUES = [1000, 5000, 15000, 50000, "full"]
WINDOW = 64
SEV = 5
SEED = 0


def _fused_auroc_per_corruption(mdd, all_phi, a, b, seq_lens_pool, spatial, use_spatial):
    clean_fused = mdd.score_branches(all_phi["clean"][a:b],
                                      spatial[a:b] if use_spatial else None)["fused"]
    out = {}
    for corruption in cfg.CORRUPTIONS:
        run = f"{corruption}_L{SEV}"
        if run not in all_phi:
            continue
        phi_run = all_phi[run][a:b]
        sp_run = all_phi.get_phi_spatial(run)
        sp_run = sp_run[a:b] if (use_spatial and sp_run is not None) else None
        corr_fused = mdd.score_branches(phi_run, sp_run)["fused"]
        cs = aggregate_by_windows(clean_fused, seq_lens_pool, WINDOW)
        ts = aggregate_by_windows(corr_fused, seq_lens_pool, WINDOW)
        y = np.concatenate([np.zeros(len(cs)), np.ones(len(ts))])
        s = np.concatenate([cs, ts])
        auroc, _ = auroc_fpr95(y, s)
        out[corruption] = auroc
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", choices=["sensitivity", "final"], default="sensitivity")
    args = ap.parse_args()
    pool = args.pool

    print(f"MDD hyperparameter sensitivity (k_pca/k_nn/n_ref), {pool} pool, "
          f"fused, W={WINDOW}, L{SEV}.")
    all_phi = LazyPhiDict(cache_size=1)
    clean = all_phi["clean"]
    seq_lens = all_phi.get_seq_lens("clean")
    n = len(clean)
    cuts = split_boundaries(n, POOL_FRACS, seq_lens)
    bounds = [0] + cuts + [n]
    ranges = {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(POOL_NAMES)}
    a, b = ranges[pool]
    fa, fb = ranges["fit"]
    ca, cb = ranges["calib"]

    sids = _seq_ids(seq_lens)
    seq_ids_pool = sorted(set(sids[a:b].tolist()))
    seq_lens_pool = [seq_lens[i] for i in seq_ids_pool]

    spatial = all_phi.get_phi_spatial("clean")
    use_spatial = spatial is not None
    sp_fit = spatial[fa:fb] if use_spatial else None
    sp_calib = spatial[ca:cb] if use_spatial else None
    phi_fit, phi_calib = clean[fa:fb], clean[ca:cb]

    rows = []

    print("\n-- k_pca sweep --")
    for v in K_PCA_VALUES:
        mdd = MDD(k_pca=v, use_spatial=use_spatial).fit(phi_fit, phi_calib, sp_fit, sp_calib)
        per_c = _fused_auroc_per_corruption(mdd, all_phi, a, b, seq_lens_pool, spatial, use_spatial)
        for c, auroc in per_c.items():
            rows.append({"knob": "k_pca", "value": v, "corruption": c, "auroc": auroc})
        print(f"  k_pca={v}: MACRO={np.mean(list(per_c.values())):.4f}")

    print("\n-- k_nn sweep --")
    for v in K_NN_VALUES:
        mdd = MDD(k_nn=v, use_spatial=use_spatial).fit(phi_fit, phi_calib, sp_fit, sp_calib)
        per_c = _fused_auroc_per_corruption(mdd, all_phi, a, b, seq_lens_pool, spatial, use_spatial)
        for c, auroc in per_c.items():
            rows.append({"knob": "k_nn", "value": v, "corruption": c, "auroc": auroc})
        print(f"  k_nn={v}: MACRO={np.mean(list(per_c.values())):.4f}")

    print("\n-- n_ref sweep (REAL cap: post-fit random subsample of RCF reference) --")
    mdd_ref = MDD(use_spatial=use_spatial).fit(phi_fit, phi_calib, sp_fit, sp_calib)
    full_ref_dir = mdd_ref.ref_dir.copy()
    full_ref_r = mdd_ref.ref_r.copy()
    n_full = len(full_ref_dir)
    rng = np.random.default_rng(SEED)
    for v in N_REF_VALUES:
        if v == "full":
            mdd_ref.ref_dir, mdd_ref.ref_r = full_ref_dir, full_ref_r
        else:
            idx = rng.choice(n_full, size=min(v, n_full), replace=False)
            mdd_ref.ref_dir, mdd_ref.ref_r = full_ref_dir[idx], full_ref_r[idx]
        per_c = _fused_auroc_per_corruption(mdd_ref, all_phi, a, b, seq_lens_pool, spatial, use_spatial)
        for c, auroc in per_c.items():
            rows.append({"knob": "n_ref", "value": v, "corruption": c, "auroc": auroc})
        print(f"  n_ref={v} (of {n_full}): MACRO={np.mean(list(per_c.values())):.4f}")

    df = pd.DataFrame(rows)
    res_dir = cfg.OUTPUT_DIR / "results"
    dest = res_dir / f"mdd_hparam_sensitivity_{pool}.csv"
    df.to_csv(dest, index=False)
    print(f"\nWrote {dest} ({len(df)} rows)")

    for knob in ["k_pca", "k_nn", "n_ref"]:
        sub = df[df.knob == knob]
        piv = sub.pivot_table(index="value", columns="corruption", values="auroc")
        piv["MACRO"] = piv.mean(axis=1)
        print(f"\n=== {knob} sweep (fused, W={WINDOW}, L{SEV}, {pool} pool) ===")
        print(piv.round(3).to_string())


if __name__ == "__main__":
    main()
