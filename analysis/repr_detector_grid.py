"""Representation x Detector grid — removes the single-detector confound in
`representation_ablation.py`.

That stage fits exactly ONE scorer per representation (`fit_mahalanobis`), so
its headline claim ("phi is the right signal") is entangled with "Mahalanobis is
the right density model": a representation can lose for either reason and the
table cannot tell them apart.

This script crosses every representation with every detector, and adds two
things that need no density model at all:

  * a TWO-SIDED variant of each scorer. The paper's central claim is contraction
    inversion (`spatial_dropout` at rho = -1); a one-sided score cannot express
    it, so the one- vs two-sided delta measures the claim directly.
  * ENERGY DISTANCE between clean and corrupt in the representation space
    (Szekely & Rizzo 2004) — detector-free, so it ranks representations with the
    Mahalanobis assumption removed entirely.
  * HOTELLING T^2 / Q-RESIDUAL (Jackson & Mudholkar 1979) as two additional
    detectors — the multivariate-SPC decomposition of Mahalanobis distance into
    an in-subspace (T^2) and orthogonal-residual (Q/SPE) half. MDD's own
    radius-vs-RCF split is structurally this decomposition; running it as an
    explicit baseline states the delta rather than waiting for a reviewer to.

SPLIT (2026-08-29): every cell uses the project's canonical 4-way pool split
(`vmem_utils.pool_ranges`) — fit 50% / calib 10% / sensitivity 10% / final 30%,
sequence-aligned — the SAME split as MDD and every number in
`unified_numbers/README.md`. Detectors fit on `fit`; two-sided medians and any
model selection use `calib`/`sensitivity`; every reported AUROC comes from
`final`. Previously this script used its own 70/30 split, which made its numbers
non-comparable with the MDD tables.

Output: outputs/results/repr_detector_grid.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    LAYER_SPECS, load_pt, auroc_aupr_fpr95, pool_ranges,
    slice_phi_stat, slice_phi_layer,
)
from analysis.vmem_scorers import (
    mahalanobis_scorer, pca_mahalanobis_scorer, knn_scorer,
)
from analysis.gpu_fit import GpuPCA

# The banked phi/phi_spatial cover all five severities; only the scoring config
# was ever trimmed to [1,3,5]. Score the full grid (matches unified_numbers).
cfg.SEVERITIES = [1, 2, 3, 4, 5]


# ── lean row-indexed loading ──
# The obvious implementation (LazyPhiDict -> full array -> subsample) materializes
# ~5 GB per run (2.9 GB phi + 2.0 GB phi_spatial, full length) just to keep ~20k
# of 343k rows. Indexing the mmap-backed tensor FIRST faults only the pages we
# actually want: peak RAM drops to ~200 MB and load time from minutes to seconds.
SPATIAL_DIR = cfg.REPO_ROOT / "phi_spatial"


def _spatial_path(run):
    """The 1488-D re-extraction lives in a top-level folder, one file per run,
    a couple of which carry a ' (1)' download suffix."""
    for name in (f"{run}.pt", f"{run} (1).pt"):
        p = SPATIAL_DIR / name
        if p.exists():
            return p
    return None


def load_pools(run, caps, rng, pools):
    """Return {pool: (phi_rows, spatial_rows)} sampled from each canonical pool.

    `caps` maps pool name -> max rows to keep (per-pool subsampling keeps the
    O(n^2) energy-distance and kNN terms tractable). Pool boundaries come from
    the run's own seq_lens, so they land on the same sequence edges as MDD's.
    """
    d = load_pt(cfg.PHI_DIR / f"{run}.pt")
    phi_t = d["phi"]
    n = phi_t.shape[0]
    ranges = pool_ranges(n, d.get("seq_lens", None))

    idx = {}
    for pool in pools:
        lo, hi = ranges[pool]
        k = min(caps.get(pool, 0), hi - lo)
        idx[pool] = (np.sort(rng.choice(np.arange(lo, hi), k, replace=False))
                     if k > 0 else None)

    def take(t, i):
        return None if i is None else np.array(t[i].numpy(), dtype=np.float32)

    out = {p: [take(phi_t, idx[p]), None] for p in pools}
    del phi_t, d

    sp_path = _spatial_path(run)
    if sp_path is not None:
        ds = load_pt(sp_path)
        sp_t = ds["phi_spatial"]
        if sp_t.shape[0] == n:      # row-aligned with phi, or it is not comparable
            for p in pools:
                out[p][1] = take(sp_t, idx[p])
        del sp_t, ds
    return {p: tuple(v) for p, v in out.items()}


# ── representations (phi-derived only: spike/ann/temporal dirs are all empty) ──
def build_representations(phi, spatial):
    if phi is None:
        return {}
    reps = {
        "full_membrane": phi,
        "membrane_mean": slice_phi_stat(phi, "mu"),
        "membrane_var": slice_phi_stat(phi, "var"),
        "membrane_kurtosis": slice_phi_stat(phi, "kurtosis"),
    }
    for i, spec in enumerate(LAYER_SPECS):
        reps[f"layer{i + 1}"] = slice_phi_layer(phi, i)
    if spatial is not None:
        # phi_spatial ALONE: the spatial-dispersion read-out on its own, the
        # signal GAP throws away. Never compared against the phi slices before.
        reps["phi_spatial"] = spatial
        # phi + phi_spatial: the union the shipped MDD actually sees. Its delta
        # over each half is the question "do the two carry complementary
        # information, or is one subsumed?" -- which no table has answered.
        # Each half is z-scored on its own scale first, otherwise the raw
        # magnitude difference between the moment blocks and the dispersion
        # blocks lets whichever is larger dominate a Euclidean/Mahalanobis fit
        # for a units reason rather than an information one.
        reps["phi_plus_spatial"] = np.concatenate([phi, spatial], axis=1)
    return reps


PAIR_KEY = "phi_plus_spatial"


def fit_standardizer(fit_reps):
    """Column mean/std of `phi_plus_spatial` on the FIT pool.

    Computed ONCE, from the fit pool, and reused for every other pool and every
    corruption run. Re-deriving it per-call would be a real bug: after the fit
    rows have been standardized in place their mean/std are 0/1, so a second
    call would apply an identity map to raw corruption rows and leave the two
    sides on different scales.
    """
    if PAIR_KEY not in fit_reps:
        return None
    return fit_reps[PAIR_KEY].mean(0), fit_reps[PAIR_KEY].std(0) + 1e-8


def apply_standardizer(stats, reps_list):
    """Apply the fit-pool z-scoring of `phi_plus_spatial` to each rep dict.

    Only the concatenated representation needs it: raw phi and raw phi_spatial
    differ in magnitude by orders of magnitude, so an un-normalised concat lets
    whichever block is larger dominate a Euclidean/Mahalanobis fit for a UNITS
    reason rather than an information one. Using fit-pool statistics only means
    no eval or corruption data touches the transform.
    """
    if stats is None:
        return
    mu, sd = stats
    for reps in reps_list:
        if reps.get(PAIR_KEY) is not None:
            reps[PAIR_KEY] = (reps[PAIR_KEY] - mu) / sd


# ── detectors: every factory is scorer(clean_fit) -> score(x) ──
def _fit_pca_spc(clean, n_components=50):
    """Shared PCA fit for the Hotelling T^2 / Q-residual (SPE) pair (Jackson &
    Mudholkar, "Control Procedures for Residuals Associated With Principal
    Component Analysis", Technometrics 21:341-349, 1979). `n_components`
    matches `pca_mahalanobis_scorer`'s default for consistency with the rest of
    the grid. Eigenvalues are the retained-PC variances of the FIT data (not
    returned by GpuPCA directly), needed to studentize T^2 per axis."""
    nc = max(1, min(n_components, clean.shape[1], len(clean) - 1))
    pca = GpuPCA.fit(clean, nc, op="Hotelling/SPE PCA fit (SVD)")
    proj = pca.transform(clean)
    eigvals = np.maximum(proj.var(axis=0, ddof=1), 1e-8)
    return pca, eigvals


def hotelling_t2_factory(clean, **_):
    """Hotelling's T^2: Mahalanobis distance WITHIN the retained PCA subspace,
    sum_k proj_k(x)^2 / lambda_k. The in-model half of the T^2/Q decomposition."""
    pca, eigvals = _fit_pca_spc(clean)
    def score(x):
        proj = pca.transform(x)
        return (proj ** 2 / eigvals).sum(axis=1)
    return score


def q_residual_factory(clean, **_):
    """Q-residual / SPE: squared reconstruction error ORTHOGONAL to the
    retained PCA subspace, ||x-mean||^2 - ||proj(x)||^2 -- exact without a
    full D-dim reconstruction because PCA components are orthonormal
    (Pythagorean identity). The out-of-model half of the T^2/Q decomposition."""
    pca, _ = _fit_pca_spc(clean)
    mean = pca.mean_
    def score(x):
        proj = pca.transform(x)
        centered_sq = ((x - mean) ** 2).sum(axis=1)
        return centered_sq - (proj ** 2).sum(axis=1)
    return score


def knn_factory(clean, select_on=None, **_):
    """Sun et al. (ICML 2022) kNN, with k resolved on the SENSITIVITY pool.

    See DetectorKNN in evaluate_ann_baselines.py for why neither published k
    (50 @ 50k refs, 1000 @ 1.28M refs) transfers to our ~204k reference by
    authority. The selection criterion is unsupervised (tightest clean band),
    so no corrupted data or label is consulted.
    """
    from analysis.evaluate_ann_baselines import DetectorKNN
    det = DetectorKNN(k=None)
    det.fit(clean, None, select_on=select_on if select_on is not None else clean)
    knn_factory.last_note = getattr(det, "_k_note", "")
    return lambda x: det.score(x, None)


def neco_factory(clean, **_):
    """NECO (Ammar et al., ICLR 2024) as a FEATURE-space detector.

    use_maxlogit=False is mandatory here and is not a formula deviation: the
    multiply is defined only against a trained in-distribution class posterior,
    and phi has no logits at all. Dimension d follows the paper's own
    >=90%-explained-variance rule (NOT the hardcoded 100 an earlier version used).
    """
    from analysis.evaluate_ann_baselines import DetectorNECO
    det = DetectorNECO(use_maxlogit=False)
    det.fit(clean, None)
    neco_factory.last_note = det.dim_note
    return lambda x: det.score(x, None)


DETECTORS = {
    "mahalanobis": lambda c, **kw: mahalanobis_scorer(c),
    "pca_mahalanobis": lambda c, **kw: pca_mahalanobis_scorer(c),
    "knn": knn_factory,
    "hotelling_t2": hotelling_t2_factory,
    "q_residual": q_residual_factory,
    "neco": neco_factory,
}

# `scp` is deliberately NOT here. SCP's rule fed phi instead of real spike
# counts was run once and RETRACTED (Docs/ablation_attack_plan.md S6c): a
# baseline is only valid on the cited method's own signal. Faithful SCP now
# lives in analysis/extract_scp_scores.py and needs the last-layer per-neuron
# spike extraction. Do not re-add it to this grid.


def two_sided(score_fn, calib_ref):
    """|s - median(s_calib)| — turns any one-sided score two-sided.

    A one-sided density score inverts on contractions (corrupt sits CLOSER to
    the clean mean than held-out clean does), which is exactly why kNN and OCSVM
    land below chance in `ood_metrics.csv`. Deviation in either direction is
    anomalous; this measures that without changing the underlying detector.

    The median comes from the CALIB pool — never from the eval rows it will be
    applied to, which would let the reference absorb the very shift it is meant
    to detect.
    """
    med = float(np.median(score_fn(calib_ref)))
    return lambda x: np.abs(score_fn(x) - med)


def energy_distance(a, b, cap=4000, seed=0):
    """Szekely-Rizzo energy distance: 2*E|A-B| - E|A-A'| - E|B-B'|.

    Detector-free two-sample statistic on the RAW representation vectors — no
    density model, no Gaussian assumption, so it ranks representations with the
    Mahalanobis confound removed. Subsampled: the pairwise terms are O(n^2).
    """
    from scipy.spatial.distance import cdist
    rng = np.random.default_rng(seed)
    a = a[rng.choice(len(a), min(cap, len(a)), replace=False)].astype(np.float32)
    b = b[rng.choice(len(b), min(cap, len(b)), replace=False)].astype(np.float32)
    return float(2 * cdist(a, b).mean() - cdist(a, a).mean() - cdist(b, b).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-cap", type=int, default=20000,
                    help="Clean frames sampled from the FIT pool per detector.")
    ap.add_argument("--eval-cap", type=int, default=20000,
                    help="Frames sampled from the FINAL pool per run (reported).")
    ap.add_argument("--calib-cap", type=int, default=5000,
                    help="Frames from CALIB, used for the two-sided median.")
    ap.add_argument("--sens-cap", type=int, default=5000,
                    help="Frames from SENSITIVITY, used for kNN's k selection.")
    ap.add_argument("--detectors", default=",".join(DETECTORS))
    ap.add_argument("--no-energy", action="store_true",
                    help="Skip the energy-distance column (~28 s per rep x run).")
    ap.add_argument("--output", type=Path,
                    default=cfg.OUTPUT_DIR / "results" / "repr_detector_grid.csv")
    args = ap.parse_args()

    dets = [d for d in args.detectors.split(",") if d in DETECTORS]
    if not (cfg.PHI_DIR / "clean.pt").exists():
        print(f"Error: clean.pt missing from {cfg.PHI_DIR}.")
        return

    caps = {"fit": args.fit_cap, "calib": args.calib_cap,
            "sensitivity": args.sens_cap, "final": args.eval_cap}
    rng = np.random.default_rng(42)
    clean = load_pools("clean", caps, rng, list(caps))

    fit_reps = build_representations(*clean["fit"])
    calib_reps = build_representations(*clean["calib"])
    sens_reps = build_representations(*clean["sensitivity"])
    eval_reps = build_representations(*clean["final"])
    # Fit-pool statistics, captured BEFORE fit_reps is transformed in place.
    std_stats = fit_standardizer(fit_reps)
    apply_standardizer(std_stats, [fit_reps, calib_reps, sens_reps, eval_reps])

    print(f"Grid: {len(fit_reps)} representations x {len(dets)} detectors | "
          f"pools fit={len(clean['fit'][0])} calib={len(clean['calib'][0])} "
          f"sens={len(clean['sensitivity'][0])} final={len(clean['final'][0])}")
    print(f"Representations: {', '.join(fit_reps)}")

    # Fit once per (representation, detector); reuse across all runs.
    fitted, notes = {}, {}
    for rep, X in fit_reps.items():
        for det in dets:
            try:
                fn = DETECTORS[det](X, select_on=sens_reps.get(rep))
                note = getattr(DETECTORS[det], "last_note", "")
                if note:
                    notes[(rep, det)] = note
                fitted[(rep, det)] = (fn, fn(eval_reps[rep]),
                                      two_sided(fn, calib_reps[rep]))
            except Exception as e:
                print(f"  [!] {rep}/{det} fit failed: {type(e).__name__}: {e}")
    print(f"Fitted {len(fitted)} (representation, detector) pairs.")
    for (rep, det), note in sorted(notes.items()):
        print(f"    {rep}/{det}: {note}")

    rows = []
    runs = [f"{c}_L{s}" for c in cfg.CORRUPTIONS for s in cfg.SEVERITIES
            if (cfg.PHI_DIR / f"{c}_L{s}.pt").exists()]

    for run in tqdm(runs, desc="grid"):
        # Corruption runs contribute eval rows only, from the SAME final pool.
        c_phi, c_sp = load_pools(run, {"final": args.eval_cap}, rng, ["final"])["final"]
        corr_reps = build_representations(c_phi, c_sp)
        apply_standardizer(std_stats, [corr_reps])   # same fit-pool transform
        corruption, sev = run.rsplit("_L", 1)

        for (rep, det), (fn, clean_scores, ts_fn) in fitted.items():
            corr = corr_reps.get(rep)
            if corr is None:
                continue
            for sided, cs, ts in (
                ("one", clean_scores, fn(corr)),
                ("two", ts_fn(eval_reps[rep]), ts_fn(corr)),
            ):
                m = auroc_aupr_fpr95(cs, ts)
                if m is None:
                    continue
                rows.append({"representation": rep, "detector": det, "sided": sided,
                             "corruption": corruption, "severity": int(sev),
                             "granularity": "frame", "auroc": m[0], "aupr": m[1],
                             "fpr95": m[2], "pool": "final",
                             "fit_cap": args.fit_cap, "eval_cap": args.eval_cap})

        if not args.no_energy:
            for rep, corr in corr_reps.items():
                if rep not in eval_reps:
                    continue
                rows.append({"representation": rep, "detector": "energy_distance",
                             "sided": "n/a", "corruption": corruption,
                             "severity": int(sev), "granularity": "frame",
                             "auroc": np.nan, "aupr": np.nan, "fpr95": np.nan,
                             "energy": energy_distance(eval_reps[rep], corr),
                             "pool": "final",
                             "fit_cap": args.fit_cap, "eval_cap": args.eval_cap})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
