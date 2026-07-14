"""Corruption / MDD qualitative + quantitative figures — single clean generator.

Produces the full figure set into outputs/graphs/ from the real pipeline artifacts:
the φ caches (loaded lazily, one run at a time) + outputs/results/mdd_window_sweep.csv.
It fits the baseline MDD (plain max) and the improved MDD (radius'=max(iso,emp_maha) +
studentized fusion) itself, computes the per-severity baseline-vs-improved table (also
written to outputs/results/final_results.csv), and renders every figure (each guarded so
one failure can't abort the rest). Leakage-safe: MDD fits on clean TRAIN sequences only;
clean held-out negatives vs corrupted held-out positives.

Run:  python analysis/plot_corruption_graphs.py
"""
import sys
import warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":14,"axes.titlesize":15,"axes.labelsize":14,"xtick.labelsize":12,"ytick.labelsize":12,"legend.fontsize":12})
import pandas as pd

# ───────────────────────── unified pastel theme ─────────────────────────
# One soft palette drives every figure so series colors read consistently across
# the whole figure set. CORR_COLOR pins one pastel hue per corruption (the same
# color for a corruption in every plot); CLEAN_COLOR is the shared clean
# reference, and the matplotlib default cycle is made pastel so small-multiple /
# bar panels that don't pass an explicit color stay on-theme too.
PASTEL = ["#7fc7c2", "#f4a39a", "#9ab9e8", "#f6c177", "#a7d49b",
          "#c9a9dd", "#f3a0c0", "#c2bca0"]   # teal, coral, blue, peach, green, lavender, pink, khaki
_THEME_ORDER = ["hot_pixel", "event_rate_shift", "temporal_jitter",
                "event_flood", "polarity_flip", "spatial_dropout"]
CORR_COLOR = {c: PASTEL[i % len(PASTEL)] for i, c in enumerate(_THEME_ORDER)}
CLEAN_COLOR = "#9bb0c9"   # muted slate-blue — clean reference everywhere (not a corruption hue)
BASE_BAR = "#c2c2c2"      # neutral grey — "baseline" series
HILITE_BAR = "#9ab9e8"    # soft blue   — "improved" / highlighted series
POS_BAR = "#8fbf9f"       # soft green  — above chance
NEG_BAR = "#e8a39b"       # soft red    — below chance
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=PASTEL)


def _pastel_violin(parts, color=CLEAN_COLOR):
    """Recolor a violinplot's bodies + whiskers to the pastel theme."""
    for b in parts.get("bodies", []):
        b.set_facecolor(color); b.set_edgecolor("#5f6b7a"); b.set_alpha(0.6)
    for k in ("cmedians", "cmins", "cmaxes", "cbars"):
        if k in parts:
            parts[k].set_color("#5f6b7a")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import (
    LazyPhiDict, TRAIN_RATIO, split_boundary, seq_lens_after_cut,
    aggregate_by_seq, auroc_fpr95, maha_d2, LAYER_SPECS, slice_phi_stat)
from analysis.mdd import MDD, DEVICE

CALIB_FRAC = 0.15
ALPHA = 0.5                       # studentized-fusion strength of the improved MDD
GEOM_SEV = 5                      # severity used for the geometry / L5 panels
GEOM_N = 5000                     # subsample per run for scatter/KDE/violin panels
W64 = 64
GRAPH_DIR = cfg.OUTPUT_DIR / "graphs"
RES_DIR = cfg.OUTPUT_DIR / "results"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
_sub = lambda a, n=GEOM_N: a if len(a) <= n else a[rng.choice(len(a), n, replace=False)]


def _auc(cs, ts):
    y = np.concatenate([np.zeros(len(cs)), np.ones(len(ts))])
    return auroc_fpr95(y, np.concatenate([cs, ts]))[0]

def _seqauc(cs, ts, csl, tsl):
    ca, ta = aggregate_by_seq(cs, csl), aggregate_by_seq(ts, tsl)
    return _auc(ca, ta) if ca is not None and ta is not None else float("nan")

def _winauc(cs, ts, csl, tsl, w=W64):
    def agg(s, sl):
        s = np.asarray(s, np.float64)
        if not sl or int(np.sum(sl)) != len(s):
            return None
        out, st = [], 0
        for L in sl:
            seq = s[st:st + L]; st += L
            for i in range(0, L, w):
                ch = seq[i:i + w]
                if len(ch) >= max(1, w // 2):
                    out.append(ch.mean())
        return np.asarray(out) if out else None
    cw, tw = agg(cs, csl), agg(ts, tsl)
    return _auc(cw, tw) if cw is not None and tw is not None else float("nan")

def _save(fig, name):
    fig.tight_layout(); fig.savefig(GRAPH_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig); print("wrote", name)


# ───────────────────── CSV-derived summary figures (unified here) ─────────────
# These read the banked result CSVs (no phi); each skips cleanly if its CSV is
# absent. Folded in so this is the ONE figure generator for the whole paper.
_CORR_ORDER = ["hot_pixel", "event_rate_shift", "temporal_jitter",
               "event_flood", "polarity_flip", "spatial_dropout"]
_CLBL = [c.replace("_", "\n") for c in _CORR_ORDER]


def _fig_detectability(sev=5):                       # 25
    f = RES_DIR / "final_results.csv"
    if not f.exists(): return
    d = pd.read_csv(f); d = d[d.severity == sev].set_index("corruption")
    cols = ["improved_frame", "improved_w64", "improved_seq"]
    names = ["per-frame (W=1)", "windowed (W=64)", "full sequence"]
    colors = ["#cfe0f3", "#9ab9e8", "#5f86c9"]   # pastel blue gradient: W=1 / W=64 / full-seq
    x = np.arange(len(_CORR_ORDER)); w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for i, (c, nm, cl) in enumerate(zip(cols, names, colors)):
        vals = [d.loc[k, c] if k in d.index else np.nan for k in _CORR_ORDER]
        ax.bar(x + (i - 1) * w, vals, w, label=nm, color=cl, edgecolor="white")
    ax.axhline(0.5, ls="--", lw=0.8, color="grey")
    ax.text(len(_CORR_ORDER) - 0.5, 0.505, "chance", fontsize=11, color="grey", va="bottom", ha="right")
    ax.axhline(0.85, ls=":", lw=0.9, color="darkgreen")
    ax.text(len(_CORR_ORDER) - 0.5, 0.86, "solved (0.85)", fontsize=11, color="darkgreen", va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(_CLBL, fontsize=12)
    ax.set_ylim(0.3, 1.02); ax.set_ylabel("fused AUROC (L%d)" % sev)
    ax.set_title("Detection by decision granularity: four solved, two residuals", fontsize=14)
    ax.legend(fontsize=11, ncol=3, loc="lower center", frameon=False)
    _save(fig, "25_detectability_summary_L5.png")


def _fig_sensitivity(sev=5, window=64):              # 24
    f = RES_DIR / "mdd_sensitivity.csv"
    if not f.exists(): return
    df = pd.read_csv(f)
    df = df[(df.severity == sev) & (df.window == window) & (df.branch == "fused")]
    df = df[df.param != "n_ref"]  # superseded below: the shipped n_ref knob is a
    # no-op (RCF always uses the full reference outside --fast), so this column
    # was bit-identical across values. real_nref_sweep.csv ACTUALLY caps the RCF
    # reference (see real_nref_sweep.py) and is the real n_ref sensitivity.
    nf = RES_DIR / "real_nref_sweep.csv"
    real_nref = pd.read_csv(nf) if nf.exists() else None
    if real_nref is not None:
        real_nref = real_nref[(real_nref.severity == sev) & (real_nref.window == window)
                               & (real_nref.branch == "fused")].copy()
        real_nref["param"] = "n_ref"
        real_nref["value"] = real_nref["n_ref"]

    # Farthest-point-sampled reference selection (real_nref_sweep_fps.py): greedy
    # max-min cosine-distance selection instead of random subsampling. Shown
    # alongside the random-subsample panel so the recovery is visible directly.
    ff = RES_DIR / "real_nref_sweep_fps.csv"
    fps_nref = pd.read_csv(ff) if ff.exists() else None
    if fps_nref is not None:
        fps_nref = fps_nref[(fps_nref.severity == sev) & (fps_nref.window == window)
                             & (fps_nref.branch == "fused")].copy()
        fps_nref["param"] = "n_ref_fps"
        fps_nref["value"] = fps_nref["n_ref"]

    params = [("k_pca", "PCA dim $k_{pca}$"), ("k_nn", "RCF neighbours $k_{nn}$"),
              ("n_ref", "clean reference $n_{ref}$ (random)"),
              ("n_ref_fps", "clean reference $n_{ref}$ (FPS)")]
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4), sharey=True)
    for ax, (p, title) in zip(axes, params):
        if p == "n_ref_fps":
            sub = fps_nref
        elif p == "n_ref":
            sub = real_nref
        else:
            sub = df[df.param == p]
        if sub is None:
            continue
        for j, c in enumerate(_CORR_ORDER):
            cc = sub[sub.corruption == c].sort_values("value")
            if not cc.empty:
                ax.plot(cc.value, cc.auroc, "-o", ms=4, lw=1.5,
                        color=CORR_COLOR.get(c, PASTEL[j % len(PASTEL)]),
                        label=c.replace("_", " "))
        ax.set_xscale("log"); ax.set_title(title, fontsize=13)
        ax.axhline(0.5, ls="--", lw=0.8, color="grey"); ax.grid(alpha=0.25)
    axes[0].set_ylabel("fused AUROC (W=%d, L%d)" % (window, sev)); axes[0].set_ylim(0.40, 1.02)
    axes[-1].legend(fontsize=10, loc="center right", frameon=False)
    fig.suptitle("MDD is insensitive to k_pca/k_nn; farthest-point sampling recovers most of the\n"
                 "random-subsample n_ref sensitivity at far fewer reference points", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.90]); fig.savefig(GRAPH_DIR / "24_sensitivity_L5.png", dpi=170, bbox_inches="tight")
    plt.close(fig); print("wrote", "24_sensitivity_L5.png")


def _fig_representation(sev=5):                       # 26
    f = RES_DIR / "representation_metrics.csv"
    if not f.exists(): return
    df = pd.read_csv(f); df = df[df.severity == sev]
    order = ["full_membrane", "membrane_mean", "membrane_var", "membrane_kurtosis",
             "spike", "spike_entropy", "ANN", "logits"]
    disp = {"full_membrane": r"full $\varphi$", "membrane_mean": r"$\mu$",
            "membrane_var": r"$\sigma^2$", "membrane_kurtosis": r"$\kappa$",
            "spike": "spike rate", "spike_entropy": "spike entropy",
            "ANN": "ANN feat.", "logits": "logits"}
    piv = df.pivot_table(index="representation", columns="corruption", values="auroc")
    piv = piv.reindex(index=[o for o in order if o in piv.index], columns=_CORR_ORDER)
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(_CORR_ORDER))); ax.set_xticklabels(_CLBL, fontsize=12)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([disp.get(r, r) for r in piv.index], fontsize=13)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12,
                        color="white" if abs(v - 0.5) > 0.30 else "black")
    ax.set_title("Membrane representation ablation (Mahalanobis AUROC, L5)", fontsize=14)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="AUROC")
    fig.savefig(GRAPH_DIR / "26_representation_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig); print("wrote", "26_representation_ablation.png")


def _fig_detectors():                                 # 27
    f = RES_DIR / "ood_metrics.csv"
    if not f.exists(): return
    m = pd.read_csv(f).groupby("detector").auroc.mean().sort_values(ascending=False)
    names = {"pca": "PCA-Mahal", "mahalanobis": "Mahalanobis", "gmm": "GMM",
             "ae": "MLP-AE", "flow": "RealNVP", "ocsvm": "OCSVM", "knn": "kNN"}
    fig, ax = plt.subplots(figsize=(8.6, 4.2)); x = np.arange(len(m))
    ax.bar(x, m.values, color=[POS_BAR if v >= 0.5 else NEG_BAR for v in m.values], edgecolor="white")
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    for i, v in enumerate(m.values): ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=12)
    ax.set_xticks(x); ax.set_xticklabels([names.get(d, d) for d in m.index], rotation=20, fontsize=12)
    ax.set_ylim(0.40, 0.62); ax.set_ylabel("mean AUROC")
    ax.set_title("Classical OOD detectors on static $\\varphi$ (mean over 6$\\times$5)", fontsize=14)
    _save(fig, "27_detector_comparison.png")


def _corrs_from_df(df):
    """Ordered corruption list straight from a banked CSV (no phi needed)."""
    present = set(df.corruption.unique())
    return [c for c in _CORR_ORDER if c in present] or sorted(present)


GEOM_CACHE = "geom_cache.joblib"   # phi-derived stashes for the geometry/distribution panels


def _geom_figures(C, CORRS, COLORS):
    """Geometry / distribution panels (05-21) rebuilt from the cached phi-derived
    stashes (results/geom_cache.joblib). No phi load, no MDD refit -- so these can
    be re-themed / restyled freely. C is the dict written by a full (phi) run."""
    muf, sdf, pmean, comps = C["muf"], C["sdf"], C["pmean"], C["comps"]
    empP, empmu = C["empP"], C["empmu"]
    vc, muv, sdv = C["vc"], C["muv"], C["sdv"]
    G, L5 = C["G"], C["L5"]
    clean_imat, clean_imat_cal = C["clean_imat"], C["clean_imat_cal"]
    clean_fused_i, clean_esl = C["clean_fused_i"], C["clean_esl"]
    project = lambda phi: (((phi - muf) / sdf) - pmean) @ comps.T
    d2 = lambda phi: (lambda d: np.einsum("ij,jk,ik->i", d, empP, d))(project(phi) - empmu)
    znorm = lambda phi: np.linalg.norm(project(phi), axis=1)

    # -- geometry panels (L5) --
    zc, dc = znorm(G["clean"][0]), d2(G["clean"][0])
    try:  # 05 energy KDE
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, c in zip(axes.ravel(), CORRS):
            zt = znorm(G[c][0]); bins = np.linspace(0, np.percentile(np.r_[zc, zt], 99), 60)
            ax.hist(zc, bins, density=True, alpha=.55, label="clean", color=CLEAN_COLOR)
            ax.hist(zt, bins, density=True, alpha=.55, label=c, color=COLORS[c])
            ax.set_title(c, fontsize=14); ax.set_xlabel("||z|| (PCA-64 energy)")
        axes[0, 0].legend(fontsize=12); fig.suptitle("Membrane energy ||z||: clean vs corrupt (L5)", y=1.0)
        _save(fig, "05_radial_energy_kde_L5.png")
    except Exception as e: print("05 fail:", e)

    try:  # 06 mahalanobis d2 hist
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        for ax, c in zip(axes.ravel(), CORRS):
            dt = d2(G[c][0]); hi = np.percentile(np.r_[dc, dt], 98); bins = np.linspace(0, hi, 60)
            ax.hist(dc, bins, density=True, alpha=.55, label="clean", color=CLEAN_COLOR)
            ax.hist(dt, bins, density=True, alpha=.55, label=c, color=COLORS[c])
            ax.axvline(np.median(dc), c=CLEAN_COLOR, ls="--"); ax.axvline(np.median(dt), c=COLORS[c], ls="--")
            ax.set_title(c, fontsize=14); ax.set_xlabel("Mahalanobis d^2")
        axes[0, 0].legend(fontsize=12); fig.suptitle("Mahalanobis d^2: contractions sit at SMALLER d^2 (inversion) (L5)", y=1.0)
        _save(fig, "06_mahalanobis_d2_hist_L5.png")
    except Exception as e: print("06 fail:", e)

    try:  # 07 shared-PCA scatter
        Zc = project(G["clean"][0])[:, :2]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
        for ax, c in zip(axes.ravel(), CORRS):
            Zt = project(G[c][0])[:, :2]
            ax.scatter(Zc[:, 0], Zc[:, 1], s=4, alpha=.3, color=CLEAN_COLOR, label="clean")
            ax.scatter(Zt[:, 0], Zt[:, 1], s=4, alpha=.3, color=COLORS[c], label=c); ax.set_title(c, fontsize=14)
        axes[0, 0].legend(fontsize=12, markerscale=2); fig.suptitle("Shared clean-PCA(2): clean vs corrupt (L5)", y=1.0)
        _save(fig, "07_pca_scatter_shared_L5.png")
    except Exception as e: print("07 fail:", e)

    try:  # 08 drift arrows
        Cc = project(G["clean"][0])[:, :2].mean(0)
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.scatter(*project(G["clean"][0])[:, :2].T, s=4, alpha=.2, color="gray")
        ax.scatter(*Cc, c="black", s=80, marker="*", zorder=5, label="clean")
        for c in CORRS:
            Ct = project(G[c][0])[:, :2].mean(0)
            ax.annotate("", xy=Ct, xytext=Cc, arrowprops=dict(arrowstyle="->", color=COLORS[c], lw=2))
            ax.scatter(*Ct, color=COLORS[c], s=40, label=c, zorder=6)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=12, ncol=2)
        ax.set_title("Clean->corrupt centroid drift (shared PCA, L5)")
        _save(fig, "08_pca_drift_arrows_L5.png")
    except Exception as e: print("08 fail:", e)

    # -- 09 fingerprint (log) + 10 similarity, from per-corruption mean shifts --
    try:
        groups = []
        for spec in LAYER_SPECS:
            for mn, (a, b) in [("mu", (0, spec["C"])), ("var", (spec["C"], 2*spec["C"])), ("kurt", (2*spec["C"], 3*spec["C"]))]:
                groups.append((f'L{spec["idx"]+1}:{mn}', list(range(spec["phi_start"]+a, spec["phi_start"]+b))))
        M = np.array([[np.abs(L5[c]["shift"][cs]).mean() for (_, cs) in groups] for c in CORRS])
        fig, ax = plt.subplots(figsize=(11, 4.2)); im = ax.imshow(np.log10(M + 1e-3), cmap="magma", aspect="auto")
        ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], rotation=45, fontsize=11, ha="right")
        ax.set_yticks(range(len(CORRS))); ax.set_yticklabels(CORRS)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=10, color="w")
        fig.colorbar(im, label="log10 mean |standardized shift|"); ax.set_title("Channel-mean shift by layer/moment (log) — residuals ~0")
        _save(fig, "09_corruption_fingerprint_heatmap.png")
    except Exception as e: print("09 fail:", e)

    try:  # 10 similarity matrix of mean-shift directions
        V = np.array([L5[c]["shift"] for c in CORRS]); Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        S = Vn @ Vn.T
        fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(S, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(CORRS))); ax.set_xticklabels(CORRS, rotation=45, ha="right", fontsize=12)
        ax.set_yticks(range(len(CORRS))); ax.set_yticklabels(CORRS, fontsize=12)
        for i in range(len(CORRS)):
            for j in range(len(CORRS)): ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=11)
        fig.colorbar(im, label="cosine"); ax.set_title("Corruption similarity (mean-shift direction)")
        _save(fig, "10_corruption_similarity_matrix.png")
    except Exception as e: print("10 fail:", e)

    try:  # 11 LDA excluding the two saturating extremes
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        keep = [c for c in CORRS if c not in ("hot_pixel", "event_flood")]
        X = [G["clean"][0]] + [G[c][0] for c in keep]; lab = ["clean"] + keep
        y = sum(([i] * len(x) for i, x in enumerate(X)), [])
        Z = LDA(n_components=2).fit_transform(np.vstack(X), y)
        fig, ax = plt.subplots(figsize=(7.5, 6)); y = np.array(y)
        for i, nm in enumerate(lab): ax.scatter(Z[y == i, 0], Z[y == i, 1], s=5, alpha=.4, label=nm)
        ax.legend(fontsize=12, markerscale=2); ax.set_xlabel("LD1"); ax.set_ylabel("LD2")
        ax.set_title("LDA(phi) excl. hot_pixel/event_flood — residual structure")
        _save(fig, "11_lda_scatter.png")
    except Exception as e: print("11 fail:", e)

    try:  # 13 d2 violins (log)
        data = [np.log10(np.maximum(d2(G["clean"][0]), 1e-3))] + [np.log10(np.maximum(d2(G[c][0]), 1e-3)) for c in CORRS]
        fig, ax = plt.subplots(figsize=(9, 5)); _pastel_violin(ax.violinplot(data, showmedians=True))
        ax.set_xticks(range(1, len(CORRS) + 2)); ax.set_xticklabels(["clean"] + CORRS, rotation=30, ha="right")
        ax.set_ylabel("log10 d^2"); ax.set_title("d^2 distribution (log): contraction medians dip toward clean")
        _save(fig, "13_energy_d2_violins.png")
    except Exception as e: print("13 fail:", e)

    # -- 16 alpha tradeoff --
    try:
        alphas = np.linspace(0, 1.5, 16); stud = lambda M, a: M.max(1) - a * np.median(M, 1)
        sel = [c for c in ["polarity_flip", "spatial_dropout", "temporal_jitter", "event_rate_shift"] if c in L5]
        fig, ax = plt.subplots(figsize=(8, 5))
        for c in sel:
            ys = [_winauc(stud(clean_imat, a), stud(L5[c]["imat"], a), clean_esl, L5[c]["esl"]) for a in alphas]
            ax.plot(alphas, ys, "o-", color=COLORS[c], label=c, lw=2, ms=3)
        ax.axvline(ALPHA, ls=":", c="k", label=f"default a={ALPHA}")
        ax.set_xlabel("studentization a (fused = max - a*median)"); ax.set_ylabel(f"AUROC (W={W64}, L5)")
        ax.set_title(f"Fusion α-tradeoff (W={W64}): dropout up, polarity down"); ax.legend(fontsize=12); ax.grid(alpha=.3)
        _save(fig, "16_alpha_tradeoff.png")
    except Exception as e: print("16 fail:", e)

    # -- 17 fusion combiners --
    try:
        scal = [np.sort(clean_imat_cal[:, j]) for j in range(4)]
        rank = lambda M: np.stack([np.searchsorted(scal[j], M[:, j], "right") / len(scal[j]) for j in range(4)], 1)
        combs = {"max": lambda M: M.max(1), "mean": lambda M: M.mean(1),
                 "rank_max": lambda M: rank(M).max(1), f"studentized\n(max-{ALPHA}med)": lambda M: M.max(1) - ALPHA * np.median(M, 1)}
        san = lambda a: np.nan_to_num(np.asarray(a, float), posinf=1e30, neginf=-1e30)
        dd = {nm: {c: _winauc(san(fn(clean_imat)), san(fn(L5[c]["imat"])), clean_esl, L5[c]["esl"]) for c in CORRS} for nm, fn in combs.items()}
        dd = pd.DataFrame(dd); x = np.arange(len(CORRS)); w = 0.2; fig, ax = plt.subplots(figsize=(9, 5))
        for i, nm in enumerate(combs): ax.bar(x + (i - 1.5) * w, dd[nm].reindex(CORRS), w, label=nm)
        ax.axhline(0.5, ls="--", c="gray"); ax.set_xticks(x); ax.set_xticklabels(CORRS, rotation=30, ha="right")
        ax.set_ylabel(f"AUROC (W={W64}, L5)"); ax.legend(fontsize=11, ncol=2); ax.set_ylim(0.3, 1.05)
        ax.set_title(f"Fusion combiners across corruptions (W={W64}, L5)")
        _save(fig, "17_fusion_combiners.png")
    except Exception as e: print("17 fail:", e)

    # -- 18 ROC (improved fused, per-frame L5) --
    try:
        from sklearn.metrics import roc_curve
        fig, ax = plt.subplots(figsize=(6.5, 6.2))
        for c in CORRS:
            y = np.r_[np.zeros(len(clean_fused_i)), np.ones(len(L5[c]["fused_i"]))]
            s = np.r_[clean_fused_i, L5[c]["fused_i"]]; fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, color=COLORS[c], lw=2, label=f"{c} ({auroc_fpr95(y, s)[0]:.2f})")
        ax.plot([0, 1], [0, 1], "--", c="gray"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.set_title("ROC: improved fused, clean vs corrupt (L5, per-frame)"); ax.legend(fontsize=12)
        _save(fig, "18_roc_curves_L5.png")
    except Exception as e: print("18 fail:", e)

    # -- 20 sigma^2 shrinkage violin --
    try:
        shrink = lambda phi: ((phi[:, vc] - muv) / sdv).mean(1)
        data = [shrink(G["clean"][0])] + [shrink(G[c][0]) for c in CORRS]
        fig, ax = plt.subplots(figsize=(9, 5)); _pastel_violin(ax.violinplot(data, showmedians=True))
        ax.set_xticks(range(1, len(CORRS) + 2)); ax.set_xticklabels(["clean"] + CORRS, rotation=30, ha="right")
        ax.axhline(0, ls="--", c="gray"); ax.set_ylabel("mean standardized variance channel")
        ax.set_title("Variance-block shift: spatial_dropout sits LOW (under-dispersion)")
        _save(fig, "20_sigma2_shrinkage_violin.png")
    except Exception as e: print("20 fail:", e)

    # -- 21 phi_spatial KDE --
    try:
        sp_dim = G["clean"][1].shape[1]; half = sp_dim // 2
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        pairs = [(0, half, "spatial_var"), (half, sp_dim, "spatial_pr")]
        show = [("clean", "clean", CLEAN_COLOR)] + [(c, c, CORR_COLOR.get(c, col)) for c, col in
                [("event_flood", "crimson"), ("spatial_dropout", "green")] if c in G]
        for ax, (lo, hi, nm) in zip(axes, pairs):
            for key, lbl, col in show:
                v = G[key][1][:, lo:hi].mean(1); bins = np.linspace(np.percentile(v, 1), np.percentile(v, 99), 60)
                ax.hist(v, bins, density=True, alpha=.5, label=lbl, color=col)
            ax.set_title(nm); ax.set_xlabel(f"per-frame mean {nm}")
        axes[0].legend(fontsize=12); fig.suptitle("phi_spatial: flood explodes spatial_var, dropout barely moves it", y=1.02)
        _save(fig, "21_phi_spatial_kde.png")
    except Exception as e: print("21 fail:", e)


def _csv_figures(df, CORRS, COLORS):
    """Every figure the banked result CSVs support --- no phi, no MDD refit.
    Reads mdd_window_sweep.csv itself; takes the already-loaded final_results
    table (df) for the severity/granularity panels and the per-detector CSVs for
    the ablation summaries. Called both standalone (CSV-only mode) and after the
    phi geometry pass, so the quantitative figures stay identical either way."""
    swpath = RES_DIR / "mdd_window_sweep.csv"
    sweep = pd.read_csv(swpath) if swpath.exists() else None
    if sweep is not None:
        sweep["window"] = sweep["window"].astype(str)
    worder = [str(w) for w in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]] + ["full"]

    # -- window-sweep figures (mdd_window_sweep.csv) --
    if sweep is not None:
        try:  # 01 fused AUROC vs window
            d = sweep[(sweep.branch == "fused") & (sweep.severity == 5)]
            piv = d.pivot_table(index="window", values="auroc", columns="corruption").reindex([w for w in worder if w in set(d.window)])
            fig, ax = plt.subplots(figsize=(9, 5.2)); x = range(len(piv))
            for c in CORRS:
                if c in piv: ax.plot(list(x), piv[c], "o-", color=COLORS[c], label=c, lw=2, ms=4)
            ax.set_xticks(list(x)); ax.set_xticklabels(piv.index, rotation=45); ax.axhline(0.5, ls="--", c="gray")
            ax.set_ylim(0.3, 1.02); ax.set_xlabel("aggregation window (frames)"); ax.set_ylabel("fused AUROC")
            ax.set_title("Fused AUROC vs aggregation window (L5)"); ax.legend(fontsize=12, ncol=2)
            _save(fig, "01_window_sweep_fused_L5.png")
        except Exception as e: print("01 fail:", e)

        try:  # 02 per-branch window small multiples
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
            wmap = {**{str(w): i for i, w in enumerate([1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512])}, "full": 16}
            for ax, c in zip(axes.ravel(), CORRS):
                d = sweep[(sweep.corruption == c) & (sweep.severity == 5)]
                for b in ["radius", "rcf", "l4", "spatial", "fused"]:
                    db = d[d.branch == b].copy()
                    if db.empty: continue
                    db["wk"] = db.window.map(wmap); db = db.dropna(subset=["wk"]).sort_values("wk")
                    ax.plot(db.wk, db.auroc, "-o", lw=1.5, ms=3, label=b)
                ax.axhline(0.5, ls="--", c="gray", lw=.8); ax.set_title(c, fontsize=14); ax.set_ylim(0.3, 1.02); ax.grid(alpha=.25)
            tick_i = [0, 3, 7, 11, 15, 16]; tick_l = ["1", "8", "32", "128", "512", "full"]
            for ax in axes.ravel(): ax.set_xticks(tick_i); ax.set_xticklabels(tick_l)
            for ax in axes[-1]: ax.set_xlabel("aggregation window (frames)")
            for ax in axes[:, 0]: ax.set_ylabel("AUROC")
            axes[0, 0].legend(fontsize=12)
            fig.suptitle("Per-branch AUROC vs window, by corruption (L5)", y=1.0)
            _save(fig, "02_window_branches_L5.png")
        except Exception as e: print("02 fail:", e)

        try:  # 04 branch x corruption heatmap (W=64, L5)
            d = sweep[(sweep.severity == 5) & (sweep.window == str(W64))]
            piv = d.pivot_table(index="corruption", columns="branch", values="auroc").reindex(index=CORRS, columns=["radius","rcf","l4","spatial","fused"])
            fig, ax = plt.subplots(figsize=(7, 4.5)); im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.3, vmax=1, aspect="auto")
            ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns); ax.set_yticks(range(len(CORRS))); ax.set_yticklabels(CORRS)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12)
            fig.colorbar(im, label="AUROC"); ax.set_title(f"Branch × corruption AUROC (L5, W={W64})")
            _save(fig, "04_branch_heatmap_L5.png")
        except Exception as e: print("04 fail:", e)

        try:  # 19 window heatmap (fused)
            d = sweep[(sweep.branch == "fused") & (sweep.severity == 5)]
            piv = d.pivot_table(index="corruption", columns="window", values="auroc").reindex(index=CORRS, columns=[w for w in worder if w in set(d.window)])
            fig, ax = plt.subplots(figsize=(11, 4.2)); im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.3, vmax=1, aspect="auto")
            ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=45, fontsize=11)
            ax.set_yticks(range(len(CORRS))); ax.set_yticklabels(CORRS)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10)
            fig.colorbar(im, label="fused AUROC"); ax.set_xlabel("window"); ax.set_title("Fused AUROC: corruption x window (L5)")
            _save(fig, "19_window_heatmap_L5.png")
        except Exception as e: print("19 fail:", e)

    # -- severity / granularity figures (final_results.csv) --
    if df is not None:
        try:  # 03 severity sweep (improved W=64 from the table)
            fig, ax = plt.subplots(figsize=(7.5, 5))
            for c in CORRS:
                dc = df[df.corruption == c].sort_values("severity")
                ax.plot(dc.severity, dc.improved_w64, "o-", color=COLORS[c], label=c, lw=2)
            ax.axhline(0.5, ls="--", c="gray"); ax.set_ylim(0.3, 1.02); ax.grid(alpha=.25)
            ax.set_xticks(sorted(df.severity.unique()))
            ax.set_xlabel("severity"); ax.set_ylabel(f"fused AUROC (W={W64}, ≈0.64 s)")
            ax.set_title(f"Severity sweep (improved MDD, W={W64} window)"); ax.legend(fontsize=12, ncol=2)
            _save(fig, "03_severity_sweep_w64.png")
        except Exception as e: print("03 fail:", e)

        try:  # 15 baseline vs improved bars (L5, W=64)
            d = df[df.severity == GEOM_SEV].set_index("corruption").reindex(CORRS)
            x = np.arange(len(CORRS)); w = 0.38; fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(x - w/2, d.baseline_w64, w, label="baseline (max)", color=BASE_BAR)
            ax.bar(x + w/2, d.improved_w64, w, label=f"improved (max-{ALPHA}*med)", color=HILITE_BAR)
            for i, im in enumerate(d.improved_w64): ax.text(i + w/2, im + 0.01, f"{im:.2f}", ha="center", fontsize=11)
            ax.axhline(0.5, ls="--", c="gray"); ax.set_ylim(0.3, 1.05); ax.set_xticks(x)
            ax.set_xticklabels(CORRS, rotation=30, ha="right"); ax.set_ylabel(f"AUROC (W={W64}, L5)")
            ax.legend(); ax.set_title(f"Baseline vs improved MDD (L5, W={W64})")
            _save(fig, "15_baseline_vs_improved_bars.png")
        except Exception as e: print("15 fail:", e)

        try:  # 22 improvement by severity (W=64)
            d = df.copy(); d["delta"] = d.improved_w64 - d.baseline_w64
            piv = d.pivot_table(index="corruption", columns="severity", values="delta").reindex(CORRS)
            m = np.nanmax(np.abs(piv.values)); fig, ax = plt.subplots(figsize=(6, 4.2))
            im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-m, vmax=m, aspect="auto")
            ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels([f"L{c}" for c in piv.columns])
            ax.set_yticks(range(len(CORRS))); ax.set_yticklabels(CORRS)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if np.isfinite(v): ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=11)
            fig.colorbar(im, label=f"improved − baseline (W={W64})"); ax.set_title(f"Improvement by corruption × severity (W={W64})")
            _save(fig, "22_improvement_by_severity.png")
        except Exception as e: print("22 fail:", e)

        try:  # 23 best (improved) results: severity x granularity, per corruption
            gran = [("improved_frame", "W=1 (frame)", "o-"),
                    ("improved_w64", "W=64", "s--"),
                    ("improved_seq", "full sequence", "^-")]
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=True)
            for ax, c in zip(axes.ravel(), CORRS):
                dc = df[df.corruption == c].sort_values("severity")
                for col, lbl, st in gran:
                    ax.plot(dc.severity, dc[col], st, lw=2, ms=5, label=lbl)
                ax.axhline(0.5, ls="--", c="gray", lw=.8); ax.set_title(c, fontsize=15); ax.set_ylim(0.4, 1.02)
                ax.set_xticks(sorted(df.severity.unique()))
            axes[0, 0].legend(fontsize=13)
            for ax in axes[-1]: ax.set_xlabel("severity")
            for ax in axes[:, 0]: ax.set_ylabel("AUROC (improved MDD)")
            fig.suptitle("Best MDD results: AUROC vs severity at W=1 / W=64 / full-sequence, per corruption", y=1.0)
            _save(fig, "23_best_results_by_severity.png")
        except Exception as e: print("23 fail:", e)

    # -- per-detector / ablation summary figures (their own CSVs) --
    for _fn in (_fig_detectability, _fig_sensitivity, _fig_representation, _fig_detectors):
        try: _fn()
        except Exception as e: print(f"{_fn.__name__} fail:", e)


def main(no_phi=False):
    RES_DIR.mkdir(parents=True, exist_ok=True)
    ap = None
    if not no_phi:
        ap = LazyPhiDict(cache_size=1)
        if "clean" not in ap:
            print(f"clean.pt missing from {cfg.PHI_DIR} -- falling back to CSV-only mode")
            ap = None

    # ---- CSV-only path: rebuild the quantitative figures straight from banked CSVs ----
    if ap is None:
        fr = RES_DIR / "final_results.csv"
        df = pd.read_csv(fr) if fr.exists() else None
        if df is None:
            print(f"{fr} missing -- no banked results to plot from."); return
        CORRS = _corrs_from_df(df)
        COLORS = {c: CORR_COLOR.get(c, PASTEL[i % len(PASTEL)]) for i, c in enumerate(CORRS)}
        _csv_figures(df, CORRS, COLORS)
        # geometry/distribution panels -> rebuilt from the cached phi stashes, if present
        cache_path = RES_DIR / GEOM_CACHE
        if cache_path.exists():
            import joblib
            C = joblib.load(cache_path)
            gcorrs = C.get("CORRS", CORRS)
            gcolors = {c: CORR_COLOR.get(c, PASTEL[i % len(PASTEL)]) for i, c in enumerate(gcorrs)}
            _geom_figures(C, gcorrs, gcolors)
            print(f"(geometry figures rebuilt from results/{GEOM_CACHE})")
        else:
            print(f"(no results/{GEOM_CACHE} yet -- run once *with* phi to create it; "
                  "geometry/distribution panels skipped)")
        print(f"\nDONE (no phi) -- figures in {GRAPH_DIR}")
        return

    # ---- full path: phi present -> geometry figures + (re)compute final_results.csv ----
    CORRS = [c for c in cfg.CORRUPTIONS
             if any(f"{c}_L{s}" in ap for s in cfg.SEVERITIES)]
    COLORS = {c: CORR_COLOR.get(c, PASTEL[i % len(PASTEL)]) for i, c in enumerate(CORRS)}

    # ---- clean split + fit baseline & improved MDD ----
    clean = ap["clean"]; csl_all = ap.get_seq_lens("clean"); csp = ap.get_phi_spatial("clean")
    cut = split_boundary(len(clean), TRAIN_RATIO, csl_all)
    fit_end = max(1, int(cut * (1 - CALIB_FRAC)))
    phi_fit, phi_cal, phi_eval = clean[:fit_end], clean[fit_end:cut], clean[cut:]
    sp_fit, sp_cal, sp_eval = csp[:fit_end], csp[fit_end:cut], csp[cut:]
    clean_esl = seq_lens_after_cut(csl_all, cut)
    mdd_b = MDD(use_spatial=True, use_emp_radius=False, fusion_alpha=0.0).fit(phi_fit, phi_cal, sp_fit, sp_cal)
    mdd_i = MDD(use_spatial=True, use_emp_radius=True, fusion_alpha=ALPHA).fit(phi_fit, phi_cal, sp_fit, sp_cal)
    BR = ["radius", "rcf", "l4", "spatial"]
    imat = lambda phi, sp: (lambda d: np.stack([d[b] for b in BR], 1))(mdd_i.score_branches(phi, sp))

    # projection / empirical-d2 for geometry
    muf, sdf, pmean, comps = mdd_i.mu_f, mdd_i.sd_f, mdd_i.pca_mean, mdd_i.pca_comps
    project = lambda phi: (((phi - muf) / sdf) - pmean) @ comps.T
    Zf = project(phi_fit); empP = np.linalg.pinv(np.cov(Zf.T)); empmu = Zf.mean(0)
    d2 = lambda phi: (lambda d: np.einsum("ij,jk,ik->i", d, empP, d))(project(phi) - empmu)
    znorm = lambda phi: np.linalg.norm(project(phi), axis=1)
    mu0, sd0 = phi_fit.mean(0), phi_fit.std(0) + 1e-9

    # clean stashes
    G = {"clean": (_sub(phi_eval), _sub(sp_eval))}
    clean_imat = imat(phi_eval, sp_eval); clean_imat_cal = imat(phi_cal, sp_cal)
    clean_fused_i = mdd_i.score(phi_eval, sp_eval)
    L5 = {}        # corruption -> dict(imat, esl, fused_i, mean_shift)

    # ---- Phase 1: severity table + L5 stashes ----
    rows = []
    for c in CORRS:
        for s in cfg.SEVERITIES:
            run = f"{c}_L{s}"
            if run not in ap:
                continue
            phi = ap[run]; sp = ap.get_phi_spatial(run); sl = ap.get_seq_lens(run)
            rc = split_boundary(len(phi), TRAIN_RATIO, sl); pe, pse = phi[rc:], sp[rc:]
            esl = seq_lens_after_cut(sl, rc)
            fb, fi = mdd_b.score(pe, pse), mdd_i.score(pe, pse)
            cb, ci = mdd_b.score(phi_eval, sp_eval), clean_fused_i
            rows.append(dict(corruption=c, severity=s,
                baseline_frame=_auc(cb, fb), improved_frame=_auc(ci, fi),
                baseline_w64=_winauc(cb, fb, clean_esl, esl), improved_w64=_winauc(ci, fi, clean_esl, esl),
                baseline_seq=_seqauc(cb, fb, clean_esl, esl), improved_seq=_seqauc(ci, fi, clean_esl, esl)))
            if s == GEOM_SEV:
                G[c] = (_sub(pe), _sub(pse))
                L5[c] = dict(imat=imat(pe, pse), esl=esl, fused_i=fi,
                             shift=(pe.mean(0) - mu0) / sd0)
            print(f"  {run:24s} seq {rows[-1]['baseline_seq']:.3f}->{rows[-1]['improved_seq']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RES_DIR / "final_results.csv", index=False)
    print("wrote results/final_results.csv")

    # ---- cache the phi-derived stashes so the geometry/distribution figures can be
    #      re-themed / restyled later with no phi load + MDD refit
    #      (`python analysis/plot_corruption_graphs.py --no-phi` reuses this) ----
    vc = slice_phi_stat(np.arange(phi_fit.shape[1])[None, :].astype(np.float32), "var").astype(int).ravel()
    muv, sdv = phi_fit[:, vc].mean(0), phi_fit[:, vc].std(0) + 1e-9
    C = dict(
        CORRS=list(CORRS),
        muf=np.asarray(muf), sdf=np.asarray(sdf), pmean=np.asarray(pmean), comps=np.asarray(comps),
        empP=np.asarray(empP), empmu=np.asarray(empmu),
        vc=np.asarray(vc), muv=np.asarray(muv), sdv=np.asarray(sdv),
        G={k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in G.items()},
        L5={c: {kk: (list(vv) if kk == "esl" else np.asarray(vv)) for kk, vv in d.items()}
            for c, d in L5.items()},
        clean_imat=np.asarray(clean_imat), clean_imat_cal=np.asarray(clean_imat_cal),
        clean_fused_i=np.asarray(clean_fused_i), clean_esl=list(clean_esl))
    import joblib
    cache_path = RES_DIR / GEOM_CACHE
    joblib.dump(C, cache_path, compress=3)
    print(f"wrote results/{GEOM_CACHE} ({cache_path.stat().st_size/1e6:.0f} MB)")

    # geometry / distribution panels (05-21), rendered from the cache just written
    _geom_figures(C, CORRS, COLORS)

    # ---- quantitative figures from the banked CSVs (themed; shared with CSV-only mode) ----
    _csv_figures(df, CORRS, COLORS)

    print(f"\nDONE — figures in {GRAPH_DIR}")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="Generate corruption / MDD figures.")
    _ap.add_argument("--phi-dir", default=None,
                     help="dir holding the phi .pt files (the analysis input; default: outputs/phi)")
    _ap.add_argument("--output-dir", default=None,
                     help="dir for the graphs/ + results/ outputs (default: outputs/)")
    _ap.add_argument("--no-phi", action="store_true",
                     help="skip phi loading + MDD refit; rebuild only the quantitative "
                          "figures from the banked result CSVs (no geometry/distribution panels)")
    _a = _ap.parse_args()
    if _a.phi_dir:
        cfg.PHI_DIR = Path(_a.phi_dir)
    if _a.output_dir:
        GRAPH_DIR = Path(_a.output_dir) / "graphs"
        RES_DIR = Path(_a.output_dir) / "results"
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    main(no_phi=_a.no_phi)
