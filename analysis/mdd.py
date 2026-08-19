"""Manifold-Decomposition Detector (MDD).

A single unsupervised OOD detector that scores each frame on orthogonal axes of
the clean static-phi manifold and fuses them with a studentized max
(max - alpha*median). Built from the diagnosis that event-camera corruptions
split into geometry classes:

  * DILATIONS (hot_pixel, event_rate_shift, temporal_jitter) push phi OUTWARD —
    caught by distance from the clean mean.
  * CONTRACTIONS (spatial_dropout, event_flood) pull phi toward the clean mode —
    they INVERT distance/density detectors ("more normal than normal"); detected
    only by noticing under-dispersion (a two-sided, direction-conditioned score).
  * INVISIBLE (polarity_flip) — no signature in V_mem; out of scope.

Branches (each standardized on held-out clean, so a plain max is a true OR):
  B1 radius  : max( | ||z|| - E||z|| | / sd ,  emp_maha )   two-sided membrane
               energy OR'd with an empirical (UNSHRUNK) covariance Mahalanobis in
               PCA-k -- the latter catches the covariance ROTATION polarity_flip
               induces that Ledoit-Wolf shrinkage washes out (toggle: use_emp_radius)
  B2 RCF     : | ||z|| - E[||z|| | dir] | / sd   two-sided, conditioned on
               direction via cosine-kNN -> the contraction detector
  B3 L4 d^2  : Ledoit-Wolf Mahalanobis on the layer-4 block (deep timing signal
               the pooled radius averages away; rescues temporal_jitter).
               Scored TWO-SIDED (|d2 - med_cal| / MAD_cal, maha_two_sided=True)
               so contractions that push d2 BELOW the clean band fire too
  B4 spatial : (optional, only when phi_spatial is available) Ledoit-Wolf
               Mahalanobis (two-sided, as B3) on the spatial-dispersion base
               features GAP discards, OR'd (max, ssb_fold pattern) with the
               spatial-ORGANIZATION score when phi_spatial carries the compact
               block: midrank-ECDF-calibrated flatness + persistence components
               encoding the local-texture-floor and motion-non-stationarity
               priors (use_org=True; see monitor._compact_spatial_stats)
  B5 ssb     : (opt-in, use_ssb=True + phi_synth at fit) linear logistic head vs
               SIMULATED corruptions of fit-side sequences (CutPaste-style
               self-supervision, no real corruption labels) -> cancels the
               scene-activity nuisance axis that hides spatial_dropout; one dot
               product per frame (see __init__ comment)

IMPROVEMENTS (validated 2026-06-23, NOW WIRED IN below; default on, each toggleable;
full writeup + experiment timeline in Docs/mdd_improvements.md). Two changes target
the residual corruptions without adding a branch or re-extracting:
  (A) radius' = max(radius, emp_maha)  -- fold an empirical (UNSHRUNK) covariance
      Mahalanobis in PCA-k INTO the radius branch; it catches the covariance
      rotation a polarity_flip induces that Ledoit-Wolf shrinkage washes out.
      Disable with MDD(use_emp_radius=False).
  (B) fused = max(branches) - alpha*median(branches)  (alpha=0.5) -- studentized max
      instead of the plain max, which sat BELOW the best single branch by inflating
      the clean floor. MDD(fusion_alpha=0.0) restores the plain max.

Result difference (per-sequence AUROC, L5; baseline max -> improved alpha=0.5):
  corruption          baseline  improved
  hot_pixel             1.000     1.000
  event_rate_shift      0.952     0.956
  temporal_jitter       0.947     0.952
  event_flood           1.000     1.000
  polarity_flip         0.609     0.607  (0.620 at alpha=0 with radius')
  spatial_dropout       0.560     0.613  (up to 0.689 at alpha=1.25)
  --- worst-corruption floor  0.560 -> 0.607 ; mean 0.845 -> 0.855 ---
The two residuals stay below the 0.8 "solved" bar: spatial_dropout (contraction) is
capped ~0.69 and polarity_flip (model is polarity-symmetric by design) ~0.66 -- the
phi information ceiling; alpha trades polarity vs dropout (one static combiner can't
maximize both).

See Docs/novel.md for the full derivation and the validated numbers.
"""
import sys
from pathlib import Path
import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from analysis.vmem_utils import LAYER_SPECS, _subsample, _cap_subset, chunked_apply, device_for, maha_d2
from analysis.gpu_fit import ledoit_wolf_precision

DEVICE = device_for("MDD (manifold-decomposition detector)")


def l4_columns(phi_dim):
    """Column indices of the deepest available PLIF layer's [mu,var,kurt] block."""
    valid = [s for s in LAYER_SPECS if s["phi_end"] <= phi_dim]
    spec = valid[-1] if valid else LAYER_SPECS[0]
    end = min(spec["phi_end"], phi_dim)
    return list(range(spec["phi_start"], end))


# ---- spatial-organization compact block --------------------------------------
# monitor._compact_spatial_stats appends 20 dims per PLIF layer to phi_spatial:
# [cohflat_mu_w25, cohflat_mu_w12, cohflat_var_w25, cohflat_var_w12,
#  cohq_mu(3), cohq_var(3), cohfrac_mu(3), cohfrac_var(3), persq(2), persfrac(2)]
# Older phi_spatial files (2*sum_C dims, no block) still work — the org score is
# simply disabled for them.
SP_BASE_DIM = 2 * sum(s["C"] for s in LAYER_SPECS)  # [spatial_var|spatial_pr] blocks
SP_COMPACT_PER_LAYER = 20
# a-priori anomaly direction of each flatness dim (never chosen from labels):
# low windowed relative-std / low coherence quantiles = "too flat" (sign -1),
# high flat-pixel fraction = "too flat" as well but the stat GROWS (sign +1).
_ORG_FLAT_SIGNS = np.array([-1] * 4 + [-1] * 6 + [1] * 6, np.float32)
_ORG_PERSFRAC = slice(18, 20)


def _ledoit_wolf(fit_arr, n_fit=5000):
    # `n_fit` is IGNORED outside --fast: `_subsample` is a no-op there, so the
    # covariance fits on the FULL set (more samples = more faithful precision).
    # The param survives only to cap the fit for quick --fast smoke tests.
    f = _subsample(fit_arr, n_fit)
    try:
        cov = ledoit_wolf_precision(f, op="MDD Ledoit-Wolf fit")
        return cov.location_.astype(np.float32), cov.precision_.astype(np.float32)
    except Exception:
        return f.mean(0).astype(np.float32), np.eye(f.shape[1], dtype=np.float32)


def _maha_d2(x, mu, P):
    return maha_d2(x, mu, P, DEVICE)


class MDD:
    def __init__(self, k_pca=64, k_nn=64, n_ref=15000, use_spatial=True,
                 use_emp_radius=True, fusion_alpha=0.5, maha_pp=False,
                 use_rptm=False, rptm_m=512, rptm_q=0.05, rptm_seed=0,
                 use_ssb=False, ssb_cap=150000, ssb_l2=1.0, ssb_fold=False,
                 maha_two_sided=True, use_org=True):
        self.k_pca = k_pca
        self.k_nn = k_nn
        # n_ref is IGNORED outside --fast: the RCF reference uses the FULL clean
        # set (see fit(), where `_subsample` is a no-op). It's projected to k_pca
        # dims (~61 MB) and the _rcf similarity matrix is chunked, so the full
        # reference is a bounded speed cost, not an OOM risk. Kept only to cap
        # the reference for --fast smoke tests.
        self.n_ref = n_ref
        self.use_spatial = use_spatial
        # (A) fold an empirical-covariance PCA Mahalanobis into the radius branch;
        # (B) studentized fusion (max - fusion_alpha*median). See module docstring
        # / Docs/mdd_improvements.md. fusion_alpha=0 -> plain max (legacy).
        self.use_emp_radius = use_emp_radius
        self.fusion_alpha = float(fusion_alpha)
        # Mahalanobis++ (Müller et al., ICML 2025): L2-normalise features onto the
        # unit sphere before the covariance-SHAPE Mahalanobis sub-branches
        # (emp-cov radius, L4, spatial), tightening the shared-covariance Gaussian
        # fit by removing feature-norm variation. Off by default so the validated
        # numbers in the docstring are unchanged; opt-in A/B knob. NOT applied to
        # the isotropic radius energy or RCF — those ARE norm detectors and
        # normalising their input would erase their signal.
        self.maha_pp = bool(maha_pp)
        # RPTM (Random-Projection copula Tail-Mass): a distribution-free, opt-in
        # alternative to the empirical-covariance Mahalanobis term of the radius
        # branch. Whitens the PCA-k manifold, projects it onto `rptm_m` random
        # directions, and scores the two-sided rank TAIL-MASS (fraction of a frame's
        # projections landing outside the clean [q,1-q] band). A polarity_flip
        # ROTATES the clean covariance; whitening turns that rotation into
        # per-direction rank deviations that tail-mass reads off with NO covariance
        # inversion and NO kNN. When on it SUPERSEDES use_emp_radius as the radius
        # branch's second term, reusing the branch's existing SVD basis.
        #
        # Verdict (window-64 head-to-head, all severities; Docs/mdd_rptm.md): the
        # RADIUS BRANCH gets strictly stronger and covariance-free (L5 jitter
        # 0.78->0.86, rate_shift 0.89->0.91) and the polarity_flip residual improves
        # (fused L5 0.579->0.591); at the FUSED level it is a WASH (+0.0007 mean,
        # 6 win / 4 tie / 8 loss) because rcf/l4/spatial already overlap. It is
        # cheaper than RCF but NOT cheaper than emp-cov, and it cannot replace RCF
        # (RCF still owns spatial_dropout). Default OFF so shipped/paper numbers are
        # unchanged; flip on for the covariance-free radius + the polarity gain.
        self.use_rptm = bool(use_rptm)
        self.rptm_m = int(rptm_m)
        self.rptm_q = float(rptm_q)
        self.rptm_seed = int(rptm_seed)
        # SSB (Synthetic-corruption Self-supervised Branch), opt-in, default OFF.
        # A linear logistic head separating clean fit frames from `phi_synth` —
        # phi of a KNOWN corruption simulator applied to fit-side sequences
        # (CutPaste/DRAEM-style self-supervision: no real test corruption or label
        # is ever seen; the simulator is the repo's own event_corruption library).
        # Motivation (2026-07-10 oracle probe): spatial_dropout IS linearly
        # decodable from phi on held-out scenes (LR 0.87 / MLP 0.91) but its
        # signature lies ALONG the scene-activity nuisance axis (clean-vs-clean
        # null AUROC 0.755, direction dominated by L4:mu), so clean-only distance
        # scores cannot see it — a paired same-scene contrast cancels the nuisance
        # axis by construction. Inference cost is one 2112-D dot product per frame
        # (~2k MACs, vs ~1M for the RCF kNN): latency-negligible. Requires passing
        # `phi_synth` to fit(); scored one-sided (logit, higher = corruption-like)
        # and calibrated on held-out clean like every other branch.
        self.use_ssb = bool(use_ssb)
        self.ssb_cap = int(ssb_cap)
        self.ssb_l2 = float(ssb_l2)
        # ssb_fold: fold the calibrated ssb score INTO the spatial branch via max
        # instead of exposing a 5th branch. Keeps the branch count at 4 so the
        # studentized median is unchanged; the 2026-07-10 sweep found this equal
        # or slightly better than the 5-branch fusion at every alpha (per-seq L5
        # dropout 0.854 vs 0.846 at alpha=1.0). With ssb on, alpha=1.0 is the
        # sweep's Pareto point (alpha still trades polarity vs dropout).
        self.ssb_fold = bool(ssb_fold)
        # maha_two_sided (validated 2026-08-19, 60-seq probe + full-phi check):
        # score the l4 and spatial Mahalanobis branches as |d2 - median_cal| /
        # MAD_cal instead of raw d2. A raw d2 is one-sided — it only fires when
        # frames sit FARTHER from the clean mean than clean does; a distribution
        # that CONTRACTS toward the mode ("more normal than normal") pushes d2
        # BELOW the clean band and raw d2 inverts. Two-sided catches both tails:
        # on the existing full-phi data it lifts event_rate_shift per-seq from
        # 0.502 to 0.820 (l4) / 0.540 to 0.886 (spatial) with no loss on
        # temporal_jitter, event_flood, or hot_pixel. False -> legacy raw d2.
        self.maha_two_sided = bool(maha_two_sided)
        # use_org (validated 2026-08-19): score the spatial-organization compact
        # block (appended to phi_spatial by monitor._compact_spatial_stats) and
        # fold it into the spatial branch via max (ssb_fold pattern — branch
        # count stays 4). Components: per-layer persistence-fraction AND (min
        # over its two thresholds) + one signed-z flatness max; each component
        # is midrank-ECDF-calibrated against the clean fit set so weak-magnitude
        # / strong-rank components survive the max (scale-free), then the max of
        # the component p-values is the raw score. All clean-only; anomaly
        # directions are fixed a priori by the natural-data priors the block
        # encodes (local-texture floor, motion non-stationarity), never chosen
        # from corruption labels. Auto-disabled on phi_spatial files without
        # the compact block.
        self.use_org = bool(use_org)
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, phi_fit, phi_calib, phi_spatial_fit=None, phi_spatial_calib=None,
            phi_synth=None):
        """Fit on clean `phi_fit`; calibrate branch scales on disjoint clean
        `phi_calib`. Both must be held out of the eval negatives.

        `phi_synth` (optional, used only when use_ssb=True): phi rows of a
        corruption SIMULATOR applied to fit/calib-side sequences — never eval
        scenes, never real test corruptions. Enables the SSB linear head."""
        phi_fit = np.asarray(phi_fit, dtype=np.float32)
        D = phi_fit.shape[1]

        # standardize per-feature
        self.mu_f = phi_fit.mean(0).astype(np.float32)
        self.sd_f = (phi_fit.std(0) + 1e-9).astype(np.float32)
        std_fit = (phi_fit - self.mu_f) / self.sd_f

        # PCA (denoise) via torch SVD on a CAPPED subsample. SVD is a single
        # non-tileable GPU allocation (~O(n*D) for U plus cuSOLVER workspace),
        # so it cannot be chunked — and a 64-component basis saturates well
        # before 20k samples, so more data buys nothing. _cap_subset (not the
        # now-no-op _subsample) keeps this bounded and OOM-safe at full scale.
        k = max(1, min(self.k_pca, D, len(std_fit) - 1))
        sub = torch.from_numpy(_cap_subset(std_fit, 20000)).float().to(DEVICE)
        self.pca_mean = sub.mean(0).cpu().numpy().astype(np.float32)
        _, S, Vh = torch.linalg.svd(sub - sub.mean(0), full_matrices=False)
        self.pca_comps = Vh[:k].cpu().numpy().astype(np.float32)   # (k, D)
        # per-component std (whitening scale for RPTM); free from the same SVD.
        self.pca_sv = ((S[:k] / np.sqrt(len(sub))).clamp_min(1e-6)
                       .cpu().numpy().astype(np.float32))

        Zfit = self._project(phi_fit)
        rfit = np.linalg.norm(Zfit, axis=1) + 1e-9
        self.rg_mu = float(rfit.mean())
        self.rg_sd = float(rfit.std() + 1e-9)

        # (A) empirical (UNSHRUNK) covariance Mahalanobis in PCA space, folded into
        # the radius branch (radius' = max(isotropic radius, emp_maha)). It catches
        # the covariance ROTATION polarity_flip induces, which Ledoit-Wolf shrinkage
        # washes out. Pre-standardized on the calib split so the max() in
        # _branches_raw combines two comparable scales before the final calibration.
        if self.use_emp_radius and not self.use_rptm:
            Zemp = self._mpp(Zfit)
            self.emp_mu = Zemp.mean(0).astype(np.float32)
            self.emp_P = np.linalg.pinv(np.cov(Zemp.T)).astype(np.float32)
            emp_cal = _maha_d2(self._mpp(self._project(phi_calib)), self.emp_mu, self.emp_P)
            self.emp_cal_m = float(np.mean(emp_cal))
            self.emp_cal_s = float(np.std(emp_cal) + 1e-9)

        # RPTM: distribution-free replacement for the emp-cov radius term. Draw
        # rptm_m random directions in whitened PCA-k space, sort the clean
        # projections per direction (the whole "model"), and pre-standardize the
        # two-sided tail-mass on the calib split so max() in _branches_raw combines
        # comparable scales. Supersedes emp-cov when use_rptm is on.
        if self.use_rptm:
            rng = np.random.default_rng(self.rptm_seed)
            m = min(self.rptm_m, max(1, k))
            W = rng.standard_normal((k, self.rptm_m)).astype(np.float32)
            W /= (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
            self.rptm_W = W
            Pfit = (Zfit / self.pca_sv) @ W               # (n, rptm_m)
            self.rptm_ref = np.sort(Pfit, axis=0).astype(np.float32)   # sorted per dir
            rp_cal = self._rptm(self._project(phi_calib))
            self.rptm_cal_m = float(np.mean(rp_cal))
            self.rptm_cal_s = float(np.std(rp_cal) + 1e-9)

        # RCF reference (directions + radii on the FULL clean set; _subsample is
        # a no-op outside --fast, so self.n_ref does not cap the reference here).
        ref = _subsample(phi_fit, self.n_ref)
        Zref = self._project(ref)
        rref = np.linalg.norm(Zref, axis=1) + 1e-9
        self.ref_dir = (Zref / rref[:, None]).astype(np.float32)
        self.ref_r = rref.astype(np.float32)

        # SSB: linear logistic head, clean fit frames vs synthetic-corruption
        # frames, in the SAME per-feature standardization as the other branches.
        # Cap both classes so the (convex) fit stays bounded at full scale.
        self.has_ssb = bool(self.use_ssb and phi_synth is not None)
        if self.has_ssb:
            from analysis.gpu_fit import logreg_fit
            syn = (np.asarray(phi_synth, np.float32) - self.mu_f) / self.sd_f
            neg = _cap_subset(std_fit, self.ssb_cap)
            pos = _cap_subset(syn, self.ssb_cap)
            Xs = np.concatenate([neg, pos], axis=0)
            ys = np.concatenate([np.zeros(len(neg), np.float32),
                                 np.ones(len(pos), np.float32)])
            clf = logreg_fit(Xs, ys, op="MDD SSB logistic head", l2=self.ssb_l2)
            self.ssb_w = clf.coef_[0].astype(np.float32)
            self.ssb_b = float(clf.intercept_[0])

        # L4 deep-layer Mahalanobis
        self.l4_cols = l4_columns(D)
        self.l4_mu, self.l4_P = _ledoit_wolf(self._mpp(phi_fit[:, self.l4_cols]))
        if self.maha_two_sided:
            # clean-calib center/scale of d2 so the branch can score BOTH tails
            l4_cal = _maha_d2(np.asarray(phi_calib, np.float32)[:, self.l4_cols],
                              self.l4_mu, self.l4_P)
            self.l4_med = float(np.median(l4_cal))
            self.l4_mad = float(np.median(np.abs(l4_cal - self.l4_med)) + 1e-9)

        # optional spatial branch (Mahalanobis on the [spatial_var|spatial_pr]
        # base; the org compact block, when present, gets its own fold score)
        self.has_spatial = bool(self.use_spatial and phi_spatial_fit is not None)
        if self.has_spatial:
            ps = self._sanitize_spatial(phi_spatial_fit)
            n_l = len(LAYER_SPECS)
            self.sp_base = (SP_BASE_DIM
                            if ps.shape[1] == SP_BASE_DIM + SP_COMPACT_PER_LAYER * n_l
                            else ps.shape[1])
            self.has_org = bool(self.use_org and self.sp_base < ps.shape[1])
            base = ps[:, :self.sp_base]
            self.sp_mu_f = base.mean(0).astype(np.float32)
            self.sp_sd_f = (base.std(0) + 1e-9).astype(np.float32)
            self.sp_mu, self.sp_P = _ledoit_wolf(self._mpp((base - self.sp_mu_f) / self.sp_sd_f))
            if self.maha_two_sided:
                sp_cal = _maha_d2(self._mpp(self._std_spatial(phi_spatial_calib)),
                                  self.sp_mu, self.sp_P)
                self.sp_med = float(np.median(sp_cal))
                self.sp_mad = float(np.median(np.abs(sp_cal - self.sp_med)) + 1e-9)
            if self.has_org:
                cb = ps[:, self.sp_base:]
                L = cb.shape[1] // SP_COMPACT_PER_LAYER
                flat_fit = cb.reshape(len(cb), L, SP_COMPACT_PER_LAYER)[:, :, :16]
                self.org_fmu = flat_fit.mean(0).astype(np.float32)   # (L, 16)
                self.org_fsd = (flat_fit.std(0) + 1e-9).astype(np.float32)
                # sorted clean reference per component for the midrank ECDF
                # (capped: the rank calibration saturates long before 100k rows)
                comp = self._org_components(cb)
                self.org_ref = np.sort(_cap_subset(comp, 100000), axis=0)

        # calibrate each branch's scale on held-out clean
        self._fitted = True
        B = self._branches_raw(phi_calib, phi_spatial_calib)
        self.branch_names = list(B.keys())
        self.cal_mean = {k: float(np.mean(v)) for k, v in B.items()}
        self.cal_std = {k: float(np.std(v) + 1e-9) for k, v in B.items()}
        return self

    # -------------------------------------------------------------- internals
    def _mpp(self, x):
        """Mahalanobis++ unit-sphere projection (no-op unless maha_pp is on).

        Applied to the inputs of the covariance-shape Mahalanobis sub-branches so
        fit and score normalise identically. eps guards a true zero-norm row."""
        if not getattr(self, "maha_pp", False):
            return x
        x = np.asarray(x, np.float32)
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

    def _project(self, x):
        mean_t = torch.from_numpy(self.pca_mean).to(DEVICE)
        comps_t = torch.from_numpy(self.pca_comps).to(DEVICE)
        mu_f_t = torch.from_numpy(self.mu_f).to(DEVICE)
        sd_f_t = torch.from_numpy(self.sd_f).to(DEVICE)
        def fn(c):
            cs = (c - mu_f_t) / sd_f_t
            return (cs - mean_t) @ comps_t.T
        return chunked_apply(fn, np.ascontiguousarray(x, np.float32),
                             DEVICE, n_ref=comps_t.shape[1])

    def _rcf(self, Z):
        r = np.linalg.norm(Z, axis=1) + 1e-9
        u = (Z / r[:, None]).astype(np.float32)
        ref_dir = torch.from_numpy(self.ref_dir).to(DEVICE)
        ref_r = torch.from_numpy(self.ref_r).to(DEVICE)
        k = max(1, min(self.k_nn, ref_dir.shape[0]))
        def fn(c):
            sim = c @ ref_dir.T
            _, idx = torch.topk(sim, k, dim=1)
            nbr = ref_r[idx]
            # unbiased=False: this is the spread of the k retrieved radii, not a
            # sample-variance estimate, and unbiased std is NaN when k resolves
            # to 1 (tiny reference set). Keeps the score finite at the edge.
            return torch.stack([nbr.mean(1), nbr.std(1, unbiased=False) + 1e-6], dim=1)
        out = chunked_apply(fn, u, DEVICE, n_ref=ref_dir.shape[0])  # (N, 2)
        return np.abs(r - out[:, 0]) / out[:, 1]

    def _rptm(self, Z):
        """Two-sided rank tail-mass over the whitened random projections.

        `Z` is the PCA projection (N, k). Whitens it, projects onto the fixed
        random directions, ranks each projection against its sorted clean column,
        and returns the fraction of a frame's projections landing in the clean
        tails (u < q or u > 1-q). Distribution-free; no covariance, no kNN."""
        q = self.rptm_q
        ref_t = torch.from_numpy(self.rptm_ref).to(DEVICE).T.contiguous()  # (m, Nref)
        Nref = ref_t.shape[1]
        W = torch.from_numpy(self.rptm_W).to(DEVICE)                       # (k, m)
        sv = torch.from_numpy(self.pca_sv).to(DEVICE)                      # (k,)
        def fn(zc):
            P = (zc / sv) @ W                                             # (chunk, m)
            u = torch.searchsorted(ref_t, P.T.contiguous(), right=True).float() / (Nref + 1.0)
            return ((u > 1 - q) | (u < q)).float().mean(0)               # (chunk,)
        return chunked_apply(fn, np.ascontiguousarray(Z, np.float32), DEVICE, n_ref=Nref)

    def _branches_raw(self, phi, phi_spatial=None):
        phi = np.asarray(phi, dtype=np.float32)
        Z = self._project(phi)
        r = np.linalg.norm(Z, axis=1)
        radius = np.abs(r - self.rg_mu) / self.rg_sd
        if getattr(self, "use_rptm", False):
            # OR the isotropic energy with the distribution-free RPTM tail-mass so
            # the radius branch fires on covariance rotations (polarity_flip) with
            # no covariance inversion — supersedes the emp-cov term.
            rp = (self._rptm(Z) - self.rptm_cal_m) / self.rptm_cal_s
            radius = np.maximum(radius, rp)
        elif getattr(self, "use_emp_radius", False):
            # OR the isotropic energy with the (pre-standardized) empirical-cov
            # Mahalanobis so the radius branch also fires on covariance rotations.
            emp = (_maha_d2(self._mpp(Z), self.emp_mu, self.emp_P) - self.emp_cal_m) / self.emp_cal_s
            radius = np.maximum(radius, emp)
        l4 = _maha_d2(phi[:, self.l4_cols], self.l4_mu, self.l4_P)
        if getattr(self, "maha_two_sided", False):
            l4 = np.abs(l4 - self.l4_med) / self.l4_mad
        out = {
            "radius": radius,
            "rcf": self._rcf(Z),
            "l4": l4,
        }
        if getattr(self, "has_spatial", False) and phi_spatial is not None:
            sp = _maha_d2(self._mpp(self._std_spatial(phi_spatial)), self.sp_mu, self.sp_P)
            if getattr(self, "maha_two_sided", False):
                sp = np.abs(sp - self.sp_med) / self.sp_mad
            out["spatial"] = sp
            if getattr(self, "has_org", False):
                ps = self._sanitize_spatial(phi_spatial)
                out["org"] = self._org_score(ps[:, self.sp_base:])
        if getattr(self, "has_ssb", False):
            # one 2112-D dot product per frame; one-sided logit (higher = more
            # like the simulated corruption), calibrated like every branch.
            out["ssb"] = ((phi - self.mu_f) / self.sd_f) @ self.ssb_w + self.ssb_b
        return out

    @staticmethod
    def _sanitize_spatial(ps):
        # Defense-in-depth: keep Mahalanobis finite if phi_spatial ever carries a
        # non-finite value (so one bad frame can't NaN-out the whole eval stage).
        # phi_spatial is float32 now, so this is normally a no-op.
        return np.nan_to_num(np.asarray(ps, np.float32), posinf=3.0e38, neginf=-3.0e38)

    def _std_spatial(self, ps):
        return ((self._sanitize_spatial(ps)[:, :self.sp_base] - self.sp_mu_f)
                / self.sp_sd_f)

    def _org_components(self, compact):
        """(N, 20*L) compact block -> (N, L+1) raw components.

        Per layer: the persistence-fraction AND (min over its two thresholds —
        both must be elevated, which suppresses single-threshold clean noise).
        Plus one flatness component: signed z (fit-standardized, a-priori
        directions) maxed over every flatness dim of every layer."""
        compact = np.asarray(compact, np.float32)
        N = len(compact)
        L = compact.shape[1] // SP_COMPACT_PER_LAYER
        blocks = compact.reshape(N, L, SP_COMPACT_PER_LAYER)
        pf = blocks[:, :, _ORG_PERSFRAC].min(axis=2)                    # (N, L)
        z = _ORG_FLAT_SIGNS * (blocks[:, :, :16] - self.org_fmu) / self.org_fsd
        return np.concatenate([pf, z.reshape(N, -1).max(1, keepdims=True)], axis=1)

    def _org_score(self, compact):
        """Midrank-ECDF max over the org components. Rank calibration is
        scale-free: a component whose clean spread is tiny but whose ranking
        separates cleanly contributes on equal footing with the rest — a
        z-scaled max would bury it under the widest component's clean tail."""
        C = self._org_components(compact)
        Nf = len(self.org_ref)
        p = np.empty_like(C)
        for j in range(C.shape[1]):
            lo = np.searchsorted(self.org_ref[:, j], C[:, j], side="left")
            hi = np.searchsorted(self.org_ref[:, j], C[:, j], side="right")
            p[:, j] = (lo + hi) / (2.0 * Nf)
        return p.max(axis=1)

    # ---------------------------------------------------------------- scoring
    def score_branches(self, phi, phi_spatial=None):
        """Per-branch CALIBRATED scores plus the fused score. Returns a dict.

        Fusion is the STUDENTIZED max (max - fusion_alpha*median): subtracting a
        robust noise-floor estimate suppresses the clean-floor inflation that made
        the plain max sit below the best single branch. fusion_alpha=0 -> plain max.
        """
        if not self._fitted:
            raise RuntimeError("MDD.score_branches called before fit().")
        B = self._branches_raw(phi, phi_spatial)
        cal = {k: (v - self.cal_mean[k]) / self.cal_std[k] for k, v in B.items()}
        if (getattr(self, "has_ssb", False) and getattr(self, "ssb_fold", False)
                and "ssb" in cal and "spatial" in cal):
            cal["spatial"] = np.maximum(cal["spatial"], cal.pop("ssb"))
        # fold the (calibrated) org score into the spatial branch — same
        # pattern as ssb_fold, keeps the branch count (and thus the
        # studentized median) unchanged.
        if "org" in cal and "spatial" in cal:
            cal["spatial"] = np.maximum(cal["spatial"], cal.pop("org"))
        stack = np.stack(list(cal.values()), axis=1)
        mx = np.max(stack, axis=1)
        a = getattr(self, "fusion_alpha", 0.0)
        cal["fused"] = mx - a * np.median(stack, axis=1) if a else mx
        return cal

    def score(self, phi, phi_spatial=None):
        """Single fused OOD score per frame (higher = more OOD)."""
        return self.score_branches(phi, phi_spatial)["fused"]
