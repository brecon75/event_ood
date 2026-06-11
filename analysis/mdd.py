"""Manifold-Decomposition Detector (MDD).

A single unsupervised OOD detector that scores each frame on orthogonal axes of
the clean static-phi manifold and fuses them with a calibrated max. Built from
the diagnosis that event-camera corruptions split into geometry classes:

  * DILATIONS (hot_pixel, event_rate_shift, temporal_jitter) push phi OUTWARD —
    caught by distance from the clean mean.
  * CONTRACTIONS (spatial_dropout, event_flood) pull phi toward the clean mode —
    they INVERT distance/density detectors ("more normal than normal"); detected
    only by noticing under-dispersion (a two-sided, direction-conditioned score).
  * INVISIBLE (polarity_flip) — no signature in V_mem; out of scope.

Branches (each standardized on held-out clean, so a plain max is a true OR):
  B1 radius  : | ||z|| - E||z|| | / sd          two-sided membrane energy
  B2 RCF     : | ||z|| - E[||z|| | dir] | / sd   two-sided, conditioned on
               direction via cosine-kNN -> the contraction detector
  B3 L4 d^2  : Ledoit-Wolf Mahalanobis on the layer-4 block (deep timing signal
               the pooled radius averages away; rescues temporal_jitter)
  B4 spatial : (optional, only when phi_spatial is available) Ledoit-Wolf
               Mahalanobis on the spatial-dispersion features GAP discards ->
               the per-frame detector for spatial_dropout / event_flood

See Docs/novel.md for the full derivation and the validated numbers.
"""
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.covariance import LedoitWolf

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from analysis.vmem_utils import LAYER_SPECS, _subsample

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def l4_columns(phi_dim):
    """Column indices of the deepest available PLIF layer's [mu,var,kurt] block."""
    valid = [s for s in LAYER_SPECS if s["phi_end"] <= phi_dim]
    spec = valid[-1] if valid else LAYER_SPECS[0]
    end = min(spec["phi_end"], phi_dim)
    return list(range(spec["phi_start"], end))


def _ledoit_wolf(fit_arr, n_fit=5000):
    f = _subsample(fit_arr, n_fit)
    try:
        cov = LedoitWolf().fit(f)
        return cov.location_.astype(np.float32), cov.precision_.astype(np.float32)
    except Exception:
        return f.mean(0).astype(np.float32), np.eye(f.shape[1], dtype=np.float32)


def _maha_d2(x, mu, P, chunk=50000):
    out = []
    mu_t = torch.from_numpy(mu).to(DEVICE)
    P_t = torch.from_numpy(P).to(DEVICE)
    for c in torch.split(torch.from_numpy(x).float(), chunk):
        d = c.to(DEVICE) - mu_t
        out.append(torch.einsum("ni,ij,nj->n", d, P_t, d).cpu())
    return torch.cat(out).numpy()


class MDD:
    def __init__(self, k_pca=64, k_nn=64, n_ref=15000, use_spatial=True):
        self.k_pca = k_pca
        self.k_nn = k_nn
        self.n_ref = n_ref
        self.use_spatial = use_spatial
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, phi_fit, phi_calib, phi_spatial_fit=None, phi_spatial_calib=None):
        """Fit on clean `phi_fit`; calibrate branch scales on disjoint clean
        `phi_calib`. Both must be held out of the eval negatives."""
        phi_fit = np.asarray(phi_fit, dtype=np.float32)
        D = phi_fit.shape[1]

        # standardize per-feature
        self.mu_f = phi_fit.mean(0).astype(np.float32)
        self.sd_f = (phi_fit.std(0) + 1e-9).astype(np.float32)
        std_fit = (phi_fit - self.mu_f) / self.sd_f

        # PCA (denoise) via torch SVD on a subsample
        k = max(1, min(self.k_pca, D, len(std_fit) - 1))
        sub = torch.from_numpy(_subsample(std_fit, 20000)).float().to(DEVICE)
        self.pca_mean = sub.mean(0).cpu().numpy().astype(np.float32)
        _, _, Vh = torch.linalg.svd(sub - sub.mean(0), full_matrices=False)
        self.pca_comps = Vh[:k].cpu().numpy().astype(np.float32)   # (k, D)

        Zfit = self._project(phi_fit)
        rfit = np.linalg.norm(Zfit, axis=1) + 1e-9
        self.rg_mu = float(rfit.mean())
        self.rg_sd = float(rfit.std() + 1e-9)

        # RCF reference (directions + radii on a clean subsample)
        ref = _subsample(phi_fit, self.n_ref)
        Zref = self._project(ref)
        rref = np.linalg.norm(Zref, axis=1) + 1e-9
        self.ref_dir = (Zref / rref[:, None]).astype(np.float32)
        self.ref_r = rref.astype(np.float32)

        # L4 deep-layer Mahalanobis
        self.l4_cols = l4_columns(D)
        self.l4_mu, self.l4_P = _ledoit_wolf(phi_fit[:, self.l4_cols])

        # optional spatial branch
        self.has_spatial = bool(self.use_spatial and phi_spatial_fit is not None)
        if self.has_spatial:
            ps = self._sanitize_spatial(phi_spatial_fit)
            self.sp_mu_f = ps.mean(0).astype(np.float32)
            self.sp_sd_f = (ps.std(0) + 1e-9).astype(np.float32)
            self.sp_mu, self.sp_P = _ledoit_wolf((ps - self.sp_mu_f) / self.sp_sd_f)

        # calibrate each branch's scale on held-out clean
        self._fitted = True
        B = self._branches_raw(phi_calib, phi_spatial_calib)
        self.branch_names = list(B.keys())
        self.cal_mean = {k: float(np.mean(v)) for k, v in B.items()}
        self.cal_std = {k: float(np.std(v) + 1e-9) for k, v in B.items()}
        return self

    # -------------------------------------------------------------- internals
    def _project(self, x, chunk=50000):
        mean_t = torch.from_numpy(self.pca_mean).to(DEVICE)
        comps_t = torch.from_numpy(self.pca_comps).to(DEVICE)
        out = []
        for c in torch.split(torch.from_numpy(np.asarray(x, np.float32)), chunk):
            cs = (c.to(DEVICE) - torch.from_numpy(self.mu_f).to(DEVICE)) \
                / torch.from_numpy(self.sd_f).to(DEVICE)
            out.append(((cs - mean_t) @ comps_t.T).cpu())
        return torch.cat(out).numpy()

    def _rcf(self, Z, chunk=4000):
        r = np.linalg.norm(Z, axis=1) + 1e-9
        u = torch.from_numpy((Z / r[:, None]).astype(np.float32)).to(DEVICE)
        ref_dir = torch.from_numpy(self.ref_dir).to(DEVICE)
        ref_r = torch.from_numpy(self.ref_r).to(DEVICE)
        k = max(1, min(self.k_nn, ref_dir.shape[0]))
        mus, sds = [], []
        for c in torch.split(u, chunk):
            sim = c @ ref_dir.T
            _, idx = torch.topk(sim, k, dim=1)
            nbr = ref_r[idx]
            mus.append(nbr.mean(1).cpu())
            # unbiased=False: this is the spread of the k retrieved radii, not a
            # sample-variance estimate, and unbiased std is NaN when k resolves
            # to 1 (tiny reference set). Keeps the score finite at the edge.
            sds.append((nbr.std(1, unbiased=False) + 1e-6).cpu())
        mu = torch.cat(mus).numpy()
        sd = torch.cat(sds).numpy()
        return np.abs(r - mu) / sd

    def _branches_raw(self, phi, phi_spatial=None):
        phi = np.asarray(phi, dtype=np.float32)
        Z = self._project(phi)
        r = np.linalg.norm(Z, axis=1)
        out = {
            "radius": np.abs(r - self.rg_mu) / self.rg_sd,
            "rcf": self._rcf(Z),
            "l4": _maha_d2(phi[:, self.l4_cols], self.l4_mu, self.l4_P),
        }
        if getattr(self, "has_spatial", False) and phi_spatial is not None:
            out["spatial"] = _maha_d2(self._std_spatial(phi_spatial), self.sp_mu, self.sp_P)
        return out

    @staticmethod
    def _sanitize_spatial(ps):
        # Defense-in-depth: keep Mahalanobis finite if phi_spatial ever carries a
        # non-finite value (so one bad frame can't NaN-out the whole eval stage).
        # phi_spatial is float32 now, so this is normally a no-op.
        return np.nan_to_num(np.asarray(ps, np.float32), posinf=3.0e38, neginf=-3.0e38)

    def _std_spatial(self, ps):
        return (self._sanitize_spatial(ps) - self.sp_mu_f) / self.sp_sd_f

    # ---------------------------------------------------------------- scoring
    def score_branches(self, phi, phi_spatial=None):
        """Per-branch CALIBRATED scores plus the fused max. Returns a dict."""
        if not self._fitted:
            raise RuntimeError("MDD.score_branches called before fit().")
        B = self._branches_raw(phi, phi_spatial)
        cal = {k: (v - self.cal_mean[k]) / self.cal_std[k] for k, v in B.items()}
        cal["fused"] = np.max(np.stack([cal[k] for k in B], axis=1), axis=1)
        return cal

    def score(self, phi, phi_spatial=None):
        """Single fused OOD score per frame (higher = more OOD)."""
        return self.score_branches(phi, phi_spatial)["fused"]
