import sys
import numpy as np
import torch
from pathlib import Path

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from vmem_benchmark import benchmark_config as cfg
from sklearn.metrics import roc_auc_score, roc_curve

LAYER_SPECS = [
    {"name": "SNN Block 1", "idx": 0, "C": 64},
    {"name": "SNN Block 2", "idx": 1, "C": 128},
    {"name": "SNN Block 3", "idx": 2, "C": 256},
    {"name": "SNN Block 4", "idx": 3, "C": 256},
]

# Compute phi slice boundaries
_off = 0
for _s in LAYER_SPECS:
    _s["phi_start"] = _off
    _s["mu_end"]    = _off + _s["C"]
    _s["var_end"]   = _off + 2 * _s["C"]
    _s["phi_end"]   = _off + 3 * _s["C"]
    _off = _s["phi_end"]

TOTAL_PHI_DIM   = _off
MAX_FIT_SAMPLES = 3000
PLIF_THETA      = 1.0
TRAIN_RATIO     = getattr(cfg, "CLEAN_TRAIN_RATIO", 0.7)
TABLE_DIR       = cfg.OUTPUT_DIR / "tables"


# ─────────────────────────────────────────────────────────────────────────────
# Shared clean train/eval split.
#
# Frames within a sequence are temporally correlated, so random frame-level
# splits leak near-duplicate frames between train and eval. Every script must
# use these helpers: the split is a CONTIGUOUS cut, aligned to a sequence
# boundary whenever per-sequence frame counts ('seq_lens' saved by extract.py)
# are available. The cut is deterministic, so all detectors and analyses fit
# and evaluate on exactly the same frames.
# ─────────────────────────────────────────────────────────────────────────────

def split_boundary(n: int, train_ratio: float = TRAIN_RATIO, seq_lens=None) -> int:
    """Row index where train ends and eval begins."""
    cut = int(n * train_ratio)
    if seq_lens:
        edges = np.cumsum(seq_lens)
        # Only trust seq_lens that actually match the array, and only align
        # to INTERIOR sequence boundaries: the last edge equals n, and a cut
        # there would leave eval empty (with a single sequence it degenerated
        # to a 1-frame eval set). With no interior boundary, fall back to the
        # plain ratio cut within the sequence.
        if edges[-1] == n and len(edges) > 1:
            interior = edges[:-1]
            cut = int(interior[np.argmin(np.abs(interior - cut))])
    if n > 1:
        cut = max(1, min(cut, n - 1))  # never let train or eval be empty
    else:
        cut = n  # degenerate input: everything goes to train, eval is empty
    return cut


def split_train_eval(arr, train_ratio: float = TRAIN_RATIO, seq_lens=None):
    """Split rows of `arr` into (train, eval) on whole sequences."""
    cut = split_boundary(len(arr), train_ratio, seq_lens)
    return arr[:cut], arr[cut:]


def load_phi_seq_lens(run_name: str = "clean", artifact_dir=None):
    """Per-sequence frame counts saved by extract.py, or None for legacy files.

    `artifact_dir` defaults to the phi directory but any per-run artifact
    directory (temporal_phi, spike, ...) works — they all share the metadata.
    """
    f = (artifact_dir or cfg.PHI_DIR) / f"{run_name}.pt"
    if not f.exists():
        return None
    try:
        d = torch.load(f, map_location="cpu", weights_only=True)
        return d.get("seq_lens", None)
    except Exception:
        return None


def load_phi_spatial(run_name: str = "clean"):
    """Spatial-dispersion features (`phi_spatial`) saved alongside phi, or None.

    Present only for runs extracted after collect_phi_spatial() was added;
    legacy / pre-spatial phi files return None so callers can degrade
    gracefully to the GAP'd-phi-only MDD branches.
    """
    f = cfg.PHI_DIR / f"{run_name}.pt"
    if not f.exists():
        return None
    try:
        d = torch.load(f, map_location="cpu", weights_only=True)
        ps = d.get("phi_spatial", None)
        return ps.float().numpy() if ps is not None else None
    except Exception:
        return None


def seq_lens_after_cut(seq_lens, cut):
    """Per-sequence frame counts for the rows [cut:] of a run.

    Used to aggregate held-out clean scores (rows after the train/eval cut)
    by recording. A cut that lands inside a sequence keeps that sequence's
    tail as the first (partial) block. Returns None when seq_lens is absent.
    """
    if not seq_lens:
        return None
    out = []
    start = 0
    for L in seq_lens:
        end = start + L
        if end <= cut:
            start = end
            continue
        out.append(end - max(start, cut))
        start = end
    return out or None


def aggregate_by_seq(scores, seq_lens):
    """Mean of per-frame `scores` within each sequence (per-recording pooling).

    Returns a 1-D array with one entry per sequence, or None when seq_lens is
    missing or does not sum to len(scores) (so callers fall back to per-frame).
    """
    scores = np.asarray(scores)
    if not seq_lens or int(np.sum(seq_lens)) != len(scores):
        return None
    out, start = [], 0
    for L in seq_lens:
        out.append(float(scores[start:start + L].mean()))
        start += L
    return np.asarray(out)


class LazyPhiDict:
    """Lazy loader for per-run phi arrays with a small LRU cache.

    Caching every run would hold the entire benchmark (~tens of GB) in RAM,
    so only the most recently used runs are kept; 'clean' is accessed
    constantly and therefore stays resident in practice.
    """

    def __init__(self, cache_size: int = 3):
        from collections import OrderedDict
        self._cache = OrderedDict()
        self._cache_size = max(1, cache_size)
        self._available = {f.stem: f for f in sorted(cfg.PHI_DIR.glob("*.pt"))}
        self._seq_lens = {}
        self.fast_mode = "--fast" in sys.argv

    def __contains__(self, key):
        return key in self._available

    def __len__(self):
        return len(self._available)

    def __getitem__(self, key):
        if key not in self._available:
            raise KeyError(key)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        f = self._available[key]
        try:
            d = torch.load(f, map_location="cpu", weights_only=True)
            arr = d["phi"].float().numpy()
            self._seq_lens[key] = d.get("seq_lens", None)
            if self.fast_mode:
                rng = np.random.default_rng(42)
                n = min(len(arr), 2000)
                # Sorted subsample keeps rows in temporal order; sequence
                # boundaries no longer apply, so drop seq_lens in fast mode.
                arr = arr[np.sort(rng.choice(len(arr), n, replace=False))]
                self._seq_lens[key] = None
            self._cache[key] = arr
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return arr
        except Exception as e:
            print(f"[!] Could not load {f.name}: {e}")
            raise e

    def get_seq_lens(self, key):
        """Per-sequence frame counts for a loaded run (None if unavailable)."""
        if key not in self._seq_lens and key in self:
            self[key]  # trigger load
        return self._seq_lens.get(key, None)

    def get_phi_spatial(self, key):
        """Spatial-dispersion features for a run, or None (legacy / fast mode).

        Loaded on demand from the same .pt file as phi. Not cached in the LRU
        (the MDD reads it once per run); in --fast mode the phi rows are a
        subsample so the full-length phi_spatial no longer aligns and is
        dropped.
        """
        if key not in self._available or self.fast_mode:
            return None
        try:
            d = torch.load(self._available[key], map_location="cpu", weights_only=True)
            ps = d.get("phi_spatial", None)
            return ps.float().numpy() if ps is not None else None
        except Exception:
            return None

    def keys(self):
        return self._available.keys()

    def __iter__(self):
        return iter(self._available)

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

def load_all_temporal_phi():
    out = {}
    tdir = cfg.OUTPUT_DIR / "temporal_phi"
    if not tdir.exists():
        return out
    for f in sorted(tdir.glob("*.pt")):
        try:
            d = torch.load(f, map_location="cpu", weights_only=True)
            out[d["run"]] = d["temporal_phi"].float().numpy()
        except Exception as e:
            print(f"[!] Could not load temporal phi {f.name}: {e}")
    return out

def load_traj_as_temporal_phi(run_name: str, theta: float = PLIF_THETA) -> np.ndarray:
    p = cfg.TRAJ_DIR / f"{run_name}.pt"
    if not p.exists():
        return None
    try:
        data = torch.load(p, map_location="cpu", weights_only=True)
        trajs = data["trajs"]
        parts = []
        for idx in sorted(trajs.keys()):
            V = trajs[idx]  # (T, B, D)
            T, B, D = V.shape
            if T < 2:
                continue
            V_scalar = V.float().mean(-1)  # (T, B)
            margin = theta - V_scalar
            m_mean = margin.mean(0)
            m_min  = margin.min(0).values
            m_var  = margin.var(0)

            dV      = V_scalar[1:] - V_scalar[:-1]
            dV_mean = dV.abs().mean(0)
            dV_var  = dV.var(0)

            std  = V_scalar.std(0).clamp(min=1e-8)
            Vc   = V_scalar - V_scalar.mean(0, keepdim=True)
            autocorr = (Vc[:-1] * Vc[1:]).mean(0) / std ** 2

            fft_mag  = torch.fft.rfft(V_scalar, dim=0).abs() ** 2
            total_e  = fft_mag.sum(0).clamp(min=1e-8)
            hf_e     = fft_mag[max(1, T // 4):].sum(0)
            hf_ratio = hf_e / total_e

            layer_feat = torch.stack(
                [m_mean, m_min, m_var, dV_mean, dV_var, autocorr, hf_ratio], dim=1
            )  # (B, 7)
            parts.append(layer_feat)

        if not parts:
            return None
        return torch.cat(parts, dim=1).numpy()
    except Exception as e:
        print(f"  [!] Failed to compute temporal phi from trajectory {run_name}: {e}")
        return None



def _valid_layers(phi_dim):
    return [s for s in LAYER_SPECS if s["phi_start"] < phi_dim]

def slice_phi_layer(phi: np.ndarray, layer_idx: int) -> np.ndarray:
    spec = LAYER_SPECS[layer_idx]
    start = spec["phi_start"]
    end   = min(spec["phi_end"], phi.shape[1])
    if start >= phi.shape[1]:
        return phi[:, :0]
    return phi[:, start:end]

def slice_phi_stat(phi: np.ndarray, stat: str) -> np.ndarray:
    if stat not in ("mu", "var", "kurtosis"):
        raise ValueError(f"Unknown phi stat '{stat}'; expected mu/var/kurtosis")
    parts = []
    for spec in _valid_layers(phi.shape[1]):
        s, C = spec["phi_start"], spec["C"]
        layer_phi = phi[:, s: min(spec["phi_end"], phi.shape[1])]
        lc = layer_phi.shape[1]
        if stat == "mu":
            parts.append(layer_phi[:, :min(C, lc)])
        elif stat == "var":
            parts.append(layer_phi[:, min(C, lc): min(2 * C, lc)])
        elif stat == "kurtosis":
            parts.append(layer_phi[:, min(2 * C, lc): min(3 * C, lc)])
    return np.concatenate(parts, axis=1) if parts else phi[:, :0]

def auroc_fpr95(y_true: np.ndarray, y_score: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx   = int(np.argmax(tpr >= 0.95))
    fpr95 = float(fpr[idx])
    return float(auroc), fpr95

def _get_present(all_phi, corruptions=None, severities=None):
    corruptions = corruptions or cfg.CORRUPTIONS
    severities  = severities  or cfg.SEVERITIES
    return [c for c in corruptions
            if any(f"{c}_L{s}" in all_phi for s in severities)]

def _subsample(arr: np.ndarray, n: int = MAX_FIT_SAMPLES) -> np.ndarray:
    fast_mode = "--fast" in sys.argv
    if fast_mode:
        n = min(n, 500)
    if len(arr) <= n:
        return arr
    rng = np.random.default_rng(42)
    return arr[rng.choice(len(arr), n, replace=False)]
