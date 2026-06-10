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
TABLE_DIR       = cfg.OUTPUT_DIR / "tables"

class LazyPhiDict:
    def __init__(self):
        self._cache = {}
        self._available = {}
        self.fast_mode = "--fast" in sys.argv
        for f in sorted(cfg.PHI_DIR.glob("*.pt")):
            try:
                run_name = f.stem
                self._available[run_name] = f
            except Exception:
                pass

    def __contains__(self, key):
        return key in self._available

    def __len__(self):
        return len(self._available)

    def __getitem__(self, key):
        if key not in self._available:
            raise KeyError(key)
        if key in self._cache:
            return self._cache[key]
        
        f = self._available[key]
        try:
            d = torch.load(f, map_location="cpu", weights_only=True)
            arr = d["phi"].float().numpy()
            if self.fast_mode:
                rng = np.random.default_rng(42)
                n = min(len(arr), 2000)
                arr = arr[rng.choice(len(arr), n, replace=False)]
            self._cache[key] = arr
            return arr
        except Exception as e:
            print(f"[!] Could not load {f.name}: {e}")
            raise e

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
    return np.concatenate(parts, axis=1) if parts else phi

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
