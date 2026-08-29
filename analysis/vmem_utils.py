import sys
import numpy as np
import torch
from pathlib import Path

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from vmem_benchmark import benchmark_config as cfg
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

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
FAST_MODE       = "--fast" in sys.argv   # smoke-test mode, shared across all callers
TRAIN_RATIO     = getattr(cfg, "CLEAN_TRAIN_RATIO", 0.7)
TABLE_DIR       = cfg.OUTPUT_DIR / "tables"


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers — memory-mapped .pt loading.
#
# φ files store `phi` (~3 GB/run) AND `phi_spatial` (~2 GB/run) in one archive,
# but most stages use only `phi`. A plain torch.load deserialises the WHOLE file
# into RAM, so every φ-only stage paid to read ~2 GB/run of phi_spatial it never
# touches, and `load_phi_seq_lens` deserialised the entire ~5 GB clean.pt just to
# read a tiny list. `mmap=True` faults only the pages actually accessed: reading
# `phi` never touches phi_spatial's bytes, and reading `seq_lens` touches neither.
# This both cuts disk reads (~40%/run for φ-only stages) and lets the OS page
# cache serve repeated reads across stages. Falls back to an eager load for
# legacy-format files or platforms where mmap is unsupported.
# ─────────────────────────────────────────────────────────────────────────────

def load_pt(path, mmap: bool = True):
    """torch.load with per-tensor lazy mmap faulting and a graceful fallback."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=mmap)
    except (RuntimeError, ValueError, TypeError):
        return torch.load(path, map_location="cpu", weights_only=True)


def atomic_save(obj, path):
    """Save via a temp file + rename so a crash mid-save never leaves a
    partial .pt that the exists()-based resume check would skip forever."""
    tmp = path.with_name(f"_tmp_{path.name}")
    torch.save(obj, tmp)
    tmp.replace(path)


def materialize_f32(t) -> np.ndarray:
    """Copy ONE tensor's data into a writable float32 numpy array, reading only
    that tensor's bytes off disk (so a sibling phi_spatial is never faulted in).
    np.array(..., copy) also strips the read-only flag of an mmap-backed view, so
    downstream torch.from_numpy stays warning-free."""
    return np.array(t.numpy(), dtype=np.float32)


_SEQ_LENS_CACHE = {}   # path-string -> seq_lens (memoise the metadata reads)


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


def split_boundaries(n: int, fracs, seq_lens):
    """Interior cut points for a >=2-way sequence-aligned split of `n` rows.

    `fracs` (summing to 1) give each split's target share of `n`. Every cut is
    snapped to a whole-sequence boundary from `seq_lens` (cumulative sums of
    per-sequence frame counts), so no sequence is ever divided across two
    splits — unlike `split_boundary`, this has no ratio-only fallback: without
    `seq_lens` there is no way to guarantee that, so it raises instead of
    silently risking a leak. Raises if there aren't enough sequences to give
    every split at least one whole sequence.
    """
    fracs = list(fracs)
    if abs(sum(fracs) - 1.0) > 1e-6:
        raise ValueError(f"fracs must sum to 1, got {sum(fracs)}")
    if not seq_lens or int(np.sum(seq_lens)) != n:
        raise ValueError("split_boundaries requires seq_lens summing to n (no leak-safe fallback)")
    edges = np.cumsum(seq_lens)
    interior = edges[:-1]   # candidate cut points; excludes the final edge (== n)
    targets = np.cumsum(fracs)[:-1] * n
    cuts, prev = [], 0
    for t in targets:
        remaining = interior[interior > prev]
        if len(remaining) == 0:
            raise ValueError("not enough sequences to honor every split")
        cut = int(remaining[np.argmin(np.abs(remaining - t))])
        cuts.append(cut)
        prev = cut
    return cuts


# ── the canonical 4-way pool split ───────────────────────────────────────────
# ONE split scheme for every reported number in the project (see
# unified_numbers/README.md). Defined here rather than in mdd_split_sensitivity
# so the representation/detector/information stages can share it without
# importing an MDD module.
#
#   fit         50%  detector statistics (covariance, PCA basis, kNN reference)
#   calib       10%  score standardization / threshold calibration
#   sensitivity 10%  hyperparameter sweeps + model selection; never fit/calib
#   final       30%  every reported AUROC/AUPR/FPR95/CI
#
# Cuts are sequence-aligned, so no recording contributes frames to two pools.
POOL_NAMES = ["fit", "calib", "sensitivity", "final"]
POOL_FRACS = [0.50, 0.10, 0.10, 0.30]


def pool_ranges(n: int, seq_lens):
    """{pool_name: (start, stop)} for the canonical 4-way split of `n` rows."""
    bounds = [0] + split_boundaries(n, POOL_FRACS, seq_lens) + [n]
    return {name: (bounds[i], bounds[i + 1]) for i, name in enumerate(POOL_NAMES)}


def pool_slice(arr, seq_lens, pool: str):
    """Rows of `arr` belonging to one canonical pool."""
    a, b = pool_ranges(len(arr), seq_lens)[pool]
    return arr[a:b]


def split_train_eval(arr, train_ratio: float = TRAIN_RATIO, seq_lens=None):
    """Split rows of `arr` into (train, eval) on whole sequences."""
    cut = split_boundary(len(arr), train_ratio, seq_lens)
    return arr[:cut], arr[cut:]


def held_out_eval(arr, train_ratio: float = TRAIN_RATIO, seq_lens=None):
    """Held-out eval slice [cut:] of a run.

    Used for corrupted POSITIVES so they are drawn from the same held-out
    sequences as the clean negatives: a corruption run is cut at the same
    sequence-aligned boundary as clean, and only the eval tail is scored. This
    keeps clean-vs-corrupt AUROC on matched held-out content (the train-portion
    sequences, whose clean twins the detector was fitted on, are excluded)."""
    return split_train_eval(arr, train_ratio, seq_lens)[1]


def load_phi_seq_lens(run_name: str = "clean", artifact_dir=None):
    """Per-sequence frame counts saved by extract.py, or None for legacy files.

    `artifact_dir` defaults to the phi directory but any per-run artifact
    directory (temporal_phi, spike, ...) works — they all share the metadata.
    """
    f = (artifact_dir or cfg.PHI_DIR) / f"{run_name}.pt"
    ck = str(f)
    if ck in _SEQ_LENS_CACHE:
        return _SEQ_LENS_CACHE[ck]
    if not f.exists():
        return None
    try:
        # mmap: faults only the small metadata, not the multi-GB phi tensors.
        d = load_pt(f)
        sl = d.get("seq_lens", None)
    except Exception:
        sl = None
    _SEQ_LENS_CACHE[ck] = sl
    return sl


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
        # mmap: faults only phi_spatial's pages, not the sibling phi tensor.
        d = load_pt(f)
        ps = d.get("phi_spatial", None)
        return materialize_f32(ps) if ps is not None else None
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
        self.fast_mode = FAST_MODE

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
            # mmap + materialize only phi: a sibling phi_spatial (~2 GB/run) is
            # never read off disk, and seq_lens faults just its tiny metadata.
            d = load_pt(f)
            arr = materialize_f32(d["phi"])
            self._seq_lens[key] = d.get("seq_lens", None)
            del d
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
            # mmap: faults only phi_spatial's pages, not the sibling phi tensor.
            d = load_pt(self._available[key])
            ps = d.get("phi_spatial", None)
            return materialize_f32(ps) if ps is not None else None
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

def calc_fpr95(y_true: np.ndarray, y_score: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = int(np.argmax(tpr >= 0.95))
    return float(fpr[idx])

def auroc_aupr_fpr95(clean_scores: np.ndarray, corrupt_scores: np.ndarray):
    """AUROC / AUPR / FPR@95 for one clean-vs-corrupt score comparison.

    Returns (auroc, aupr, fpr95), or None if fewer than 2 classes are present
    or a metric fails to compute — mirrors the `continue`-on-guard every
    evaluate_*.py corruption-scoring loop used before this was consolidated."""
    y_true = np.concatenate([np.zeros(len(clean_scores)), np.ones(len(corrupt_scores))])
    y_score = np.concatenate([clean_scores, corrupt_scores])
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return (float(roc_auc_score(y_true, y_score)),
                float(average_precision_score(y_true, y_score)),
                calc_fpr95(y_true, y_score))
    except Exception:
        return None

def _get_present(all_phi, corruptions=None, severities=None):
    corruptions = corruptions or cfg.CORRUPTIONS
    severities  = severities  or cfg.SEVERITIES
    return [c for c in corruptions
            if any(f"{c}_L{s}" in all_phi for s in severities)]

def _subsample(arr: np.ndarray, n: int = MAX_FIT_SAMPLES) -> np.ndarray:
    """Fit-set subsampler.

    Default (infinite-compute) behaviour: NO cap — detectors fit on the FULL
    clean set, so the `n` argument is ignored. `--fast` restores a hard 500-row
    cap for quick smoke tests. (Previously this always capped at
    MAX_FIT_SAMPLES=3000 to bound fit time; that compute cap has been removed so
    every fitting path — Mahalanobis, kNN, GMM, OCSVM, flow, AE, MDD reference —
    sees all available data.)"""
    if FAST_MODE:
        cap = min(n if n is not None else len(arr), 500)
        if len(arr) <= cap:
            return arr
        rng = np.random.default_rng(42)
        return arr[rng.choice(len(arr), cap, replace=False)]
    return arr


# Full-covariance EM in 2112-D scales ~O(n*k*d^2) per iteration and benefits
# little from more data (a 5-component model), so GMM keeps an explicit cap even
# though every other detector now fits on the full set.
GMM_FIT_SAMPLES = 20000

# RBF OneClassSVM (libsvm) FIT is ~O(n^2) in time/memory, so it keeps an explicit
# cap too. Scoring is GPU-chunked (see evaluate_detectors.score_ocsvm), so only
# the one-time fit is bounded here.
OCSVM_FIT_SAMPLES = 20000


def _cap_subset(arr: np.ndarray, n: int = GMM_FIT_SAMPLES, seed: int = 42) -> np.ndarray:
    """Deterministic HARD cap to `n` rows. Unlike `_subsample` (no-op outside
    --fast), this always caps — for fits whose cost scales badly with n."""
    if FAST_MODE:
        n = min(n if n is not None else len(arr), 500)
    if n is None or len(arr) <= n:
        return arr
    rng = np.random.default_rng(seed)
    return arr[rng.choice(len(arr), n, replace=False)]


# ─────────────────────────────────────────────────────────────────────────────
# VRAM-aware GPU chunking. Single source of truth for every tileable GPU loop,
# so chunk sizes auto-scale from a laptop card up to a cluster GPU and never need
# hardcoded magic numbers. `query_chunk_rows` sizes the FIRST chunk from FREE
# VRAM (a fraction, to leave room for op temporaries / fragmentation);
# `chunked_apply` then halves on any CUDA OOM the estimate failed to foresee.
# This only protects *tileable* ops (a batch dim to split). Single large
# allocations (full-cov GMM, big SVDs) are bounded by the *_FIT_SAMPLES caps
# instead — chunking cannot help those.
# ─────────────────────────────────────────────────────────────────────────────

def device_for(op: str, verbose: bool = True) -> str:
    """Return 'cuda'/'cpu' and announce which device `op` runs on, so a cluster
    log makes it obvious whether each operation actually hit the GPU."""
    if torch.cuda.is_available():
        if verbose:
            print(f"  [GPU] {op}", flush=True)
        return "cuda"
    if verbose:
        print(f"  [CPU] {op}", flush=True)
    return "cpu"


# Fraction of FREE VRAM one (chunk x n_ref) buffer is sized to occupy. Larger =
# bigger chunks = fewer kernel launches / better GPU saturation, at the cost of a
# few `chunked_apply` OOM-halving probes when an op's temporaries don't fit (the
# halving then converges to the largest chunk that DOES fit — so this is a "fill
# the GPU without OOM" dial, safe to push up). Default tuned conservatively;
# bump it via `set_vram_chunk_frac` (the combined runner exposes --vram-frac).
_VRAM_CHUNK_FRAC = 0.25
# Upper clamp on rows/chunk. Scales with the frac so a big GPU + a small n_ref op
# (e.g. Mahalanobis) actually gets large chunks instead of being capped low.
_CHUNK_ROW_CAP = 131072


def set_vram_chunk_frac(frac: float):
    """Set the global free-VRAM fraction used to size GPU chunks, and scale the
    row cap with it so the dial actually takes effect on large GPUs."""
    global _VRAM_CHUNK_FRAC, _CHUNK_ROW_CAP
    _VRAM_CHUNK_FRAC = max(0.05, min(0.95, float(frac)))
    _CHUNK_ROW_CAP = int(131072 * max(1.0, _VRAM_CHUNK_FRAC / 0.25))


def query_chunk_rows(n_ref: int, device, frac: float = None,
                     bytes_per_cell: int = 4) -> int:
    """Rows per chunk so a (chunk x n_ref) float32 buffer uses ~`frac` of FREE
    GPU memory. CPU has no VRAM limit → a large fixed chunk. Call this AFTER the
    reference tensor is already on the device, so its memory is excluded.
    `frac=None` uses the tunable global `_VRAM_CHUNK_FRAC`."""
    if str(device) != "cuda" or not torch.cuda.is_available():
        return 50000
    frac = _VRAM_CHUNK_FRAC if frac is None else frac
    try:
        free, _ = torch.cuda.mem_get_info()
        rows = int(free * frac / (max(1, n_ref) * bytes_per_cell))
        return int(np.clip(rows, 256, _CHUNK_ROW_CAP))
    except Exception:
        return 8192


def chunked_apply(fn, X, device, n_ref: int = 1, init_chunk: int = None,
                  min_chunk: int = 128):
    """Apply `fn` to row-chunks of `X`, concatenating results along dim 0.

    `fn` receives a chunk already moved to `device` and returns a tensor (1-D
    scores or a 2-D matrix). The chunk size starts from `query_chunk_rows`
    (or `init_chunk`) and halves on CUDA OOM until the work fits, so callers
    never hardcode a chunk size or risk an OOM crash. Returns a numpy array when
    `X` is a numpy array, else a CPU tensor."""
    is_np = isinstance(X, np.ndarray)
    Xt = torch.from_numpy(X) if is_np else X
    N = Xt.shape[0]
    chunk = init_chunk or query_chunk_rows(n_ref, device)
    out, i = [], 0
    while i < N:
        end = min(i + chunk, N)
        sub = Xt[i:end].to(device)
        try:
            r = fn(sub)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            del sub
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if chunk <= min_chunk:
                raise
            chunk = max(min_chunk, chunk // 2)
            continue
        out.append(r.detach().to("cpu"))
        del sub
        i = end
    if out:
        res = torch.cat(out, dim=0)
    else:
        # N == 0: probe `fn` on a zero-row slice so the result keeps its
        # trailing dims. A bare empty((0,)) is 1-D and breaks 2-D consumers
        # (e.g. MDD._rcf does out[:, 1]) when a sequence is fully dropped or an
        # eval split is empty.
        try:
            res = fn(Xt[:0].to(device)).detach().to("cpu")
        except Exception:
            res = torch.empty((0,))
    return res.numpy() if is_np else res


def maha_d2(x, mu, P, device):
    """Squared Mahalanobis distance (x-mu)^T P (x-mu), chunked on `device`.

    Single source of truth for the rebuild-each-call Mahalanobis scoring used by
    `evaluate_detectors.score_mahalanobis` and `mdd._maha_d2` (the scorer-factory
    `vmem_scorers.mahalanobis_scorer` intentionally keeps its own cached mu/P
    closure to avoid re-uploading across many score calls)."""
    mu_t = torch.from_numpy(np.ascontiguousarray(mu, dtype=np.float32)).to(device)
    P_t = torch.from_numpy(np.ascontiguousarray(P, dtype=np.float32)).to(device)

    def fn(chunk):
        d = chunk - mu_t
        return ((d @ P_t) * d).sum(dim=1)

    return chunked_apply(fn, np.ascontiguousarray(x, dtype=np.float32),
                         device, n_ref=P_t.shape[0])


def knn_score(ref, k, X, device, ref_block: int = 16384,
              init_query: int = None, min_chunk: int = 128, reduce: str = "mean"):
    """Distance to the k nearest `ref` neighbours, double-chunked.

    `reduce="mean"` returns the mean of the k nearest distances (the φ-detector
    convention); `reduce="kth"` returns the distance to the k-th nearest
    neighbour (the deep-kNN / Sun-et-al. convention used by the ANN baseline).

    The reference stays on CPU and is streamed to the device in `ref_block`
    rows, so the full clean reference (which can be ~2 GB) is never resident on
    the GPU at once — an allocation fix, NOT a data cap (the entire reference is
    still used). Query rows are chunked too, halving on CUDA OOM. Single source
    of truth for the GPU kNN used by `evaluate_detectors.score_knn`,
    `vmem_scorers.knn_scorer`, and the ANN-baseline `DetectorKNN`."""
    if reduce not in ("mean", "kth"):
        raise ValueError(f"reduce must be 'mean' or 'kth', got {reduce!r}")
    ref_cpu = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float32))
    n_ref = ref_cpu.shape[0]
    k = max(1, min(k, n_ref))
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    N = Xt.shape[0]
    # First-chunk size assumes one ref_block resident at a time.
    qchunk = init_query or query_chunk_rows(min(n_ref, ref_block), device)
    out, i = [], 0
    while i < N:
        end = min(i + qchunk, N)
        try:
            q = Xt[i:end].to(device)
            best = None  # running (m, k) smallest distances seen so far
            for j in range(0, n_ref, ref_block):
                rb = ref_cpu[j:j + ref_block].to(device)
                dm = torch.cdist(q, rb)               # (m, b)
                if best is not None:
                    dm = torch.cat([best, dm], dim=1)
                kk = min(k, dm.shape[1])
                best, _ = torch.topk(dm, kk, largest=False, dim=1)
                del rb, dm
            # best holds the k smallest distances; its max is the k-th nearest.
            reduced = best.max(dim=1).values if reduce == "kth" else best.mean(dim=1)
            out.append(reduced.to("cpu"))
            del q, best
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if qchunk <= min_chunk:
                raise
            qchunk = max(min_chunk, qchunk // 2)
            continue
        i = end
    return (torch.cat(out).numpy() if out
            else np.empty((0,), dtype=np.float32))
