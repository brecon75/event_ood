"""Unit + integration tests for the analysis/ folder.

Run:  python run_analysis_tests.py
Exit code 0 = all tests passed.
"""
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

PASS, FAIL = [], []


def test(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  PASS  {name}")
        except Exception:
            FAIL.append(name)
            print(f"  FAIL  {name}")
            traceback.print_exc()
    return deco


print("\n=== vmem_utils: split logic ===")
from analysis.vmem_utils import (
    split_boundary, split_train_eval, slice_phi_layer, slice_phi_stat,
    auroc_fpr95, _subsample, load_phi_seq_lens, LazyPhiDict, LAYER_SPECS,
    TOTAL_PHI_DIM,
)


@test("split_boundary: plain ratio cut, no seq_lens")
def _():
    assert split_boundary(100, 0.7) == 70
    assert split_boundary(10, 0.5) == 5


@test("split_boundary: degenerate sizes n=0,1,2")
def _():
    assert split_boundary(0, 0.7) == 0
    assert split_boundary(1, 0.7) == 1          # everything to train
    assert split_boundary(2, 0.7) == 1          # both sides non-empty


@test("split_boundary: extreme ratios never empty a side (n>1)")
def _():
    assert split_boundary(100, 0.0) == 1
    assert split_boundary(100, 1.0) == 99


@test("split_boundary: single sequence falls back to ratio cut (1192-frame regression)")
def _():
    # Previously degenerated to 1191/1192 (one eval frame).
    assert split_boundary(1192, 0.7, seq_lens=[1192]) == 834


@test("split_boundary: snaps to NEAREST interior sequence edge")
def _():
    # cut=70, interior edges [30, 60] -> 60 (old code rounded up past the end)
    assert split_boundary(100, 0.7, seq_lens=[30, 30, 40]) == 60
    # cut=50, interior edges [30, 60] -> 60 (nearest)
    assert split_boundary(100, 0.5, seq_lens=[30, 30, 40]) == 60
    # cut=35 -> 30
    assert split_boundary(100, 0.35, seq_lens=[30, 30, 40]) == 30


@test("split_boundary: mismatched seq_lens are ignored")
def _():
    assert split_boundary(100, 0.7, seq_lens=[10, 10]) == 70  # sums to 20 != 100


@test("split_train_eval: partition is exact and ordered")
def _():
    arr = np.arange(100).reshape(50, 2)
    tr, ev = split_train_eval(arr, 0.7)
    assert len(tr) == 35 and len(ev) == 15
    assert np.array_equal(np.vstack([tr, ev]), arr)
    t = torch.arange(20).reshape(10, 2)
    tr, ev = split_train_eval(t, 0.7)
    assert torch.equal(torch.cat([tr, ev]), t)


print("\n=== vmem_utils: phi slicing ===")


def make_phi(n=4):
    """phi with mu=layer_idx+0.1, var=+0.2, kurt=+0.3 so slices are checkable."""
    phi = np.zeros((n, TOTAL_PHI_DIM), dtype=np.float32)
    for li, s in enumerate(LAYER_SPECS):
        phi[:, s["phi_start"]:s["mu_end"]] = li + 0.1
        phi[:, s["mu_end"]:s["var_end"]] = li + 0.2
        phi[:, s["var_end"]:s["phi_end"]] = li + 0.3
    return phi


@test("slice_phi_layer: returns each layer's 3*C block")
def _():
    phi = make_phi()
    for li, s in enumerate(LAYER_SPECS):
        sl = slice_phi_layer(phi, li)
        assert sl.shape == (4, 3 * s["C"])
        assert np.all(sl[:, :s["C"]] == li + 0.1)
        assert np.all(sl[:, 2 * s["C"]:] == li + 0.3)


@test("slice_phi_stat: mu/var/kurt blocks across layers")
def _():
    phi = make_phi()
    mu = slice_phi_stat(phi, "mu")
    var = slice_phi_stat(phi, "var")
    kur = slice_phi_stat(phi, "kurtosis")
    n_ch = sum(s["C"] for s in LAYER_SPECS)
    assert mu.shape == var.shape == kur.shape == (4, n_ch)
    assert np.all(np.isin(np.unique(mu), [0.1, 1.1, 2.1, 3.1]))
    assert np.all(np.isin(np.unique(kur), [0.3, 1.3, 2.3, 3.3]))


@test("slice_phi_stat: truncated phi handled, invalid stat raises")
def _():
    phi = make_phi()[:, :64]  # only layer-1 mu present
    assert slice_phi_stat(phi, "mu").shape == (4, 64)
    assert slice_phi_stat(phi, "var").shape[1] == 0
    try:
        slice_phi_stat(phi, "median")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


@test("slice_phi_layer: layer beyond truncated phi returns empty")
def _():
    phi = make_phi()[:, :64]
    assert slice_phi_layer(phi, 3).shape == (4, 0)


print("\n=== vmem_utils: metrics ===")


@test("auroc_fpr95: perfect / inverted / single-class")
def _():
    y = np.array([0, 0, 0, 1, 1, 1])
    a, f = auroc_fpr95(y, np.array([0, 1, 2, 10, 11, 12]))
    assert a == 1.0 and f == 0.0
    a, f = auroc_fpr95(y, np.array([10, 11, 12, 0, 1, 2]))
    assert a == 0.0
    a, f = auroc_fpr95(np.zeros(5), np.arange(5))
    assert np.isnan(a) and np.isnan(f)


@test("auroc_fpr95: random scores ~0.5")
def _():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 5000)
    a, _ = auroc_fpr95(y, rng.normal(size=10000))
    assert abs(a - 0.5) < 0.03


@test("_subsample: identity when small, deterministic when large")
def _():
    small = np.arange(10)
    assert _subsample(small, n=20) is small
    big = np.arange(10000)
    s1, s2 = _subsample(big, n=100), _subsample(big, n=100)
    assert len(s1) == 100 and np.array_equal(s1, s2)


print("\n=== representation_ablation: Mahalanobis factory ===")
from analysis.representation_ablation import (
    fit_mahalanobis, get_mahalanobis_scores, calc_fpr95,
)


@test("fit_mahalanobis == get_mahalanobis_scores (wrapper equivalence)")
def _():
    rng = np.random.default_rng(1)
    tr, te = rng.normal(size=(500, 20)), rng.normal(size=(100, 20))
    s1 = fit_mahalanobis(tr)(te)
    s2 = get_mahalanobis_scores(tr, te)
    assert np.allclose(s1, s2)


@test("fit_mahalanobis: separates shifted OOD (AUROC > 0.95)")
def _():
    rng = np.random.default_rng(2)
    tr = rng.normal(size=(1000, 16))
    clean, ood = rng.normal(size=(300, 16)), rng.normal(loc=2.0, size=(300, 16))
    sc = fit_mahalanobis(tr)
    y = np.concatenate([np.zeros(300), np.ones(300)])
    s = np.concatenate([sc(clean), sc(ood)])
    a, _ = auroc_fpr95(y, s)
    assert a > 0.95


@test("fit_mahalanobis: scoring once vs twice gives identical results (stateless closure)")
def _():
    rng = np.random.default_rng(3)
    tr, te = rng.normal(size=(200, 8)), rng.normal(size=(50, 8))
    sc = fit_mahalanobis(tr)
    assert np.allclose(sc(te), sc(te))


print("\n=== vmem_scorers vs sklearn references ===")
from analysis.vmem_scorers import mahalanobis_scorer, knn_scorer, gmm_scorer
from analysis.vmem_utils import _subsample as _sub


@test("mahalanobis_scorer: matches LedoitWolf reference, detects shift")
def _():
    from sklearn.covariance import LedoitWolf
    rng = np.random.default_rng(4)
    clean = rng.normal(size=(800, 12)).astype(np.float32)
    test = rng.normal(loc=1.0, size=(200, 12)).astype(np.float32)
    sc = mahalanobis_scorer(clean)(test)
    cov = LedoitWolf().fit(_sub(clean))
    d = test - cov.location_
    ref = np.einsum("ni,ij,nj->n", d, cov.precision_, d)
    assert np.allclose(sc, ref, rtol=1e-3, atol=1e-2)


@test("knn_scorer: matches sklearn NearestNeighbors mean distance")
def _():
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(5)
    clean = rng.normal(size=(400, 10)).astype(np.float32)
    test = rng.normal(size=(100, 10)).astype(np.float32)
    sc = knn_scorer(clean, k=5)(test)
    nn = NearestNeighbors(n_neighbors=5).fit(_sub(clean))
    ref = nn.kneighbors(test)[0].mean(axis=1)
    assert np.allclose(sc, ref, rtol=1e-3, atol=1e-3)


@test("gmm_scorer: matches sklearn -score_samples")
def _():
    from sklearn.mixture import GaussianMixture
    rng = np.random.default_rng(6)
    clean = rng.normal(size=(400, 8)).astype(np.float32)
    test = rng.normal(size=(100, 8)).astype(np.float32)
    sc = gmm_scorer(clean, n_components=5)(test)
    gmm = GaussianMixture(n_components=5, covariance_type="full",
                          reg_covar=1e-4, random_state=42, max_iter=300)
    gmm.fit(_sub(clean))
    ref = -gmm.score_samples(test.astype(np.float64))
    assert np.max(np.abs(sc - ref)) < 0.05, np.max(np.abs(sc - ref))


@test("knn_scorer: degenerate 1-sample reference set does not crash")
def _():
    clean = np.zeros((1, 4), dtype=np.float32)
    s = knn_scorer(clean, k=5)(np.ones((3, 4), dtype=np.float32))
    assert s.shape == (3,) and np.all(s > 0)


print("\n=== vmem_models: RealNVP correctness ===")
from analysis.vmem_models import CouplingLayer, RealNVP, TemporalAutoencoder


@test("CouplingLayer: returned log-det matches autograd Jacobian")
def _():
    torch.manual_seed(0)
    layer = CouplingLayer(dim=6).double()
    x = torch.randn(1, 6, dtype=torch.float64)
    for flip in (False, True):
        y, s_sum = layer(x, flip=flip)
        J = torch.autograd.functional.jacobian(
            lambda v: layer(v, flip=flip)[0], x)
        J = J.squeeze(0).squeeze(1)  # (6, 6)
        ref = torch.linalg.slogdet(J)[1]
        assert torch.allclose(s_sum.squeeze(), ref, atol=1e-8), (flip, s_sum, ref)


@test("RealNVP: total log-det matches autograd; log_prob finite & invariant-ish")
def _():
    torch.manual_seed(0)
    flow = RealNVP(dim=4, n_layers=4).double()
    x = torch.randn(1, 4, dtype=torch.float64)
    z, log_det = flow(x)
    J = torch.autograd.functional.jacobian(lambda v: flow(v)[0], x)
    ref = torch.linalg.slogdet(J.squeeze(0).squeeze(1))[1]
    assert torch.allclose(log_det.squeeze(), ref, atol=1e-8)
    lp = flow.log_prob(torch.randn(32, 4, dtype=torch.float64))
    assert torch.isfinite(lp).all()


@test("TemporalAutoencoder: reconstruction shape matches input")
def _():
    ae = TemporalAutoencoder(dim=704)
    x = torch.randn(3, 10, 704)
    assert ae(x).shape == x.shape


print("\n=== extract_offline_features: margin histogram ===")
from analysis.extract_offline_features import extract_margin_hist


@test("extract_margin_hist: matches naive per-sample digitize histogram")
def _():
    torch.manual_seed(1)
    N, T = 3, 10
    sum_C = sum(s["C"] for s in LAYER_SPECS)
    tgap = torch.rand(N, T, sum_C) * 7 - 3  # values in [-3, 4]
    out = extract_margin_hist(tgap, theta=1.0, bins=20)
    assert out.shape == (N, len(LAYER_SPECS) * 20)
    edges = np.linspace(-2, 2, 21)[1:-1]
    c0 = 0
    for li, s in enumerate(LAYER_SPECS):
        V = tgap[:, :, c0:c0 + s["C"]].numpy()
        c0 += s["C"]
        for n in range(N):
            m = (V[n] - 1.0).reshape(-1)
            idx = np.digitize(m, edges)
            ref = np.bincount(idx, minlength=20).astype(np.float64)
            ref = ref / (ref.sum() + 1e-8)
            got = out[n, li * 20:(li + 1) * 20]
            assert np.allclose(got, ref, atol=1e-6)


print("\n=== fusion_features: alignment ===")
from analysis.fusion_features import align_and_concat


@test("align_and_concat: truncates to min rows, preserves order; None when empty")
def _():
    d = {"a": np.arange(12).reshape(6, 2), "b": np.arange(8).reshape(4, 2)}
    out = align_and_concat(d, ["a", "b"])
    assert out.shape == (4, 4)
    assert np.array_equal(out[:, :2], d["a"][:4])
    assert align_and_concat(d, ["missing"]) is None


print("\n=== reliability: detection metric ===")
from analysis.reliability import compute_detection_metric


@test("compute_detection_metric: counts boxes with obj*cls > 0.3; None-safe")
def _():
    det = torch.zeros(2, 3, 7)
    det[0, 0, 4], det[0, 0, 5] = 0.9, 0.8   # 0.72 -> counted
    det[0, 1, 4], det[0, 1, 6] = 0.5, 0.5   # 0.25 -> not counted
    det[1, 2, 4], det[1, 2, 5] = 1.0, 0.4   # 0.40 -> counted
    out = compute_detection_metric(det)
    assert list(out) == [1, 1]
    assert compute_detection_metric(None).shape == (1,)


print("\n=== extract_ann_baselines: atomic save ===")
from analysis.extract_ann_baselines import atomic_save


@test("atomic_save: file loadable, no temp residue")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "x.pt"
        atomic_save({"a": torch.ones(3)}, p)
        assert torch.load(p, weights_only=True)["a"].sum() == 3
        assert not list(Path(td).glob("_tmp_*"))


print("\n=== integration: real output files ===")
from vmem_benchmark import benchmark_config as cfg


@test("LazyPhiDict: loads real phi runs, seq_lens consistent")
def _():
    d = LazyPhiDict()
    assert "clean" in d and len(d) >= 1
    arr = d["clean"]
    assert arr.ndim == 2 and arr.shape[1] == TOTAL_PHI_DIM
    assert np.isfinite(arr).all(), "phi contains non-finite values"
    sl = d.get_seq_lens("clean")
    if sl is not None:
        assert sum(sl) == len(arr)


@test("real clean split is no longer degenerate (eval >= 10% of frames)")
def _():
    d = LazyPhiDict()
    arr = d["clean"]
    cut = split_boundary(len(arr), seq_lens=d.get_seq_lens("clean"))
    assert len(arr) - cut >= 0.1 * len(arr), f"eval side only {len(arr)-cut} frames"


@test("all saved phi/fused/ann artifacts are finite and row-consistent per run")
def _():
    for f in cfg.PHI_DIR.glob("*.pt"):
        d = torch.load(f, map_location="cpu", weights_only=True)
        phi = d["phi"]
        assert torch.isfinite(phi.float()).all(), f.name
        fused = cfg.OUTPUT_DIR / "features/fused" / f.name
        if fused.exists():
            fd = torch.load(fused, map_location="cpu", weights_only=True)
            mf = fd.get("membrane_fused")
            if mf is not None:
                assert torch.isfinite(mf.float()).all(), f.name
                assert len(mf) == len(phi), f"{f.name}: fused rows != phi rows"


print(f"\n{'='*50}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  Failed:", ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
