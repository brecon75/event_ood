"""Extraction correctness: VmemMonitor.collect_phi / collect_phi_spatial /
collect_spikes / collect_temporal_phi. Golden moments are hand-computed.

The monitor is fed synthetic membrane tensors via stub_monitor; no model runs.
"""
import numpy as np
import pytest
import torch


# ───────────────────── collect_phi: golden moments ─────────────────────

def test_collect_phi_golden_mean_var_kurt(stub_monitor):
    # One channel, one pixel, values 0..9 over T=10. Hand-computed:
    #   mean=4.5, var(pop)=8.25, excess kurtosis = 120.8625/68.0625 - 3 = -1.22433
    V = torch.arange(10, dtype=torch.float32).reshape(10, 1, 1, 1, 1)
    phi = stub_monitor({0: V}).collect_phi().numpy()
    assert phi.shape == (1, 3)
    assert phi[0, 0] == pytest.approx(4.5, abs=1e-5)        # mu
    assert phi[0, 1] == pytest.approx(8.25, abs=1e-5)       # var (population)
    assert phi[0, 2] == pytest.approx(-1.22433, abs=1e-4)   # excess kurtosis


def test_collect_phi_width_concats_layers(stub_monitor, make_membrane):
    m = stub_monitor({0: make_membrane(C=64, seed=1),
                      1: make_membrane(C=128, seed=2)})
    phi = m.collect_phi().numpy()
    assert phi.shape == (1, 3 * (64 + 128))
    assert np.isfinite(phi).all()


def test_collect_phi_empty_returns_empty(stub_monitor):
    m = stub_monitor({})
    m._v = {}
    assert m.collect_phi().numel() == 0


def test_collect_phi_constant_input_no_nan(stub_monitor):
    # Zero temporal variance -> var clamped, kurt finite (not NaN/inf).
    V = torch.full((10, 1, 4, 8, 8), 0.7)
    phi = stub_monitor({0: V}).collect_phi().numpy()
    assert np.isfinite(phi).all()


# ─────────────────── collect_phi_spatial: the GAP-recovered signal ───────────────────

def test_spatial_var_zero_for_uniform_map(stub_monitor):
    # All 4 pixels equal (constant over T) -> spatial variance 0, PR = 1.
    V = torch.ones(4, 1, 1, 2, 2)
    sp = stub_monitor({0: V}).collect_phi_spatial().numpy()
    assert sp.shape == (1, 2)                      # [spatial_var, spatial_pr] x 1 channel
    assert sp[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert sp[0, 1] == pytest.approx(1.0, abs=1e-3)


def test_spatial_var_positive_for_patchy_map(stub_monitor):
    # mu_pix = [2,2,0,0] (constant over T) -> population var = 1.0
    V = torch.zeros(4, 1, 1, 2, 2)
    V[:, 0, 0, 0, :] = 2.0
    sp = stub_monitor({0: V}).collect_phi_spatial().numpy()
    assert sp[0, 0] == pytest.approx(1.0, abs=1e-5)


def test_spatial_pr_low_when_energy_concentrated(stub_monitor):
    # Only 1 of 4 pixels has temporal variance -> participation ratio ~= 1/4.
    V = torch.zeros(10, 1, 1, 2, 2)
    V[:, 0, 0, 0, 0] = torch.arange(10, dtype=torch.float32)
    sp = stub_monitor({0: V}).collect_phi_spatial().numpy()
    assert sp[0, 1] == pytest.approx(0.25, abs=1e-2)


def test_spatial_pr_in_unit_interval(stub_monitor, make_membrane):
    sp = stub_monitor({0: make_membrane(C=16, seed=3)}).collect_phi_spatial().numpy()
    pr = sp[0, 16:]
    assert np.all(pr > 0) and np.all(pr <= 1.0 + 1e-4)


def test_spatial_width_matches_two_stats_per_channel(stub_monitor, make_membrane):
    m = stub_monitor({0: make_membrane(C=64, seed=1), 1: make_membrane(C=128, seed=2)})
    assert m.collect_phi_spatial().shape == (1, 2 * (64 + 128))


# ─────────────────── collect_spikes / collect_temporal_phi ───────────────────

def test_collect_spikes_binary_entropy(stub_monitor):
    m = stub_monitor({0: torch.zeros(10, 1, 2, 4, 4)})
    # spike maps: channel 0 fires half the time (p=0.5 -> entropy 1),
    # channel 1 never fires (p=0 -> entropy 0 after clamp).
    sp = torch.zeros(10, 1, 2, 4, 4)
    sp[:5, 0, 0] = 1.0
    m._spikes = {0: [sp]}
    out = m.collect_spikes()
    assert out["spike_rate"][0, 0] == pytest.approx(0.5, abs=1e-6)
    assert out["spike_entropy"][0, 0] == pytest.approx(1.0, abs=1e-4)
    assert out["spike_entropy"][0, 1] == pytest.approx(0.0, abs=1e-4)
    assert np.isfinite(out["spike_entropy"].numpy()).all()


def test_collect_temporal_phi_shape_and_finite(stub_monitor, make_membrane):
    m = stub_monitor({0: make_membrane(C=8, seed=5)})
    tphi = m.collect_temporal_phi().numpy()
    assert tphi.shape == (1, 7)                    # 7 handcrafted stats x 1 layer
    assert np.isfinite(tphi).all()
