"""free_rider_ablation pure units: raw input stats + weight randomization."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from analysis.free_rider_ablation import extract_raw_input_stats, randomize_weights


def test_extract_raw_input_stats_golden_4d():
    # (N=1, TC=1, H=1, W=2) with pixel values [0, 9] -> mu=4.5, var=20.25, kurt=-2
    x = torch.tensor([[[[0.0, 9.0]]]])          # mean 4.5, dev ±4.5
    out = extract_raw_input_stats(x).numpy()
    assert out.shape == (1, 3)                  # [mu, var, kurt] x 1 channel
    assert out[0, 0] == pytest.approx(4.5)
    assert out[0, 1] == pytest.approx(20.25)
    # two symmetric points -> excess kurtosis = -2
    assert out[0, 2] == pytest.approx(-2.0, abs=1e-4)


def test_extract_raw_input_stats_5d_shape():
    x = torch.randn(3, 10, 2, 8, 8)             # (N, T, C, H, W)
    out = extract_raw_input_stats(x)
    assert out.shape == (3, 3 * 10 * 2)         # mu|var|kurt over T*C channels
    assert torch.isfinite(out).all()


def test_extract_raw_input_stats_bad_ndim_raises():
    with pytest.raises(ValueError):
        extract_raw_input_stats(torch.randn(4, 8))   # 2D not supported


def test_randomize_weights_changes_conv_and_resets_bn():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(2, 4, 3), nn.BatchNorm2d(4), nn.Linear(4, 2))
    # dirty the BN running stats so we can confirm they are reset
    net[1].running_mean.fill_(7.0)
    net[1].running_var.fill_(3.0)
    conv_before = net[0].weight.detach().clone()
    randomize_weights(net)
    assert not torch.equal(net[0].weight, conv_before)          # conv re-init
    assert torch.allclose(net[1].running_mean, torch.zeros(4))  # BN stats reset
    assert torch.allclose(net[1].running_var, torch.ones(4))
