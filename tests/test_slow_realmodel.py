"""Optional end-to-end test against the REAL model + checkpoint.

Skipped by default. Enable with:  RUN_SLOW=1 pytest -m slow
Verifies one real forward pass produces a finite 2112-D phi and 1408-D
phi_spatial, exercising monitor + model wiring that the stub cannot.
"""
import os

import numpy as np
import pytest
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not os.environ.get("RUN_SLOW"),
                       reason="slow real-model test; set RUN_SLOW=1 to run"),
]


def test_real_model_forward_produces_finite_phi():
    from vmem_benchmark import benchmark_config as cfg
    if not getattr(cfg, "CKPT_PATH", None) or not os.path.exists(cfg.CKPT_PATH):
        pytest.skip("model checkpoint not available")

    from vmem_benchmark.model_loader import load_model
    from vmem_benchmark.monitor import VmemMonitor
    from spikingjelly.clock_driven import functional

    module, backbone = load_model(cfg.DEVICE)
    mon = VmemMonitor(backbone)
    functional.reset_net(backbone)
    mon.reset()

    x = torch.zeros((1, 20, 240, 304), device=cfg.DEVICE)
    with torch.no_grad():
        padded = module.input_padder.pad_tensor_ev_repr(x)
        module.mdl.forward_backbone(x=padded, h_c={0: None, 1: None})

    phi = mon.collect_phi().numpy()
    spatial = mon.collect_phi_spatial().numpy()
    assert phi.shape[1] == 2112 and np.isfinite(phi).all()
    assert spatial.shape[1] == 1408 and np.isfinite(spatial).all()
