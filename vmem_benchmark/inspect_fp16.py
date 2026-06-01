import torch
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from model_loader import load_model
from spikingjelly.clock_driven import functional

module, backbone = load_model("cuda")
functional.reset_net(backbone)
h_c = {0: None, 1: None}
batch = torch.randn((1, 20, 240, 304)).float().cuda()

# Warmup
with torch.no_grad():
    module.mdl.forward_backbone(x=batch, h_c=h_c)

# FP32 Test
functional.reset_net(backbone)
h_c = {0: None, 1: None}
t0 = time.time()
with torch.no_grad():
    for _ in range(10):
        module.mdl.forward_backbone(x=batch, h_c=h_c)
print("FP32 time:", time.time() - t0)

# FP16 Test
functional.reset_net(backbone)
h_c = {0: None, 1: None}
t0 = time.time()
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
    for _ in range(10):
        module.mdl.forward_backbone(x=batch, h_c=h_c)
print("FP16 time:", time.time() - t0)
