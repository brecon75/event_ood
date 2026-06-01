import torch
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "event_corruption"))

from model_loader import load_model
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode
from spikingjelly.clock_driven import functional

module, backbone = load_model("cpu")

def hook(module, input, output):
    print("Has v:", hasattr(module, 'v'))
    if hasattr(module, 'v') and module.v is not None:
        print("v shape:", module.v.shape)
    print("Has v_seq:", hasattr(module, 'v_seq'))
    if hasattr(module, 'v_seq') and module.v_seq is not None:
        print("v_seq shape:", module.v_seq.shape)
    # Stop after one layer
    sys.exit(0)

for name, mod in backbone.named_modules():
    if isinstance(mod, MultiStepParametricLIFNode):
        mod.register_forward_hook(hook)

functional.reset_net(backbone)
h_c = {0: None, 1: None}
with torch.no_grad():
    module.mdl.forward_backbone(x=torch.randn((1, 20, 240, 304)).float(), h_c=h_c)
