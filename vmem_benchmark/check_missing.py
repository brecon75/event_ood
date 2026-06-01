import torch
import benchmark_config as cfg
from model_loader import load_model, _remap_legacy_backbone_keys

module, _ = load_model()
ckpt = torch.load(str(cfg.CKPT_PATH), map_location='cpu')
sd_in_file = _remap_legacy_backbone_keys(ckpt.get('state_dict', ckpt))
model_keys = set(module.state_dict().keys())

missing = model_keys - set(sd_in_file.keys())
print("\n--- EXACT MISSING KEYS ---")
for k in sorted(list(missing)):
    print(f"  {k}")
print("--------------------------")
