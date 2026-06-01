"""
model_loader.py — Load the HybridDetection checkpoint and return the bare
backbone nn.Module ready for hook registration.

Usage (standalone audit):
    python model_loader.py
"""
import sys
import torch
from pathlib import Path

# ── add HybridDetection to sys.path so its imports resolve ──────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "HybridDetection"))

from omegaconf import OmegaConf
from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_model_module
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode

import benchmark_config as cfg


# ── legacy key remapping (mirrors validation.py exactly) ────────────────────

def _remap_legacy_backbone_keys(state_dict: dict) -> dict:
    remapped = dict(state_dict)
    for suffix in (
        "conv3x3_dws.weight", "conv3x3_dws.bias",
        "conv1x1.weight",     "conv1x1.bias",
    ):
        old_k = f"mdl.backbone.lstm_3.{suffix}"
        new_k = f"mdl.backbone.lstm_1.{suffix}"
        if new_k not in remapped and old_k in remapped:
            remapped[new_k] = remapped[old_k]

    for block in ("1", "2"):
        for suffix in (
            "conv.conv.weight",
            "conv.norm.weight",
            "conv.norm.bias",
            "conv.norm.running_mean",
            "conv.norm.running_var",
            "conv.norm.num_batches_tracked",
        ):
            old_k = f"mdl.backbone.ann_features_{block}_2.0.{suffix}"
            new_k = f"mdl.backbone.ann_features_{block}.1.{suffix}"
            if new_k not in remapped and old_k in remapped:
                remapped[new_k] = remapped[old_k]

    return remapped


# ── public API ───────────────────────────────────────────────────────────────

def load_model(device: str = cfg.DEVICE):
    """
    Return (module, backbone) where:
      module   — full Module_Hybrid (pl.LightningModule)
      backbone — module.mdl.backbone (the SNN+ANN backbone nn.Module)

    The checkpoint is loaded with strict=False so extra/missing keys
    (e.g. LSTM legacy renames) don't raise.
    """
    # Build a minimal Hydra-compatible config using the gen1 val config
    hydra_cfg = OmegaConf.load(cfg.HYBRID_DIR / "config" / "val.yaml")
    dataset_cfg = OmegaConf.load(cfg.HYBRID_DIR / "config" / "dataset" / "gen1.yaml")
    dataset_base_cfg = OmegaConf.load(cfg.HYBRID_DIR / "config" / "dataset" / "base.yaml")
    dataset_cfg = OmegaConf.merge(dataset_base_cfg, dataset_cfg)

    model_cfg_dir = cfg.HYBRID_DIR / "config" / "model"
    model_cfg = OmegaConf.load(model_cfg_dir / "hybrid.yaml")
    
    # Load the default parameters for hybrid_yolox
    default_model_cfg = OmegaConf.load(model_cfg_dir / "hybrid_yolox" / "default.yaml")
    # The default.yaml has a 'model' key at the top, we want to merge its content
    model_cfg = OmegaConf.merge(model_cfg, default_model_cfg.model)

    # Find the experiment override if present
    experiment_dir = cfg.HYBRID_DIR / "config" / "experiment" / "gen1"
    experiment_cfg = OmegaConf.create({})
    if experiment_dir.exists():
        small_yaml = experiment_dir / "small.yaml"
        if small_yaml.exists():
            experiment_cfg = OmegaConf.load(small_yaml)

    full_config = OmegaConf.merge(
        OmegaConf.create({
            "dataset":  dataset_cfg,
            "model":    model_cfg,
            "hardware": {"gpus": 0, "num_workers": {"eval": 4}},
            "batch_size": {"eval": cfg.BATCH_SIZE},
            "training": {"precision": 32},
            "checkpoint": str(cfg.CKPT_PATH),
            "checkpoint_load_strict": False,
            "use_test_set": True,
        }),
        experiment_cfg,
    )
    dynamically_modify_train_config(full_config)

    module = fetch_model_module(config=full_config)

    ckpt = torch.load(str(cfg.CKPT_PATH), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = _remap_legacy_backbone_keys(state_dict)

    model_keys    = set(module.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered_state.keys())
    unexpected = set(state_dict.keys()) - model_keys
    module.load_state_dict(filtered_state, strict=False)

    print(f"[model_loader] Loaded checkpoint: {cfg.CKPT_PATH.name}")
    print(f"  Keys loaded   : {len(filtered_state)}")
    print(f"  Missing keys  : {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")

    module = module.to(device).eval()
    return module, module.mdl.backbone


def audit_plif_layers(model) -> int:
    """
    Print every MultiStepParametricLIFNode in the model with its index,
    qualified name, and learned tau value.  Returns total count.
    """
    print("\n-- PLIF layer audit ------------------------------------------")
    idx = 0
    for name, mod in model.named_modules():
        if isinstance(mod, MultiStepParametricLIFNode):
            print(f"  [{idx}]  {name:<55}")
            idx += 1
    print(f"\n  Total PLIF nodes found: {idx}")
    print("-------------------------------------------------------------\n")
    return idx


if __name__ == "__main__":
    module, backbone = load_model()
    n = audit_plif_layers(module)
    print(f"Set PLIF_LAYERS in config.py to a subset of [0..{n-1}] to reduce memory.")
