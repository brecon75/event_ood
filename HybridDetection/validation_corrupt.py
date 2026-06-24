"""
Corruption-aware inference for the Hybrid (Neftci CVPR'25) detection model.

This is the MOTIVATION experiment for the Vmem-phi paper: take the *pretrained*
detector (no retraining), feed it clean vs. corrupted event input, and quantify
how far mAP drops. It establishes empirically that event-camera OOD corruptions
break a SOTA detector -- which is why detecting them matters.

It is an additive sibling of `validation.py`: it reuses the same Hydra config,
checkpoint loading, data module, and the Prophesee mAP evaluator unchanged. The
ONLY behavioural change is a one-method override that applies a corruption to the
event histogram (`data[DataType.EV_REPR]`) before the backbone sees it, so no
read-only HybridDetection model code is modified.

Usage (single run, append one row to results/neftci_map_degradation.csv):
    python validation_corrupt.py dataset=gen1 +experiment/gen1=no_lstm.yaml \
        dataset.path=/path/to/gen1 checkpoint=/path/to/gen1_mAP36.ckpt \
        checkpoint_load_strict=False use_test_set=True hardware.gpus=0 \
        +corruption=hot_pixel +severity=5

    # clean baseline (corruption pass-through):
    ... +corruption=clean

The driver `run_map_degradation.sh` loops over clean + the 6x5 grid; the raw CSV
is turned into the degradation table by `analysis/summarize_map_degradation.py`.
"""

import os

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy('file_system')

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelSummary

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from modules.detection_hybrid import Module_Hybrid
from data.utils.types import DataType

# --- wire the project's corruption library onto sys.path (no benchmark_config dep) ---
# HybridDetection/ is a sub-repo of the project root; event_corruption/ is a sibling.
_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent
sys.path.insert(0, str(REPO_ROOT / "event_corruption"))
from corrupt.registry import apply_corruption, CORRUPTIONS, SEVERITIES  # noqa: E402

RESULTS_CSV = REPO_ROOT / "vmem_benchmark" / "outputs" / "results" / "neftci_map_degradation.csv"
# The Prophesee evaluator (utils/.../coco_eval.py) returns these keys; they are
# logged with a 'test/' or 'val/' prefix by run_psee_evaluator.
AP_KEYS = ('AP', 'AP_50', 'AP_75', 'AP_S', 'AP_M', 'AP_L')


def _corrupt_batch_tensor(ev: torch.Tensor, name: str, severity: int,
                          base_seed: int, step: int) -> torch.Tensor:
    """Corrupt one (B, 20, H, W) event-histogram tensor, sample-by-sample.

    Each sample in the batch is an independent recording, so it gets its own
    seed [base_seed, step, b] -- otherwise every sequence would receive the
    identical hot-pixel set / jitter shift / flood patch.
    """
    device = ev.device
    arr = ev.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        # Round + clip (not bare astype) so counts > 255 don't wrap mod 256.
        arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    out = np.empty_like(arr)
    for b in range(arr.shape[0]):
        rng = np.random.default_rng([base_seed, step, b])
        # corruptions expect (N, 20, H, W); a single sample is (20, H, W) -> add N axis
        out[b] = apply_corruption(arr[b][None], None, name, severity, rng)[0]
    return torch.from_numpy(out).to(device=device, dtype=ev.dtype)


class Module_Hybrid_Corrupt(Module_Hybrid):
    """Hybrid detector with input corruption injected at the single data hook.

    `get_data_from_batch` is the one place every step reads its event tensors
    from, so overriding it corrupts the input for train/val/test uniformly with
    zero duplication of the step logic.
    """

    def configure_corruption(self, name: str, severity: int, seed: int = 42):
        self._corruption_name = name
        self._corruption_severity = int(severity)
        self._corruption_seed = int(seed)
        self._corruption_step = 0
        self._corruption_active = name not in (None, "", "clean", "none")
        return self

    def get_data_from_batch(self, batch):
        data = batch['data']
        if getattr(self, "_corruption_active", False):
            ev_list = data[DataType.EV_REPR]  # list over time of (B, 20, H, W)
            data[DataType.EV_REPR] = [
                _corrupt_batch_tensor(ev, self._corruption_name,
                                      self._corruption_severity,
                                      self._corruption_seed,
                                      self._corruption_step + i)
                for i, ev in enumerate(ev_list)
            ]
            self._corruption_step += len(ev_list)
        return data


def _remap_legacy_backbone_keys(state_dict: dict) -> dict:
    """Map older backbone checkpoint keys to current names (copied verbatim from
    validation.py so this script is self-contained)."""
    remapped = dict(state_dict)
    for suffix in ('conv3x3_dws.weight', 'conv3x3_dws.bias', 'conv1x1.weight', 'conv1x1.bias'):
        old_k = f'mdl.backbone.lstm_3.{suffix}'
        new_k = f'mdl.backbone.lstm_1.{suffix}'
        if new_k not in remapped and old_k in remapped:
            remapped[new_k] = remapped[old_k]
    for block in ('1', '2'):
        for suffix in (
            'conv.conv.weight', 'conv.norm.weight', 'conv.norm.bias',
            'conv.norm.running_mean', 'conv.norm.running_var',
            'conv.norm.num_batches_tracked',
        ):
            old_k = f'mdl.backbone.ann_features_{block}_2.0.{suffix}'
            new_k = f'mdl.backbone.ann_features_{block}.1.{suffix}'
            if new_k not in remapped and old_k in remapped:
                remapped[new_k] = remapped[old_k]
    return remapped


def _extract_ap(metrics: dict) -> dict:
    """Pull the AP_* keys out of Lightning's returned metric dict, stripping the
    'test/'/'val/' prefix. Returns {key: float} with NaN for anything missing."""
    flat = {}
    for k, v in metrics.items():
        short = k.split('/')[-1]
        if short in AP_KEYS:
            flat[short] = float(v.item() if hasattr(v, 'item') else v)
    return {k: flat.get(k, float('nan')) for k in AP_KEYS}


def _append_row(row: dict, results_csv: Path):
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ['corruption', 'severity', *AP_KEYS]
    write_header = not results_csv.exists()
    with open(results_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in fields})


@hydra.main(config_path='config', config_name='val', version_base='1.2')
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    corruption = str(config.get('corruption', 'clean'))
    severity = int(config.get('severity', 5))
    corruption_seed = int(config.get('corruption_seed', 42))
    is_clean = corruption in ('clean', 'none')
    if not is_clean:
        if corruption not in CORRUPTIONS:
            raise ValueError(f"Unknown corruption '{corruption}'. "
                             f"Valid: clean | {list(CORRUPTIONS)}")
        if severity not in SEVERITIES:
            raise ValueError(f"severity must be in {SEVERITIES}, got {severity}")

    print('------ Corruption run ------')
    print(f'corruption={corruption}  severity={severity}  seed={corruption_seed}')
    print('----------------------------')

    gpus = config.hardware.gpus
    assert isinstance(gpus, int), 'no more than 1 GPU supported'
    gpus = [gpus]

    data_module = fetch_data_module(config=config)
    logger = CSVLogger(save_dir='./validation_logs')
    ckpt_path = Path(config.checkpoint)

    # Build the corruption-aware module instead of the stock one.
    module = Module_Hybrid_Corrupt(config).configure_corruption(
        corruption, severity, corruption_seed)

    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    state_dict = _remap_legacy_backbone_keys(state_dict)
    model_keys = set(module.state_dict().keys())
    filtered_state = {k: v for k, v in state_dict.items() if k in model_keys}
    module.load_state_dict(filtered_state, strict=config.checkpoint_load_strict)

    trainer = pl.Trainer(
        accelerator='gpu',
        callbacks=[ModelSummary(max_depth=2)],
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
        move_metrics_to_cpu=False,
        # smoke control: pass `+limit_test_batches=4` (or limit_val_batches) on the CLI
        limit_test_batches=config.get('limit_test_batches', 1.0),
        limit_val_batches=config.get('limit_val_batches', 1.0),
    )
    with torch.inference_mode():
        if config.use_test_set:
            results = trainer.test(model=module, datamodule=data_module)
        else:
            results = trainer.validate(model=module, datamodule=data_module)

    metrics = results[0] if results else dict(trainer.callback_metrics)
    ap = _extract_ap(metrics)
    # `output_dir` (Hydra `+output_dir=...`) redirects the results CSV; the test
    # data folder is the standard `dataset.path`.
    out_dir = config.get('output_dir', None)
    results_csv = (Path(out_dir) / 'neftci_map_degradation.csv') if out_dir else RESULTS_CSV
    row = {'corruption': corruption, 'severity': severity, **ap}
    _append_row(row, results_csv)
    print(f'[map-degradation] {corruption} sev{severity}: '
          f"AP={ap['AP']:.4f} AP_50={ap['AP_50']:.4f}  -> {results_csv}")


if __name__ == '__main__':
    main()
