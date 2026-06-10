import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "event_corruption"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "HybridDetection"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "vmem_benchmark"))
from vmem_benchmark import benchmark_config as cfg
from event_corruption.pipeline.loader import load_histogram
from vmem_benchmark.corruption_wrap import apply_corruption_to_tensor

def get_resnet_feature_extractor(in_channels=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if in_channels != 3:
        w = model.conv1.weight
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight.data = w.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)

    class FeatureAndLogit(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.fc = base.fc
            self.base.fc = nn.Identity()

        def forward(self, x):
            f = self.base(x)
            l = self.fc(f)
            return f, l

    return FeatureAndLogit(model)


@torch.no_grad()
def run_batched(model, x, device, batch_size=64):
    feats, logits = [], []
    for chunk in torch.split(x, batch_size):
        f, l = model(chunk.to(device))
        feats.append(f.cpu())
        logits.append(l.cpu())
    return torch.cat(feats, dim=0), torch.cat(logits, dim=0)


def main():
    print("Extracting ANN Baselines (Event Image & Voxel Grid)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    resnet_img = get_resnet_feature_extractor(in_channels=2).to(device)
    resnet_img.eval()

    resnet_vox = get_resnet_feature_extractor(in_channels=20).to(device)
    resnet_vox.eval()

    out_dir_img = cfg.ANN_DIR / "event_image"
    out_dir_vox = cfg.ANN_DIR / "voxel_grid"
    out_dir_img.mkdir(parents=True, exist_ok=True)
    out_dir_vox.mkdir(parents=True, exist_ok=True)

    input_dir = getattr(cfg, "INPUT_DIR", None)
    if input_dir is None:
        input_dir = cfg.GEN1_ROOT / cfg.SPLIT

    label_files = sorted(input_dir.glob("*/labels_v2/labels.npz"))
    seq_dirs = [p.parent.parent for p in label_files]

    max_seq = getattr(cfg, "MAX_SEQUENCES", None)
    if max_seq is not None:
        seq_dirs = seq_dirs[:max_seq]

    runs = [("clean", None, 0)]
    for c_name in cfg.CORRUPTIONS:
        for sev in cfg.SEVERITIES:
            runs.append((f"{c_name}_L{sev}", c_name, sev))

    # Runs in the outer loop so each run's features are saved (and freed)
    # before the next one starts — accumulating all 31 runs of per-frame
    # features in RAM is not feasible.
    for run_name, c_name, severity in tqdm(runs, desc="Runs"):
        out_img = out_dir_img / f"{run_name}.pt"
        out_vox = out_dir_vox / f"{run_name}.pt"
        if out_img.exists() and out_vox.exists():
            continue

        img_feats, img_logits = [], []
        vox_feats, vox_logits = [], []

        for seq_idx, seq_dir in enumerate(tqdm(seq_dirs, desc=run_name, leave=False)):
            try:
                hist_np, _ = load_histogram(seq_dir)
            except Exception as e:
                print(f"  Skipping {seq_dir.name}: {e}")
                continue

            if c_name is not None:
                arr = apply_corruption_to_tensor(
                    torch.from_numpy(hist_np), c_name, severity, seed=[42, seq_idx]
                ).float()
            else:
                arr = torch.from_numpy(hist_np).float()

            # Scale uint8 counts into [0, 1] before the ImageNet-pretrained
            # backbone — raw counts up to 255 would saturate the features.
            arr = arr / 255.0

            C = arr.shape[1]
            c_half = C // 2

            # Event image PER FRAME: collapse the 10 time bins of each
            # polarity -> (N, 2, H, W). (Frames must NOT be summed together —
            # one feature per frame, aligned with the per-frame phi rows.)
            img_2c = torch.stack(
                [arr[:, :c_half].sum(dim=1), arr[:, c_half:].sum(dim=1)], dim=1
            )

            # Voxel grid PER FRAME: keep all 20 (bin, polarity) channels.
            vox = arr  # (N, 20, H, W)

            f_img, l_img = run_batched(resnet_img, img_2c, device)
            f_vox, l_vox = run_batched(resnet_vox, vox, device)
            img_feats.append(f_img);  img_logits.append(l_img)
            vox_feats.append(f_vox);  vox_logits.append(l_vox)

        if img_feats:
            torch.save(
                {"feat": torch.cat(img_feats, dim=0), "logit": torch.cat(img_logits, dim=0)},
                out_img,
            )
        if vox_feats:
            torch.save(
                {"feat": torch.cat(vox_feats, dim=0), "logit": torch.cat(vox_logits, dim=0)},
                out_vox,
            )

if __name__ == "__main__":
    main()
