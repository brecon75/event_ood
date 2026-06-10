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

    # Accumulators: run_name -> list of {feat, logit}
    img_feats_all = {run_name: [] for run_name, _, _ in runs}
    vox_feats_all = {run_name: [] for run_name, _, _ in runs}

    # Outer loop over sequences — each sequence is loaded exactly once
    for seq_dir in tqdm(seq_dirs, desc="Sequences"):
        try:
            hist_np, _ = load_histogram(seq_dir)
        except Exception:
            continue

        C = hist_np.shape[1]
        c_half = C // 2

        for run_name, c_name, severity in runs:
            if c_name is not None:
                arr = apply_corruption_to_tensor(
                    torch.from_numpy(hist_np), c_name, severity
                ).float()
            else:
                arr = torch.from_numpy(hist_np).float()

            # Event image: sum over T, split into 2 polarity channels
            img_sum = arr.sum(dim=0)  # (C, H, W)
            img_2c = torch.stack(
                [img_sum[:c_half].sum(0), img_sum[c_half:].sum(0)], dim=0
            ).unsqueeze(0)  # (1, 2, H, W)

            # Voxel grid: sum over T into 20 channels
            vox_img = arr.sum(dim=0).unsqueeze(0)  # (1, 20, H, W)

            with torch.no_grad():
                feat_img, logit_img = resnet_img(img_2c.to(device))
                feat_vox, logit_vox = resnet_vox(vox_img.to(device))

            img_feats_all[run_name].append({"feat": feat_img.cpu(), "logit": logit_img.cpu()})
            vox_feats_all[run_name].append({"feat": feat_vox.cpu(), "logit": logit_vox.cpu()})

    # Save one file per run
    print("Saving...")
    for run_name, _, _ in tqdm(runs, desc="Saving runs"):
        if img_feats_all[run_name]:
            torch.save(
                {
                    "feat": torch.cat([d["feat"] for d in img_feats_all[run_name]], dim=0),
                    "logit": torch.cat([d["logit"] for d in img_feats_all[run_name]], dim=0),
                },
                out_dir_img / f"{run_name}.pt",
            )
        if vox_feats_all[run_name]:
            torch.save(
                {
                    "feat": torch.cat([d["feat"] for d in vox_feats_all[run_name]], dim=0),
                    "logit": torch.cat([d["logit"] for d in vox_feats_all[run_name]], dim=0),
                },
                out_dir_vox / f"{run_name}.pt",
            )

if __name__ == "__main__":
    main()
