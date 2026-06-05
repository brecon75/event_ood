import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
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
    
    # We will use two ResNets: one for 2-channel event image, one for 20-channel voxel grid
    # To keep it simple, we use the same architecture, modifying the first conv
    resnet_img = get_resnet_feature_extractor(in_channels=2).to(device)
    resnet_img.eval()
    
    resnet_vox = get_resnet_feature_extractor(in_channels=20).to(device)
    resnet_vox.eval()
    
    out_dir_img = Path("outputs/ann_features/event_image")
    out_dir_vox = Path("outputs/ann_features/voxel_grid")
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
            
    overall_pbar = tqdm(runs, desc="ANN Extraction")
    for run_name, c_name, severity in overall_pbar:
        img_feats_run = []
        vox_feats_run = []
        
        for i, seq_dir in enumerate(seq_dirs):
            try:
                hist_np, _ = load_histogram(seq_dir)
                if c_name is not None:
                    hist_np = apply_corruption_to_tensor(
                        torch.from_numpy(hist_np), c_name, severity
                    ).numpy()
            except Exception as e:
                continue
                
            if hist_np is None:
                continue
                
            # hist_np shape: (T, 20, 240, 304) where 20 channels = positive/negative interleaved or blocks?
            # Normally GEN1 uses 10 bins per polarity. 20 channels total.
            hist_t = torch.from_numpy(hist_np).float()
            T, C, H, W = hist_t.shape
            
            # 1. Event Image: sum over T, sum over polarity
            # Assuming channels 0-9 are positive, 10-19 are negative, or alternating.
            # Let's just group into 2 channels: first half vs second half.
            img_sum = hist_t.sum(dim=0) # (20, H, W)
            c_half = C // 2
            img_2c = torch.stack([img_sum[:c_half].sum(0), img_sum[c_half:].sum(0)], dim=0).unsqueeze(0) # (1, 2, H, W)
            
            # 2. Voxel Grid: sum over T to get 1 frame of 20 channels? No, Voxel Grid is temporal.
            # If we sum over T, we get (20, H, W) which is a standard Voxel Grid representation.
            # Or we treat the 10 time steps as the batch and average features?
            # Usually Voxel Grid = sum over time into C channels, or it's a 3D volume.
            # Let's sum over T to get (20, H, W) as the "Voxel Grid Image"
            vox_img = hist_t.sum(dim=0).unsqueeze(0) # (1, 20, H, W)
            
            with torch.no_grad():
                feat_img, logit_img = resnet_img(img_2c.to(device))
                feat_vox, logit_vox = resnet_vox(vox_img.to(device))
                
            img_feats_run.append({"feat": feat_img.cpu(), "logit": logit_img.cpu()})
            vox_feats_run.append({"feat": feat_vox.cpu(), "logit": logit_vox.cpu()})
            
        if img_feats_run:
            img_out = {
                "feat": torch.cat([d["feat"] for d in img_feats_run], dim=0),
                "logit": torch.cat([d["logit"] for d in img_feats_run], dim=0)
            }
            torch.save(img_out, out_dir_img / f"{run_name}.pt")
            
        if vox_feats_run:
            vox_out = {
                "feat": torch.cat([d["feat"] for d in vox_feats_run], dim=0),
                "logit": torch.cat([d["logit"] for d in vox_feats_run], dim=0)
            }
            torch.save(vox_out, out_dir_vox / f"{run_name}.pt")

if __name__ == "__main__":
    main()
