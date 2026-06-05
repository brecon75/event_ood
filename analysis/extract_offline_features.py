import torch
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.vmem_utils import LAYER_SPECS
from analysis.vmem_models import TemporalAutoencoder, train_temporal_ae_model, prepare_temporal_ae_input

def extract_margin_hist(trajs, theta=1.0, bins=20):
    """
    Computes margin histogram from raw trajectories.
    margin = Vmem - threshold.
    Histogram over [-2theta, +2theta] with 20 bins per layer.
    """
    parts = []
    for idx in sorted(trajs.keys()):
        V = trajs[idx].float() # (T, N, D)
        T, N, D = V.shape
        
        margin = V - theta
        margin = margin.view(T*N, D)
        
        # We need histogram per sample (N)
        # Reshape to (N, T*D)
        margin = margin.view(T, N, D).transpose(0, 1).reshape(N, -1)
        
        hists = []
        for n in range(N):
            h = torch.histc(margin[n], bins=bins, min=-2*theta, max=2*theta)
            h = h / (h.sum() + 1e-8)
            hists.append(h)
            
        layer_hists = torch.stack(hists, dim=0) # (N, bins)
        parts.append(layer_hists)
        
    if not parts:
        return None
    return torch.cat(parts, dim=-1).numpy() # (N, n_layers * bins)


def main():
    print("Extracting offline features from trajectories...")
    
    out_dir_margin = Path("outputs/features/margin_hist")
    out_dir_latent = Path("outputs/features/trajectory_latent")
    out_dir_margin.mkdir(parents=True, exist_ok=True)
    out_dir_latent.mkdir(parents=True, exist_ok=True)
    
    clean_traj_file = cfg.TRAJ_DIR / "clean.pt"
    if not clean_traj_file.exists():
        print("Error: clean.pt trajectory not found.")
        return
        
    clean_data = torch.load(clean_traj_file, weights_only=True, map_location="cpu")
    clean_trajs = clean_data["trajs"]
    
    print("Training Temporal AE on clean trajectories...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = train_temporal_ae_model(clean_trajs, epochs=100, device=device)
    ae.eval()
    
    def extract_latent(trajs):
        x = prepare_temporal_ae_input(trajs).to(device)
        with torch.no_grad():
            z = ae.encoder(x)
            z = z.view(z.shape[0], -1) # Flatten (N, latent_dim)
        return z.cpu().numpy()
        
    print("Extracting margin_hist and trajectory_latent for all runs...")
    
    for f in cfg.TRAJ_DIR.glob("*.pt"):
        run_name = f.stem
        try:
            d = torch.load(f, weights_only=True, map_location="cpu")
            trajs = d["trajs"]
            
            # Margin Histogram
            m_hist = extract_margin_hist(trajs)
            if m_hist is not None:
                torch.save({"margin_hist": torch.from_numpy(m_hist)}, out_dir_margin / f"{run_name}.pt")
                
            # Trajectory Latent
            latent = extract_latent(trajs)
            if latent is not None:
                torch.save({"trajectory_latent": torch.from_numpy(latent)}, out_dir_latent / f"{run_name}.pt")
                
        except Exception as e:
            print(f"Failed on {run_name}: {e}")
            
    print("Offline extraction complete.")

if __name__ == "__main__":
    main()
