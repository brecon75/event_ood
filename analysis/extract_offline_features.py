import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.vmem_utils import LAYER_SPECS, split_train_eval
from analysis.vmem_models import TemporalAutoencoder, train_temporal_ae_model

def extract_margin_hist(tgap, theta=1.0, bins=20):
    """
    Computes margin histogram from compressed GAP trajectories.
    margin = Vmem - threshold.
    Histogram over [-2theta, +2theta] with 20 bins per layer.
    """
    parts = []
    # tgap shape: (N, T, sum_C)
    N, T, sum_C = tgap.shape
    
    c_offset = 0
    for idx, spec in enumerate(LAYER_SPECS):
        C = spec["C"]
        V = tgap[:, :, c_offset : c_offset + C]  # (N, T, C)
        c_offset += C
        
        margin = (V - theta).reshape(N, -1)  # (N, T*C)

        # Vectorised histogram over all N samples in one call
        boundaries = torch.linspace(-2 * theta, 2 * theta, bins + 1)[1:-1]  # (bins-1,) interior edges
        bin_idx = torch.bucketize(margin.contiguous(), boundaries)  # (N, T*C), values in [0, bins-1]
        layer_hists = torch.zeros(N, bins)
        layer_hists.scatter_add_(1, bin_idx, torch.ones_like(margin))
        layer_hists = layer_hists / (layer_hists.sum(dim=1, keepdim=True) + 1e-8)
        parts.append(layer_hists)
        
    if not parts:
        return None
    return torch.cat(parts, dim=-1).numpy()  # (N, n_layers * bins)


def main():
    print("Extracting offline features from compressed GAP trajectories...")
    
    tgap_dir = cfg.OUTPUT_DIR / "temporal_gap"
    out_dir_margin = cfg.OUTPUT_DIR / "features/margin_hist"
    out_dir_latent = cfg.OUTPUT_DIR / "features/trajectory_latent"
    out_dir_margin.mkdir(parents=True, exist_ok=True)
    out_dir_latent.mkdir(parents=True, exist_ok=True)
    
    clean_gap_file = tgap_dir / "clean.pt"
    if not clean_gap_file.exists():
        print(f"Error: clean.pt temporal GAP file not found in {tgap_dir}.")
        return
        
    clean_data = torch.load(clean_gap_file, weights_only=True, map_location="cpu")
    clean_tgap = clean_data["temporal_gap"]  # shape (N, T, 704)

    # Train only on the clean TRAIN split so the held-out clean frames used
    # as evaluation negatives downstream were never seen by the AE.
    clean_tgap_train, _ = split_train_eval(
        clean_tgap, seq_lens=clean_data.get("seq_lens", None))

    print(f"Training Temporal AE on clean GAP trajectories "
          f"({len(clean_tgap_train)}/{len(clean_tgap)} train frames)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ae = train_temporal_ae_model(clean_tgap_train, epochs=100, device=device)
    ae.eval()
    
    def extract_latent(tgap, batch_size=4096):
        # Process in batches — a full run is ~10 GB and would OOM the GPU
        # if moved over in one piece.
        outs = []
        with torch.no_grad():
            for chunk in torch.split(tgap, batch_size):
                z = ae.encoder(chunk.to(device))
                outs.append(z.view(z.shape[0], -1).cpu())
        return torch.cat(outs, dim=0).numpy()
        
    print("Extracting margin_hist and trajectory_latent for all runs...")
    
    for f in tqdm(list(tgap_dir.glob("*.pt")), desc="Extracting offline features"):
        if f.name.startswith("_tmp_"):
            continue
        run_name = f.stem
        try:
            d = torch.load(f, weights_only=True, map_location="cpu")
            tgap = d["temporal_gap"]  # shape (N, T, 704)
            
            # Margin Histogram
            m_hist = extract_margin_hist(tgap)
            if m_hist is not None:
                torch.save({"margin_hist": torch.from_numpy(m_hist)}, out_dir_margin / f"{run_name}.pt")
                
            # Trajectory Latent
            latent = extract_latent(tgap)
            if latent is not None:
                torch.save({"trajectory_latent": torch.from_numpy(latent)}, out_dir_latent / f"{run_name}.pt")
                
        except Exception as e:
            print(f"Failed on {run_name}: {e}")
            
    print("Offline extraction complete.")

if __name__ == "__main__":
    main()
