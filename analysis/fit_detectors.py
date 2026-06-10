import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
from tqdm import tqdm
from pathlib import Path
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.representation_ablation import load_all_features, extract_representation
from analysis.vmem_utils import (
    MAX_FIT_SAMPLES, TRAIN_RATIO, split_boundary, load_phi_seq_lens, _subsample
)
from analysis.vmem_models import RealNVP, train_flow_model

class SimpleAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

def fit_ae(X, epochs=50, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_dim = X.shape[1]
    model = SimpleAE(input_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    pbar = tqdm(range(epochs), desc="Training Autoencoder (fit)", leave=False)
    for epoch in pbar:
        for batch in loader:
            x = batch[0]
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()

    return model.cpu()

def main():
    print("Fitting detectors on clean data...")
    all_feats = load_all_features()

    if 'clean' not in all_feats:
        print("Error: 'clean' run not found. Cannot fit detectors.")
        return

    # We fit on the membrane_fused representation as the primary target, fallback to full_membrane
    rep = 'membrane_fused'
    X_clean = extract_representation(all_feats['clean'], rep)
    if X_clean is None:
        print("Warning: membrane_fused not found, falling back to full_membrane")
        rep = 'full_membrane'
        X_clean = extract_representation(all_feats['clean'], rep)
    if X_clean is None:
        print("Error: neither membrane_fused nor full_membrane available for the "
              "clean run. Run extract.py / fusion_features.py first.")
        return

    # Sequence-aware train/eval split, shared with evaluate_detectors.py.
    # Detectors are fitted ONLY on the train portion; the eval portion is the
    # held-out clean negative set at evaluation time.
    seq_lens = load_phi_seq_lens("clean")
    cut = split_boundary(len(X_clean), TRAIN_RATIO, seq_lens)
    X_train = X_clean[:cut]
    print(f"Clean split: {cut} train / {len(X_clean) - cut} eval frames "
          f"({'sequence-aligned' if seq_lens else 'contiguous fallback'}).")

    out_dir = cfg.DETECTOR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record the split so evaluation uses exactly the same boundary.
    with open(out_dir / "split.json", "w") as f:
        json.dump({
            "representation": rep,
            "n_clean": int(len(X_clean)),
            "train_end": int(cut),
            "train_ratio": TRAIN_RATIO,
            "sequence_aligned": bool(seq_lens),
        }, f, indent=2)

    detectors = {}

    print("Fitting Mahalanobis...")
    try:
        cov = LedoitWolf().fit(X_train)
        detectors['mahalanobis'] = cov
    except Exception as e:
        print(f"Mahalanobis failed: {e}")

    print("Fitting PCA (Linear AE)...")
    pca = PCA(n_components=min(X_train.shape[0], X_train.shape[1], 64)).fit(X_train)
    detectors['pca'] = pca

    print("Fitting kNN...")
    # Subsample the reference set: exact kNN against the full clean split is
    # computationally intractable (brute-force in ~2000-D).
    X_knn = _subsample(X_train, n=MAX_FIT_SAMPLES)
    k_nn = max(1, min(5, X_knn.shape[0]))
    knn = NearestNeighbors(n_neighbors=k_nn).fit(X_knn)
    detectors['knn'] = knn

    print("Fitting GMM...")
    try:
        n_comp = min(5, X_train.shape[0])  # clamp components to available samples
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full',
                              reg_covar=1e-4, random_state=42).fit(X_train)
        detectors['gmm'] = gmm
    except Exception as e:
        print(f"GMM failed: {e}")

    print("Fitting OCSVM... (might be slow)")
    # sample for SVM if too large
    X_svm = X_train[np.random.default_rng(42).choice(
        X_train.shape[0], min(X_train.shape[0], 5000), replace=False)]
    ocsvm = OneClassSVM(gamma='auto').fit(X_svm)
    detectors['ocsvm'] = ocsvm

    print("Fitting AutoEncoder...")
    ae = fit_ae(X_train)
    torch.save(ae.state_dict(), out_dir / "ae.pt")

    print("Fitting Normalizing Flow (PCA + RealNVP)...")
    try:
        flow_pca = PCA(n_components=min(50, X_train.shape[0], X_train.shape[1]),
                       random_state=42).fit(_subsample(X_train))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        flow = train_flow_model(flow_pca.transform(X_train), device=device)
        torch.save(flow.cpu().state_dict(), out_dir / "flow.pt")
        joblib.dump(flow_pca, out_dir / "flow_pca.joblib")
    except Exception as e:
        print(f"Normalizing Flow failed: {e}")

    for name, model in detectors.items():
        joblib.dump(model, out_dir / f"{name}.joblib")

    print(f"Detectors fitted and saved to {out_dir}")

if __name__ == "__main__":
    main()
