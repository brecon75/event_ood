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
    X_train = extract_representation(all_feats['clean'], 'membrane_fused')
    if X_train is None:
        print("Warning: membrane_fused not found, falling back to full_membrane")
        X_train = extract_representation(all_feats['clean'], 'full_membrane')
    
    out_dir = cfg.DETECTOR_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    detectors = {}
    
    print("Fitting Mahalanobis...")
    try:
        cov = LedoitWolf().fit(X_train)
        detectors['mahalanobis'] = cov
    except Exception as e:
        print(f"Mahalanobis failed: {e}")
        
    print("Fitting PCA (Linear AE)...")
    pca = PCA(n_components=min(X_train.shape[1], 64)).fit(X_train)
    detectors['pca'] = pca
    
    print("Fitting kNN...")
    k_nn = min(5, X_train.shape[0])  # clamp k to available samples
    knn = NearestNeighbors(n_neighbors=k_nn).fit(X_train)
    detectors['knn'] = knn
    
    print("Fitting GMM...")
    try:
        n_comp = min(5, X_train.shape[0])  # clamp components to available samples
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42).fit(X_train)
        detectors['gmm'] = gmm
    except Exception as e:
        print(f"GMM failed: {e}")
        
    print("Fitting OCSVM... (might be slow)")
    # sample for SVM if too large
    X_svm = X_train[np.random.default_rng(42).choice(X_train.shape[0], min(X_train.shape[0], 5000), replace=False)]
    ocsvm = OneClassSVM(gamma='auto').fit(X_svm)
    detectors['ocsvm'] = ocsvm
    
    print("Fitting AutoEncoder...")
    ae = fit_ae(X_train)
    torch.save(ae.state_dict(), out_dir / "ae.pt")
    
    for name, model in detectors.items():
        joblib.dump(model, out_dir / f"{name}.joblib")
        
    print(f"Detectors fitted and saved to {out_dir}")

if __name__ == "__main__":
    main()
