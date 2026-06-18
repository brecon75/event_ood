import json
import time
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
    MAX_FIT_SAMPLES, TRAIN_RATIO, split_boundary, load_phi_seq_lens, _subsample,
    _cap_subset, GMM_FIT_SAMPLES, OCSVM_FIT_SAMPLES,
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

    # Keep the full training set on CPU (~2 GB at full data) and move only each
    # 256-row batch to the GPU. Uploading all of X at once OOMs a small GPU; it
    # still trains on 100% of the data, just not resident all at once.
    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    pbar = tqdm(range(epochs), desc="Training Autoencoder (fit)", leave=False)
    for epoch in pbar:
        for batch in loader:
            x = batch[0].to(device)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_features = int(X_train.shape[1])
    n_train = int(len(X_train))

    # Per-detector provenance: exactly which detector was fit on HOW MUCH data,
    # whether a cap was applied, and how long it took. Saved to fit_manifest.json
    # so the run is auditable after the fact.
    manifest = {
        "representation": rep,
        "n_clean_total": int(len(X_clean)),
        "n_train_available": n_train,
        "n_eval_held_out": int(len(X_clean) - cut),
        "n_features": n_features,
        "device": device,
        "fast_mode": "--fast" in sys.argv,
        "detectors": {},
    }

    def _record(name, n_used, capped, t0, extra=None):
        entry = {"n_fit_samples": int(n_used),
                 "fraction_of_train": round(n_used / max(1, n_train), 4),
                 "n_features": n_features,
                 "capped": bool(capped),
                 "fit_seconds": round(time.time() - t0, 2),
                 "device": device}
        if extra:
            entry.update(extra)
        manifest["detectors"][name] = entry
        print(f"  [fit] {name:<12} {n_used:>8}/{n_train} train samples "
              f"({entry['fraction_of_train']*100:.1f}%) x {n_features}D | "
              f"{'CAPPED' if capped else 'FULL'} | {entry['fit_seconds']}s",
              flush=True)

    print(f"Fitting 6 detectors on '{rep}' "
          f"({n_train} train x {n_features}D, device={device})...")

    print("Fitting Mahalanobis (Ledoit-Wolf, FULL data)...", flush=True)
    try:
        t0 = time.time()
        cov = LedoitWolf().fit(X_train)
        detectors['mahalanobis'] = cov
        _record('mahalanobis', n_train, False, t0)
    except Exception as e:
        print(f"Mahalanobis failed: {e}")

    print("Fitting PCA (FULL data)...", flush=True)
    t0 = time.time()
    pca = PCA(n_components=min(X_train.shape[0], X_train.shape[1], 64)).fit(X_train)
    detectors['pca'] = pca
    _record('pca', n_train, False, t0, {"n_components": int(pca.n_components_)})

    print("Fitting kNN (FULL reference)...", flush=True)
    # Reference set = the FULL clean train split (cap removed). _subsample is a
    # no-op outside --fast; brute-force kNN in ~2000-D against all clean frames
    # is the infinite-compute setting.
    t0 = time.time()
    X_knn = _subsample(X_train, n=MAX_FIT_SAMPLES)
    k_nn = max(1, min(5, X_knn.shape[0]))
    knn = NearestNeighbors(n_neighbors=k_nn).fit(X_knn)
    detectors['knn'] = knn
    _record('knn', len(X_knn), len(X_knn) < n_train, t0, {"k": int(k_nn)})

    print(f"Fitting GMM (CAPPED at {GMM_FIT_SAMPLES})...", flush=True)
    try:
        # GMM fits on a capped subset (full-cov EM in 2112-D is impractical on
        # the full split); every other detector still uses all clean data.
        t0 = time.time()
        X_gmm = _cap_subset(X_train, GMM_FIT_SAMPLES)
        n_comp = min(5, X_gmm.shape[0])  # clamp components to available samples
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full',
                              reg_covar=1e-4, random_state=42,
                              max_iter=1000).fit(X_gmm)  # ceiling raised for convergence
        detectors['gmm'] = gmm
        _record('gmm', len(X_gmm), len(X_gmm) < n_train, t0,
                {"n_components": int(n_comp), "converged": bool(gmm.converged_),
                 "n_iter": int(gmm.n_iter_)})
    except Exception as e:
        print(f"GMM failed: {e}")

    print(f"Fitting OCSVM (RBF, CAPPED at {OCSVM_FIT_SAMPLES})...", flush=True)
    try:
        # RBF OneClassSVM fit is ~O(n^2); cap it (scoring is GPU-chunked, so the
        # cap only bounds the one-time fit). Re-enabled so the scored ocsvm model
        # is consistent with the current data instead of a stale leftover file.
        t0 = time.time()
        X_svm = _cap_subset(X_train, OCSVM_FIT_SAMPLES)
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_svm)
        detectors['ocsvm'] = ocsvm
        _record('ocsvm', len(X_svm), len(X_svm) < n_train, t0,
                {"n_support_vectors": int(ocsvm.support_vectors_.shape[0])})
    except Exception as e:
        print(f"OCSVM failed: {e}")

    print("Fitting AutoEncoder (FULL data, batched)...", flush=True)
    t0 = time.time()
    ae = fit_ae(X_train)
    torch.save(ae.state_dict(), out_dir / "ae.pt")
    _record('ae', n_train, False, t0)

    print("Fitting Normalizing Flow (PCA + RealNVP, FULL data)...", flush=True)
    try:
        t0 = time.time()
        flow_pca = PCA(n_components=min(50, X_train.shape[0], X_train.shape[1]),
                       random_state=42).fit(_subsample(X_train))
        flow = train_flow_model(flow_pca.transform(X_train), device=device)
        torch.save(flow.cpu().state_dict(), out_dir / "flow.pt")
        joblib.dump(flow_pca, out_dir / "flow_pca.joblib")
        _record('flow', n_train, False, t0,
                {"pca_components": int(flow_pca.n_components_)})
    except Exception as e:
        print(f"Normalizing Flow failed: {e}")

    print("Saving detectors...", flush=True)
    for name, model in detectors.items():
        joblib.dump(model, out_dir / f"{name}.joblib")

    with open(out_dir / "fit_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Detectors fitted and saved to {out_dir}")
    print(f"Fit provenance written to {out_dir / 'fit_manifest.json'}")

if __name__ == "__main__":
    main()
