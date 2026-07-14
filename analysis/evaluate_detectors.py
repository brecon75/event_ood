import json
import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vmem_benchmark import benchmark_config as cfg
from analysis.representation_ablation import load_all_features, extract_representation, calc_fpr95
from analysis.vmem_utils import (
    TRAIN_RATIO, split_boundary, load_phi_seq_lens, chunked_apply, knn_score,
    device_for, maha_d2, held_out_eval, auroc_aupr_fpr95,
)
from analysis.fit_detectors import SimpleAE
from analysis.vmem_models import RealNVP


def _device(op="detector scoring", verbose=False):
    # Silent by default: called inside the per-run scoring loop. The stage prints
    # the device once up front (see main()).
    return device_for(op, verbose=verbose)

def score_mahalanobis(model, X):
    """Squared Mahalanobis distance, chunked on the GPU (shared `maha_d2`)."""
    return maha_d2(X, model.location_, model.precision_, _device("Mahalanobis scoring"))

def score_knn(model, X):
    """Mean distance to the k nearest clean neighbours (higher = more OOD).

    Routed through the shared `knn_score` helper, which streams the reference
    from CPU in blocks (the full ~2 GB reference is never moved to the GPU in
    one allocation) and chunks the query with OOM-halving. The saved sklearn
    model is only a container for its reference points (`_fit_X`) and
    `n_neighbors`. Falls back to the CPU sklearn path if the reference points
    are not accessible."""
    ref = getattr(model, "_fit_X", None)
    if ref is None:
        dists, _ = model.kneighbors(X)
        return dists.mean(axis=1)
    device = _device()
    return knn_score(ref, model.n_neighbors, X, device)

def score_gmm(model, X):
    """Negative GMM log-likelihood, chunked on the GPU.

    Reproduces sklearn's `-score_samples` for `covariance_type='full'` directly
    from the fitted `precisions_cholesky_` / `means_` / `weights_`, so the heavy
    full-covariance log-prob runs on the GPU instead of single-host BLAS."""
    if model.covariance_type != "full":
        return -model.score_samples(X)  # uncommon path; keep sklearn

    device = _device()
    D = model.means_.shape[1]
    means = torch.from_numpy(np.ascontiguousarray(model.means_, dtype=np.float32)).to(device)
    prec_chol = torch.from_numpy(
        np.ascontiguousarray(model.precisions_cholesky_, dtype=np.float32)).to(device)  # (k, D, D)
    log_weights = torch.from_numpy(
        np.log(np.ascontiguousarray(model.weights_, dtype=np.float32))).to(device)       # (k,)
    # log|prec_chol| per component = sum log of its diagonal (upper-triangular L).
    log_det = torch.diagonal(prec_chol, dim1=-2, dim2=-1).log().sum(dim=1)                # (k,)
    const = D * np.log(2.0 * np.pi)
    nc = means.shape[0]

    def fn(chunk):
        # weighted log-gaussian per component, stacked then logsumexp.
        comp = []
        for k in range(nc):
            y = chunk @ prec_chol[k] - means[k] @ prec_chol[k]   # (m, D)
            mahal = (y * y).sum(dim=1)                            # (m,)
            log_gauss = -0.5 * (const + mahal) + log_det[k]
            comp.append(log_gauss + log_weights[k])
        stacked = torch.stack(comp, dim=1)                       # (m, k)
        return -torch.logsumexp(stacked, dim=1)

    X = np.ascontiguousarray(X, dtype=np.float32)
    return chunked_apply(fn, X, device, n_ref=D * nc)

def score_ocsvm(model, X):
    """Negative OCSVM decision function, chunked on the GPU.

    sklearn's libsvm `decision_function` is single-threaded with no BLAS — the
    dominant cost in this stage. The RBF decision is exactly
    `dual_coef · exp(-gamma·‖x−SV‖²) + intercept`, which we evaluate on the GPU.
    Falls back to sklearn for any non-RBF kernel."""
    if model.kernel != "rbf":
        return -model.decision_function(X)

    device = _device()
    gamma = float(model._gamma)
    sv = torch.from_numpy(np.ascontiguousarray(model.support_vectors_, dtype=np.float32)).to(device)
    dual = torch.from_numpy(
        np.ascontiguousarray(model.dual_coef_[0], dtype=np.float32)).to(device)   # (n_SV,)
    intercept = float(model.intercept_[0])

    def fn(chunk):
        d2 = torch.cdist(chunk, sv).pow_(2)          # (m, n_SV)
        K = torch.exp(-gamma * d2)
        dec = K @ dual + intercept
        return -dec

    X = np.ascontiguousarray(X, dtype=np.float32)
    return chunked_apply(fn, X, device, n_ref=sv.shape[0])

def score_pca(model, X):
    """PCA reconstruction error, chunked on the GPU."""
    device = _device()
    mean = torch.from_numpy(np.ascontiguousarray(model.mean_, dtype=np.float32)).to(device)
    comp = torch.from_numpy(
        np.ascontiguousarray(model.components_, dtype=np.float32)).to(device)     # (k, D)

    def fn(chunk):
        xc = chunk - mean
        recon = (xc @ comp.T) @ comp                 # project then back-project
        return torch.norm(xc - recon, dim=1)

    X = np.ascontiguousarray(X, dtype=np.float32)
    return chunked_apply(fn, X, device, n_ref=comp.shape[1])

def score_ae(model, X):
    device = _device("AE scoring")
    model = model.to(device)
    model.eval()

    def fn(chunk):
        with torch.no_grad():
            return torch.norm(chunk - model(chunk), dim=1)

    X = np.ascontiguousarray(X, dtype=np.float32)
    return chunked_apply(fn, X, device, n_ref=X.shape[1])

def score_flow(model, X):
    """model is a (pca, flow) pair; higher score = more OOD."""
    pca, flow = model
    device = _device("flow scoring")
    flow = flow.to(device)
    flow.eval()

    def fn(chunk):
        with torch.no_grad():
            return -flow.log_prob(chunk)

    X = np.ascontiguousarray(pca.transform(X), dtype=np.float32)
    # Peak isn't the 50-D input: each RealNVP coupling net expands to a hidden
    # width across n_layers (≈ hidden_dim*n_layers*2), so size the chunk to that.
    return chunked_apply(fn, X, device, n_ref=64 * 4 * 2)

# Explicit dispatch: an unknown detector file is skipped with a warning
# instead of silently reusing the previous detector's scores.
SCORERS = {
    'mahalanobis': score_mahalanobis,
    'knn': score_knn,
    'gmm': score_gmm,
    'ocsvm': score_ocsvm,
    'pca': score_pca,
    'ae': score_ae,
    'flow': score_flow,
}

def _extract(feats, rep):
    X = extract_representation(feats, rep)
    if X is None and rep != 'full_membrane':
        X = extract_representation(feats, 'full_membrane')
    return X

def main():
    print("Evaluating detectors...")
    all_feats = load_all_features()

    if 'clean' not in all_feats:
        print("Error: 'clean' run not found.")
        return

    out_dir = cfg.DETECTOR_DIR
    if not out_dir.exists():
        print(f"Error: Detectors not found in {out_dir}. Run fit_detectors.py first.")
        return

    # Recover the exact split the detectors were fitted on.
    rep = 'membrane_fused'
    split_meta = None
    split_path = out_dir / "split.json"
    if split_path.exists():
        with open(split_path) as f:
            split_meta = json.load(f)
        rep = split_meta.get("representation", rep)

    X_clean = _extract(all_feats['clean'], rep)
    if X_clean is None:
        print("Error: clean representation not found.")
        return

    if split_meta is not None and split_meta.get("n_clean") == len(X_clean):
        cut = int(split_meta["train_end"])
    else:
        if split_meta is not None:
            print("Warning: split.json does not match current clean data; "
                  "recomputing the boundary (re-run fit_detectors.py!).")
        cut = split_boundary(len(X_clean), TRAIN_RATIO, load_phi_seq_lens("clean"))

    # Clean negatives are the HELD-OUT portion only — the detectors never saw
    # these frames during fitting, so AUROC is uncontaminated.
    X_clean_eval = X_clean[cut:]
    print(f"Clean negatives: {len(X_clean_eval)} held-out frames "
          f"(train portion: {cut}).")

    detectors = {}
    for f in out_dir.glob("*.joblib"):
        if f.stem == "flow_pca":
            continue  # belongs to the 'flow' detector, loaded below
        if f.stem not in SCORERS:
            print(f"Warning: unknown detector file '{f.name}' — skipping.")
            continue
        detectors[f.stem] = joblib.load(f)

    if (out_dir / "ae.pt").exists():
        ae = SimpleAE(X_clean.shape[1])
        ae.load_state_dict(torch.load(out_dir / "ae.pt", weights_only=True))
        detectors['ae'] = ae

    if (out_dir / "flow.pt").exists() and (out_dir / "flow_pca.joblib").exists():
        flow_pca = joblib.load(out_dir / "flow_pca.joblib")
        flow = RealNVP(dim=flow_pca.n_components_)
        flow.load_state_dict(torch.load(out_dir / "flow.pt", weights_only=True))
        detectors['flow'] = (flow_pca, flow)

    results = []
    sev3plus_data = {name: {'corrupt': []} for name in detectors.keys()}

    # Announce the scoring device once (the per-run loop stays quiet).
    device_for(f"scoring {len(detectors)} detectors x {len(X_clean_eval)} clean "
               f"+ corrupted frames")

    # Precompute clean scores on the held-out split
    clean_scores = {name: SCORERS[name](model, X_clean_eval)
                    for name, model in detectors.items()}

    # Iterate run names (NOT list(all_feats.items())): materialising the items
    # view would load every run's phi into RAM at once and defeat the lazy,
    # memory-bounded loader. Each run is loaded on demand and evicted by the LRU.
    run_names = sorted(name for name in all_feats if name != 'clean')
    for run_name in tqdm(run_names, desc="Evaluating detectors"):
        feats = all_feats[run_name]

        X_test = _extract(feats, rep)
        if X_test is None:
            print(f"Warning: no '{rep}' representation for {run_name} — skipping.")
            continue
        # Positives = held-out tail of the corruption run only, drawn from the
        # same sequences as the clean negatives (X_clean_eval). The train-portion
        # frames — whose clean twins the detectors were fitted on — are excluded.
        X_test = held_out_eval(X_test, seq_lens=load_phi_seq_lens(run_name))
        if X_test.shape[1] != X_clean.shape[1]:
            # _extract may have fallen back to a different representation for
            # this run; feeding wrong-width features into the fitted
            # detectors would crash (or worse, silently misscore).
            print(f"Warning: {run_name} features are {X_test.shape[1]}-D but "
                  f"detectors were fitted on {X_clean.shape[1]}-D — skipping.")
            continue

        for name, model in detectors.items():
            test_scores = SCORERS[name](model, X_test)

            metrics = auroc_aupr_fpr95(clean_scores[name], test_scores)
            if metrics is None:
                continue
            auroc, aupr, fpr95 = metrics

            parts = run_name.rsplit('_L', 1)
            corruption = parts[0]
            severity = int(parts[1]) if len(parts) > 1 else 0

            if severity >= 3:
                sev3plus_data[name]['corrupt'].extend(test_scores.tolist())

            results.append({
                "detector": name,
                "corruption": corruption,
                "severity": severity,
                "auroc": auroc,
                "aupr": aupr,
                "fpr95": fpr95
            })

    # Compute Sev >= 3 aggregate AUROC
    sev3plus_results = []
    for name in detectors.keys():
        if len(sev3plus_data[name]['corrupt']) > 0:
            c_scores = clean_scores[name]
            t_scores = np.array(sev3plus_data[name]['corrupt'])
            y_true = np.concatenate([np.zeros(len(c_scores)), np.ones(len(t_scores))])
            y_score = np.concatenate([c_scores, t_scores])

            auroc_3p = roc_auc_score(y_true, y_score)
            aupr_3p = average_precision_score(y_true, y_score)
            fpr95_3p = calc_fpr95(y_true, y_score)

            sev3plus_results.append({
                "detector": name,
                "severity_group": ">=3",
                "auroc": auroc_3p,
                "aupr": aupr_3p,
                "fpr95": fpr95_3p
            })

    df = pd.DataFrame(results)
    df_3p = pd.DataFrame(sev3plus_results)

    res_dir = cfg.OUTPUT_DIR / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(res_dir / "ood_metrics.csv", index=False)
    df_3p.to_csv(res_dir / "severity3plus_metrics.csv", index=False)

    print(f"Evaluation complete. Results saved to {res_dir / 'ood_metrics.csv'} and {res_dir / 'severity3plus_metrics.csv'}")

if __name__ == "__main__":
    main()
