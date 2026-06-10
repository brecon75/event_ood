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
from analysis.vmem_utils import TRAIN_RATIO, split_boundary, load_phi_seq_lens
from analysis.fit_detectors import SimpleAE
from analysis.vmem_models import RealNVP

def score_mahalanobis(model, X):
    diff = X - model.location_
    return np.einsum('ni,ij,nj->n', diff, model.precision_, diff)

def score_knn(model, X):
    dists, _ = model.kneighbors(X)
    return dists.mean(axis=1)

def score_gmm(model, X):
    return -model.score_samples(X)

def score_ocsvm(model, X):
    return -model.decision_function(X)

def score_pca(model, X):
    X_proj = model.transform(X)
    X_recon = model.inverse_transform(X_proj)
    return np.linalg.norm(X - X_recon, axis=1)

def score_ae(model, X):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        recon = model(X_t)
        scores = torch.norm(X_t - recon, dim=1).cpu().numpy()
    return scores

def score_flow(model, X):
    """model is a (pca, flow) pair; higher score = more OOD."""
    pca, flow = model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow = flow.to(device)
    flow.eval()
    X_t = torch.from_numpy(pca.transform(X)).float().to(device)
    scores = []
    with torch.no_grad():
        for chunk in torch.split(X_t, 20000):
            scores.append(-flow.log_prob(chunk))
    return torch.cat(scores).cpu().numpy()

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

    # Precompute clean scores on the held-out split
    clean_scores = {name: SCORERS[name](model, X_clean_eval)
                    for name, model in detectors.items()}

    for run_name, feats in tqdm(list(all_feats.items()), desc="Evaluating detectors"):
        if run_name == 'clean':
            continue

        X_test = _extract(feats, rep)
        if X_test is None:
            print(f"Warning: no '{rep}' representation for {run_name} — skipping.")
            continue

        for name, model in detectors.items():
            test_scores = SCORERS[name](model, X_test)

            y_true = np.concatenate([np.zeros(len(clean_scores[name])), np.ones(len(test_scores))])
            y_score = np.concatenate([clean_scores[name], test_scores])

            if len(np.unique(y_true)) < 2:
                continue
            try:
                auroc = roc_auc_score(y_true, y_score)
                aupr = average_precision_score(y_true, y_score)
                fpr95 = calc_fpr95(y_true, y_score)
            except Exception:
                continue

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
