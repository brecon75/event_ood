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
from analysis.fit_detectors import SimpleAE

def score_mahalanobis(model, X):
    diff = X - model.location_
    return np.einsum('ni,ij,nj->n', diff, model.precision_, diff)

def score_knn(model, X):
    dists, _ = model.kneighbors(X)
    return dists[:, -1] # distance to k-th neighbor

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
        
    detectors = {}
    for f in out_dir.glob("*.joblib"):
        detectors[f.stem] = joblib.load(f)
        
    if (out_dir / "ae.pt").exists():
        # we need input dim
        X_clean = extract_representation(all_feats['clean'], 'membrane_fused')
        if X_clean is None:
            X_clean = extract_representation(all_feats['clean'], 'full_membrane')
        ae = SimpleAE(X_clean.shape[1])
        ae.load_state_dict(torch.load(out_dir / "ae.pt", weights_only=True))
        detectors['ae'] = ae
        
    results = []
    sev3plus_data = {name: {'clean': [], 'corrupt': []} for name in detectors.keys()}
    
    # Precompute clean scores
    X_clean = extract_representation(all_feats['clean'], 'membrane_fused')
    if X_clean is None:
        X_clean = extract_representation(all_feats['clean'], 'full_membrane')
    clean_scores = {}
    for name, model in detectors.items():
        if name == 'mahalanobis':
            clean_scores[name] = score_mahalanobis(model, X_clean)
        elif name == 'knn':
            clean_scores[name] = score_knn(model, X_clean)
        elif name == 'gmm':
            clean_scores[name] = score_gmm(model, X_clean)
        elif name == 'ocsvm':
            clean_scores[name] = score_ocsvm(model, X_clean)
        elif name == 'pca':
            clean_scores[name] = score_pca(model, X_clean)
        elif name == 'ae':
            clean_scores[name] = score_ae(model, X_clean)
            
    for run_name, feats in tqdm(list(all_feats.items()), desc="Evaluating detectors"):
        if run_name == 'clean': continue
        
        X_test = extract_representation(feats, 'membrane_fused')
        if X_test is None:
            X_test = extract_representation(feats, 'full_membrane')
        
        for name, model in detectors.items():
            if name == 'mahalanobis':
                test_scores = score_mahalanobis(model, X_test)
            elif name == 'knn':
                test_scores = score_knn(model, X_test)
            elif name == 'gmm':
                test_scores = score_gmm(model, X_test)
            elif name == 'ocsvm':
                test_scores = score_ocsvm(model, X_test)
            elif name == 'pca':
                test_scores = score_pca(model, X_test)
            elif name == 'ae':
                test_scores = score_ae(model, X_test)
                
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
    for name, model in detectors.items():
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
