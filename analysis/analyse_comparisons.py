import sys
import csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from vmem_benchmark import benchmark_config as cfg

from analysis.vmem_utils import (
    LAYER_SPECS, TABLE_DIR, slice_phi_layer, slice_phi_stat,
    auroc_fpr95, _get_present, _valid_layers
)
from analysis.vmem_scorers import (
    mahalanobis_scorer, knn_scorer, gmm_scorer, pca_mahalanobis_scorer,
    ocsvm_scorer, normalizing_flow_scorer, autoencoder_scorer
)
from analysis.analyse_plots import (
    _plot_per_layer_heatmap, plot_statwise_ablation,
    plot_detector_comparison, plot_corruption_confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — Per-layer AUROC breakdown
# ─────────────────────────────────────────────────────────────────────────────

def split_clean(clean, train_ratio=0.7):
    rng = np.random.default_rng(42)
    indices = np.arange(len(clean))
    rng.shuffle(indices)
    split_idx = int(len(clean) * train_ratio)
    return clean[indices[:split_idx]], clean[indices[split_idx:]]


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — Per-layer AUROC breakdown
# ─────────────────────────────────────────────────────────────────────────────

def run_per_layer_auroc_table(all_phi):
    print("\n======================================================")
    print(" LEVEL 2 - Per-Layer x Per-Corruption AUROC Table")
    print("======================================================")

    clean = all_phi["clean"]
    present = _get_present(all_phi)
    if not present:
        print("  No corrupted runs found.")
        return None

    clean_train, clean_test = split_clean(clean)
    valid = _valid_layers(clean.shape[1])
    rows  = []

    for spec in valid + [{"name": "ALL (concat)", "idx": -1}]:
        is_all = spec["idx"] == -1
        layer_clean_train = clean_train if is_all else slice_phi_layer(clean_train, spec["idx"])
        layer_clean_test = clean_test if is_all else slice_phi_layer(clean_test, spec["idx"])
        scorer = mahalanobis_scorer(layer_clean_train)
        cs = scorer(layer_clean_test)

        row  = {"Layer": spec["name"]}
        avgs = []

        for c_name in present:
            aurocs = []
            for sev in cfg.SEVERITIES:
                rn = f"{c_name}_L{sev}"
                if rn not in all_phi:
                    continue
                lc = all_phi[rn] if is_all else slice_phi_layer(all_phi[rn], spec["idx"])
                yt = np.concatenate([np.zeros(len(cs)), np.ones(len(lc))])
                ys = np.concatenate([cs, scorer(lc)])
                a, _ = auroc_fpr95(yt, ys)
                aurocs.append(a)
            avg = float(np.nanmean(aurocs)) if aurocs else float("nan")
            row[c_name] = round(avg, 3)
            avgs.append(avg)

        row["AVG"] = round(float(np.nanmean(avgs)), 3)
        rows.append(row)
        print(f"  {spec['name']:<20}  AVG = {row['AVG']:.3f}   "
              + "  ".join(f"{c}: {row.get(c, float('nan')):.3f}" for c in present))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Layer"] + present + ["AVG"]
    with open(TABLE_DIR / "per_layer_auroc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved per_layer_auroc.csv")

    _plot_per_layer_heatmap(rows, present)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 2 — Stat-wise ablation (μ vs σ² vs κ vs combined)
# ─────────────────────────────────────────────────────────────────────────────

def run_statwise_ablation(all_phi):
    print("\n======================================================")
    print(" LEVEL 2 - Stat-wise Ablation (mean vs var vs kurt)")
    print("======================================================")

    clean   = all_phi["clean"]
    present = _get_present(all_phi)
    if not present:
        print("  No corrupted runs found.")
        return

    clean_train, clean_test = split_clean(clean)
    STAT_CONFIGS = [
        ("mean only",      "mu"),
        ("var only",       "var"),
        ("kurtosis only",  "kurtosis"),
        ("mean+var+kurt",  None),
    ]

    results = {}
    bar_means = {}

    for stat_label, stat_key in STAT_CONFIGS:
        clean_train_sub = clean_train if stat_key is None else slice_phi_stat(clean_train, stat_key)
        clean_test_sub = clean_test if stat_key is None else slice_phi_stat(clean_test, stat_key)
        scorer    = mahalanobis_scorer(clean_train_sub)
        cs        = scorer(clean_test_sub)
        c_aurocs  = {}
        for c_name in present:
            aurocs = []
            for sev in cfg.SEVERITIES:
                rn = f"{c_name}_L{sev}"
                if rn not in all_phi:
                    continue
                test = all_phi[rn] if stat_key is None else slice_phi_stat(all_phi[rn], stat_key)
                yt   = np.concatenate([np.zeros(len(cs)), np.ones(len(test))])
                ys   = np.concatenate([cs, scorer(test)])
                a, _ = auroc_fpr95(yt, ys)
                aurocs.append(a)
            c_aurocs[c_name] = float(np.nanmean(aurocs)) if aurocs else float("nan")
        results[stat_label]   = c_aurocs
        bar_means[stat_label] = float(np.nanmean(list(c_aurocs.values())))
        print(f"  {stat_label:<18}  grand avg AUROC = {bar_means[stat_label]:.3f}")

    plot_statwise_ablation(results, present)


# ─────────────────────────────────────────────────────────────────────────────
# Shared detector factory — call once, pass to both comparison functions
# ─────────────────────────────────────────────────────────────────────────────

def _build_detectors(clean_train):
    return {
        "Mahalanobis":      mahalanobis_scorer(clean_train),
        "kNN (k=5)":        knn_scorer(clean_train, k=5),
        "GMM":              gmm_scorer(clean_train, n_components=5),
        "PCA-Mahal":        pca_mahalanobis_scorer(clean_train, n_components=50),
        "One-Class SVM":    ocsvm_scorer(clean_train),
        "Normalizing Flow": normalizing_flow_scorer(clean_train, n_components=50),
        "Autoencoder":      autoencoder_scorer(clean_train),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 3 — Detector comparison (Mahal / kNN / GMM / PCA-Mahal)
# ─────────────────────────────────────────────────────────────────────────────

def run_detector_comparison(all_phi, detectors=None):
    print("\n======================================================")
    print(" LEVEL 3 - Detector Comparison")
    print("======================================================")

    clean   = all_phi["clean"]
    present = _get_present(all_phi)
    if not present:
        print("  No corrupted runs found.")
        return

    clean_train, clean_test = split_clean(clean)
    DETECTORS = detectors if detectors is not None else _build_detectors(clean_train)

    summary = {}
    per_corr = {det: {} for det in DETECTORS}

    for det_name, scorer in DETECTORS.items():
        cs = scorer(clean_test)
        all_aurocs, all_fprs = [], []
        for c_name in present:
            aurocs, fprs = [], []
            for sev in cfg.SEVERITIES:
                rn = f"{c_name}_L{sev}"
                if rn not in all_phi:
                    continue
                yt = np.concatenate([np.zeros(len(cs)), np.ones(len(all_phi[rn]))])
                ys = np.concatenate([cs, scorer(all_phi[rn])])
                a, f = auroc_fpr95(yt, ys)
                aurocs.append(a)
                fprs.append(f)
                all_aurocs.append(a)
                all_fprs.append(f)
            per_corr[det_name][c_name] = float(np.nanmean(aurocs)) if aurocs else float("nan")
        summary[det_name] = {
            "auroc": float(np.nanmean(all_aurocs)),
            "fpr95": float(np.nanmean(all_fprs)),
        }
        print(f"  {det_name:<18}  Avg AUROC = {summary[det_name]['auroc']:.3f}"
              f"   FPR@95 = {summary[det_name]['fpr95']:.3f}")

    print(f"\n  {'Corruption':<22} " +
          "  ".join(f"{d:<16}" for d in DETECTORS))
    print("  " + "-" * 120)
    for c_name in present:
        vals = "  ".join(
            f"{per_corr[d].get(c_name, float('nan')):<16.3f}" for d in DETECTORS)
        print(f"  {c_name:<22} {vals}")

    det_names  = list(DETECTORS.keys())
    plot_detector_comparison(summary, per_corr, det_names, present)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"Detector": d, "Avg AUROC": round(summary[d]["auroc"], 4),
             "Avg FPR@95": round(summary[d]["fpr95"], 4)} for d in det_names]
    with open(TABLE_DIR / "detector_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Detector", "Avg AUROC", "Avg FPR@95"])
        w.writeheader(); w.writerows(rows)
    print("  Saved detector_comparison.csv")


# ─────────────────────────────────────────────────────────────────────────────
# LEVEL 5 — Spearman-severity correlation
# ─────────────────────────────────────────────────────────────────────────────

def run_spearman_severity(all_phi):
    print("\n======================================================")
    print(" LEVEL 5 - Spearman-Severity Correlation")
    print("======================================================")
    print("  (rho > 0: AUROC rises with severity  |  p < 0.05: significant)")

    clean  = all_phi["clean"]
    clean_train, clean_test = split_clean(clean)
    scorer = mahalanobis_scorer(clean_train)
    cs     = scorer(clean_test)

    print(f"\n  {'Corruption':<25}  rho       p-val   n_sev")
    print("  " + "-" * 55)
    rows = []
    for c_name in cfg.CORRUPTIONS:
        aurocs, sevs = [], []
        for sev in cfg.SEVERITIES:
            rn = f"{c_name}_L{sev}"
            if rn not in all_phi:
                continue
            yt = np.concatenate([np.zeros(len(cs)), np.ones(len(all_phi[rn]))])
            ys = np.concatenate([cs, scorer(all_phi[rn])])
            a, _ = auroc_fpr95(yt, ys)
            aurocs.append(a);  sevs.append(sev)
        if len(aurocs) >= 3:
            rho, pval = spearmanr(sevs, aurocs)
            star = "*" if pval < 0.05 else " "
            print(f"  {c_name:<25}  {rho:+.3f}   {pval:.3f}{star}   {len(sevs)}")
            rows.append({"Corruption": c_name, "Spearman-rho": round(float(rho), 4),
                          "p-value": round(float(pval), 4), "n_severities": len(sevs)})
        elif aurocs:
            print(f"  {c_name:<25}  (need >=3 severity levels, have {len(aurocs)})")

    if rows:
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        with open(TABLE_DIR / "spearman_severity.csv", "w", newline="") as f:
            w = csv.DictWriter(f,
                fieldnames=["Corruption", "Spearman-rho", "p-value", "n_severities"])
            w.writeheader();  w.writerows(rows)
        print(f"\n  Saved spearman_severity.csv")


# ─────────────────────────────────────────────────────────────────────────────
# FULL RESULTS TABLE  (all detectors × all runs)
# ─────────────────────────────────────────────────────────────────────────────

def save_full_results_table(all_phi, detectors=None):
    print("\n======================================================")
    print(" Generating full results table (all detectors)...")
    print("======================================================")

    clean = all_phi["clean"]
    clean_train, clean_test = split_clean(clean)
    DETECTORS = detectors if detectors is not None else _build_detectors(clean_train)

    clean_scores = {}
    for det_name, scorer in DETECTORS.items():
        print(f"Pre-computing clean scores for {det_name}...")
        clean_scores[det_name] = scorer(clean_test)

    rows = []
    for c_name in cfg.CORRUPTIONS:
        for sev in cfg.SEVERITIES:
            rn = f"{c_name}_L{sev}"
            if rn not in all_phi:
                continue
            row  = {"Corruption": c_name, "Severity": sev}
            yt   = np.concatenate([np.zeros(len(clean_test)), np.ones(len(all_phi[rn]))])
            for det_name, scorer in DETECTORS.items():
                cs = clean_scores[det_name]
                ys = np.concatenate([cs, scorer(all_phi[rn])])
                a, f = auroc_fpr95(yt, ys)
                row[f"{det_name}_AUROC"]  = round(a, 4)
                row[f"{det_name}_FPR@95"] = round(f, 4)
            rows.append(row)

    if not rows:
        print("  No corrupted runs found. Nothing to write.")
        return

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    fn = ["Corruption", "Severity"] + [
        f"{d}_{m}" for d in DETECTORS for m in ["AUROC", "FPR@95"]]
    with open(TABLE_DIR / "full_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn);  w.writeheader();  w.writerows(rows)
    print(f"  Saved full_results.csv  ({len(rows)} rows)")

    hdr = f"  {'Corruption':<22} {'Sev':<5} " + \
          "  ".join(f"{d+'-AUROC':<18}" for d in DETECTORS)
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))
    for row in rows:
        line = f"  {row['Corruption']:<22} {row['Severity']:<5} "
        line += "  ".join(f"{row.get(d+'_AUROC', float('nan')):<18.4f}" for d in DETECTORS)
        print(line)


# ─────────────────────────────────────────────────────────────────────────────
# NEW RESEARCH IDEAS (IDEAS 3, 5, 7)
# ─────────────────────────────────────────────────────────────────────────────

def run_severity_regression(all_phi):
    print("\n======================================================")
    print(" LEVEL 7 - Severity Regression (Idea 5)")
    print("======================================================")
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import train_test_split
    
    rows = []
    all_X = []
    all_y = []
    
    for c_name in cfg.CORRUPTIONS:
        c_X = []
        c_y = []
        for sev in cfg.SEVERITIES:
            rn = f"{c_name}_L{sev}"
            if rn in all_phi:
                features = all_phi[rn]
                c_X.append(features)
                c_y.append(np.full(len(features), sev))
                
        if not c_X:
            continue
            
        c_X = np.concatenate(c_X, axis=0)
        c_y = np.concatenate(c_y, axis=0)
        
        all_X.append(c_X)
        all_y.append(c_y)
        
        # Need at least 2 samples to split
        if len(c_X) < 2:
            print(f"  {c_name:<25} (skipped: only {len(c_X)} sample)")
            continue

        test_size = max(1, int(0.3 * len(c_X)))
        train_size = len(c_X) - test_size
        if train_size < 1:
            print(f"  {c_name:<25} (skipped: too few samples to split)")
            continue

        X_train, X_test, y_train, y_test = train_test_split(c_X, c_y, test_size=0.3, random_state=42)
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        if len(np.unique(y_test)) < 2:
            # Cannot compute meaningful r2 with one class
            r2, mse = float("nan"), float("nan")
        else:
            r2 = r2_score(y_test, preds)
            mse = mean_squared_error(y_test, preds)
        print(f"  {c_name:<25} R^2 = {r2 if not np.isnan(r2) else 'N/A':>8}  MSE = {mse if not np.isnan(mse) else 'N/A'}")
        rows.append({"Scope": c_name, "R2": round(r2, 4) if not np.isnan(r2) else "", "MSE": round(mse, 4) if not np.isnan(mse) else ""})
        
    if all_X:
        all_X = np.concatenate(all_X, axis=0)
        all_y = np.concatenate(all_y, axis=0)
        
        if len(all_X) >= 2:
            X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.3, random_state=42)
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            r2 = r2_score(y_test, preds) if len(np.unique(y_test)) > 1 else float("nan")
            mse = mean_squared_error(y_test, preds)
            print(f"  {'Overall (All Corruptions)':<25} R^2 = {r2 if not np.isnan(r2) else 'N/A':>8}  MSE = {mse:.4f}")
            rows.append({"Scope": "Overall", "R2": round(r2, 4) if not np.isnan(r2) else "", "MSE": round(mse, 4)})
        
        with open(TABLE_DIR / "severity_regression.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["Scope", "R2", "MSE"])
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved severity_regression.csv")
        
        
def run_corruption_classification(all_phi):
    print("\n======================================================")
    print(" LEVEL 8 - Corruption Classification (Idea 3)")
    print("======================================================")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    
    X_list = [all_phi["clean"]]
    y_list = [np.zeros(len(all_phi["clean"]))]
    
    class_names = ["clean"] + list(cfg.CORRUPTIONS)
    
    for idx, c_name in enumerate(cfg.CORRUPTIONS, start=1):
        c_samples = []
        for sev in cfg.SEVERITIES:
            rn = f"{c_name}_L{sev}"
            if rn in all_phi:
                c_samples.append(all_phi[rn])
        if c_samples:
            c_samples = np.concatenate(c_samples, axis=0)
            X_list.append(c_samples)
            y_list.append(np.full(len(c_samples), idx))
            
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    # Guard: need at least 2 samples per class to stratify
    unique_classes, counts = np.unique(y, return_counts=True)
    if len(X) < 4 or any(c < 2 for c in counts):
        print("  Skipping: not enough samples per class for classification.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
    except ValueError:
        # Fallback to non-stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"  Accuracy (7-class classification): {acc:.4f}")
    
    unique_y = np.unique(y)
    target_names = [class_names[int(i)] for i in unique_y]
    
    report = classification_report(y_test, preds, target_names=target_names)
    print("\n  Classification Report:\n")
    print(report)
    
    plot_corruption_confusion_matrix(y_test, preds, target_names)
    
    report_dict = classification_report(y_test, preds, target_names=target_names, output_dict=True)
    with open(TABLE_DIR / "corruption_classification_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Class", "Precision", "Recall", "F1-Score", "Support"])
        w.writeheader()
        for k, v in report_dict.items():
            if isinstance(v, dict):
                w.writerow({
                    "Class": k,
                    "Precision": round(v["precision"], 4),
                    "Recall": round(v["recall"], 4),
                    "F1-Score": round(v["f1-score"], 4),
                    "Support": int(v["support"])
                })
        print("  Saved corruption_classification_report.csv")


def run_conformal_prediction(all_phi):
    print("\n======================================================")
    print(" LEVEL 9 - Conformal Prediction (Idea 6)")
    print("======================================================")
    
    clean_phi = all_phi["clean"]
    present = _get_present(all_phi)
    if not present:
        print("  No corrupted runs found.")
        return

    clean_train, clean_test = split_clean(clean_phi)
    scorer = mahalanobis_scorer(clean_train)
    cal_scores = scorer(clean_test)
    
    q_low = float(np.percentile(cal_scores, 90))
    q_high = float(np.percentile(cal_scores, 99))
    
    print(f"  Calibration thresholds derived from {len(cal_scores)} clean frames:")
    print(f"    q_low (90% quantile)  = {q_low:.4f}")
    print(f"    q_high (99% quantile) = {q_high:.4f}")
    print()

    rows = []
    c_clean = np.sum(cal_scores <= q_low) / len(cal_scores)
    c_amb = np.sum((cal_scores > q_low) & (cal_scores <= q_high)) / len(cal_scores)
    c_ood = np.sum(cal_scores > q_high) / len(cal_scores)
    print(f"  Clean: Clean={c_clean*100:.1f}%, Ambiguous={c_amb*100:.1f}%, OOD={c_ood*100:.1f}%")
    rows.append({
        "Corruption": "clean",
        "Severity": 0,
        "Clean (90% Conf.)": round(c_clean, 4),
        "Ambiguous": round(c_amb, 4),
        "OOD (99% Conf.)": round(c_ood, 4)
    })

    for c_name in present:
        for sev in cfg.SEVERITIES:
            rn = f"{c_name}_L{sev}"
            if rn not in all_phi:
                continue
            test_phi = all_phi[rn]
            test_results = scorer(test_phi)
            n = len(test_results)
            p_clean = np.sum(test_results <= q_low) / n
            p_amb = np.sum((test_results > q_low) & (test_results <= q_high)) / n
            p_ood = np.sum(test_results > q_high) / n
            
            print(f"  {c_name:<18} L{sev} : Clean={p_clean*100:.1f}%, Ambiguous={p_amb*100:.1f}%, OOD={p_ood*100:.1f}%")
            rows.append({
                "Corruption": c_name,
                "Severity": sev,
                "Clean (90% Conf.)": round(p_clean, 4),
                "Ambiguous": round(p_amb, 4),
                "OOD (99% Conf.)": round(p_ood, 4)
            })

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / "conformal_prediction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Corruption", "Severity", "Clean (90% Conf.)", "Ambiguous", "OOD (99% Conf.)"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n  Saved conformal_prediction.csv")

