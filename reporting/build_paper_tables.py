import pandas as pd
from pathlib import Path

def generate_table(df_path, out_path, index=False, caption=""):
    if not df_path.exists():
        return
    df = pd.read_csv(df_path)
    if df.empty:
        return
        
    latex_str = df.to_latex(index=index, float_format="%.3f")
    # wrap in simple table env
    latex_out = f"\\begin{{table}}[h]\n\\centering\n{latex_str}\\caption{{{caption}}}\n\\end{{table}}\n"
    
    with open(out_path, "w") as f:
        f.write(latex_out)

def main():
    print("Building paper tables...")
    res_dir = Path("results")
    out_dir = Path("paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: OOD
    generate_table(res_dir / "ood_metrics.csv", out_dir / "table1_ood.tex", 
                   caption="OOD Detection performance (AUROC, AUPR, FPR95).")
                   
    # Table 2: Layers/Representations
    generate_table(res_dir / "representation_metrics.csv", out_dir / "table2_layers.tex",
                   caption="Ablation of feature representations for OOD Detection.")
                   
    # Table 3: Reliability
    generate_table(res_dir / "reliability_metrics.csv", out_dir / "table3_reliability.tex",
                   caption="Reliability metrics (Spearman $\\rho$, AURC) across corruptions.")
                   
    # Table 4: Transfer
    # We combine cross_corruption and dsec_transfer if both exist, else just do what we can
    cc_path = res_dir / "cross_corruption.csv"
    dsec_path = res_dir / "dsec_transfer.csv"
    
    if cc_path.exists():
        generate_table(cc_path, out_dir / "table4_transfer.tex", caption="Cross-corruption generalization.")
    elif dsec_path.exists():
        generate_table(dsec_path, out_dir / "table4_transfer.tex", caption="DSEC Transfer OOD scores.")
        
    # Table 5: Models
    generate_table(res_dir / "model_comparison.csv", out_dir / "table5_models.tex",
                   caption="OOD Detection across different model architectures.")
                   
    # TASK L: Final Paper Table (table_final_main.csv)
    print("Generating table_final_main.csv...")
    try:
        # Load the 3 tables if they exist
        dfs = []
        if (res_dir / "ood_metrics.csv").exists():
            df_ood = pd.read_csv(res_dir / "ood_metrics.csv")
            # We assume model="Membrane", representation is implicit or based on filename if we merged them.
            # Wait, ood_metrics.csv has ['detector', 'corruption', 'severity', 'auroc', 'aupr', 'fpr95']
            # We aggregate across corruptions/severities.
            df_ood_agg = df_ood.groupby("detector").mean(numeric_only=True).reset_index()
            df_ood_agg["Method"] = "Membrane"
            df_ood_agg["Representation"] = "membrane_fused (or specified)"
            dfs.append(df_ood_agg)
            
        if (res_dir / "ann_baselines.csv").exists():
            df_ann = pd.read_csv(res_dir / "ann_baselines.csv")
            df_ann_agg = df_ann.groupby(["model", "representation", "detector"]).mean(numeric_only=True).reset_index()
            df_ann_agg.rename(columns={"model": "Method", "representation": "Representation"}, inplace=True)
            dfs.append(df_ann_agg)
            
        if dfs:
            combined = pd.concat(dfs, ignore_index=True)
            
            # Now load Severity >= 3 metrics to join
            if (res_dir / "severity3plus_metrics.csv").exists():
                df_sev = pd.read_csv(res_dir / "severity3plus_metrics.csv")
                # sev3plus_metrics has ['detector', 'severity_group', 'auroc', 'aupr', 'fpr95']
                df_sev = df_sev.rename(columns={"auroc": "Severity>=3 AUROC"})
                # Merge on detector (for membrane). ANN doesn't have a separate sev>=3 evaluated right now, 
                # but we can just map it where available.
                combined = pd.merge(combined, df_sev[["detector", "Severity>=3 AUROC"]], on="detector", how="left")
            else:
                combined["Severity>=3 AUROC"] = float("nan")
                
            combined.rename(columns={"auroc": "Overall AUROC", "fpr95": "FPR95", "aupr": "AUPR", "detector": "Detector"}, inplace=True)
            cols = ["Method", "Representation", "Detector", "Overall AUROC", "Severity>=3 AUROC", "FPR95", "AUPR"]
            # Filter available columns
            cols = [c for c in cols if c in combined.columns]
            final_table = combined[cols]
            
            # Sort by Severity>=3 AUROC descending
            if "Severity>=3 AUROC" in final_table.columns:
                final_table = final_table.sort_values(by="Severity>=3 AUROC", ascending=False)
            else:
                final_table = final_table.sort_values(by="Overall AUROC", ascending=False)
                
            final_table.to_csv(res_dir / "table_final_main.csv", index=False)
            generate_table(res_dir / "table_final_main.csv", out_dir / "table_final_main.tex", caption="Primary robustness benchmark results.")
    except Exception as e:
        print(f"Error generating final table: {e}")
                   
    print(f"Tables generated in {out_dir}")

if __name__ == "__main__":
    main()
