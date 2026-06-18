"""Reporting layer: build_paper_tables, build_paper_figures, analyse_plots.

Matplotlib runs headless (Agg, set in conftest). These assert the file-reading
guards and that the table/figure builders produce output without crashing."""
import pandas as pd
import pytest

import reporting.build_paper_tables as bt
import reporting.build_paper_figures as bf


# ───────────────────────── safe_read_csv / generate_table ─────────────────────────

def test_safe_read_csv_missing_empty_valid(tmp_path):
    assert bt.safe_read_csv(tmp_path / "nope.csv") is None
    empty = tmp_path / "empty.csv"; empty.write_text("")
    assert bt.safe_read_csv(empty) is None
    good = tmp_path / "good.csv"; pd.DataFrame({"a": [1, 2]}).to_csv(good, index=False)
    assert len(bt.safe_read_csv(good)) == 2


def test_generate_table_skips_empty_and_writes_latex(tmp_path):
    # empty / missing input -> no file written
    bt.generate_table(tmp_path / "missing.csv", tmp_path / "t.tex")
    assert not (tmp_path / "t.tex").exists()
    # valid input -> LaTeX table file written
    src = tmp_path / "ood.csv"
    pd.DataFrame({"detector": ["maha"], "auroc": [0.9]}).to_csv(src, index=False)
    out = tmp_path / "table.tex"
    bt.generate_table(src, out, caption="x")
    assert out.exists() and "\\begin{table}" in out.read_text()


# ───────────────────────── full builders (smoke) ─────────────────────────

def test_build_paper_tables_main(tmp_path, monkeypatch):
    from vmem_benchmark import benchmark_config as cfg
    res = tmp_path / "results"; res.mkdir()
    pd.DataFrame({"detector": ["maha", "knn"], "corruption": ["a", "a"],
                  "severity": [5, 5], "auroc": [0.9, 0.8],
                  "aupr": [0.9, 0.8], "fpr95": [0.1, 0.2]}).to_csv(
        res / "ood_metrics.csv", index=False)
    monkeypatch.setattr(cfg, "OUTPUT_DIR", tmp_path)
    bt.main()
    assert (tmp_path / "paper_tables" / "table1_ood.tex").exists()


def test_build_paper_figures_main_smoke(tmp_path, monkeypatch):
    from vmem_benchmark import benchmark_config as cfg
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(cfg, "OUTPUT_DIR", tmp_path)
    bf.main()                                    # no CSVs -> placeholder figures
    figs = list((tmp_path / "paper_figures").glob("*.pdf"))
    assert any("fig1" in f.name for f in figs)   # placeholder always emitted


def test_build_paper_figures_with_data(tmp_path, monkeypatch):
    from vmem_benchmark import benchmark_config as cfg
    res = tmp_path / "results"; res.mkdir()
    pd.DataFrame({"representation": ["mu", "var"], "auroc": [0.7, 0.8],
                  "severity": [5, 5], "corruption": ["a", "b"]}).to_csv(
        res / "representation_metrics.csv", index=False)
    monkeypatch.setattr(cfg, "OUTPUT_DIR", tmp_path)
    bf.main()
    assert (tmp_path / "paper_figures" / "fig3_representation.pdf").exists()


def test_analyse_plots_free_rider(tmp_path, monkeypatch):
    from vmem_benchmark import benchmark_config as cfg
    import analysis.analyse_plots as ap
    monkeypatch.setattr(cfg, "PLOT_DIR", tmp_path)
    # also patch the module-level TABLE/PLOT dirs the function may reference
    results = {"Trained SNN": {"hot_pixel": 0.99, "event_flood": 0.5},
               "Random SNN": {"hot_pixel": 0.6, "event_flood": 0.5},
               "Raw Input Stats": {"hot_pixel": 0.7, "event_flood": 0.55}}
    # smoke: must not raise (writes a PDF somewhere under the patched dirs)
    ap.plot_free_rider_ablation(results)
