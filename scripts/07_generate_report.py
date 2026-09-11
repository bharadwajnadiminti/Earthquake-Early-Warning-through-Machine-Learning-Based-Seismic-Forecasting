#!/usr/bin/env python3
"""
07_generate_report.py

Pulls together the target definition, feature list, model comparison table,
best model and its metrics from every prior stage's saved outputs into one
markdown report meant to be lifted straight into the MSc project report.

Input:  ../outputs/features/04_feature_engineering_report.json
        ../outputs/features/04_data_dictionary.csv
        ../outputs/models/05_split_summary.json
        ../outputs/models/05_*_best_params.json
        ../outputs/metrics/06_evaluation_metrics.json
        ../outputs/metrics/06_best_model.json
Output: ../outputs/report/07_final_report.md
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(x):
    return f"{100 * x:.2f}%"


def main():
    p = argparse.ArgumentParser(description="Generate the final markdown report (stage 07).")
    p.add_argument("--features-dir", type=str, default="../outputs/features")
    p.add_argument("--models-dir", type=str, default="../outputs/models")
    p.add_argument("--metrics-dir", type=str, default="../outputs/metrics")
    p.add_argument("--outdir", type=str, default="../outputs/report")
    args = p.parse_args()

    features_dir = Path(args.features_dir)
    models_dir = Path(args.models_dir)
    metrics_dir = Path(args.metrics_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    feat_report = load_json(features_dir / "04_feature_engineering_report.json")
    data_dict = pd.read_csv(features_dir / "04_data_dictionary.csv")
    split_summary = load_json(models_dir / "05_split_summary.json")
    eval_metrics = load_json(metrics_dir / "06_evaluation_metrics.json")
    best_model_report = load_json(metrics_dir / "06_best_model.json")
    comp_df = pd.DataFrame(best_model_report["comparison_table"]).sort_values("roc_auc", ascending=False)

    best_params = {}
    for key, fname in [
        ("AdaBoost + Decision Tree", "05_adaboost_dt_best_params.json"),
        ("AdaBoost + Random Forest", "05_adaboost_rf_best_params.json"),
        ("XGBoost", "05_xgboost_best_params.json"),
    ]:
        fpath = models_dir / fname
        if fpath.exists():
            best_params[key] = load_json(fpath)

    lines = []
    lines.append("# Automatic Earthquake Forecasting — Final Model Report\n")
    lines.append(
        "Pipeline: seismic data collection -> preprocessing -> feature extraction -> "
        "ML model training -> forecast output -> early warning alert.\n"
    )

    # --- 1. Target definition ------------------------------------------------
    lines.append("## 1. Target Definition\n")
    lines.append(
        f"**Formulation:** spatio-temporal binary classification. The study region is gridded "
        f"into **{feat_report['grid_size_deg']} deg x {feat_report['grid_size_deg']} deg** cells; "
        f"only **active cells** (>= {feat_report['min_events_per_cell']} historical catalog events) "
        f"are modeled — **{feat_report['n_active_cells']}** of {feat_report['n_total_cells_with_any_event']} "
        f"cells with any recorded event, covering **{feat_report['active_cell_event_coverage_pct']}%** "
        f"of all catalog events.\n"
    )
    lines.append(
        f"For every active cell *c* and day *t*: **y=1** if at least one event with "
        f"magnitude >= **{feat_report['mag_threshold']}** occurs in cell *c* during "
        f"**(t, t+{feat_report['forecast_horizon_days']}]** days (strictly after t — features use "
        f"only information available through end of day t); **y=0** otherwise.\n"
    )
    lines.append(
        f"**Magnitude threshold justification:** M{feat_report['mag_threshold']} is the standard "
        f"\"moderately damaging / broadly felt\" cutoff used in operational seismology and short-term "
        f"forecasting studies. It was also chosen for statistical power: at this catalog's size and "
        f"span, M4.5+ yields a workable ~{fmt_pct(feat_report['target_positive_rate'])} positive rate "
        f"at (cell, day) granularity, versus M5.0+ which would push the positive rate into the low "
        f"single digits and leave too few positive examples per cross-validation fold.\n"
    )
    lines.append(
        f"**Warm-up:** a cell's rows only start once it has accumulated >= {feat_report['warmup_min_events']} "
        f"historical events (insufficient history before that to compute meaningful rolling features).\n"
    )
    lines.append(
        f"- Final feature matrix: **{feat_report['final_row_count']:,}** (cell, day) rows, "
        f"**{feat_report['target_positive_count']:,}** positive ({fmt_pct(feat_report['target_positive_rate'])}).\n"
        f"- Date range: {feat_report['date_range'][0]} -> {feat_report['date_range'][1]}\n"
        f"- Rows dropped in per-cell warm-up: {feat_report['rows_dropped_warmup']:,}\n"
        f"- Rows dropped (no complete future label window): {feat_report['rows_dropped_no_label']:,}\n"
    )
    lines.append(
        "**Note on formulation:** this is one reasonable, standard choice — not the only one. "
        "A finer grid (0.1-0.5 deg) with a shorter horizon over a single well-studied region "
        "(e.g. California, Japan) would likely forecast better and be more directly comparable to "
        "published regional models; the global/1-degree/7-day choice here maximizes usable positive "
        "examples given the ~20-month catalog window available. See scripts/04_feature_engineering.py "
        "docstring for the full reasoning.\n"
    )

    # --- 2. Feature list -------------------------------------------------------
    lines.append("## 2. Features\n")
    lines.append(f"{len(data_dict)} columns total (including `day`, `cell_lat`, `cell_lon`, and `target`). "
                  "Full data dictionary: `outputs/features/04_data_dictionary.csv`.\n")
    lines.append("| Column | Type | % Missing | Description |")
    lines.append("|---|---|---|---|")
    for _, row in data_dict.iterrows():
        lines.append(f"| `{row['column']}` | {row['dtype']} | {row['pct_missing']}% | {row['description']} |")
    lines.append("")

    # --- 3. Train/val/test split ------------------------------------------------
    lines.append("## 3. Chronological Split\n")
    lines.append(
        f"70/15/15 train/val/test split by unique day, with a "
        f"{split_summary['purge_gap_days']}-day purge gap at each boundary (the label looks up to "
        f"{split_summary['purge_gap_days']} days into the future, so rows near a split boundary would "
        f"otherwise leak information across it).\n"
    )
    lines.append("| Split | Rows | Date range | Positive rate |")
    lines.append("|---|---|---|---|")
    lines.append(f"| Train | {split_summary['train_rows']:,} | {split_summary['train_date_range'][0]} -> "
                  f"{split_summary['train_date_range'][1]} | {fmt_pct(split_summary['train_positive_rate'])} |")
    if split_summary["val_date_range"]:
        lines.append(f"| Val | {split_summary['val_rows']:,} | {split_summary['val_date_range'][0]} -> "
                      f"{split_summary['val_date_range'][1]} | {fmt_pct(split_summary['val_positive_rate'])} |")
    if split_summary["test_date_range"]:
        lines.append(f"| Test | {split_summary['test_rows']:,} | {split_summary['test_date_range'][0]} -> "
                      f"{split_summary['test_date_range'][1]} | {fmt_pct(split_summary['test_positive_rate'])} |")
    lines.append("")

    # --- 4. Model comparison -----------------------------------------------------
    lines.append("## 4. Model Comparison (Test Set)\n")
    lines.append(comp_df.to_markdown(index=False))
    lines.append("")
    lines.append(f"Tuning: GridSearchCV, `scoring='roc_auc'`, `TimeSeriesSplit` "
                  f"(n_splits={split_summary['cv_n_splits']}, gap~{split_summary['cv_gap_rows_approx']} rows) "
                  f"on the train split only.\n")

    for name, params in best_params.items():
        lines.append(f"**{name} — best hyperparameters:**")
        lines.append("```json")
        lines.append(json.dumps(params, indent=2))
        lines.append("```\n")

    # --- 5. Best model --------------------------------------------------------
    lines.append("## 5. Best Model\n")
    lines.append(f"**{best_model_report['best_model']}**\n")
    lines.append(best_model_report["justification"] + "\n")
    bm = best_model_report["metrics"]
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| ROC-AUC | {bm['roc_auc']:.4f} |")
    lines.append(f"| PR-AUC (Average Precision) | {bm['pr_auc']:.4f} |")
    lines.append(f"| Precision @ 0.5 | {bm['precision_at_0.5']:.4f} |")
    lines.append(f"| Recall @ 0.5 | {bm['recall_at_0.5']:.4f} |")
    lines.append(f"| F1 @ 0.5 | {bm['f1_at_0.5']:.4f} |")
    lines.append(f"| Best F1 threshold | {bm['best_f1_threshold']:.4f} |")
    lines.append(f"| F1 @ best threshold | {bm['f1_at_best_f1_threshold']:.4f} |")
    lines.append(f"| Brier score (calibration) | {bm['brier_score']:.4f} |")
    lines.append(f"| Confusion matrix @ 0.5 [[TN,FP],[FN,TP]] | {bm['confusion_matrix_at_0.5']} |")
    lines.append("")
    lines.append("Plots: `outputs/metrics/06_roc_curves.png`, `06_pr_curves.png`, "
                  "`06_confusion_matrices.png`, `06_calibration_curves.png`, "
                  "`06_feature_importance_<model>.png`.\n")

    lines.append("## 6. Reproducing This Report\n")
    lines.append("```\ncd scripts\npython 01_fetch_and_update_data.py\n"
                  "python 02_generate_profile_report.py\npython 03_preprocess.py\n"
                  "python 04_feature_engineering.py\npython 05_train_models.py\n"
                  "python 06_evaluate_models.py\npython 07_generate_report.py\n```\n")
    lines.append("See the top-level `README.md` for the full pipeline description.\n")

    report_text = "\n".join(lines)
    out_path = outdir / "07_final_report.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Final report written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
