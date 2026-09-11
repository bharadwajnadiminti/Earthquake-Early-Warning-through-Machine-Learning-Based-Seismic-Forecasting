#!/usr/bin/env python3
"""
06_evaluate_models.py

Evaluates the three tuned models from stage 05 on the held-out, never-seen
TEST split (chronologically after train and val, with the same 7-day purge
gap applied in stage 05), and picks a best model.

For each model: ROC-AUC, PR-AUC (average precision), precision/recall/F1
(at threshold 0.5, and at the threshold that maximizes F1 on the test set,
reported separately and clearly labeled), a confusion matrix, and a
calibration curve (+ Brier score). All three are compared in one table.
ROC curves, PR curves, and feature-importance plots are saved as PNG.

Input:  ../outputs/04_features/04_feature_matrix.csv
        ../outputs/05_models/05_model_*.pkl, 05_imputer.pkl, 05_feature_columns.json
Output: ../outputs/06_metrics/06_evaluation_metrics.json
        ../outputs/06_metrics/06_evaluation_metrics.csv   (comparison table)
        ../outputs/06_metrics/06_roc_curves.png
        ../outputs/06_metrics/06_pr_curves.png
        ../outputs/06_metrics/06_confusion_matrices.png
        ../outputs/06_metrics/06_calibration_curves.png
        ../outputs/06_metrics/06_feature_importance_<model>.png
        ../outputs/06_metrics/06_best_model.json
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
sns.set_theme(style="whitegrid")

FORECAST_HORIZON_DAYS = 7
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

MODEL_FILES = {
    "AdaBoost + Decision Tree": "05_model_adaboost_dt.pkl",
    "AdaBoost + Random Forest": "05_model_adaboost_rf.pkl",
    "XGBoost": "05_model_xgboost.pkl",
}
COLORS = {"AdaBoost + Decision Tree": "#4C72B0", "AdaBoost + Random Forest": "#DD8452", "XGBoost": "#55A868"}


def chronological_split(df, day_col="day"):
    days = np.sort(df[day_col].unique())
    n = len(days)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train_days = days[:train_end]
    val_start = min(train_end + FORECAST_HORIZON_DAYS, val_end)
    val_days = days[val_start:val_end]
    test_start = min(val_end + FORECAST_HORIZON_DAYS, n)
    test_days = days[test_start:]
    train = df[df[day_col].isin(train_days)].copy()
    val = df[df[day_col].isin(val_days)].copy()
    test = df[df[day_col].isin(test_days)].copy()
    return train, val, test


def main():
    p = argparse.ArgumentParser(description="Evaluate the three tuned models on the test split (stage 06).")
    p.add_argument("--features", type=str, default="../outputs/04_features/04_feature_matrix.csv")
    p.add_argument("--models-dir", type=str, default="../outputs/05_models")
    p.add_argument("--outdir", type=str, default="../outputs/06_metrics")
    args = p.parse_args()

    models_dir = Path(args.models_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.features} ...")
    df = pd.read_csv(args.features)
    df["day"] = pd.to_datetime(df["day"], utc=True)
    train, val, test = chronological_split(df)
    print(f"Test split: {len(test):,} rows, {test['day'].min()} -> {test['day'].max()}, "
          f"positive rate={test['target'].mean():.4f}")

    with open(models_dir / "05_feature_columns.json", encoding="utf-8") as f:
        feat_info = json.load(f)
    feature_cols = feat_info["feature_columns"]
    base_cols = [c for c in feature_cols if not c.endswith("_missing")]
    nan_source_cols = feat_info["nan_indicator_source_columns"]

    Xte = test[base_cols].copy()
    for c in nan_source_cols:
        Xte[f"{c}_missing"] = Xte[c].isna().astype(int)
    imputer = joblib.load(models_dir / "05_imputer.pkl")
    Xte[base_cols] = imputer.transform(Xte[base_cols])
    Xte = Xte[feature_cols]
    yte = test["target"].values

    results = {}
    roc_data, pr_data, cal_data = {}, {}, {}

    for name, fname in MODEL_FILES.items():
        path = models_dir / fname
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        model = joblib.load(path)
        proba = model.predict_proba(Xte)[:, 1]
        pred_05 = (proba >= 0.5).astype(int)

        fpr, tpr, _ = roc_curve(yte, proba)
        precision_arr, recall_arr, pr_thresh = precision_recall_curve(yte, proba)
        f1_by_thresh = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-12)
        best_f1_idx = np.nanargmax(f1_by_thresh)
        best_f1_thresh = pr_thresh[min(best_f1_idx, len(pr_thresh) - 1)]
        pred_bestf1 = (proba >= best_f1_thresh).astype(int)

        cm = confusion_matrix(yte, pred_05)
        frac_pos, mean_pred = calibration_curve(yte, proba, n_bins=10, strategy="quantile")

        metrics = {
            "roc_auc": float(roc_auc_score(yte, proba)),
            "pr_auc": float(average_precision_score(yte, proba)),
            "precision_at_0.5": float(precision_score(yte, pred_05, zero_division=0)),
            "recall_at_0.5": float(recall_score(yte, pred_05, zero_division=0)),
            "f1_at_0.5": float(f1_score(yte, pred_05, zero_division=0)),
            "best_f1_threshold": float(best_f1_thresh),
            "precision_at_best_f1": float(precision_score(yte, pred_bestf1, zero_division=0)),
            "recall_at_best_f1": float(recall_score(yte, pred_bestf1, zero_division=0)),
            "f1_at_best_f1_threshold": float(f1_score(yte, pred_bestf1, zero_division=0)),
            "brier_score": float(brier_score_loss(yte, proba)),
            "confusion_matrix_at_0.5": cm.tolist(),  # [[TN, FP], [FN, TP]]
            "n_test": int(len(yte)),
            "n_positive_test": int(yte.sum()),
        }
        results[name] = metrics
        roc_data[name] = (fpr, tpr)
        pr_data[name] = (recall_arr, precision_arr)
        cal_data[name] = (mean_pred, frac_pos)

        print(f"\n{name}: ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  "
              f"F1@0.5={metrics['f1_at_0.5']:.4f}  Brier={metrics['brier_score']:.4f}")
        print(f"  Confusion matrix @0.5 [[TN,FP],[FN,TP]]: {cm.tolist()}")

        # feature importance
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                importances = None
            if importances is not None:
                fi = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(15)
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.barplot(x=fi.values, y=fi.index, ax=ax, color=COLORS.get(name, "#4C72B0"))
                ax.set_title(f"Top 15 Feature Importances — {name}")
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                fig.tight_layout()
                safe_name = name.lower().replace(" ", "_").replace("+", "").replace("__", "_")
                fig.savefig(outdir / f"06_feature_importance_{safe_name}.png", dpi=150)
                plt.close(fig)
        except Exception as e:
            print(f"  (feature importance plot skipped: {e})")

    # --- comparison table -----------------------------------------------------
    comp_rows = []
    for name, m in results.items():
        comp_rows.append({
            "model": name, "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
            "precision@0.5": m["precision_at_0.5"], "recall@0.5": m["recall_at_0.5"],
            "f1@0.5": m["f1_at_0.5"], "f1@best_thresh": m["f1_at_best_f1_threshold"],
            "brier_score": m["brier_score"],
        })
    comp_df = pd.DataFrame(comp_rows).sort_values("roc_auc", ascending=False)
    comp_df.to_csv(outdir / "06_evaluation_metrics.csv", index=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(comp_df.to_string(index=False))

    with open(outdir / "06_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # --- best model selection --------------------------------------------------
    best_name = comp_df.iloc[0]["model"]
    best_metrics = results[best_name]
    runner_up = comp_df.iloc[1] if len(comp_df) > 1 else None
    justification = (
        f"{best_name} selected as best model: highest test-set ROC-AUC "
        f"({best_metrics['roc_auc']:.4f}), the primary tuning metric specified "
        f"in the project proposal (GridSearchCV scoring='roc_auc')."
    )
    if runner_up is not None:
        justification += (
            f" Runner-up was {runner_up['model']} (ROC-AUC={runner_up['roc_auc']:.4f}). "
            f"PR-AUC (more informative than ROC-AUC under the ~"
            f"{100 * best_metrics['n_positive_test'] / best_metrics['n_test']:.1f}% test-set class "
            f"imbalance) was also checked as a tie-breaker/consistency check: "
            + ", ".join(f"{row['model']}={row['pr_auc']:.4f}" for _, row in comp_df.iterrows())
            + "."
        )
    best_model_report = {
        "best_model": best_name,
        "justification": justification,
        "metrics": best_metrics,
        "comparison_table": comp_rows,
    }
    with open(outdir / "06_best_model.json", "w", encoding="utf-8") as f:
        json.dump(best_model_report, f, indent=2)
    print(f"\nBest model: {best_name}")
    print(justification)

    # --- plots ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr) in roc_data.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})", color=COLORS.get(name))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "06_roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    base_rate = yte.mean()
    for name, (recall_arr, precision_arr) in pr_data.items():
        ax.plot(recall_arr, precision_arr, label=f"{name} (AP={results[name]['pr_auc']:.3f})", color=COLORS.get(name))
    ax.axhline(base_rate, linestyle="--", color="grey", label=f"Chance (base rate={base_rate:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(outdir / "06_pr_curves.png", dpi=150)
    plt.close(fig)

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
    for ax, (name, m) in zip(axes, results.items()):
        cm = np.array(m["confusion_matrix_at_0.5"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        ax.set_title(name)
    fig.suptitle("Confusion Matrices @ threshold=0.5 — Test Set")
    fig.tight_layout()
    fig.savefig(outdir / "06_confusion_matrices.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (mean_pred, frac_pos) in cal_data.items():
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{name} (Brier={results[name]['brier_score']:.4f})",
                color=COLORS.get(name))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curves — Test Set (10 quantile bins)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "06_calibration_curves.png", dpi=150)
    plt.close(fig)

    print(f"\nAll evaluation artifacts written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
