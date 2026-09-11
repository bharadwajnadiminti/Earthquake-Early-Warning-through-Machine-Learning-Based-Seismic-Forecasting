#!/usr/bin/env python3
"""
05_train_models.py

Chronological train/val/test split + GridSearchCV(TimeSeriesSplit) tuning
for the three candidate models specified in the project proposal:

  1. AdaBoost + Decision Tree   (estimator__max_depth in [2,5,7],
                                  n_estimators in {200,300,400,500,600},
                                  learning_rate=0.6 fixed)
  2. AdaBoost + Random Forest   (n_estimators in {200,700},
                                  estimator__max_features in {'sqrt','log2'})
  3. XGBoost                    (booster='gbtree', objective='binary:logistic',
                                  centered on max_depth=6, eta=0.03,
                                  num_boost_round=2000 with early stopping)

--------------------------------------------------------------------------
IMPLEMENTATION NOTES / DEVIATIONS FROM THE PROPOSAL DECK (documented, not
silent)
--------------------------------------------------------------------------
- algorithm='SAMME': scikit-learn 1.6+ (this project uses 1.9.1) removed the
  `algorithm` parameter from AdaBoostClassifier entirely and removed
  SAMME.R - SAMME is now the ONLY algorithm it runs. So the deck's
  algorithm='SAMME' requirement is met automatically; passing the argument
  explicitly would raise a TypeError on this sklearn version. Also renamed:
  `base_estimator` -> `estimator`.
- AdaBoost + Random Forest compute budget: nesting a full RandomForest
  (sklearn default: 100 trees each) as AdaBoost's per-round weak learner,
  boosted up to 700 rounds, means up to 70,000 trees for one candidate
  model fit. That's intractable to grid-search on ~200k rows in this
  project's time budget. The deck does not specify the *inner* RF's own
  n_estimators/max_depth, so this script fixes them small
  (n_estimators=15, max_depth=4) - a shallow forest still gives AdaBoost a
  higher-variance/lower-bias weak learner than a single decision tree
  (satisfying the intent of "AdaBoost + Random Forest" as a distinct
  candidate from "AdaBoost + Decision Tree"), while keeping the grid search
  computationally tractable. This is a deliberate, documented compute-budget
  choice, not an oversight.
- XGBoost + GridSearchCV + early stopping: sklearn's GridSearchCV cannot
  pass a *per-fold* eval_set (required for early stopping) through its
  standard fit_params interface. So XGBoost tuning here is a manual grid
  search over TimeSeriesSplit folds (same CV strategy, same roc_auc scoring,
  same reported results shape as the other two models' GridSearchCV output),
  with early stopping active inside every fold fit. The deck's max_depth=6,
  eta=0.03 are used as the center of the small grid searched around them.
- Class imbalance: stage 04 reports an ~5.8% positive rate. `class_weight=
  'balanced'` is set on the AdaBoost base estimators (DecisionTree/
  RandomForest); XGBoost uses `scale_pos_weight` computed from the TRAIN
  split's class ratio only.

--------------------------------------------------------------------------
SPLIT / LEAKAGE CONTROL
--------------------------------------------------------------------------
Chronological 70/15/15 train/val/test split by unique day. Because the
target label looks FORWARD up to FORECAST_HORIZON_DAYS (7) days past its
row's day, a `gap` of 7 days is purged between train/val and val/test so
that no training-set label is computed from events that actually fall
inside the validation or test period. The same gap (converted to an
approximate row count) is passed to TimeSeriesSplit's `gap` parameter
during cross-validation within the training set, for the same reason.

Missing values (see stage 04's data dictionary - rolling_mean_mag/max_mag
and b_value columns can be NaN when a window has no/too few events): a
`<col>_missing` indicator is added for every column with any NaN, then
NaNs are median-imputed using an imputer FIT ON THE TRAIN SPLIT ONLY and
applied unchanged to val/test (fitting on the full dataset would leak
future information backward into training).

Input:  ../outputs/04_features/04_feature_matrix.csv
Output: ../outputs/05_models/05_model_<name>.pkl               (fitted model)
        ../outputs/05_models/05_<name>_best_params.json
        ../outputs/05_models/05_<name>_cv_results.csv
        ../outputs/05_models/05_imputer.pkl
        ../outputs/05_models/05_split_summary.json
        ../outputs/05_models/05_feature_columns.json
"""

import argparse
import itertools
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FORECAST_HORIZON_DAYS = 7   # must match stage 04's horizon
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15             # remainder (~0.15) is test
CV_N_SPLITS = 5

MODELS_DIR = Path("../outputs/05_models")


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


def build_features(train, val, test, feature_cols):
    Xtr, Xval, Xte = train[feature_cols].copy(), val[feature_cols].copy(), test[feature_cols].copy()

    nan_cols = [c for c in feature_cols if Xtr[c].isna().any() or Xval[c].isna().any() or Xte[c].isna().any()]
    for c in nan_cols:
        Xtr[f"{c}_missing"] = Xtr[c].isna().astype(int)
        Xval[f"{c}_missing"] = Xval[c].isna().astype(int)
        Xte[f"{c}_missing"] = Xte[c].isna().astype(int)

    imputer = SimpleImputer(strategy="median")
    imputer.fit(Xtr[feature_cols])
    Xtr[feature_cols] = imputer.transform(Xtr[feature_cols])
    Xval[feature_cols] = imputer.transform(Xval[feature_cols])
    Xte[feature_cols] = imputer.transform(Xte[feature_cols])

    return Xtr, Xval, Xte, imputer, nan_cols


def approx_gap_rows(train_df, day_col="day"):
    n_days = train_df[day_col].nunique()
    rows_per_day = len(train_df) / max(n_days, 1)
    return int(round(FORECAST_HORIZON_DAYS * rows_per_day))


def run_gridsearch(name, estimator, param_grid, X, y, cv, n_jobs=-1):
    print(f"\n=== {name}: GridSearchCV over {param_grid} ===")
    t0 = time.time()
    gs = GridSearchCV(estimator, param_grid, scoring="roc_auc", cv=cv, n_jobs=n_jobs, refit=True, verbose=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)
    elapsed = time.time() - t0
    print(f"{name}: best CV roc_auc={gs.best_score_:.4f}  best_params={gs.best_params_}  ({elapsed:.1f}s)")
    return gs, elapsed


def train_xgboost_manual_grid(X, y, X_val, y_val, cv, grid, fixed_params):
    """Manual grid search with early stopping per fold (see module docstring
    for why this can't use sklearn's GridSearchCV directly)."""
    print(f"\n=== xgboost: manual grid search over {grid} (TimeSeriesSplit + early stopping) ===")
    t0 = time.time()
    keys = list(grid.keys())
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*grid.values())]

    results = []
    for combo in combos:
        fold_scores = []
        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X)):
            params = dict(fixed_params)
            params.update(combo)
            n_estimators = params.pop("n_estimators")
            model = XGBClassifier(n_estimators=n_estimators, **params)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                eval_set=[(X.iloc[va_idx], y.iloc[va_idx])],
                verbose=False,
            )
            proba = model.predict_proba(X.iloc[va_idx])[:, 1]
            score = roc_auc_score(y.iloc[va_idx], proba)
            fold_scores.append(score)
        mean_score = float(np.mean(fold_scores))
        results.append({**combo, "mean_test_roc_auc": mean_score, "fold_scores": fold_scores})
        print(f"  {combo} -> mean CV roc_auc={mean_score:.4f}")

    best = max(results, key=lambda r: r["mean_test_roc_auc"])
    print(f"xgboost: best CV roc_auc={best['mean_test_roc_auc']:.4f}  best_params={ {k: best[k] for k in keys} }")

    # Refit final model on the full train set with early stopping against the
    # held-out validation split (this is the val split from the outer
    # chronological split, kept separate from CV entirely).
    final_params = dict(fixed_params)
    final_params.update({k: best[k] for k in keys})
    n_estimators = final_params.pop("n_estimators")
    final_model = XGBClassifier(n_estimators=n_estimators, **final_params)
    final_model.fit(X, y, eval_set=[(X_val, y_val)], verbose=False)

    elapsed = time.time() - t0
    print(f"xgboost total tuning + refit time: {elapsed:.1f}s, best_iteration={final_model.best_iteration}")
    return final_model, results, {k: best[k] for k in keys}, elapsed


def main():
    p = argparse.ArgumentParser(description="Train + tune the three candidate models (stage 05).")
    p.add_argument("--input", type=str, default="../outputs/04_features/04_feature_matrix.csv")
    p.add_argument("--outdir", type=str, default="../outputs/05_models")
    p.add_argument("--cv-splits", type=int, default=CV_N_SPLITS)
    p.add_argument("--skip", type=str, default="", help="Comma-separated model names to skip, e.g. 'adaboost_rf'")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df["day"] = pd.to_datetime(df["day"], utc=True)
    print(f"Loaded {len(df):,} rows, {df['day'].min()} -> {df['day'].max()}")

    train, val, test = chronological_split(df)
    print(f"\nSplit sizes: train={len(train):,} ({train['day'].min()} -> {train['day'].max()}), "
          f"val={len(val):,} ({val['day'].min() if len(val) else 'NA'} -> {val['day'].max() if len(val) else 'NA'}), "
          f"test={len(test):,} ({test['day'].min() if len(test) else 'NA'} -> {test['day'].max() if len(test) else 'NA'})")
    print(f"Positive rate: train={train['target'].mean():.4f}, val={val['target'].mean():.4f}, test={test['target'].mean():.4f}")

    feature_cols = [c for c in df.columns if c not in ("day", "target")]
    Xtr, Xval, Xte, imputer, nan_cols = build_features(train, val, test, feature_cols)
    ytr, yval, yte = train["target"], val["target"], test["target"]
    all_feature_cols = list(Xtr.columns)
    print(f"\nFeature columns ({len(all_feature_cols)}): {all_feature_cols}")
    print(f"Columns needing missing-indicator + median imputation: {nan_cols}")

    joblib.dump(imputer, outdir / "05_imputer.pkl")
    with open(outdir / "05_feature_columns.json", "w", encoding="utf-8") as f:
        json.dump({"feature_columns": all_feature_cols, "nan_indicator_source_columns": nan_cols}, f, indent=2)

    gap_rows = approx_gap_rows(train)
    cv = TimeSeriesSplit(n_splits=args.cv_splits, gap=gap_rows)
    print(f"\nTimeSeriesSplit: n_splits={args.cv_splits}, gap={gap_rows} rows (~{FORECAST_HORIZON_DAYS}d purge)")

    split_summary = {
        "train_rows": int(len(train)), "val_rows": int(len(val)), "test_rows": int(len(test)),
        "train_date_range": [str(train["day"].min()), str(train["day"].max())],
        "val_date_range": [str(val["day"].min()), str(val["day"].max())] if len(val) else None,
        "test_date_range": [str(test["day"].min()), str(test["day"].max())] if len(test) else None,
        "train_positive_rate": float(train["target"].mean()),
        "val_positive_rate": float(val["target"].mean()),
        "test_positive_rate": float(test["target"].mean()),
        "purge_gap_days": FORECAST_HORIZON_DAYS,
        "cv_n_splits": args.cv_splits,
        "cv_gap_rows_approx": gap_rows,
    }
    with open(outdir / "05_split_summary.json", "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    timings = {}

    # ---------------------------------------------------------------- model 1
    if "adaboost_dt" not in skip:
        base_dt = DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight="balanced")
        ada_dt = AdaBoostClassifier(estimator=base_dt, learning_rate=0.6, random_state=RANDOM_SEED)
        grid_dt = {
            "estimator__max_depth": [2, 5, 7],
            "n_estimators": [200, 300, 400, 500, 600],
        }
        gs_dt, t_dt = run_gridsearch("adaboost_dt", ada_dt, grid_dt, Xtr, ytr, cv)
        timings["adaboost_dt"] = t_dt
        joblib.dump(gs_dt.best_estimator_, outdir / "05_model_adaboost_dt.pkl")
        with open(outdir / "05_adaboost_dt_best_params.json", "w", encoding="utf-8") as f:
            json.dump({"best_params": gs_dt.best_params_, "best_cv_roc_auc": gs_dt.best_score_,
                       "fixed_params": {"learning_rate": 0.6, "algorithm": "SAMME (only option in sklearn>=1.6)"}}, f, indent=2)
        pd.DataFrame(gs_dt.cv_results_).to_csv(outdir / "05_adaboost_dt_cv_results.csv", index=False)

    # ---------------------------------------------------------------- model 2
    if "adaboost_rf" not in skip:
        base_rf = RandomForestClassifier(n_estimators=15, max_depth=4, class_weight="balanced",
                                          random_state=RANDOM_SEED, n_jobs=1)
        ada_rf = AdaBoostClassifier(estimator=base_rf, random_state=RANDOM_SEED)
        grid_rf = {
            "n_estimators": [200, 700],
            "estimator__max_features": ["sqrt", "log2"],
        }
        gs_rf, t_rf = run_gridsearch("adaboost_rf", ada_rf, grid_rf, Xtr, ytr, cv, n_jobs=-1)
        timings["adaboost_rf"] = t_rf
        joblib.dump(gs_rf.best_estimator_, outdir / "05_model_adaboost_rf.pkl")
        with open(outdir / "05_adaboost_rf_best_params.json", "w", encoding="utf-8") as f:
            json.dump({"best_params": gs_rf.best_params_, "best_cv_roc_auc": gs_rf.best_score_,
                       "fixed_params": {"inner_rf_n_estimators": 15, "inner_rf_max_depth": 4,
                                        "reason": "see module docstring: compute-budget choice"}}, f, indent=2)
        pd.DataFrame(gs_rf.cv_results_).to_csv(outdir / "05_adaboost_rf_cv_results.csv", index=False)

    # ---------------------------------------------------------------- model 3
    if "xgboost" not in skip:
        pos = int(ytr.sum())
        neg = int(len(ytr) - pos)
        scale_pos_weight = neg / max(pos, 1)
        fixed = dict(
            objective="binary:logistic", booster="gbtree", eval_metric="auc",
            tree_method="hist", early_stopping_rounds=50, random_state=RANDOM_SEED,
            scale_pos_weight=scale_pos_weight, n_jobs=-1,
        )
        grid_xgb = {"n_estimators": [2000], "max_depth": [4, 6, 8], "learning_rate": [0.01, 0.03, 0.1]}
        model_xgb, results_xgb, best_params_xgb, t_xgb = train_xgboost_manual_grid(
            Xtr, ytr, Xval, yval, cv, grid_xgb, fixed
        )
        timings["xgboost"] = t_xgb
        joblib.dump(model_xgb, outdir / "05_model_xgboost.pkl")
        with open(outdir / "05_xgboost_best_params.json", "w", encoding="utf-8") as f:
            json.dump({"best_params": best_params_xgb,
                       "best_iteration": int(model_xgb.best_iteration) if model_xgb.best_iteration is not None else None,
                       "fixed_params": {k: v for k, v in fixed.items()},
                       "scale_pos_weight": scale_pos_weight}, f, indent=2)
        pd.DataFrame(results_xgb).to_csv(outdir / "05_xgboost_cv_results.csv", index=False)

    with open(outdir / "05_training_timings.json", "w", encoding="utf-8") as f:
        json.dump(timings, f, indent=2)
    print(f"\nAll done. Timings (seconds): {timings}")


if __name__ == "__main__":
    main()
