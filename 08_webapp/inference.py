#!/usr/bin/env python3
"""
inference.py

Shared, cached inference logic for the stage 08 forecast dashboard
(app.py). Deliberately does NOT reimplement feature engineering: it
imports scripts/04_feature_engineering.py directly (by file path, since
its filename starts with a digit and can't be a normal import target) and
reuses its exact `compute_cell_features` / `maxc_completeness_magnitude`
functions and constants. This guarantees the live dashboard can never
drift from what the models were actually trained on.

Why this is needed at all: stage 04's own committed
outputs/04_features/04_feature_matrix.csv deliberately DROPS the most
recent FORECAST_HORIZON_DAYS-1 rows per cell (`dropna(subset=["target"])`)
because their target label isn't knowable yet — that's correct for
*training* data, but it means the matrix has no row for "today", which is
exactly the row a live forecast dashboard needs. This module recomputes
just that last row per active cell (reusing the same rolling-window code),
without ever touching the target column.

Input:  ../outputs/03_processed/03_cleaned_catalog.csv (stage 03 output)
        ../outputs/05_models/05_feature_columns.json, 05_imputer.pkl,
        05_model_{adaboost_dt,adaboost_rf,xgboost}.pkl (stage 05 output)
        ../outputs/06_metrics/06_evaluation_metrics.json, 06_best_model.json
        (stage 06 output, for the metrics panel)
"""

import importlib.util
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
PROCESSED_DIR = ROOT / "outputs" / "03_processed"
MODELS_DIR = ROOT / "outputs" / "05_models"
METRICS_DIR = ROOT / "outputs" / "06_metrics"

MODEL_KEYS = {
    "adaboost_dt": ("AdaBoost + Decision Tree", "05_model_adaboost_dt.pkl"),
    "adaboost_rf": ("AdaBoost + Random Forest", "05_model_adaboost_rf.pkl"),
    "xgboost": ("XGBoost", "05_model_xgboost.pkl"),
}
BEST_MODEL_KEY = "xgboost"

FEATURE_DISPLAY_COLS = [
    "historical_event_count", "days_since_last_event",
    "rolling_count_7d", "rolling_count_30d", "rolling_count_90d",
    "rolling_max_mag_365d", "mc_cell", "b_value_90d", "b_value_365d",
]


def _load_feature_engineering_module():
    """Import scripts/04_feature_engineering.py by path (filename starts
    with a digit, so it can't be `import 04_feature_engineering`).
    Executing the module only defines functions/constants at this indent
    level - `main()` is guarded by `if __name__ == "__main__"` and never
    runs here."""
    path = SCRIPTS_DIR / "04_feature_engineering.py"
    spec = importlib.util.spec_from_file_location("stage04_feature_engineering", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_feat_mod = _load_feature_engineering_module()
FORECAST_HORIZON_DAYS = _feat_mod.FORECAST_HORIZON_DAYS
MAG_THRESHOLD = 4.5

_lock = threading.Lock()
_cache = {"forecast_df": None, "asof_date": None, "horizon_end_date": None, "n_cells": None}


def _compute_live_features(min_events_per_cell=None):
    """Recompute today's feature row for every active cell, using the
    committed cleaned catalog (outputs/03_processed/03_cleaned_catalog.csv).
    Returns a DataFrame: one row per active cell, its most recent day's
    feature values (no target column - it's not knowable for today)."""
    mep = min_events_per_cell or _feat_mod.MIN_EVENTS_PER_CELL
    mag_threshold = 4.5

    catalog_path = PROCESSED_DIR / "03_cleaned_catalog.csv"
    df = pd.read_csv(catalog_path)
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)

    df["cell_lat_floor"] = np.floor(df["latitude"]).astype(int)
    df["cell_lon_floor"] = np.floor(df["longitude"]).astype(int)

    cell_counts = df.groupby(["cell_lat_floor", "cell_lon_floor"]).size()
    active_cells = cell_counts[cell_counts >= mep].index

    df_active = df.set_index(["cell_lat_floor", "cell_lon_floor"])
    df_active = df_active.loc[df_active.index.isin(active_cells)].reset_index()

    all_days = pd.date_range(
        df["time"].dt.floor("D").min(), df["time"].dt.floor("D").max(), freq="D", tz="UTC"
    )
    asof_date = all_days[-1]

    rows = []
    for clat, clon in active_cells:
        g = df_active[(df_active["cell_lat_floor"] == clat) & (df_active["cell_lon_floor"] == clon)]
        mc_cell = _feat_mod.maxc_completeness_magnitude(g["mag"].values)
        feats = _feat_mod.compute_cell_features(g, all_days, mc_cell, mag_threshold)
        last = feats.iloc[-1].copy()
        last["cell_lat"] = clat + 0.5
        last["cell_lon"] = clon + 0.5
        last["mc_cell"] = mc_cell
        rows.append(last)

    live = pd.DataFrame(rows).drop(columns=["target"], errors="ignore")
    doy = asof_date.dayofyear
    live["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    live["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    live = live.reset_index(drop=True)
    return live, asof_date


def _load_model_artifacts():
    with open(MODELS_DIR / "05_feature_columns.json", encoding="utf-8") as f:
        feat_info = json.load(f)
    feature_cols = feat_info["feature_columns"]
    base_cols = [c for c in feature_cols if not c.endswith("_missing")]
    nan_source_cols = feat_info["nan_indicator_source_columns"]
    imputer = joblib.load(MODELS_DIR / "05_imputer.pkl")

    models = {}
    for key, (_, fname) in MODEL_KEYS.items():
        path = MODELS_DIR / fname
        if path.exists():
            models[key] = joblib.load(path)
    return feature_cols, base_cols, nan_source_cols, imputer, models


_FEATURE_COLS, _BASE_COLS, _NAN_SOURCE_COLS, _IMPUTER, _MODELS = _load_model_artifacts()


def _predict_all_models(live_df):
    """Exact same preprocessing as stage 06 (missing-indicator + median
    impute, fit on train only, column order from 05_feature_columns.json),
    then predict_proba with every loaded model."""
    X = live_df[_BASE_COLS].copy()
    for c in _NAN_SOURCE_COLS:
        X[f"{c}_missing"] = X[c].isna().astype(int)
    X[_BASE_COLS] = _IMPUTER.transform(X[_BASE_COLS])
    X = X[_FEATURE_COLS]

    proba = {}
    for key, model in _MODELS.items():
        proba[key] = model.predict_proba(X)[:, 1]
    return proba


def get_forecast(force_refresh=False):
    """Cached: (forecast_df, asof_date, horizon_end_date). forecast_df has
    one row per active cell with cell_lat/cell_lon, display features, and
    a probability column per model key."""
    with _lock:
        if _cache["forecast_df"] is not None and not force_refresh:
            return _cache["forecast_df"], _cache["asof_date"], _cache["horizon_end_date"]

        live_df, asof_date = _compute_live_features()
        proba = _predict_all_models(live_df)

        out = live_df[["cell_lat", "cell_lon"] + FEATURE_DISPLAY_COLS].copy()
        for key in MODEL_KEYS:
            out[f"proba_{key}"] = proba.get(key, np.nan)
        out = out.replace({np.nan: None})

        horizon_end_date = asof_date + pd.Timedelta(days=_feat_mod.FORECAST_HORIZON_DAYS)

        _cache["forecast_df"] = out
        _cache["asof_date"] = asof_date
        _cache["horizon_end_date"] = horizon_end_date
        _cache["n_cells"] = len(out)
        return out, asof_date, horizon_end_date


def get_recent_significant_quakes(days=30, min_mag=4.5):
    catalog_path = PROCESSED_DIR / "03_cleaned_catalog.csv"
    df = pd.read_csv(catalog_path)
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    cutoff = df["time"].max() - pd.Timedelta(days=days)
    recent = df[(df["time"] >= cutoff) & (df["mag"] >= min_mag)]
    recent = recent[["time", "latitude", "longitude", "depth", "mag", "place"]]
    recent = recent.sort_values("time", ascending=False)
    recent["time"] = recent["time"].astype(str)
    return recent.replace({np.nan: None}).to_dict(orient="records")


def get_model_metrics():
    with open(METRICS_DIR / "06_evaluation_metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)
    with open(METRICS_DIR / "06_best_model.json", encoding="utf-8") as f:
        best = json.load(f)
    return {"metrics": metrics, "best_model": best["best_model"], "justification": best["justification"]}


def model_display_names():
    return {key: name for key, (name, _) in MODEL_KEYS.items()}
