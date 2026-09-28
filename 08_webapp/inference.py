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
recent FORECAST_HORIZON_DAYS-1 rows per cell (`dropna(subset=["target_h7"])`
before the multi-horizon melt) because their target label isn't knowable
yet — that's correct for *training* data, but it means the matrix has no
row for "today", which is exactly the row a live forecast dashboard needs.
This module recomputes today's feature row per active cell (reusing the
same rolling-window code), without ever touching the target column.

Note on the "day slider" in the UI (fixed - read this if you're wondering
why this looks different from an older version): stage 04 used to label
each (cell, day) with a single fixed "does an event happen in the next 7
days" target, which meant a model trained on it could only ever produce
ONE probability per cell, no matter which day of the 7-day window you
asked about — moving the old slider just replayed the same number at
every position. Stage 04 now labels every horizon_day in 1..
FORECAST_HORIZON_DAYS separately (see its docstring), and `horizon_day` is
an input feature the models were retrained on. So this module computes
ONE feature snapshot for today (`_compute_live_features`), then expands it
into FORECAST_HORIZON_DAYS rows per cell — one per horizon_day
(`_expand_horizons`) — and predicts each separately. The slider now
directly selects horizon_day, and the numbers it shows are genuinely
different per position because the model was trained to treat "risk by
tomorrow" and "risk by day 7" as different questions.

Magnitude readout: the classifier only ever answers "will a
>=MAG_THRESHOLD event happen" — a single deterministic magnitude number
isn't something a short-term forecasting model can honestly produce.
`_magnitude_estimate` instead reports the Gutenberg-Richter-implied
magnitude distribution *conditional on* such an event occurring (using
scripts/04_feature_engineering.py's `magnitude_exceedance_prob` /
`magnitude_quantile` and each cell's own b-value) — a most-likely magnitude
and a "could reach this in a severe scenario" figure, not a point forecast.

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
FORECAST_HORIZON_DAYS = _feat_mod.FORECAST_HORIZON_DAYS  # max horizon_day (7)
MAG_THRESHOLD = 4.5

_lock = threading.Lock()
_cache = {"forecast_df": None, "asof_date": None}

_TARGET_COLS = [f"target_h{d}" for d in range(1, FORECAST_HORIZON_DAYS + 1)]


def _compute_live_features(min_events_per_cell=None, n_lookback_days=1):
    """Recompute the last `n_lookback_days` feature rows (one per day) for
    every active cell, using the committed cleaned catalog
    (outputs/03_processed/03_cleaned_catalog.csv). `compute_cell_features`
    already returns a full per-cell daily time series, so keeping the last
    N rows instead of just the last one costs essentially nothing extra.
    Returns a DataFrame: n_lookback_days rows per active cell (no target_h*
    columns - they're not knowable for any of these recent days). Default
    n_lookback_days=1 -> just today's row, which is all the live dashboard
    needs now that horizon_day (not which day this snapshot was taken) is
    what varies the forecast."""
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

    # neighbor_decay_count_hl{N}d (stage 04's spatial feature: exponentially
    # decayed activity in a cell's 8 surrounding cells) is a CROSS-cell
    # computation, done once outside the per-cell loop in stage 04's main()
    # - not something compute_cell_features itself produces. Reusing stage
    # 04's own function here (not reimplementing it) so the live dashboard
    # can never drift from what the models were actually trained on, same
    # principle as everything else in this module.
    neighbor_activity = _feat_mod.compute_neighbor_decay_activity(
        df, active_cells, all_days, _feat_mod.NEIGHBOR_DECAY_HALF_LIFE_DAYS
    )
    neighbor_col = f"neighbor_decay_count_hl{_feat_mod.NEIGHBOR_DECAY_HALF_LIFE_DAYS}d"

    rows = []
    for clat, clon in active_cells:
        g = df_active[(df_active["cell_lat_floor"] == clat) & (df_active["cell_lon_floor"] == clon)]
        mc_cell = _feat_mod.maxc_completeness_magnitude(g["mag"].values)
        feats = _feat_mod.compute_cell_features(g, all_days, mc_cell, mag_threshold)
        feats[neighbor_col] = neighbor_activity[(clat, clon)]
        recent = feats.tail(n_lookback_days).copy()
        recent["day"] = recent.index
        recent["cell_lat"] = clat + 0.5
        recent["cell_lon"] = clon + 0.5
        recent["mc_cell"] = mc_cell
        rows.append(recent)

    live = pd.concat(rows, axis=0).drop(columns=_TARGET_COLS, errors="ignore")
    doy = live["day"].dt.dayofyear
    live["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    live["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    live = live.reset_index(drop=True)
    return live, asof_date


def _expand_horizons(today_df):
    """Repeat each cell's today-row once per horizon_day in
    1..FORECAST_HORIZON_DAYS, adding `horizon_day` as a feature. This is
    the piece that makes the dashboard's slider genuinely dynamic: the
    trained model reads horizon_day like any other feature, so the SAME
    underlying rolling-window snapshot produces a different, learned
    probability for "risk by tomorrow" vs. "risk by day 7" instead of the
    single fixed-window answer the old single-horizon target could give."""
    frames = []
    for h in range(1, FORECAST_HORIZON_DAYS + 1):
        f = today_df.copy()
        f["horizon_day"] = h
        frames.append(f)
    return pd.concat(frames, axis=0, ignore_index=True)


def _magnitude_estimate(live_df):
    """Per-row Gutenberg-Richter magnitude readout, conditional on a
    >=MAG_THRESHOLD event occurring (see scripts/04_feature_engineering.py's
    "MAGNITUDE ESTIMATE" docstring section) - NOT a point prediction.
    Prefers b_value_90d (more responsive to recent activity), falls back to
    b_value_365d, then to a fixed global default when both are NaN (too few
    qualifying events in either window). Returns a DataFrame with
    mag_likely (50% exceedance - "most likely around here") and
    mag_severe (10% exceedance - "could reach this in a worse-case
    scenario")."""
    b = live_df["b_value_90d"].where(
        live_df["b_value_90d"].notna(), live_df["b_value_365d"]
    )
    b = b.fillna(_feat_mod.DEFAULT_B_VALUE).clip(lower=0.3)  # guard against a degenerate near-zero b
    mag_likely = _feat_mod.magnitude_quantile(b.to_numpy(), MAG_THRESHOLD, 0.5)
    mag_severe = _feat_mod.magnitude_quantile(b.to_numpy(), MAG_THRESHOLD, 0.1)
    return pd.DataFrame({"mag_likely": mag_likely, "mag_severe": mag_severe}, index=live_df.index)


def _check_model_matches_schema(key, model, feature_cols):
    """Fail early and legibly if a model on disk was fit on a different
    feature set than 05_feature_columns.json declares. Without this, the
    mismatch only surfaces deep inside predict_proba as sklearn's
    "feature names unseen at fit time" - which says nothing about the
    actual cause (stage 05's artifacts being from two different runs) or
    the fix. Stage 05 now writes all its artifacts together at the end so
    this shouldn't happen, but a half-finished or interrupted training run
    is exactly when a clear message matters most."""
    fitted_on = getattr(model, "feature_names_in_", None)
    if fitted_on is None:
        return  # model didn't record feature names (e.g. fit on a bare array) - nothing to check
    missing = [c for c in feature_cols if c not in set(fitted_on)]
    extra = [c for c in fitted_on if c not in set(feature_cols)]
    if missing or extra:
        raise RuntimeError(
            f"Model '{key}' ({MODEL_KEYS[key][1]}) was trained on a different feature set than "
            f"05_feature_columns.json declares.\n"
            f"  Declared but not in the model: {missing}\n"
            f"  In the model but not declared: {extra}\n"
            f"This means outputs/05_models/ holds artifacts from two different training runs - "
            f"usually because stage 05 is still running, or was interrupted partway.\n"
            f"Fix: re-run `py -3.14 05_train_models.py` from scripts/ and let it finish "
            f"(it writes every artifact at the end, so the directory stays self-consistent)."
        )


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
            model = joblib.load(path)
            _check_model_matches_schema(key, model, feature_cols)
            models[key] = model
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


def _refresh_cache():
    today_df, asof_date = _compute_live_features()
    expanded = _expand_horizons(today_df)
    proba = _predict_all_models(expanded)
    mag = _magnitude_estimate(expanded)

    out = expanded[["horizon_day", "cell_lat", "cell_lon"] + FEATURE_DISPLAY_COLS].copy()
    for key in MODEL_KEYS:
        out[f"proba_{key}"] = proba.get(key, np.nan)
    out["mag_likely"] = mag["mag_likely"]
    out["mag_severe"] = mag["mag_severe"]
    out = out.replace({np.nan: None})

    _cache["forecast_df"] = out
    _cache["asof_date"] = asof_date


def get_forecast(horizon_day=1, force_refresh=False):
    """Cached: (day_df, asof_date, horizon_end_date). day_df has one row per
    active cell for the selected horizon_day, with cell_lat/cell_lon,
    display features, a probability column per model key, and a
    Gutenberg-Richter magnitude readout (mag_likely/mag_severe). All rows
    come from a SINGLE today-feature-snapshot (asof_date) - horizon_day
    (1..FORECAST_HORIZON_DAYS) selects which "risk by day N" question the
    model answers about that same snapshot, which is what makes moving the
    slider produce genuinely different numbers (see module docstring)."""
    with _lock:
        if _cache["forecast_df"] is None or force_refresh:
            _refresh_cache()

        horizon_day = max(1, min(int(horizon_day), FORECAST_HORIZON_DAYS))

        day_df = _cache["forecast_df"]
        day_df = day_df[day_df["horizon_day"] == horizon_day].drop(columns=["horizon_day"]).reset_index(drop=True)
        asof_date = _cache["asof_date"]
        horizon_end_date = asof_date + pd.Timedelta(days=horizon_day)

        return day_df, asof_date, horizon_end_date


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
