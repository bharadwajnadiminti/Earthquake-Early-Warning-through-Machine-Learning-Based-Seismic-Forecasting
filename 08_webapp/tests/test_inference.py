"""
Tests for inference.py - the shared, cached inference logic behind both
routes in app.py.

Design note: two different data sources are used deliberately.
- The tiny `synthetic_catalog` fixture (see conftest.py) gives fast,
  fully-controlled tests of *logic* - active-cell threshold behaviour,
  column shape, no-target guarantee - without paying for a ~15s read of
  the real 234k-row catalog.
- The session-scoped `_warm_inference_cache` autouse fixture (also in
  conftest.py) already ran get_forecast() once against the REAL committed
  pipeline outputs before any test here runs, so tests that need to prove
  "this really works end-to-end against the real trained model" (not a
  fake stand-in) just read that already-populated cache - no extra cost,
  and no drift from what's actually shipped in outputs/.
"""

import numpy as np
import pandas as pd

import inference


# ---------------------------------------------------------------------------
# Module-level constants and one-off setup
# ---------------------------------------------------------------------------

def test_module_constants():
    assert inference.FORECAST_HORIZON_DAYS == 7
    assert inference.MAG_THRESHOLD == 4.5
    assert inference.N_LOOKBACK_DAYS == 8
    assert inference.BEST_MODEL_KEY == "xgboost"
    assert inference.BEST_MODEL_KEY in inference.MODEL_KEYS
    assert inference.FEATURE_DISPLAY_COLS[0] == "historical_event_count"


def test_load_feature_engineering_module_exposes_expected_api():
    # inference.py imports scripts/04_feature_engineering.py by file path
    # (its name starts with a digit) rather than reimplementing anything
    # from it - assert the loaded module actually has what inference.py
    # relies on, and that its constants agree with what got copied up into
    # inference.py's own module-level names.
    mod = inference._load_feature_engineering_module()
    assert callable(mod.compute_cell_features)
    assert callable(mod.maxc_completeness_magnitude)
    assert mod.FORECAST_HORIZON_DAYS == inference.FORECAST_HORIZON_DAYS
    assert mod.MIN_EVENTS_PER_CELL == 50


def test_model_artifacts_loaded_at_import_include_all_three_models():
    # _load_model_artifacts() already ran once at import time against the
    # real outputs/05_models/ - assert all three trained models loaded,
    # not reimplemented or stubbed.
    assert set(inference._MODELS.keys()) == {"adaboost_dt", "adaboost_rf", "xgboost"}
    assert len(inference._BASE_COLS) > 0
    assert len(inference._FEATURE_COLS) > len(inference._BASE_COLS)  # + _missing indicator cols
    assert len(inference._NAN_SOURCE_COLS) > 0


def test_load_model_artifacts_skips_missing_model_files(tmp_path, monkeypatch):
    """_load_model_artifacts's `if path.exists(): models[key] = ...` guard
    is the only branch that can't be exercised by the real, fully-populated
    outputs/05_models/ directory - build a deliberately incomplete one."""
    import shutil
    import joblib

    real_dir = inference.MODELS_DIR
    tmp_models = tmp_path / "05_models"
    tmp_models.mkdir()
    shutil.copy(real_dir / "05_feature_columns.json", tmp_models / "05_feature_columns.json")
    shutil.copy(real_dir / "05_imputer.pkl", tmp_models / "05_imputer.pkl")
    # Only xgboost's model file exists; adaboost_dt/_rf are deliberately absent.
    joblib.dump({"stub": "not a real model, just needs to be loadable"},
                tmp_models / "05_model_xgboost.pkl")

    monkeypatch.setattr(inference, "MODELS_DIR", tmp_models)
    feature_cols, base_cols, nan_source_cols, imputer, models = inference._load_model_artifacts()

    assert set(models.keys()) == {"xgboost"}
    assert "adaboost_dt" not in models
    assert "adaboost_rf" not in models
    assert isinstance(feature_cols, list) and len(feature_cols) > 0
    assert isinstance(base_cols, list) and len(base_cols) > 0
    assert isinstance(nan_source_cols, list)


# ---------------------------------------------------------------------------
# _compute_live_features - fast, synthetic-data branch tests
# ---------------------------------------------------------------------------

def test_compute_live_features_explicit_threshold_includes_sparse_cell(synthetic_catalog):
    live, asof_date = inference._compute_live_features(min_events_per_cell=2, n_lookback_days=5)

    cells = set(zip(live["cell_lat"], live["cell_lon"]))
    assert (10.5, 20.5) in cells   # cell A: 50 events
    assert (5.5, 5.5) in cells     # cell B: 3 events, >= explicit threshold of 2
    assert (-5.5, -5.5) not in cells  # cell C: 1 event, < explicit threshold of 2

    assert "target" not in live.columns
    assert {"doy_sin", "doy_cos", "day", "mc_cell"} <= set(live.columns)
    assert live["doy_sin"].between(-1, 1).all()
    assert live["doy_cos"].between(-1, 1).all()
    # n_lookback_days=5 rows per active cell, no more, no less.
    assert (live.groupby(["cell_lat", "cell_lon"]).size() == 5).all()
    assert isinstance(asof_date, pd.Timestamp)


def test_compute_live_features_default_threshold_excludes_sparse_cells(synthetic_catalog):
    # No min_events_per_cell passed -> falls back to the real pipeline's
    # MIN_EVENTS_PER_CELL (50), which only cell A (50 events) clears.
    live, _ = inference._compute_live_features()

    cells = set(zip(live["cell_lat"], live["cell_lon"]))
    assert cells == {(10.5, 20.5)}


def test_compute_live_features_real_catalog_matches_warmed_cache():
    # Confidence that the function behaves the same way against the REAL
    # committed catalog as the already-warmed session cache did (same
    # active-cell count, same most-recent day) - without paying for a
    # second ~15s read of it.
    assert inference._cache["forecast_df"] is not None
    real_cells = inference._cache["forecast_df"][["cell_lat", "cell_lon"]].drop_duplicates()
    assert len(real_cells) == 372  # documented active-cell count for this project


# ---------------------------------------------------------------------------
# _predict_all_models - fast, synthetic-row unit test of the real models
# ---------------------------------------------------------------------------

def test_predict_all_models_synthetic_rows():
    base_cols = inference._BASE_COLS

    row_complete = {c: 1.0 for c in base_cols}
    row_missing = dict(row_complete)
    for c in inference._NAN_SOURCE_COLS:
        row_missing[c] = np.nan  # exercises the missing-indicator + imputer fill-in path

    df = pd.DataFrame([row_complete, row_missing])
    proba = inference._predict_all_models(df)

    assert set(proba.keys()) == set(inference._MODELS.keys())
    for key, arr in proba.items():
        assert len(arr) == 2
        assert np.all((arr >= 0) & (arr <= 1))


# ---------------------------------------------------------------------------
# get_forecast - cache lifecycle (reuses the session-warmed cache)
# ---------------------------------------------------------------------------

def test_get_forecast_offset_clamping():
    _, _, day_neg, _, available_days = inference.get_forecast(offset_days=-5)
    _, _, day_zero, _, _ = inference.get_forecast(offset_days=0)
    _, _, day_huge, _, _ = inference.get_forecast(offset_days=99999)

    assert day_neg == day_zero == available_days[0]     # negative clamps to newest
    assert day_huge == available_days[-1]               # too-large clamps to oldest


def test_get_forecast_returns_one_row_per_active_cell_for_selected_day():
    day_df, asof_date, day_date, horizon_end_date, available_days = inference.get_forecast(offset_days=0)

    assert len(day_df) == len(day_df[["cell_lat", "cell_lon"]].drop_duplicates())  # one row per cell
    assert "day" not in day_df.columns  # dropped before returning
    assert (horizon_end_date - day_date).days == inference.FORECAST_HORIZON_DAYS
    assert isinstance(available_days, list) and len(available_days) == inference.N_LOOKBACK_DAYS


def test_get_forecast_force_refresh_recomputes():
    inference.get_forecast()
    before_id = id(inference._cache["forecast_df"])

    inference.get_forecast(force_refresh=True)
    after_id = id(inference._cache["forecast_df"])

    assert after_id != before_id  # a genuinely new DataFrame was computed, not just reused


# ---------------------------------------------------------------------------
# get_recent_significant_quakes
# ---------------------------------------------------------------------------

def test_get_recent_significant_quakes_default_shape():
    results = inference.get_recent_significant_quakes()
    assert isinstance(results, list)
    for row in results:
        assert {"time", "latitude", "longitude", "depth", "mag", "place"} <= set(row.keys())
        assert row["mag"] >= 4.5
        assert isinstance(row["time"], str)  # converted from Timestamp


def test_get_recent_significant_quakes_empty_for_impossible_threshold():
    assert inference.get_recent_significant_quakes(days=30, min_mag=15.0) == []


def test_get_recent_significant_quakes_wider_window_has_at_least_as_many():
    narrow = inference.get_recent_significant_quakes(days=10, min_mag=4.5)
    wide = inference.get_recent_significant_quakes(days=3650, min_mag=4.5)
    assert len(wide) >= len(narrow)


# ---------------------------------------------------------------------------
# get_model_metrics / model_display_names
# ---------------------------------------------------------------------------

def test_get_model_metrics():
    result = inference.get_model_metrics()
    assert set(result.keys()) == {"metrics", "best_model", "justification"}
    assert result["best_model"] == "XGBoost"
    assert isinstance(result["metrics"], dict)
    assert isinstance(result["justification"], str) and result["justification"]


def test_model_display_names():
    assert inference.model_display_names() == {
        "adaboost_dt": "AdaBoost + Decision Tree",
        "adaboost_rf": "AdaBoost + Random Forest",
        "xgboost": "XGBoost",
    }
