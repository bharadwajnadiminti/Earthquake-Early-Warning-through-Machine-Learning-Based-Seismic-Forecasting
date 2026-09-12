"""
Tests for app.py - the Flask routing layer on top of inference.py.

get_earth_quake_estimates's own branch logic (skip None, skip <= threshold,
keep > threshold) is tested with a monkeypatched inference.get_forecast so
it's deterministic and doesn't depend on today's live data happening to
contain a None probability (it normally won't - every trained model
produces a real float for every row - so without this, that branch would
simply never be exercised). Route tests (build_page) then separately prove
the real, warmed cache flows all the way through to a rendered page.
"""

import pandas as pd

import app as app_module
import inference


def test_prob_threshold_matches_reference_apps_cutoff():
    assert app_module.PROB_THRESHOLD == 0.3


def test_get_earth_quake_estimates_filters_none_and_low_probability(monkeypatch):
    fake_day_df = pd.DataFrame([
        {"cell_lat": 1.5, "cell_lon": 2.5, "proba_xgboost": None},  # missing -> skipped
        {"cell_lat": 3.5, "cell_lon": 4.5, "proba_xgboost": 0.1},   # <= threshold -> skipped
        {"cell_lat": 5.5, "cell_lon": 6.5, "proba_xgboost": 0.9},   # > threshold -> kept
    ])
    fake_day_date = "FAKE_DAY"
    fake_horizon_end = "FAKE_HORIZON_END"

    def fake_get_forecast(offset_days=0, force_refresh=False):
        return fake_day_df, None, fake_day_date, fake_horizon_end, [fake_day_date]

    monkeypatch.setattr(app_module.inference, "get_forecast", fake_get_forecast)

    points, day_date, horizon_end = app_module.get_earth_quake_estimates(0)

    assert points == [[5.5, 6.5, 0.9]]
    assert day_date == fake_day_date
    assert horizon_end == fake_horizon_end


def test_get_earth_quake_estimates_real_cache_stays_within_contract():
    # Uses the session-warmed real cache (see conftest.py) - proves the
    # function's contract holds against the actual trained pipeline, not
    # just a mock.
    points, day_date, horizon_end = app_module.get_earth_quake_estimates(0)

    assert all(p[2] > app_module.PROB_THRESHOLD for p in points)
    assert all(len(p) == 3 for p in points)
    assert (horizon_end - day_date).days == inference.FORECAST_HORIZON_DAYS


def test_get_earth_quake_estimates_different_horizons_can_differ():
    points_0, day_0, _ = app_module.get_earth_quake_estimates(0)
    points_7, day_7, _ = app_module.get_earth_quake_estimates(7)

    assert day_0 != day_7  # different forecast snapshots


def test_build_page_get_renders_default_slider_position(flask_client):
    resp = flask_client.get("/")
    assert resp.status_code == 200
    assert b"Earthquake Forecaster" in resp.data
    assert b"var earthquakeHorizon" in resp.data


def test_build_page_post_with_slider_value(flask_client):
    resp = flask_client.post("/", data={"slider_date_horizon": "3"})
    assert resp.status_code == 200
    assert b"var earthquakeHorizon" in resp.data


def test_build_page_post_without_slider_value_defaults_to_zero(flask_client):
    # request.form.get('slider_date_horizon', 0) falls back to 0 when the
    # field is absent entirely - shouldn't raise even though int(0) is
    # called on an already-int default rather than a submitted string.
    resp = flask_client.post("/", data={})
    assert resp.status_code == 200


def test_build_page_historical_route_removed(flask_client):
    # Regression guard: the "Historical replay" view was built, tried, and
    # deliberately removed again in this project's history - make sure it
    # doesn't quietly come back.
    resp = flask_client.get("/historical")
    assert resp.status_code == 404
