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
        {"cell_lat": 5.5, "cell_lon": 6.5, "proba_xgboost": 0.9, "mag_likely": 4.8, "mag_severe": 5.9},  # > threshold -> kept
    ])
    fake_asof_date = "FAKE_ASOF"
    fake_horizon_end = "FAKE_HORIZON_END"

    def fake_get_forecast(horizon_day=1, force_refresh=False):
        return fake_day_df, fake_asof_date, fake_horizon_end

    monkeypatch.setattr(app_module.inference, "get_forecast", fake_get_forecast)

    points, asof_date, horizon_end = app_module.get_earth_quake_estimates(1)

    assert points == [[5.5, 6.5, 0.9, 4.8, 5.9]]
    assert asof_date == fake_asof_date
    assert horizon_end == fake_horizon_end


def test_get_earth_quake_estimates_real_cache_stays_within_contract():
    # Uses the session-warmed real cache (see conftest.py) - proves the
    # function's contract holds against the actual trained pipeline, not
    # just a mock.
    points, asof_date, horizon_end = app_module.get_earth_quake_estimates(3)

    assert all(p[2] > app_module.PROB_THRESHOLD for p in points)
    assert all(len(p) == 5 for p in points)  # [lat, lon, proba, mag_likely, mag_severe]
    assert (horizon_end - asof_date).days == 3


def test_get_earth_quake_estimates_different_horizons_can_differ():
    # Regression guard for the bug this project was built to fix: the
    # dashboard used to show an identical forecast at every "days ahead"
    # slider position because the model only had one fixed 7-day window.
    # horizon_day is now a real input feature, so day 1 and day 7 should
    # give genuinely different risk probabilities from the SAME snapshot.
    points_1, asof_1, end_1 = app_module.get_earth_quake_estimates(1)
    points_7, asof_7, end_7 = app_module.get_earth_quake_estimates(7)

    assert asof_1 == asof_7               # same today-snapshot
    assert (end_7 - end_1).days == inference.FORECAST_HORIZON_DAYS - 1  # different horizon end dates
    probs_1 = sorted(p[2] for p in points_1)
    probs_7 = sorted(p[2] for p in points_7)
    assert probs_1 != probs_7             # NOT the same forecast replayed


def test_build_page_get_renders_default_slider_position(flask_client):
    resp = flask_client.get("/")
    assert resp.status_code == 200
    assert b"Earthquake Forecaster" in resp.data
    assert b"var earthquakeHorizon" in resp.data


def test_build_page_post_with_slider_value(flask_client):
    resp = flask_client.post("/", data={"slider_date_horizon": "3"})
    assert resp.status_code == 200
    assert b"var earthquakeHorizon" in resp.data


def test_build_page_post_without_slider_value_defaults_to_one(flask_client):
    # request.form.get('slider_date_horizon', 1) falls back to 1 when the
    # field is absent entirely - shouldn't raise even though int(1) is
    # called on an already-int default rather than a submitted string.
    resp = flask_client.post("/", data={})
    assert resp.status_code == 200


def test_build_page_historical_route_removed(flask_client):
    # Regression guard: the "Historical replay" view was built, tried, and
    # deliberately removed again in this project's history - make sure it
    # doesn't quietly come back.
    resp = flask_client.get("/historical")
    assert resp.status_code == 404
