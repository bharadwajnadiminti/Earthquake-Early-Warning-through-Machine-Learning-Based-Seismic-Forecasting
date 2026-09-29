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

from pathlib import Path

import pandas as pd

import app as app_module
import inference

WEBAPP_DIR = Path(__file__).resolve().parent.parent


def test_prob_threshold_matches_reference_apps_cutoff():
    assert app_module.PROB_THRESHOLD == 0.3


def test_get_earth_quake_estimates_filters_none_and_low_probability(monkeypatch):
    fake_day_df = pd.DataFrame([
        {"cell_lat": 1.5, "cell_lon": 2.5, "proba_xgboost": None},  # missing -> skipped
        {"cell_lat": 3.5, "cell_lon": 4.5, "proba_xgboost": 0.1},   # <= threshold -> skipped
        {"cell_lat": 5.5, "cell_lon": 6.5, "proba_xgboost": 0.9},   # > threshold -> kept
    ])
    fake_asof_date = "FAKE_ASOF"
    fake_horizon_end = "FAKE_HORIZON_END"

    def fake_get_forecast(horizon_day=1, force_refresh=False):
        return fake_day_df, fake_asof_date, fake_horizon_end

    monkeypatch.setattr(app_module.inference, "get_forecast", fake_get_forecast)

    lat_lng, asof_date, horizon_end = app_module.get_earth_quake_estimates(1)

    # Exactly the literal the reference app's template expects, for the one
    # cell above the threshold and no others.
    assert lat_lng == "new google.maps.LatLng(5.5,6.5),"
    assert asof_date == fake_asof_date
    assert horizon_end == fake_horizon_end


def test_get_earth_quake_estimates_real_cache_stays_within_contract():
    # Uses the session-warmed real cache (see conftest.py) - proves the
    # function's contract holds against the actual trained pipeline, not
    # just a mock.
    lat_lng, asof_date, horizon_end = app_module.get_earth_quake_estimates(3)

    assert isinstance(lat_lng, str)
    assert lat_lng.startswith("new google.maps.LatLng(")
    assert lat_lng.endswith(",")
    # One LatLng literal per cell shown, nothing malformed in between.
    parts = [p for p in lat_lng.split("),") if p]
    assert all(p.startswith("new google.maps.LatLng(") for p in parts)
    assert (horizon_end - asof_date).days == 3


def test_get_earth_quake_estimates_different_horizons_can_differ():
    # Regression guard for the bug this project was built to fix: the
    # dashboard used to show an identical forecast at every "days ahead"
    # slider position because the model only had one fixed 7-day window.
    # horizon_day is now a real input feature, so day 1 and day 7 should
    # put a genuinely different set of cells on the map.
    lat_lng_1, asof_1, end_1 = app_module.get_earth_quake_estimates(1)
    lat_lng_7, asof_7, end_7 = app_module.get_earth_quake_estimates(7)

    assert asof_1 == asof_7               # same today-snapshot
    assert (end_7 - end_1).days == inference.FORECAST_HORIZON_DAYS - 1  # different horizon end dates
    assert lat_lng_1 != lat_lng_7         # NOT the same forecast replayed


def test_build_page_get_renders_default_slider_position(flask_client):
    resp = flask_client.get("/")
    assert resp.status_code == 200
    assert b"Earthquake Forecaster" in resp.data
    assert b"google.maps.visualization.HeatmapLayer" in resp.data
    assert b"function getPoints()" in resp.data


def test_build_page_post_with_slider_value(flask_client):
    resp = flask_client.post("/", data={"slider_date_horizon": "3"})
    assert resp.status_code == 200
    assert b"new google.maps.LatLng(" in resp.data  # real forecast points injected


def test_build_page_never_leaks_a_hardcoded_api_key():
    """The reference app shipped a live Google Maps key in its page source.
    This project reads it from an untracked file instead, so the TEMPLATE must
    never contain a literal key - only the injected placeholder's value."""
    template = (WEBAPP_DIR / "templates" / "index.html").read_text(encoding="utf-8")
    assert "AIza" not in template, "a literal Google API key is committed in the template"
    assert "{{ google_maps_key }}" in template


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
