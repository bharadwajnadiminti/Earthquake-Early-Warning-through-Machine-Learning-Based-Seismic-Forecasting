#!/usr/bin/env python
"""
app.py

Stage 08: forecast dashboard. UI/interaction follows the third-party
reference app this project's dashboard was modeled on - title bar, single
world map, one "select future date" slider, 1-7 days out - kept as close
to that original as possible (see templates/index.html). That reference
project is not vendored here (it was never committed - third-party
material, not part of this deliverable). The DATA PULLING AND MODEL
BUILDING behind it is this project's own verified pipeline, not the
reference app's from-scratch-every-restart approach: see inference.py,
which reuses scripts/04_feature_engineering.py's own feature code and
loads the actual GridSearchCV-tuned model saved by stage 05
(outputs/05_models/05_model_xgboost.pkl - the best model per stage 06),
rather than retraining a fresh, unevaluated, untuned model blind on every
restart against only the last rolling month of all-magnitude data.

Slider semantics: the model predicts P(M>=4.5 event by day N) directly for
N = 1..7 - `horizon_day` is a real input feature the models were retrained
on (scripts/04_feature_engineering.py), so moving the slider asks the SAME
today-snapshot a genuinely different question at each position ("risk by
tomorrow" vs. "risk by day 7"), rather than replaying one fixed-window
number (an earlier version of this project's target only had one 7-day
window total, so every slider position looked identical - see
inference.py's module docstring for the fix). Each point's popup also shows
a Gutenberg-Richter magnitude readout (most-likely / severe-scenario), not
a single deterministic magnitude - see inference.py's `_magnitude_estimate`
for why a point value would be dishonest here.

No external API key required (Leaflet + Esri tiles, not the Google Maps JS
API the original had a key hardcoded for - tested that exact original code,
Google has discontinued `google.maps.visualization.HeatmapLayer` entirely
as of Maps JS API v3.65, unrelated to the key's validity or this project).
Points, not a blended heatmap: one circle marker per active cell, colored/
sized continuously by that cell's own predicted probability, so real day-
to-day changes are visible even when they don't cross an old bucket edge.

Run locally:
    cd 08_webapp
    pip install -r requirements.txt -r ../requirements.txt
    python app.py
    -> http://127.0.0.1:5000
"""

from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, render_template, request

import inference

app = Flask(__name__)

PROB_THRESHOLD = 0.3  # same cutoff the reference app used to decide what to show

# The reference app had its Google Maps API key written into the page source.
# This repo is public, so the key lives in an untracked local file instead
# (08_webapp/.google_maps_key, see .gitignore). Absent that file the page still
# renders - Google just shows its "for development purposes only" watermark.
_KEY_FILE = Path(__file__).resolve().parent / ".google_maps_key"
GOOGLE_MAPS_KEY = _KEY_FILE.read_text(encoding="utf-8").strip() if _KEY_FILE.exists() else ""


def get_earth_quake_estimates(horizon_day):
    """Slider value (1-7, "days ahead") -> the exact string the reference
    app's template expects: a comma-separated run of
    `new google.maps.LatLng(lat,lon),` literals, injected raw into
    getPoints()'s array and consumed by google.maps.visualization.HeatmapLayer.

    Same PROB_THRESHOLD cutoff the reference used. What differs is only what
    sits behind the number: this project's tuned, stage-06-evaluated
    multi-horizon model rather than a model retrained from scratch on every
    restart (see inference.py)."""
    day_df, asof_date, horizon_end_date = inference.get_forecast(horizon_day=horizon_day)

    lat_lng_string = ''
    for row in day_df.to_dict(orient="records"):
        p = row.get(f"proba_{inference.BEST_MODEL_KEY}")
        if p is not None and p > PROB_THRESHOLD:
            lat_lng_string += f"new google.maps.LatLng({row['cell_lat']},{row['cell_lon']}),"
    return lat_lng_string, asof_date, horizon_end_date


@app.route("/", methods=['POST', 'GET'])
def build_page():
    horizon_int = int(request.form.get('slider_date_horizon', 1)) if request.method == 'POST' else 1
    # The slider's own range starts at 0 ("today"), which has no forecast
    # window at all - the reference app rendered a blank map there. Clamp into
    # the model's real 1..7 horizons instead so no slider position is dead.
    horizon_day = max(1, min(horizon_int, inference.FORECAST_HORIZON_DAYS))
    earthquake_horizon, _, _ = get_earth_quake_estimates(horizon_day)

    # Label the slider off the wall clock, exactly as the reference did, so the
    # date shown matches what the slider's own oninput handler computes.
    horizon_date = datetime.today() + timedelta(days=horizon_int)

    return render_template(
        'index.html',
        date_horizon=horizon_date.strftime('%m/%d/%Y'),
        earthquake_horizon=earthquake_horizon,
        current_value=horizon_int,
        days_out_to_predict=inference.FORECAST_HORIZON_DAYS,
        google_maps_key=GOOGLE_MAPS_KEY,
    )


if __name__ == "__main__":
    print("Precomputing today's forecast for all 7 horizon days from the trained pipeline "
          "(first load can take ~20s)...")
    inference.get_forecast()
    print("Ready. Starting Flask dev server on http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
