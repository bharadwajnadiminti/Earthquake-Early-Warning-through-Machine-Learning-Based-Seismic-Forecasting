#!/usr/bin/env python
"""
app.py

Stage 08: forecast dashboard. UI/interaction is the reference app's own
(temp/EarthquakeForecasting/Webapp) - title bar, single world map, one
"select future date" slider, 0-7 days out - kept as close to the original
as possible (see templates/index.html). The DATA PULLING AND MODEL
BUILDING behind it is this project's own verified pipeline, not the
reference app's from-scratch-every-restart approach: see inference.py,
which reuses scripts/04_feature_engineering.py's own feature code and
loads the actual GridSearchCV-tuned model saved by stage 05
(outputs/05_models/05_model_xgboost.pkl - the best model per stage 06),
rather than retraining a fresh, unevaluated, untuned model blind on every
restart against only the last rolling month of all-magnitude data.

Slider semantics: the model predicts one probability per (cell, day) for
"does an M>=4.5 event happen in the *following* 7 days" - it does not
decompose that window into a day-by-day breakdown the way the reference
app's slider implied. So instead of a lookup hack, the slider here scrubs
through real forecast *snapshots* from the last 8 days, chosen so the
slider's own label ("Select future date: today + N") lines up exactly
with the *end* of the 7-day window being shown - move the slider to N and
you're looking at the model's actual view of cumulative risk through
that date, computed from that snapshot's real rolling features, not a
fabricated per-day split.

No external API key required (Leaflet + OpenStreetMap tiles + leaflet.heat,
not the Google Maps JS API the original had a key hardcoded for). This
isn't just a style choice: tested the reference app's exact original code
(same key, same `google.maps.visualization.HeatmapLayer` call) and Google
has discontinued that feature entirely -
`Error: The Heatmap Layer functionality in the Maps JavaScript API is no
longer available in the Maps JavaScript API as of version 3.65.` - a
console exception on Google's own SDK, unrelated to the key's validity or
this project. The original code cannot render a heatmap with ANY key
anymore. Leaflet + leaflet.heat is the closest still-working equivalent.

Run locally:
    cd 08_webapp
    pip install -r requirements.txt -r ../requirements.txt
    python app.py
    -> http://127.0.0.1:5000
"""

from datetime import datetime, timedelta

from flask import Flask, render_template, request

import inference

app = Flask(__name__)

PROB_THRESHOLD = 0.3  # same cutoff the reference app used to decide what to show


def get_earth_quake_estimates(horizon_int):
    """Slider value (0-7, "days from today") -> Leaflet.heat [lat, lon, weight]
    points for the best model's (XGBoost) predicted probability, from the
    forecast snapshot whose 7-day horizon ends exactly on today+horizon_int
    (see module docstring)."""
    offset_days = inference.N_LOOKBACK_DAYS - 1 - horizon_int
    day_df, _, day_date, horizon_end_date, _ = inference.get_forecast(offset_days=offset_days)

    points = []
    for row in day_df.to_dict(orient="records"):
        p = row.get(f"proba_{inference.BEST_MODEL_KEY}")
        if p is not None and p > PROB_THRESHOLD:
            points.append([row["cell_lat"], row["cell_lon"], float(p)])
    return points, day_date, horizon_end_date


@app.route("/", methods=['POST', 'GET'])
def build_page():
    horizon_int = int(request.form.get('slider_date_horizon', 0)) if request.method == 'POST' else 0
    points, day_date, horizon_end_date = get_earth_quake_estimates(horizon_int)
    horizon_date = datetime.today() + timedelta(days=horizon_int)

    return render_template(
        'index.html',
        date_horizon=horizon_date.strftime('%m/%d/%Y'),
        earthquake_horizon=points,
        current_value=horizon_int,
        days_out_to_predict=inference.N_LOOKBACK_DAYS - 1,
    )


if __name__ == "__main__":
    print("Precomputing the last week's forecast snapshots from the trained pipeline "
          "(first load can take ~20s)...")
    inference.get_forecast()
    print("Ready. Starting Flask dev server on http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
