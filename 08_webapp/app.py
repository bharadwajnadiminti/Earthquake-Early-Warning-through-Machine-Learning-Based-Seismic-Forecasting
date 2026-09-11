#!/usr/bin/env python3
"""
app.py

Stage 08: forecast dashboard ("forecast output -> early warning alert",
the last two steps of the proposal deck's pipeline). Serves live M4.5+/
7-day risk predictions from the three trained models (stage 05) over an
interactive map, plus a model-comparison panel (stage 06) and a recent-
actual-earthquakes overlay for ground-truth context.

Deliberately built against the already-verified pipeline artifacts rather
than retraining anything: loads the saved models/imputer (stage 05) and
recomputes only *today's* feature row per active cell (see inference.py)
by reusing stage 04's own feature-engineering code, so the dashboard can
never compute a feature differently than the models were trained on.

Run locally:
    pip install -r ../requirements.txt -r requirements.txt
    python app.py
    -> http://127.0.0.1:5000

No external API key required (Leaflet + OpenStreetMap tiles, unlike the
Google-Maps-API-key-in-the-HTML approach in temp/EarthquakeForecasting/Webapp).
"""

from flask import Flask, jsonify, render_template, request

import inference

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", model_names=inference.model_display_names(),
                            best_model_key=inference.BEST_MODEL_KEY)


@app.route("/api/forecast")
def api_forecast():
    force_refresh = request.args.get("refresh", "0") == "1"
    df, asof_date, horizon_end_date = inference.get_forecast(force_refresh=force_refresh)
    return jsonify({
        "asof_date": str(asof_date.date()),
        "horizon_end_date": str(horizon_end_date.date()),
        "horizon_days": inference.FORECAST_HORIZON_DAYS,
        "mag_threshold": inference.MAG_THRESHOLD,
        "n_cells": len(df),
        "model_names": inference.model_display_names(),
        "best_model_key": inference.BEST_MODEL_KEY,
        "cells": df.to_dict(orient="records"),
    })


@app.route("/api/recent-quakes")
def api_recent_quakes():
    days = int(request.args.get("days", 30))
    min_mag = float(request.args.get("min_mag", 4.5))
    quakes = inference.get_recent_significant_quakes(days=days, min_mag=min_mag)
    return jsonify({"days": days, "min_mag": min_mag, "count": len(quakes), "quakes": quakes})


@app.route("/api/metrics")
def api_metrics():
    return jsonify(inference.get_model_metrics())


if __name__ == "__main__":
    print("Precomputing today's forecast (first load can take ~20s - reuses stage 04's "
          "feature code over the full catalog history for every active cell)...")
    inference.get_forecast()
    print("Ready. Starting Flask dev server on http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)
