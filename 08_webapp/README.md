# Stage 08 — Forecast dashboard (web app)

Implements the proposal's last two pipeline steps — **forecast output ->
early warning alert** — as a local interactive dashboard, combining:

- **UI/interaction**: ported from `temp/EarthquakeForecasting/Webapp/`
  as-is (title bar, single world map, one "select future date" slider) -
  see `templates/index.html`. The only change is swapping the original's
  Google Maps JS API (which had a real-looking key hardcoded in the page
  source) for Leaflet + leaflet.heat over free OpenStreetMap tiles - no
  API key needed, and nothing gets committed to a public repo that
  shouldn't be.
- **Data pulling and model building**: this project's own verified
  pipeline, NOT the reference app's from-scratch-every-restart approach.
  `inference.py` reuses `scripts/04_feature_engineering.py`'s own feature
  code (imported by file path) to compute each active cell's real rolling
  features, and loads the actual `GridSearchCV`-tuned XGBoost model saved
  by stage 05 (`outputs/05_models/05_model_xgboost.pkl` - the best model
  per stage 06's evaluation), instead of retraining a fresh, untuned,
  unevaluated model blind on every restart against only the last rolling
  month of all-magnitude USGS data.

## Run locally

```bash
cd 08_webapp
py -3.14 -m pip install -r requirements.txt -r ../requirements.txt
py -3.14 app.py
```

Then open **http://127.0.0.1:5000**. First load takes ~15-20s (computing
the last 8 days of features for all 372 active cells, reusing stage 04's
rolling-window code); cached in memory afterward.

## The slider

The reference app's slider ("Select future date: today + N", N in 0-7)
implied the underlying model could break its forecast down day-by-day.
Ours can't — it predicts one probability per (cell, day) for "does an
M&ge;4.5 event happen in the *following* 7 days", not a day-by-day split.
Rather than fake that, the slider here is wired to real forecast
**snapshots** from the last 8 days, chosen so the slider's own label lines
up exactly with the *end* of the 7-day window being shown: move it to N
and you're looking at the model's actual view (real rolling features,
real prediction) of cumulative risk through today+N. See `app.py`'s
docstring for the exact offset math.

One behavioral difference from the original: at slider position 0 the
reference app showed a blank map (its own lookup table happened to have
no row for exactly "today" at that position). Here position 0 shows a
real forecast snapshot (today-7's), so the map is never blank.

## Files

- `app.py` — Flask routes; maps the slider to a forecast snapshot and
  renders the heatmap points.
- `inference.py` — loads the stage 05 model/imputer/feature-columns and
  computes live features per active cell for the last 8 days (see its own
  docstring for why the committed `outputs/04_features/` matrix alone
  isn't enough — it deliberately drops the very rows a live dashboard
  needs).
- `templates/index.html` — the page, ported from the reference app with
  Leaflet swapped in for Google Maps.

## Known limitations

Same catalog-window and coarse-grid caveats as the rest of the project
(top-level README's "Known limitations") — this dashboard visualizes the
same model, it doesn't change what it can or can't see. Only the best
model (XGBoost) is shown, matching the reference app's single-heatmap UI
(no model switcher).
