# Stage 08 — Forecast dashboard (web app)

Implements the last two steps of the project pipeline —
**forecast output -> early warning alert** — as an interactive local web
app, using the models trained and evaluated in stages 05/06. Modeled after
`temp/EarthquakeForecasting/Webapp` (a single-model heatmap-and-slider demo
that retrains from scratch on every restart and hardcodes a Google Maps API
key in the page source), but built on this project's own verified pipeline
and with several things that reference app doesn't have.

## Run locally

```bash
cd 08_webapp
py -3.14 -m pip install -r requirements.txt -r ../requirements.txt
py -3.14 app.py
```

Then open **http://127.0.0.1:5000** in a browser. First load takes ~15-20s
(computing today's feature row for all 372 active cells reuses stage 04's
own rolling-window code over the full catalog history — see "How it
works" below); the result is cached in memory for the life of the process.

No API key or external account needed — the map uses
[Leaflet](https://leafletjs.com/) with free OpenStreetMap tiles.

## Features

- **Interactive risk map.** One marker per active (>=50-event) grid cell,
  sized/colored by the selected model's predicted P(M>=4.5 in the next 7
  days). Click a marker for the full feature breakdown behind that
  prediction (rolling event counts, b-value, days since last event,
  magnitude of completeness) and a side-by-side probability comparison
  across all three trained models.
- **Model switcher.** Toggle between AdaBoost+Decision Tree,
  AdaBoost+Random Forest, and XGBoost (the best model, preselected) without
  a page reload — all three models' predictions are computed once and sent
  to the browser together.
- **Early-warning alert banner.** Lists every cell at or above an
  adjustable probability threshold (slider, default 0.5); updates live as
  you move the slider or switch models.
- **Top-risk table.** The 15 highest-probability cells, ranked, with
  supporting stats.
- **Model comparison panel.** Test-set ROC-AUC/PR-AUC/F1/Brier for all
  three models (from stage 06's `06_evaluation_metrics.json`), best model
  highlighted, with the same justification text stage 07 writes into the
  final report.
- **Recent-earthquakes overlay.** Actual M4.5+ events from the last 30 days
  (from the stage 03 cleaned catalog) plotted as separate markers, toggle-
  able, so predicted risk can be eyeballed against recent ground truth.
- **Manual refresh.** The "Refresh" button recomputes today's features from
  whatever is currently in `outputs/03_processed/03_cleaned_catalog.csv` —
  re-run stages 01 (fetch new USGS data) and 03 (re-clean) first, then hit
  Refresh, to see a forecast that reflects newly-fetched data without
  restarting the server.

## How it works (and why it's not just re-running stage 04)

Stage 04's own committed feature matrix (`outputs/04_features/`)
deliberately **drops the most recent days per cell** — their target label
(did an M4.5+ event happen in the following 7 days?) isn't knowable yet, so
those rows are useless for *training*. But that's exactly the row a live
dashboard needs: "today's" features, with an unknown future.

`inference.py` does not reimplement feature engineering — it imports
`scripts/04_feature_engineering.py` **by file path** (its name starts with
a digit, so it can't be a normal Python import) and calls the same
`compute_cell_features()` / `maxc_completeness_magnitude()` functions and
constants stage 04 uses, then keeps only the last (most recent) row per
cell instead of dropping it. This guarantees the dashboard can never
compute a feature differently than the models were actually trained on —
single source of truth, not a parallel reimplementation that could drift.

Prediction preprocessing (missing-indicator columns + median imputation,
fit on the train split back in stage 05, column ordering from
`05_feature_columns.json`) is likewise copied verbatim from stage 06's
evaluation code.

## Endpoints

- `GET /` — the dashboard page.
- `GET /api/forecast` — JSON: every active cell's lat/lon, display
  features, and per-model probability. `?refresh=1` forces recomputation.
- `GET /api/recent-quakes?days=30&min_mag=4.5` — recent actual events for
  the ground-truth overlay.
- `GET /api/metrics` — the stage 06 model comparison table + best-model
  justification.

## Known limitations

Same catalog-window and coarse-grid caveats as the rest of the project
(see the top-level README's "Known limitations") — this dashboard
visualizes the same model, it doesn't change what it can or can't see.
The "Refresh" button re-derives features from whatever's on disk; it does
not itself call the USGS API (run stage 01 first for genuinely new data).
