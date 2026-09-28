# Stage 08 — Forecast dashboard (web app)

Implements the proposal's last two pipeline steps — **forecast output ->
early warning alert** — as a local interactive dashboard, combining:

- **UI/interaction**: ported from `temp/EarthquakeForecasting/Webapp/`
  as-is (title bar, single world map, one "select future date" slider) -
  see `templates/index.html`. The original's Google Maps JS API (a real-
  looking key hardcoded in the page source, and since discontinued for
  the heatmap feature it used) is swapped for Leaflet + Esri tiles - no
  API key needed. One circle marker per active cell, colored/sized
  continuously by that cell's own predicted probability, rather than a
  blended heatmap - what you see is exactly what the model predicted for
  that cell, nothing smoothed in between.
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
today's features for all 372 active cells, reusing stage 04's rolling-
window code, then predicting all 7 horizon days at once); cached in memory
afterward.

## The slider

The reference app's slider ("Select future date: today + N", N in 0-7)
implied the underlying model could break its forecast down day-by-day. An
earlier version of *this* project's model couldn't either — it predicted
one probability per (cell, day) for "does an M&ge;4.5 event happen in the
*following* 7 days" as a single fixed window, so every slider position
showed the exact same number.

That's fixed at the source: stage 04 (`scripts/04_feature_engineering.py`)
now labels every horizon_day in 1..7 separately and `horizon_day` is a real
input feature the models were retrained on. So the slider (1-7, "days
ahead") now asks today's live feature snapshot a genuinely different
question at each position — P(M&ge;4.5 by tomorrow) vs. P(M&ge;4.5 by day
7) are different, learned probabilities, not a lookup trick. Each point's
popup also shows a magnitude readout (Gutenberg-Richter-implied "most
likely" / "severe scenario" magnitude, conditional on an event happening —
see `inference.py`'s `_magnitude_estimate`), not a single deterministic
number, since a point magnitude forecast isn't something this kind of
model can honestly produce. See `inference.py`'s module docstring for the
full before/after.

## Files

- `app.py` — Flask routes; maps the slider (horizon_day) to a forecast and
  builds the per-cell points (with magnitude readout) sent to the page.
- `inference.py` — loads the stage 05 model/imputer/feature-columns,
  computes today's live feature snapshot per active cell, expands it into
  one row per horizon_day, and predicts each (see its own docstring for
  why the committed `outputs/04_features/` matrix alone isn't enough — it
  deliberately drops the very row a live dashboard needs).
- `templates/index.html` — the page, ported from the reference app with
  Leaflet swapped in for Google Maps.
- `static/js/risk-scale.js` — the color/radius scale for a predicted
  probability, pulled out of the page's inline script specifically so it
  can be unit tested with plain Node (see Tests below) instead of only
  ever being checked by eye in a browser.

## Tests

100% line + branch coverage on `app.py` and `inference.py`, plus a small
Node suite for the one piece of client-side JS that's pure logic
(`static/js/risk-scale.js`).

```bash
# Python (pytest + coverage, config in pytest.ini / .coveragerc)
cd 08_webapp
py -3.14 -m pip install -r requirements.txt -r requirements-dev.txt -r ../requirements.txt
py -3.14 -m pytest --cov=. --cov-report=term-missing
# -> 24 passed, app.py 100%, inference.py 100% (line + branch)

# JS (Node's built-in test runner - no npm install needed)
node --test tests/js/risk-scale.test.js
# -> 12 passed
```

Notes on how the suite gets to 100% honestly rather than by testing
around the hard parts:
- `inference.get_forecast()` rereads the full cleaned catalog and
  recomputes rolling-window features for all 372 active cells (~15s) -
  a session-scoped autouse fixture (`tests/conftest.py`) warms this cache
  **once** for the whole run; most tests then just read it.
- Branch logic that real data can't reliably exercise (a missing model
  file in `outputs/05_models/`, a `None` predicted probability) is tested
  against small, deliberately-constructed synthetic inputs instead of
  hoping real data happens to hit that case.
- The `if __name__ == "__main__":` dev-server bootstrap in `app.py` is
  excluded via `.coveragerc` (`exclude_lines`), not tested around — there
  is nothing a unit test can assert about "a real server actually started"
  short of an integration test that boots one, which is a different kind
  of test than this suite is for.

## Known limitations

Same catalog-window and coarse-grid caveats as the rest of the project
(top-level README's "Known limitations") — this dashboard visualizes the
same model, it doesn't change what it can or can't see. Only the best
model (XGBoost) is shown, matching the reference app's single-heatmap UI
(no model switcher).
