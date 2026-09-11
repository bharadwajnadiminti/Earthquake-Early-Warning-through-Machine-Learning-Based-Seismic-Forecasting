# Automatic Earthquake Forecasting — ML Pipeline

MSc Data Science project. Binary spatio-temporal forecasting of significant
(M4.5+) earthquakes from the USGS earthquake catalog, using AdaBoost+Decision
Tree, AdaBoost+Random Forest, and XGBoost, tuned with `GridSearchCV` +
`TimeSeriesSplit`.

Pipeline: **seismic data collection -> preprocessing -> feature extraction ->
ML model training -> forecast output -> early warning alert.**

## Quick start

```bash
pip install -r requirements.txt
cd scripts
python 01_fetch_and_update_data.py      # optional: refresh dataset/usgs_earthquake_catalog.csv
python 02_generate_profile_report.py    # optional: re-run EDA (needs Python <3.14, see note below)
python 03_preprocess.py
python 04_feature_engineering.py
python 05_train_models.py
python 06_evaluate_models.py
python 07_generate_report.py
cd ../08_webapp                         # optional: interactive forecast dashboard
python -m pip install -r requirements.txt -r ../requirements.txt
python app.py                           # -> http://127.0.0.1:5000
```

Everything after stage 02 has already been run once; its outputs are
committed under `outputs/`. Re-running any stage overwrites that stage's
(and only that stage's) numbered output files, so you can re-run from any
point forward without repeating earlier stages.

> **Python version note:** this pipeline was built and run on Python 3.14.6.
> `ydata-profiling` (stage 02 only) does not yet ship a build for Python
> >=3.14 — stage 02's outputs are already generated and committed under
> `outputs/02_profiling/`; you only need a separate Python <3.14 environment
> with `ydata-profiling==4.18.4` if you want to *re-run* stage 02 (e.g. on
> a refreshed catalog).

## Folder structure

```
project/
  dataset/
    usgs_earthquake_catalog.csv          canonical dataset (updated in place by stage 01)
    archive/                             superseded raw snapshot, not read by the pipeline
  scripts/
    01_fetch_and_update_data.py          fetch/upsert new USGS events (unchanged logic; renamed + path-fixed)
    02_generate_profile_report.py        ydata-profiling EDA report (unchanged logic; renamed + path-fixed)
    03_preprocess.py                     cleaning: dedupe, missing values, dtypes, UTC, chronological sort
    04_feature_engineering.py            spatio-temporal target + time/magnitude/energy/spatial features
    05_train_models.py                   chronological split + GridSearchCV(TimeSeriesSplit) tuning, 3 models
    06_evaluate_models.py                test-set metrics, curves, feature importance, best-model selection
    07_generate_report.py                assembles the final markdown report
  08_webapp/                             stage 08: interactive forecast dashboard (Flask + Leaflet),
                                          "forecast output -> early warning alert" made visual —
                                          see 08_webapp/README.md for features and how it works
  outputs/
    02_profiling/                        stage 02: usgs_earthquake_profile.{html,json}
    03_processed/                        stage 03: 03_cleaned_catalog.csv, 03_cleaning_report.json
    04_features/                         stage 04: 04_feature_matrix.csv, 04_data_dictionary.csv,
                                          04_feature_engineering_report.json
    05_models/                           stage 05: 05_model_<name>.pkl, 05_<name>_best_params.json,
                                          05_<name>_cv_results.csv, 05_imputer.pkl,
                                          05_feature_columns.json, 05_split_summary.json
    06_metrics/                          stage 06: 06_evaluation_metrics.{json,csv}, 06_best_model.json,
                                          06_roc_curves.png, 06_pr_curves.png,
                                          06_confusion_matrices.png, 06_calibration_curves.png,
                                          06_feature_importance_<model>.png
    07_report/                           stage 07: 07_final_report.md
  requirements.txt
  README.md
```

## Target definition (read before touching stages 04-07)

The proposal specifies binary classifiers but leaves the exact forecasting
target open. This project uses a **sliding daily forecast, spatio-temporal
binary formulation**, defined and justified in full in
`scripts/04_feature_engineering.py`'s docstring. Summary:

- **Spatial grid:** 1 deg x 1 deg lat/lon cells, restricted to **active
  cells** (>=50 historical events) — 372 of 4,609 cells with any event,
  covering ~90% of the catalog. Inactive (mostly ocean/aseismic) cells are
  excluded: not enough history for a meaningful rolling forecast, and
  including them would balloon the sample count ~60x for a trivial
  always-zero label.
- **Temporal resolution:** daily, per active cell.
- **Forecast horizon:** 7 days.
- **Label:** for (cell *c*, day *t*), **y=1** if >=1 event with magnitude
  >= **4.5** occurs in cell *c* during (*t*, *t*+7] days; else **y=0**.
- **Magnitude threshold (M4.5):** standard "moderately damaging / broadly
  felt" cutoff in operational seismology, and empirically the better choice
  for this catalog's size: M4.5+ gives 13,857 events (5.9% of the catalog)
  vs. M5.0+'s 3,525 (1.5%) — at (cell, day) granularity M4.5+ yields a
  workable ~5.8% positive rate, while M5.0+ would leave too few positives
  per cross-validation fold given only ~20 months of catalog. `--mag-threshold`
  on stage 04 lets you re-run with M5.0 as a sensitivity check.
- **Warm-up:** a cell only starts emitting rows once it has >=5 historical
  events.

**Flagged alternative:** if the real interest is a specific well-studied
region (California, Japan, etc.) rather than global coverage, a finer grid
(0.1-0.5 deg) with a shorter horizon would likely forecast better and be
more directly comparable to published regional short-term models (e.g.
ETAS-style). The global/1-degree/7-day choice here was made to maximize
usable positive examples given the ~20-month catalog window actually
available (see "Known limitations" below).

## What stage 03 actually cleaned (cited from stage 02's profiling)

All decisions below cite the specific ydata-profiling finding they respond
to (`outputs/02_profiling/usgs_earthquake_profile.json`):

| Finding (profiling alert) | Action taken |
|---|---|
| `[mag]`/`[magType]` missing, 20 rows (0.01%) | Rows dropped — `mag` defines the label, can't be used without it. |
| `[type]` has a constant value ("earthquake") | Filter re-asserted defensively (stage 01's `--eventtype` filter already did this). |
| `[nst]`/`[gap]`/`[dmin]` ~8.8% missing each; `[horizontalError]` 14.0%; `[magError]` 9.0%; `[magNst]` 8.8%; `[depthError]` 0.15% + highly skewed (gamma1=117.6) | **Not imputed, not used downstream.** These are per-event network/instrumentation quality metadata — they describe how well an event was *measured after the fact*, not a forecasting signal, and are reported inconsistently across the catalog's 15 contributing networks. Left as-is (real missingness preserved) in the cleaned file for traceability only. |
| No duplicate rows found; `id` 100% unique | Dedupe-by-`id` step kept anyway (defensive — stage 01 does incremental upserts and could reintroduce a stale duplicate on a future run). |
| lat/lon/depth: no out-of-bounds values found | Physical-bounds guardrail (lat in [-90,90], lon in [-180,180], depth in [-5,700]km) kept anyway, for the same defensive reason. |

Note: stage 02's profiling was run in `minimal` mode (large dataset), which
skips ydata-profiling's correlation/interaction computation — no
correlation matrix was available to consult; the cleaning decisions above
rely on the univariate stats/alerts and on domain reasoning instead.

## Features (stage 04)

Full column-by-column reference: `outputs/04_features/04_data_dictionary.csv`
(also embedded in `outputs/07_report/07_final_report.md`). Categories:

- **Time-based:** rolling event counts (7/30/90/365d) per cell, days since
  the cell's last event, cyclical day-of-year encoding.
- **Magnitude-based:** rolling mean/max magnitude (7/30/90/365d), and the
  **Gutenberg-Richter b-value** (Aki/Utsu maximum-likelihood estimator, over
  events >= the cell's own completeness magnitude Mc, computed via the
  maximum-curvature method + 0.2 correction) on 90d and 365d trailing
  windows, with a reliability count column alongside each.
- **Energy-based:** log10 rolling radiated seismic energy (7/30/90/365d),
  via the Gutenberg-Richter energy-magnitude relation
  `log10(E_joules) = 1.5*mag + 4.8`.
- **Spatial:** cell centroid lat/lon, per-cell completeness magnitude,
  per-cell historical event count as of day *t* ("maturity" of the cell's
  seismicity record).

Some rolling-magnitude and b-value columns are legitimately `NaN` (no event
in the window, or too few events for a reliable b-value estimate — see
`outputs/04_features/04_data_dictionary.csv` for exact missing rates). Stage 05
adds a `<col>_missing` indicator for every such column and median-imputes,
**fit on the train split only**, before any model sees the data.

## Modeling (stage 05)

Chronological 70/15/15 train/val/test split **by day**, with a 7-day purge
gap at each split boundary (the label looks up to 7 days into the future,
so rows right at a boundary would otherwise leak across it). The same
purge is approximated via `TimeSeriesSplit(gap=...)` for the 5-fold CV used
inside `GridSearchCV`, run on the train split only.

Three candidates, exactly as specified in the proposal deck, with two
compatibility/compute notes documented in `05_train_models.py`'s docstring:

1. **AdaBoost + Decision Tree** — grid: `estimator__max_depth` in
   `[2, 5, 7]`, `n_estimators` in `[200, 300, 400, 500, 600]`,
   `learning_rate=0.6` fixed. *(`algorithm='SAMME'` from the deck: scikit-learn
   1.6+ removed the `algorithm` parameter entirely — SAMME is now the only
   algorithm AdaBoostClassifier runs, so this requirement is automatically
   satisfied on this project's sklearn 1.9.1; also `base_estimator` ->
   `estimator`.)*
2. **AdaBoost + Random Forest** — grid: `n_estimators` in `[200, 700]`,
   `estimator__max_features` in `['sqrt', 'log2']`. *(Compute-budget note:
   the inner RandomForest's own size isn't specified in the deck; fixed at
   15 trees/max_depth=4 — a full 100-tree RF boosted 700 rounds would mean
   fitting up to 70,000 trees per candidate and was intractable to
   grid-search in this project's time budget.)*
3. **XGBoost** — `booster='gbtree'`, `objective='binary:logistic'`,
   `n_estimators=2000` with `early_stopping_rounds=50`, grid searched over
   `max_depth` in `[4, 6, 8]` and `learning_rate` (eta) in
   `[0.01, 0.03, 0.1]` (centered on the deck's `max_depth=6, eta=0.03`).
   *(Implementation note: sklearn's `GridSearchCV` can't pass a per-fold
   `eval_set` needed for early stopping, so this is a manual grid search
   over the same `TimeSeriesSplit` folds and the same `roc_auc` scoring —
   functionally equivalent to `GridSearchCV`, just hand-rolled to keep early
   stopping working.)*

Class imbalance (~5.8% positive): `class_weight='balanced'` on the AdaBoost
base estimators; XGBoost uses `scale_pos_weight` from the train split's
class ratio.

All random seeds are fixed to **42** throughout (numpy, scikit-learn,
XGBoost, the train/val/test split is otherwise deterministic by date).

## Evaluation (stage 06) and best model (stage 07)

Test-set (never used in tuning) ROC-AUC, PR-AUC, precision/recall/F1 (at
threshold 0.5, and at the F1-maximizing threshold), a confusion matrix, and
a calibration curve + Brier score, for all three models, plus ROC/PR/
feature-importance plots. The best model is selected on test ROC-AUC (the
metric GridSearchCV tuned on) with PR-AUC checked as a consistency
tie-breaker given the class imbalance. Full numbers: run stage 06/07 and see
`outputs/06_metrics/06_best_model.json` and `outputs/07_report/07_final_report.md`.

## Forecast dashboard (stage 08)

`08_webapp/` implements the proposal's last two pipeline steps — **forecast
output -> early warning alert** — as a local interactive dashboard: an
OpenStreetMap/Leaflet map of every active cell's live predicted risk
(color/size-coded, switchable across all three trained models), an
early-warning alert banner with an adjustable probability threshold, a
top-risk table, the stage 06 model comparison panel, and a recent-actual-
M4.5+-quakes overlay for ground-truth context. It does not reimplement
feature engineering — it imports `scripts/04_feature_engineering.py`'s own
functions to compute "today"'s feature row per cell (the one row stage
04's training matrix deliberately omits, since its target is unknowable),
so the dashboard can never drift from what the models were trained on. No
external API key is needed. Run it with `python 08_webapp/app.py`; see
`08_webapp/README.md` for the full feature list and how it works.

## Known limitations (worth citing in a viva)

- **Short catalog window (~20 months).** All three models' generalization
  claims are bounded by this — a single unusual sequence (a large
  mainshock/aftershock swarm) can dominate a fold. TimeSeriesSplit with a
  purge gap avoids *leakage*, but it does not manufacture more independent
  time periods than the catalog actually has.
- **Cold-start rolling features.** Even after the 5-event warm-up, a cell's
  earliest rows are estimated from a short history; the 365-day windows in
  particular don't fully "fill up" until a year into the catalog.
- **b-value NaN rate.** `b_value_90d`/`b_value_365d` are NaN whenever a
  cell/window has too few qualifying events (see
  `outputs/04_features/04_data_dictionary.csv` for the exact rate) —
  legitimate missingness, handled via missing-indicator + median imputation
  (stage 05), not an error.
- **Global, coarse grid.** See "Flagged alternative" above — a regional,
  finer-grained model would likely outperform this one; the global model
  was chosen to maximize usable training data given the short catalog.
- **AdaBoost+RF inner-forest size** and **XGBoost's manual grid search**
  are both documented compute/API compatibility deviations from a literal
  reading of the proposal deck — see "Modeling" above and the relevant
  script docstrings for exact justification.
