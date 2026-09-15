# Automatic Earthquake Forecasting — Final Model Report

Pipeline: seismic data collection -> preprocessing -> feature extraction -> ML model training -> forecast output -> early warning alert.

## 1. Target Definition

**Formulation:** spatio-temporal binary classification. The study region is gridded into **1.0 deg x 1.0 deg** cells; only **active cells** (>= 50 historical catalog events) are modeled — **372** of 4609 cells with any recorded event, covering **89.74%** of all catalog events.

For every active cell *c* and day *t*: **y=1** if at least one event with magnitude >= **4.5** occurs in cell *c* during **(t, t+7]** days (strictly after t — features use only information available through end of day t); **y=0** otherwise.

**Magnitude threshold justification:** M4.5 is the standard "moderately damaging / broadly felt" cutoff used in operational seismology and short-term forecasting studies. It was also chosen for statistical power: at this catalog's size and span, M4.5+ yields a workable ~5.83% positive rate at (cell, day) granularity, versus M5.0+ which would push the positive rate into the low single digits and leave too few positive examples per cross-validation fold.

**Warm-up:** a cell's rows only start once it has accumulated >= 5 historical events (insufficient history before that to compute meaningful rolling features).

- Final feature matrix: **209,557** (cell, day) rows, **12,219** positive (5.83%).
- Date range: 2025-01-01 00:00:00+00:00 -> 2026-09-04 00:00:00+00:00
- Rows dropped in per-cell warm-up: 18,107
- Rows dropped (no complete future label window): 2,604

**Note on formulation:** this is one reasonable, standard choice — not the only one. A finer grid (0.1-0.5 deg) with a shorter horizon over a single well-studied region (e.g. California, Japan) would likely forecast better and be more directly comparable to published regional models; the global/1-degree/7-day choice here maximizes usable positive examples given the ~20-month catalog window available. See scripts/04_feature_engineering.py docstring for the full reasoning.

## 2. Features

34 columns total (including `day`, `cell_lat`, `cell_lon`, and `target`). Full data dictionary: `outputs/04_features/04_data_dictionary.csv`.

| Column | Type | % Missing | Description |
|---|---|---|---|
| `day` | datetime64[us, UTC] | 0.0% | Forecast origin date (UTC midnight). Features use data through end of this day; target looks forward. |
| `cell_lat` | float64 | 0.0% | Latitude of the 1.0 deg grid cell centroid. |
| `cell_lon` | float64 | 0.0% | Longitude of the 1.0 deg grid cell centroid. |
| `mc_cell` | float64 | 0.0% | Per-cell magnitude of completeness (MAXC + 0.2 correction), static, computed from full cell history. |
| `historical_event_count` | float64 | 0.0% | Cumulative count of catalog events in this cell from the start of the catalog through day t (cell 'maturity'). |
| `days_since_last_event` | float64 | 0.0% | Days since the most recent event in this cell, as of day t (inter-event-time proxy). |
| `rolling_count_7d` | float64 | 0.0% | Count of events in this cell in the trailing 7-day window ending on day t (inclusive) - seismicity rate. |
| `rolling_mean_mag_7d` | float64 | 31.354% | Mean magnitude of events in this cell in the trailing 7-day window. NaN if no events in the window. |
| `rolling_max_mag_7d` | float64 | 31.354% | Max magnitude of events in this cell in the trailing 7-day window. NaN if no events in the window. |
| `log_energy_7d` | float64 | 0.0% | log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing 7-day window). |
| `rolling_count_30d` | float64 | 0.0% | Count of events in this cell in the trailing 30-day window ending on day t (inclusive) - seismicity rate. |
| `rolling_mean_mag_30d` | float64 | 7.974% | Mean magnitude of events in this cell in the trailing 30-day window. NaN if no events in the window. |
| `rolling_max_mag_30d` | float64 | 7.974% | Max magnitude of events in this cell in the trailing 30-day window. NaN if no events in the window. |
| `log_energy_30d` | float64 | 0.0% | log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing 30-day window). |
| `rolling_count_90d` | float64 | 0.0% | Count of events in this cell in the trailing 90-day window ending on day t (inclusive) - seismicity rate. |
| `rolling_mean_mag_90d` | float64 | 1.712% | Mean magnitude of events in this cell in the trailing 90-day window. NaN if no events in the window. |
| `rolling_max_mag_90d` | float64 | 1.712% | Max magnitude of events in this cell in the trailing 90-day window. NaN if no events in the window. |
| `log_energy_90d` | float64 | 0.0% | log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing 90-day window). |
| `rolling_count_365d` | float64 | 0.0% | Count of events in this cell in the trailing 365-day window ending on day t (inclusive) - seismicity rate. |
| `rolling_mean_mag_365d` | float64 | 0.021% | Mean magnitude of events in this cell in the trailing 365-day window. NaN if no events in the window. |
| `rolling_max_mag_365d` | float64 | 0.021% | Max magnitude of events in this cell in the trailing 365-day window. NaN if no events in the window. |
| `log_energy_365d` | float64 | 0.0% | log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing 365-day window). |
| `decay_count_hl3d` | float64 | 0.0% | Exponentially recency-weighted running event count for this cell (half-life 3 days: an event this many days old retains half its weight, 6 days old a quarter, decaying forever rather than dropping to zero outside a fixed window). Prioritizes very recent activity far more than the flat rolling_count windows above. |
| `decay_log_energy_hl3d` | float64 | 0.0% | log10(1 + exponentially recency-weighted running radiated-energy sum, half-life 3 days) - same decay idea as decay_count_hl3d applied to energy instead of raw counts. |
| `decay_count_hl10d` | float64 | 0.0% | Exponentially recency-weighted running event count for this cell (half-life 10 days: an event this many days old retains half its weight, 20 days old a quarter, decaying forever rather than dropping to zero outside a fixed window). Prioritizes very recent activity far more than the flat rolling_count windows above. |
| `decay_log_energy_hl10d` | float64 | 0.0% | log10(1 + exponentially recency-weighted running radiated-energy sum, half-life 10 days) - same decay idea as decay_count_hl10d applied to energy instead of raw counts. |
| `neighbor_decay_count_hl3d` | float64 | 0.0% | Exponentially recency-weighted running event count (half-life 3 days), summed over this cell's 8 surrounding 1x1 degree cells (Moore neighborhood) - NOT this cell's own events. Captures geographic spillover (cluster/aftershock migration into neighboring territory) that cell_lat/cell_lon alone, used only as a static coordinate, cannot represent. |
| `b_value_90d` | float64 | 54.46% | Gutenberg-Richter b-value (Aki/Utsu MLE) over the trailing 90-day window, using events >= mc_cell. NaN if fewer than 10 qualifying events in the window (unreliable estimate). |
| `b_value_90d_n_events` | float64 | 0.0% | Number of events >= mc_cell used in the 90-day b-value estimate (reliability indicator for b_value_90d). |
| `b_value_365d` | float64 | 26.186% | Gutenberg-Richter b-value (Aki/Utsu MLE) over the trailing 365-day window, using events >= mc_cell. NaN if fewer than 10 qualifying events in the window (unreliable estimate). |
| `b_value_365d_n_events` | float64 | 0.0% | Number of events >= mc_cell used in the 365-day b-value estimate (reliability indicator for b_value_365d). |
| `doy_sin` | float64 | 0.0% | sin(2*pi*day_of_year/365.25) - cyclical seasonal encoding. |
| `doy_cos` | float64 | 0.0% | cos(2*pi*day_of_year/365.25) - cyclical seasonal encoding. |
| `target` | int64 | 0.0% | 1 if >=1 event with magnitude >= 4.5 occurs in this cell during (day t, day t+7]; else 0. |

## 3. Chronological Split

70/15/15 train/val/test split by unique day, with a 7-day purge gap at each boundary (the label looks up to 7 days into the future, so rows near a split boundary would otherwise leak information across it).

| Split | Rows | Date range | Positive rate |
|---|---|---|---|
| Train | 141,391 | 2025-01-01 00:00:00+00:00 -> 2026-03-04 00:00:00+00:00 | 5.74% |
| Val | 31,429 | 2026-03-12 00:00:00+00:00 -> 2026-06-04 00:00:00+00:00 | 5.81% |
| Test | 31,557 | 2026-06-12 00:00:00+00:00 -> 2026-09-04 00:00:00+00:00 | 6.25% |

## 4. Model Comparison (Test Set)

| model                    |   roc_auc |   pr_auc |   precision@0.5 |   recall@0.5 |   f1@0.5 |   f1@best_thresh |   brier_score |
|:-------------------------|----------:|---------:|----------------:|-------------:|---------:|-----------------:|--------------:|
| XGBoost                  |  0.907933 | 0.44142  |        0.230204 |     0.904714 | 0.36702  |         0.434299 |      0.136331 |
| AdaBoost + Random Forest |  0.890338 | 0.275273 |        0.237482 |     0.896604 | 0.375504 |         0.379739 |      0.147555 |
| AdaBoost + Decision Tree |  0.862712 | 0.244046 |        0.229744 |     0.873796 | 0.363828 |         0.404388 |      0.145926 |

Tuning: GridSearchCV, `scoring='roc_auc'`, `TimeSeriesSplit` (n_splits=5, gap~2312 rows) on the train split only.

**AdaBoost + Decision Tree — best hyperparameters:**
```json
{
  "best_params": {
    "estimator__max_depth": 5,
    "n_estimators": 200
  },
  "best_cv_roc_auc": 0.8848660433359644,
  "fixed_params": {
    "learning_rate": 0.6,
    "algorithm": "SAMME (only option in sklearn>=1.6)"
  }
}
```

**AdaBoost + Random Forest — best hyperparameters:**
```json
{
  "best_params": {
    "estimator__max_features": "log2",
    "n_estimators": 200
  },
  "best_cv_roc_auc": 0.883058611685559,
  "fixed_params": {
    "inner_rf_n_estimators": 15,
    "inner_rf_max_depth": 4,
    "reason": "see module docstring: compute-budget choice"
  }
}
```

**XGBoost — best hyperparameters:**
```json
{
  "best_params": {
    "n_estimators": 2000,
    "max_depth": 4,
    "learning_rate": 0.03
  },
  "best_iteration": 170,
  "fixed_params": {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "auc",
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "scale_pos_weight": 16.432005917889285,
    "n_jobs": -1
  },
  "scale_pos_weight": 16.432005917889285
}
```

## 5. Best Model

**XGBoost**

XGBoost selected as best model: highest test-set ROC-AUC (0.9079), the primary tuning metric specified in the project proposal (GridSearchCV scoring='roc_auc'). Runner-up was AdaBoost + Random Forest (ROC-AUC=0.8903). PR-AUC (more informative than ROC-AUC under the ~6.3% test-set class imbalance) was also checked as a tie-breaker/consistency check: XGBoost=0.4414, AdaBoost + Random Forest=0.2753, AdaBoost + Decision Tree=0.2440.

| Metric | Value |
|---|---|
| ROC-AUC | 0.9079 |
| PR-AUC (Average Precision) | 0.4414 |
| Precision @ 0.5 | 0.2302 |
| Recall @ 0.5 | 0.9047 |
| F1 @ 0.5 | 0.3670 |
| Best F1 threshold | 0.8174 |
| F1 @ best threshold | 0.4343 |
| Brier score (calibration) | 0.1363 |
| Confusion matrix @ 0.5 [[TN,FP],[FN,TP]] | [[23615, 5969], [188, 1785]] |

Plots: `outputs/06_metrics/06_roc_curves.png`, `06_pr_curves.png`, `06_confusion_matrices.png`, `06_calibration_curves.png`, `06_feature_importance_<model>.png`.

## 6. Forecast Dashboard (Web App)

`08_webapp/` implements the proposal's last two pipeline steps - **forecast output -> early warning alert** - as a local interactive dashboard: a live risk map (switchable across all three models above), an adjustable-threshold early-warning banner, a top-risk table, this same model comparison panel, and a recent-actual-earthquakes overlay for ground-truth context. It reuses `scripts/04_feature_engineering.py`'s own functions to compute each active cell's current feature row (the one row the training matrix above deliberately omits, since its label isn't knowable yet), so it can never compute a feature differently than the models were trained on. Run with `python 08_webapp/app.py` -> http://127.0.0.1:5000 (no external API key required). See `08_webapp/README.md` for the full feature list.

## 7. Reproducing This Report

```
cd scripts
python 01_fetch_and_update_data.py
python 02_generate_profile_report.py
python 03_preprocess.py
python 04_feature_engineering.py
python 05_train_models.py
python 06_evaluate_models.py
python 07_generate_report.py
cd ../08_webapp && python app.py  # optional: forecast dashboard
```

See the top-level `README.md` for the full pipeline description.
