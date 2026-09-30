# Automatic Earthquake Forecasting — Final Model Report

Pipeline: seismic data collection -> preprocessing -> feature extraction -> ML model training -> forecast output -> early warning alert.

## 1. Target Definition

**Formulation:** spatio-temporal binary classification. The study region is gridded into **1.0 deg x 1.0 deg** cells; only **active cells** (>= 50 historical catalog events) are modeled — **375** of 4647 cells with any recorded event, covering **89.82%** of all catalog events.

**Multi-horizon target:** for every active cell *c*, day *t*, and horizon_day *h* in 1..7: **y=1** if at least one event with magnitude >= **4.5** occurs in cell *c* during **(t, t+h]** days (strictly after t — features use only information available through end of day t); **y=0** otherwise. Each (cell, day) row is expanded into 7 output rows (one per horizon_day), with `horizon_day` itself included as an ordinary input feature. This lets one trained model answer "risk by tomorrow" and "risk by day 7" as genuinely different questions — an earlier version of this project used a single fixed 7-day window only, which meant a day-by-day forecast dashboard could only ever show one repeated number. See §4b below for evidence this was fixed.

**Magnitude threshold justification:** M4.5 is the standard "moderately damaging / broadly felt" cutoff used in operational seismology and short-term forecasting studies. It was also chosen for statistical power: at this catalog's size and span, M4.5+ yields a workable ~5.77% positive rate at (cell, day) granularity for the full 7-day horizon, versus M5.0+ which would push the positive rate into the low single digits and leave too few positive examples per cross-validation fold.

**Warm-up:** a cell's rows only start once it has accumulated >= 5 historical events (insufficient history before that to compute meaningful rolling features).

- Final feature matrix: **1,522,325** (cell, day, horizon_day) rows (7 horizon_day rows per (cell, day)), **55,631** positive (3.65% averaged across all horizons).
- Date range: 2025-01-01 00:00:00+00:00 -> 2026-09-21 00:00:00+00:00
- Rows dropped in per-cell warm-up: 18,400
- (cell, day) rows dropped (no complete 7-day future window): 2,625
- **Positive rate by horizon_day** (rises with horizon_day - a longer window gives more chances for an event, and is the reason a single model trained on this can tell "risk by tomorrow" apart from "risk by day 7"): day 1: 1.24%, day 2: 2.18%, day 3: 3.01%, day 4: 3.77%, day 5: 4.47%, day 6: 5.14%, day 7: 5.77%

**Note on formulation:** this is one reasonable, standard choice — not the only one. A finer grid (0.1-0.5 deg) with a shorter horizon over a single well-studied region (e.g. California, Japan) would likely forecast better and be more directly comparable to published regional models; the global/1-degree/7-day choice here maximizes usable positive examples given the ~20-month catalog window available. See scripts/04_feature_engineering.py docstring for the full reasoning.

## 2. Features

35 columns total (including `day`, `cell_lat`, `cell_lon`, and `target`). Full data dictionary: `outputs/04_features/04_data_dictionary.csv`.

| Column | Type | % Missing | Description |
|---|---|---|---|
| `day` | datetime64[us, UTC] | 0.0% | Forecast origin date (UTC midnight). Features use data through end of this day; target looks forward. |
| `horizon_day` | int64 | 0.0% | Forecast horizon in days, 1..7. Each (cell, day) origin row is repeated once per horizon_day so a single model can be asked about different lookout windows (risk by tomorrow vs. risk by day 7) instead of only ever answering one fixed-length window. Treated as an ordinary numeric input feature by stage 05. |
| `cell_lat` | float64 | 0.0% | Latitude of the 1.0 deg grid cell centroid. |
| `cell_lon` | float64 | 0.0% | Longitude of the 1.0 deg grid cell centroid. |
| `mc_cell` | float64 | 0.0% | Per-cell magnitude of completeness (MAXC + 0.2 correction), static, computed from full cell history. |
| `historical_event_count` | float64 | 0.0% | Cumulative count of catalog events in this cell from the start of the catalog through day t (cell 'maturity'). |
| `days_since_last_event` | float64 | 0.0% | Days since the most recent event in this cell, as of day t (inter-event-time proxy). |
| `rolling_count_7d` | float64 | 0.0% | Count of events in this cell in the trailing 7-day window ending on day t (inclusive) - seismicity rate. |
| `rolling_mean_mag_7d` | float64 | 31.84% | Mean magnitude of events in this cell in the trailing 7-day window. NaN if no events in the window. |
| `rolling_max_mag_7d` | float64 | 31.84% | Max magnitude of events in this cell in the trailing 7-day window. NaN if no events in the window. |
| `log_energy_7d` | float64 | 0.0% | log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing 7-day window). |
| `rolling_count_30d` | float64 | 0.0% | Count of events in this cell in the trailing 30-day window ending on day t (inclusive) - seismicity rate. |
| `rolling_mean_mag_30d` | float64 | 8.318% | Mean magnitude of events in this cell in the trailing 30-day window. NaN if no events in the window. |
| `rolling_max_mag_30d` | float64 | 8.318% | Max magnitude of events in this cell in the trailing 30-day window. NaN if no events in the window. |
| `log_energy_30d` | float64 | 0.0% | log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing 30-day window). |
| `rolling_count_90d` | float64 | 0.0% | Count of events in this cell in the trailing 90-day window ending on day t (inclusive) - seismicity rate. |
| `rolling_mean_mag_90d` | float64 | 1.8% | Mean magnitude of events in this cell in the trailing 90-day window. NaN if no events in the window. |
| `rolling_max_mag_90d` | float64 | 1.8% | Max magnitude of events in this cell in the trailing 90-day window. NaN if no events in the window. |
| `log_energy_90d` | float64 | 0.0% | log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing 90-day window). |
| `rolling_count_365d` | float64 | 0.0% | Count of events in this cell in the trailing 365-day window ending on day t (inclusive) - seismicity rate. |
| `rolling_mean_mag_365d` | float64 | 0.035% | Mean magnitude of events in this cell in the trailing 365-day window. NaN if no events in the window. |
| `rolling_max_mag_365d` | float64 | 0.035% | Max magnitude of events in this cell in the trailing 365-day window. NaN if no events in the window. |
| `log_energy_365d` | float64 | 0.0% | log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing 365-day window). |
| `decay_count_hl3d` | float64 | 0.0% | Exponentially recency-weighted running event count for this cell (half-life 3 days: an event this many days old retains half its weight, 6 days old a quarter, decaying forever rather than dropping to zero outside a fixed window). Prioritizes very recent activity far more than the flat rolling_count windows above. |
| `decay_log_energy_hl3d` | float64 | 0.0% | log10(1 + exponentially recency-weighted running radiated-energy sum, half-life 3 days) - same decay idea as decay_count_hl3d applied to energy instead of raw counts. |
| `decay_count_hl10d` | float64 | 0.0% | Exponentially recency-weighted running event count for this cell (half-life 10 days: an event this many days old retains half its weight, 20 days old a quarter, decaying forever rather than dropping to zero outside a fixed window). Prioritizes very recent activity far more than the flat rolling_count windows above. |
| `decay_log_energy_hl10d` | float64 | 0.0% | log10(1 + exponentially recency-weighted running radiated-energy sum, half-life 10 days) - same decay idea as decay_count_hl10d applied to energy instead of raw counts. |
| `neighbor_decay_count_hl3d` | float64 | 0.0% | Exponentially recency-weighted running event count (half-life 3 days), summed over this cell's 8 surrounding 1x1 degree cells (Moore neighborhood) - NOT this cell's own events. Captures geographic spillover (cluster/aftershock migration into neighboring territory) that cell_lat/cell_lon alone, used only as a static coordinate, cannot represent. |
| `b_value_90d` | float64 | 54.795% | Gutenberg-Richter b-value (Aki/Utsu MLE) over the trailing 90-day window, using events >= mc_cell. NaN if fewer than 10 qualifying events in the window (unreliable estimate). |
| `b_value_90d_n_events` | float64 | 0.0% | Number of events >= mc_cell used in the 90-day b-value estimate (reliability indicator for b_value_90d). |
| `b_value_365d` | float64 | 26.562% | Gutenberg-Richter b-value (Aki/Utsu MLE) over the trailing 365-day window, using events >= mc_cell. NaN if fewer than 10 qualifying events in the window (unreliable estimate). |
| `b_value_365d_n_events` | float64 | 0.0% | Number of events >= mc_cell used in the 365-day b-value estimate (reliability indicator for b_value_365d). |
| `doy_sin` | float64 | 0.0% | sin(2*pi*day_of_year/365.25) - cyclical seasonal encoding. |
| `doy_cos` | float64 | 0.0% | cos(2*pi*day_of_year/365.25) - cyclical seasonal encoding. |
| `target` | int64 | 0.0% | 1 if >=1 event with magnitude >= 4.5 occurs in this cell during (day t, day t+horizon_day]; else 0. |

## 3. Chronological Split

70/15/15 train/val/test split by unique day, with a 7-day purge gap at each boundary (the label looks up to 7 days into the future, so rows near a split boundary would otherwise leak information across it).

| Split | Rows | Date range | Positive rate |
|---|---|---|---|
| Train | 1,027,922 | 2025-01-01 00:00:00+00:00 -> 2026-03-16 00:00:00+00:00 | 3.61% |
| Val | 227,192 | 2026-03-24 00:00:00+00:00 -> 2026-06-18 00:00:00+00:00 | 3.67% |
| Test | 230,657 | 2026-06-26 00:00:00+00:00 -> 2026-09-21 00:00:00+00:00 | 3.81% |

## 4. Model Comparison (Test Set)

| model                    |   roc_auc |   pr_auc |   precision@0.5 |   recall@0.5 |   f1@0.5 |   f1@best_thresh |   brier_score |
|:-------------------------|----------:|---------:|----------------:|-------------:|---------:|-----------------:|--------------:|
| XGBoost                  |  0.916443 | 0.388763 |        0.152688 |     0.90075  | 0.261115 |         0.37396  |      0.130036 |
| AdaBoost + Decision Tree |  0.89695  | 0.248166 |        0.179813 |     0.825716 | 0.295316 |         0.350858 |      0.113315 |
| AdaBoost + Random Forest |  0.89219  | 0.188364 |        0.149343 |     0.903479 | 0.256318 |         0.316171 |      0.154374 |

Tuning: GridSearchCV, `scoring='roc_auc'`, `TimeSeriesSplit` (n_splits=5, gap~16353 rows) on the train split only.

**AdaBoost + Decision Tree — best hyperparameters:**
```json
{
  "best_params": {
    "estimator__max_depth": 5,
    "n_estimators": 200
  },
  "best_cv_roc_auc": 0.8967716586800625,
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
    "estimator__max_features": "sqrt",
    "n_estimators": 200
  },
  "best_cv_roc_auc": 0.9002137153341645,
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
  "best_iteration": 184,
  "fixed_params": {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "eval_metric": "auc",
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "random_state": 42,
    "scale_pos_weight": 26.67918787193365,
    "n_jobs": -1
  },
  "scale_pos_weight": 26.67918787193365
}
```

## 4b. Risk by Forecast Horizon (Test Set)

Direct evidence that the multi-horizon target (Section 1) actually gives day-by-day variation instead of one repeated number: mean predicted P(event by horizon_day) for each model, split out by horizon_day. It should climb monotonically with horizon_day (a longer window gives more chances for an event) and differ visibly between horizon_day=1 and horizon_day=7 — the two things a single-fixed-window model could never do.

| model                    |   horizon_day |     n |   positive_rate |   mean_predicted_proba |   roc_auc |   pr_auc |
|:-------------------------|--------------:|------:|----------------:|-----------------------:|----------:|---------:|
| AdaBoost + Decision Tree |             1 | 32951 |          0.0134 |                 0.2192 |    0.8961 |   0.1608 |
| AdaBoost + Decision Tree |             2 | 32951 |          0.0236 |                 0.2337 |    0.8974 |   0.2012 |
| AdaBoost + Decision Tree |             3 | 32951 |          0.0322 |                 0.2343 |    0.8962 |   0.2396 |
| AdaBoost + Decision Tree |             4 | 32951 |          0.0398 |                 0.3083 |    0.8876 |   0.231  |
| AdaBoost + Decision Tree |             5 | 32951 |          0.0465 |                 0.3097 |    0.8866 |   0.2468 |
| AdaBoost + Decision Tree |             6 | 32951 |          0.0527 |                 0.3097 |    0.8854 |   0.2679 |
| AdaBoost + Decision Tree |             7 | 32951 |          0.0588 |                 0.3097 |    0.8845 |   0.2882 |
| AdaBoost + Random Forest |             1 | 32951 |          0.0134 |                 0.2806 |    0.9001 |   0.097  |
| AdaBoost + Random Forest |             2 | 32951 |          0.0236 |                 0.2826 |    0.8951 |   0.15   |
| AdaBoost + Random Forest |             3 | 32951 |          0.0322 |                 0.3087 |    0.8915 |   0.1746 |
| AdaBoost + Random Forest |             4 | 32951 |          0.0398 |                 0.3345 |    0.8804 |   0.1668 |
| AdaBoost + Random Forest |             5 | 32951 |          0.0465 |                 0.3345 |    0.88   |   0.1911 |
| AdaBoost + Random Forest |             6 | 32951 |          0.0527 |                 0.3345 |    0.8803 |   0.2141 |
| AdaBoost + Random Forest |             7 | 32951 |          0.0588 |                 0.3345 |    0.8809 |   0.2361 |
| XGBoost                  |             1 | 32951 |          0.0134 |                 0.1606 |    0.9155 |   0.2637 |
| XGBoost                  |             2 | 32951 |          0.0236 |                 0.2045 |    0.9112 |   0.3235 |
| XGBoost                  |             3 | 32951 |          0.0322 |                 0.2357 |    0.9079 |   0.353  |
| XGBoost                  |             4 | 32951 |          0.0398 |                 0.2585 |    0.9072 |   0.3766 |
| XGBoost                  |             5 | 32951 |          0.0465 |                 0.2757 |    0.905  |   0.3906 |
| XGBoost                  |             6 | 32951 |          0.0527 |                 0.2895 |    0.9037 |   0.4062 |
| XGBoost                  |             7 | 32951 |          0.0588 |                 0.2943 |    0.903  |   0.4221 |

Plot: `outputs/06_metrics/06_risk_by_horizon.png`.

## 5. Best Model

**XGBoost**

XGBoost selected as best model: highest test-set ROC-AUC (0.9164), the primary tuning metric specified in the project proposal (GridSearchCV scoring='roc_auc'). Runner-up was AdaBoost + Decision Tree (ROC-AUC=0.8969). PR-AUC (more informative than ROC-AUC under the ~3.8% test-set class imbalance) was also checked as a tie-breaker/consistency check: XGBoost=0.3888, AdaBoost + Decision Tree=0.2482, AdaBoost + Random Forest=0.1884.

| Metric | Value |
|---|---|
| ROC-AUC | 0.9164 |
| PR-AUC (Average Precision) | 0.3888 |
| Precision @ 0.5 | 0.1527 |
| Recall @ 0.5 | 0.9008 |
| F1 @ 0.5 | 0.2611 |
| Best F1 threshold | 0.8644 |
| F1 @ best threshold | 0.3740 |
| Brier score (calibration) | 0.1300 |
| Confusion matrix @ 0.5 [[TN,FP],[FN,TP]] | [[177894, 43967], [873, 7923]] |

Plots: `outputs/06_metrics/06_roc_curves.png`, `06_pr_curves.png`, `06_confusion_matrices.png`, `06_calibration_curves.png`, `06_feature_importance_<model>.png`.

## 6. Forecast Dashboard (Web App)

`08_webapp/` implements the proposal's last two pipeline steps - **forecast output -> early warning alert** - as a local interactive dashboard: a live risk map with a "days ahead" slider (1-7) that now shows a genuinely different, model-predicted probability at each position (see Section 4b - this required the multi-horizon target change in Section 1; an earlier version showed one repeated number at every slider position), plus a per-cell Gutenberg-Richter magnitude readout (most-likely / severe-scenario, not a single deterministic value). It reuses `scripts/04_feature_engineering.py`'s own functions to compute each active cell's current feature row (the one row the training matrix above deliberately omits, since its label isn't knowable yet), so it can never compute a feature differently than the models were trained on. Run with `python 08_webapp/app.py` -> http://127.0.0.1:5000 (no external API key required). See `08_webapp/README.md` for the full feature list.

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
