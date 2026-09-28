# outputs/

One directory per pipeline stage. **Results are committed; bulk data is not** —
so the metrics, plots and final report are browsable straight from GitHub,
while the multi-hundred-megabyte intermediates are rebuilt on demand.

| Stage | Committed | Not committed (rebuild with) |
|---|---|---|
| `02_profiling/` | `usgs_earthquake_profile.html` | `...json` (~107MB) — `02_generate_profile_report.py` |
| `03_processed/` | `03_cleaning_report.json` | `03_cleaned_catalog.csv` (~51MB) — `03_preprocess.py` |
| `04_features/` | `04_data_dictionary.csv`, `04_feature_engineering_report.json` | `04_feature_matrix.csv` (~557MB) — `04_feature_engineering.py` |
| `05_models/` | all — trained models, best params, CV results, imputer, split summary, train log | — |
| `06_metrics/` | all — metrics JSON/CSV, ROC/PR/calibration/confusion/importance plots, per-horizon breakdown | — |
| `07_report/` | `07_final_report.md` | — |

## Full rebuild

```bash
cd scripts
py -3.14 01_fetch_and_update_data.py   # populates ../dataset/ (see its README)
py -3.14 03_preprocess.py
py -3.14 04_feature_engineering.py
py -3.14 05_train_models.py            # slowest stage by far (~1.5-2h)
py -3.14 06_evaluate_models.py
py -3.14 07_generate_report.py
```

Stage 02 is optional and needs a separate Python <3.14 environment
(`ydata-profiling` has no 3.14 build yet) — its committed HTML output is
what the cleaning decisions in stage 03 cite, so you only need to re-run it
against a refreshed catalog.

Each stage overwrites only its own numbered outputs, so you can re-run from
any point forward without repeating earlier stages.
