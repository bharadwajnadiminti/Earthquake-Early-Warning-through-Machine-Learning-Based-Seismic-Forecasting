#!/usr/bin/env python3
"""
03_preprocess.py

Cleans the raw USGS earthquake catalog and writes a chronologically sorted,
UTC-normalized, deduplicated dataset ready for feature engineering (stage 04).

Every cleaning decision below is driven by the ydata-profiling report from
stage 02 (outputs/02_profiling/usgs_earthquake_profile.json), not guessed —
each step cites the specific finding it responds to:

  1. Deduplicate by event `id`, keep the row with the latest `updated`
     timestamp. Profiling's `duplicates` section reports NONE at the full-row
     level, and `id` is flagged "has unique values" (100% unique) — so this
     is a defensive no-op given the current file, but it's kept because
     stage 01 does incremental upserts and a future re-run could reintroduce
     revised duplicates before this stage catches them.

  2. Drop rows missing `mag` or `magType`. Profiling alerts: "[mag] ...
     missing values" and "[magType] ... missing values" — both 20 rows
     (0.01%). `mag` defines the forecasting target (Section: target
     definition in README), so a row without it cannot be used; 0.01% loss
     is negligible.

  3. Enforce the `eventtype` filter. Profiling alert: "[type] has a constant
     value" — every row is already 'earthquake' (stage 01's default
     --eventtype filter did its job). This step re-asserts that filter
     defensively rather than trusting it blindly.

  4. Physical-plausibility bounds check on latitude/longitude/depth
     (lat in [-90,90], lon in [-180,180], depth in [-5,700] km — the deepest
     verified subduction-zone earthquakes are ~700km; -5km covers legitimate
     above-sea-level/shallow volcanic events, which the USGS catalog does
     report as small negative depths). Manual inspection of this catalog
     found zero violations, so nothing is currently dropped by this check —
     it exists as a guardrail against future bad data from stage 01, not to
     fix a problem seen today.

  5. NOT imputed, and NOT used as ML features downstream (documented here
     rather than silently dropped, so the decision is auditable):
     `nst`, `gap`, `dmin` (each ~8.8% missing per profiling), `horizontalError`
     (14.0% missing), `magError` (9.0% missing), `magNst` (8.8% missing),
     and `depthError` (0.15% missing, and profiling flags it as extremely
     skewed: gamma1 = 117.6). These are per-event network/instrumentation
     quality metadata (station count, azimuthal gap, timing residuals) —
     they describe how well *this* event was measured after the fact, not
     a precursor signal available for forecasting *future* events, and they
     are reported inconsistently across the 15 contributing networks in this
     catalog (`net` column), which would make imputation fabricate
     comparability across networks that doesn't exist. They are kept as-is
     (with their real missingness) in the cleaned file purely for traceability/
     audit; stage 04 does not read them.

  6. Sort chronologically by `time` ascending and reset the index. This
     dataset is used throughout with TimeSeriesSplit / chronological
     train-val-test splits, so a stable chronological order is required to
     prevent any accidental leakage from row order.

  7. Normalize `time` and `updated` to timezone-aware UTC datetimes (the
     raw CSV stores them as ISO-8601 strings with a 'Z' suffix — already UTC,
     but stored as text, not a datetime dtype).

Input:  ../dataset/usgs_earthquake_catalog.csv
Output: ../outputs/03_processed/03_cleaned_catalog.csv
        ../outputs/03_processed/03_cleaning_report.json  (row counts at every step)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NUMERIC_COLS = [
    "latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms",
    "horizontalError", "depthError", "magError", "magNst",
]
DATETIME_COLS = ["time", "updated"]

LAT_BOUNDS = (-90.0, 90.0)
LON_BOUNDS = (-180.0, 180.0)
DEPTH_BOUNDS = (-5.0, 700.0)  # km; see docstring point 4


def main():
    p = argparse.ArgumentParser(description="Clean the raw USGS earthquake catalog (stage 03).")
    p.add_argument("--input", type=str, default="../dataset/usgs_earthquake_catalog.csv")
    p.add_argument("--output", type=str, default="../outputs/03_processed/03_cleaned_catalog.csv")
    p.add_argument("--report", type=str, default="../outputs/03_processed/03_cleaning_report.json")
    p.add_argument("--eventtype", type=str, default="earthquake",
                   help="Event type to keep. Default 'earthquake' (matches stage 01's default filter).")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    steps = []

    def log(name, df):
        steps.append({"step": name, "rows": int(len(df))})
        print(f"  [{name}] {len(df):,} rows")

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path, dtype=str)
    log("00_raw_loaded", df)

    # --- dtypes -----------------------------------------------------------
    for col in DATETIME_COLS:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 1. dedupe by id, keep latest `updated` ----------------------------
    before = len(df)
    df = df.sort_values("updated").drop_duplicates(subset="id", keep="last")
    log("01_deduped_by_id", df)
    removed_dupes = before - len(df)

    # --- 2. drop rows missing mag or magType --------------------------------
    before = len(df)
    df = df.dropna(subset=["mag", "magType"])
    log("02_dropped_missing_mag_or_magtype", df)
    removed_missing_mag = before - len(df)

    # --- 2b. drop rows missing time (can't be placed on the time axis) -----
    before = len(df)
    df = df.dropna(subset=["time"])
    log("02b_dropped_missing_time", df)
    removed_missing_time = before - len(df)

    # --- 3. enforce eventtype filter ---------------------------------------
    before = len(df)
    if args.eventtype:
        df = df[df["type"] == args.eventtype]
    log("03_filtered_eventtype", df)
    removed_eventtype = before - len(df)

    # --- 4. physical bounds check -------------------------------------------
    before = len(df)
    in_bounds = (
        df["latitude"].between(*LAT_BOUNDS)
        & df["longitude"].between(*LON_BOUNDS)
        & df["depth"].between(*DEPTH_BOUNDS)
    )
    df = df[in_bounds]
    log("04_physical_bounds_filter", df)
    removed_bounds = before - len(df)

    # --- 6. sort chronologically ---------------------------------------------
    df = df.sort_values("time").reset_index(drop=True)
    log("05_sorted_chronologically", df)

    print(f"\nDate range after cleaning: {df['time'].min()} -> {df['time'].max()}")
    print(f"Magnitude range: {df['mag'].min():.2f} -> {df['mag'].max():.2f}")

    df.to_csv(out_path, index=False)
    print(f"\nCleaned catalog written to: {out_path.resolve()} ({len(df):,} rows, {len(df.columns)} columns)")

    report = {
        "input_file": str(in_path),
        "output_file": str(out_path),
        "steps": steps,
        "removed": {
            "duplicate_ids": int(removed_dupes),
            "missing_mag_or_magtype": int(removed_missing_mag),
            "missing_time": int(removed_missing_time),
            "non_matching_eventtype": int(removed_eventtype),
            "out_of_physical_bounds": int(removed_bounds),
        },
        "final_row_count": int(len(df)),
        "final_date_range": [str(df["time"].min()), str(df["time"].max())],
        "not_imputed_not_used_downstream": {
            "columns": ["nst", "gap", "dmin", "horizontalError", "magError", "magNst", "depthError"],
            "reason": (
                "Per-event network/instrumentation quality metadata, ~8.8-14.0% "
                "missing (per stage-02 profiling alerts) and inconsistent across "
                "the 15 contributing networks in `net`; not predictive of FUTURE "
                "events, so left as-is (real missingness preserved) and not read "
                "by stage 04 feature engineering."
            ),
            "missing_pct_at_source": {
                "nst": 8.76, "gap": 8.76, "dmin": 8.79,
                "horizontalError": 14.04, "magError": 9.04, "magNst": 8.83,
                "depthError": 0.15,
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Cleaning report written to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
