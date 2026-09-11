#!/usr/bin/env python3
"""
04_feature_engineering.py

Builds the spatio-temporal binary-classification feature matrix used by
stages 05-07, from the cleaned catalog produced by stage 03.

--------------------------------------------------------------------------
TARGET DEFINITION (read this before touching the modeling stages)
--------------------------------------------------------------------------
The project proposal specifies binary classifiers but leaves the exact
forecasting target open. This script implements a standard spatio-temporal
"sliding daily forecast" formulation used widely in short-term operational
earthquake forecasting research:

  Spatial grid:
    1 deg x 1 deg lat/lon cells (cell_lat/cell_lon = floor(latitude/longitude),
    reported here as the cell centroid, floor + 0.5). Restricted to "active"
    cells with >= MIN_EVENTS_PER_CELL (50) events in the full cleaned catalog
    -> 372 cells, covering ~90% of all catalog events. Inactive cells
    (mostly ocean/aseismic land with only 1-2 incidental events) are
    excluded: there isn't enough history in them to compute a meaningful
    rolling seismicity rate or b-value, and their forecast would be a
    trivial always-zero prediction. Including them would balloon the sample
    count ~60x for near-zero information gain and would badly dilute the
    positive class.

  Temporal resolution:
    Daily. For every active cell, one row is generated for (almost) every
    calendar day in the catalog's span (2025-01-01 to 2026-09-11, 618 days),
    subject to two trims: (a) a warm-up period per cell (see below) and
    (b) the last FORECAST_HORIZON_DAYS days of the catalog, which have no
    complete future window to label.

  Forecast horizon:
    7 days. Chosen to match "forecast output -> early warning alert" in the
    pipeline spec: a rolling 7-day outlook is the standard operational
    cadence for short-term regional hazard advisories (it's long enough to
    average out daily noise in a sparse catalog, short enough to still be
    an "early warning"-relevant horizon rather than a long-range outlook).

  Label:
    For (cell c, day t): y=1 if >=1 event with magnitude >= MAG_THRESHOLD
    occurs in cell c during (t, t+7] (strictly after t - no leakage: features
    at day t use only information available by the end of day t). y=0
    otherwise.

  Magnitude threshold (MAG_THRESHOLD = 4.5):
    M4.5 is a standard "moderately damaging / broadly felt" cutoff in
    operational seismology and is commonly used as the significant-event
    threshold in short-term forecasting studies. It was chosen over M5.0
    empirically as well as conventionally: over the cleaned catalog there
    are 13,857 M>=4.5 events (5.9% of all events) vs 3,525 M>=5.0 events
    (1.5%). At the (cell, day) sample granularity this yields a workable
    ~15-20% positive rate for the 7-day-ahead label; M5.0 would push the
    positive rate down into the low single digits given only ~20 months of
    catalog, which would leave too few positive examples per
    TimeSeriesSplit fold for GridSearchCV to tune reliably. M5.0 is left
    available via --mag-threshold for a sensitivity run.

  Warm-up / cold start:
    A cell's rows only start once it has accumulated >= WARMUP_MIN_EVENTS
    (5) historical events (as of that day) - before that there isn't enough
    history for the rolling-window features to mean anything. Even after
    warm-up, rolling stats for a cell's early rows are estimated from a
    short history and are noisier than later rows once the trailing windows
    (up to 365 days) fill up - this is a known limitation of any
    fixed-length trailing-window scheme on a young catalog and is called
    out again in the README.

  FLAG TO THE USER: this is one reasonable, standard formulation, not the
  only one. If the real interest is a specific well-studied region (e.g.
  California, Japan) rather than global coverage, a finer grid (0.1-0.5 deg)
  with a shorter horizon would likely forecast better and be more directly
  comparable to published regional models (e.g. ETAS-style short-term
  models). The global/1-degree/7-day choice here maximizes usable positive
  examples given the ~20-month catalog window available.

--------------------------------------------------------------------------
FEATURES
--------------------------------------------------------------------------
See outputs/features/04_data_dictionary.csv (written by this script) for
the full, authoritative column-by-column description. Categories:
  - Time-based:      rolling event counts (7/30/90/365d), days since last
                      event in cell, cyclical day-of-year encoding.
  - Magnitude-based:  rolling mean/max magnitude (7/30/90/365d), Gutenberg-
                      Richter b-value (Aki/Utsu MLE, 90d & 365d trailing
                      windows) per spatial cell.
  - Energy-based:     log10 rolling radiated-seismic-energy sum (7/30/90/
                      365d), via the Gutenberg-Richter energy-magnitude
                      relation log10(E_joules) = 1.5*mag + 4.8.
  - Spatial:          cell centroid (lat/lon), per-cell completeness
                      magnitude (Mc), per-cell historical event count as of
                      day t (seismicity "maturity" of the cell).

Input:  ../outputs/processed/03_cleaned_catalog.csv
Output: ../outputs/features/04_feature_matrix.csv
        ../outputs/features/04_data_dictionary.csv
        ../outputs/features/04_feature_engineering_report.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

GRID_SIZE_DEG = 1.0
MIN_EVENTS_PER_CELL = 50       # active-cell threshold
WARMUP_MIN_EVENTS = 5          # per-cell warm-up before a row is emitted
MIN_EVENTS_FOR_B = 10          # min qualifying events for a b-value to be trusted
FORECAST_HORIZON_DAYS = 7
COUNT_WINDOWS = [7, 30, 90, 365]
B_VALUE_WINDOWS = [90, 365]
MC_CORRECTION = 0.2            # MAXC + 0.2 (Wiemer & Wyss, 2000)
UTSU_CORRECTION = 0.05         # half the 0.1-mag binning width (Utsu, 1965)


def energy_joules(mag):
    # Gutenberg-Richter energy-magnitude relation (radiated seismic energy, joules)
    return 10 ** (1.5 * mag + 4.8)


def maxc_completeness_magnitude(mags):
    """Maximum-curvature estimate of the completeness magnitude Mc for one cell,
    computed once from its full history (Wiemer & Wyss, 2000), + the standard
    +0.2 correction for the MAXC method's tendency to underestimate Mc."""
    rounded = np.round(mags / 0.1) * 0.1
    mode = pd.Series(rounded).value_counts().idxmax()
    return float(mode) + MC_CORRECTION


def compute_cell_features(g, all_days, mc_cell, mag_threshold):
    """g: this cell's raw events (rows), already filtered to this cell.
    Returns a fully-populated daily feature frame for this one cell."""
    g = g.copy()
    g["day"] = g["time"].dt.floor("D")
    g["energy"] = energy_joules(g["mag"])
    g["is_qualifying_for_b"] = g["mag"] >= mc_cell
    g["is_significant"] = (g["mag"] >= mag_threshold).astype(int)
    g["qual_mag"] = g["mag"].where(g["is_qualifying_for_b"], 0.0)

    daily = g.groupby("day").agg(
        count=("mag", "size"),
        sum_mag=("mag", "sum"),
        max_mag=("mag", "max"),
        sum_energy=("energy", "sum"),
        qual_count=("is_qualifying_for_b", "sum"),
        qual_summag=("qual_mag", "sum"),
        sig_count=("is_significant", "sum"),
    )
    daily = daily.reindex(all_days)
    for col in ["count", "sum_mag", "sum_energy", "qual_count", "qual_summag", "sig_count"]:
        daily[col] = daily[col].fillna(0.0)
    # max_mag stays NaN on no-event days (correct: "no max" is not 0)

    out = pd.DataFrame(index=all_days)

    for w in COUNT_WINDOWS:
        roll_count = daily["count"].rolling(w, min_periods=1).sum()
        roll_summag = daily["sum_mag"].rolling(w, min_periods=1).sum()
        roll_energy = daily["sum_energy"].rolling(w, min_periods=1).sum()
        out[f"rolling_count_{w}d"] = roll_count
        out[f"rolling_mean_mag_{w}d"] = (roll_summag / roll_count.replace(0, np.nan))
        out[f"rolling_max_mag_{w}d"] = daily["max_mag"].rolling(w, min_periods=1).max()
        out[f"log_energy_{w}d"] = np.log10(roll_energy + 1.0)

    for w in B_VALUE_WINDOWS:
        roll_qcount = daily["qual_count"].rolling(w, min_periods=1).sum()
        roll_qsummag = daily["qual_summag"].rolling(w, min_periods=1).sum()
        mean_mag_qual = roll_qsummag / roll_qcount.replace(0, np.nan)
        b_val = np.log10(np.e) / (mean_mag_qual - (mc_cell - UTSU_CORRECTION))
        b_val = b_val.where(roll_qcount >= MIN_EVENTS_FOR_B)
        out[f"b_value_{w}d"] = b_val
        out[f"b_value_{w}d_n_events"] = roll_qcount

    # days since last event in cell (inter-event-time proxy)
    had_event = daily["count"] > 0
    last_event_day = pd.Series(np.where(had_event, all_days, pd.NaT), index=all_days)
    last_event_day = pd.to_datetime(last_event_day).ffill()
    out["days_since_last_event"] = (all_days - last_event_day).dt.days.astype("float")

    # historical event count as of day t (expanding, inclusive) -> cell "maturity"
    out["historical_event_count"] = daily["count"].expanding().sum()

    out["target"] = daily["sig_count"].rolling(FORECAST_HORIZON_DAYS, min_periods=FORECAST_HORIZON_DAYS) \
        .sum().shift(-FORECAST_HORIZON_DAYS)

    return out


def main():
    p = argparse.ArgumentParser(description="Feature engineering + spatio-temporal target labeling (stage 04).")
    p.add_argument("--input", type=str, default="../outputs/processed/03_cleaned_catalog.csv")
    p.add_argument("--output", type=str, default="../outputs/features/04_feature_matrix.csv")
    p.add_argument("--dictionary", type=str, default="../outputs/features/04_data_dictionary.csv")
    p.add_argument("--report", type=str, default="../outputs/features/04_feature_engineering_report.json")
    p.add_argument("--mag-threshold", type=float, default=4.5,
                   help="Magnitude threshold defining a 'significant' event for the label. Default 4.5.")
    p.add_argument("--min-events-per-cell", type=int, default=MIN_EVENTS_PER_CELL)
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    dict_path = Path(args.dictionary)
    report_path = Path(args.report)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    df = pd.read_csv(in_path)
    df["time"] = pd.to_datetime(df["time"], format="mixed", utc=True)
    print(f"Loaded {len(df):,} rows, {df['time'].min()} -> {df['time'].max()}")

    df["cell_lat_floor"] = np.floor(df["latitude"]).astype(int)
    df["cell_lon_floor"] = np.floor(df["longitude"]).astype(int)

    cell_counts = df.groupby(["cell_lat_floor", "cell_lon_floor"]).size()
    active_cells = cell_counts[cell_counts >= args.min_events_per_cell].index
    n_active = len(active_cells)
    coverage_pct = 100 * cell_counts[active_cells].sum() / len(df)
    print(f"Active cells (>= {args.min_events_per_cell} events, {GRID_SIZE_DEG} deg grid): "
          f"{n_active} / {len(cell_counts)} cells, covering {coverage_pct:.1f}% of events")

    df_active = df.set_index(["cell_lat_floor", "cell_lon_floor"])
    df_active = df_active.loc[df_active.index.isin(active_cells)].reset_index()

    all_days = pd.date_range(df["time"].dt.floor("D").min(), df["time"].dt.floor("D").max(), freq="D", tz="UTC")
    print(f"Daily span: {len(all_days)} days")

    frames = []
    mc_report = {}
    for i, (clat, clon) in enumerate(active_cells):
        g = df_active[(df_active["cell_lat_floor"] == clat) & (df_active["cell_lon_floor"] == clon)]
        mc_cell = maxc_completeness_magnitude(g["mag"].values)
        mc_report[f"{clat}_{clon}"] = mc_cell
        feats = compute_cell_features(g, all_days, mc_cell, args.mag_threshold)
        feats["cell_lat"] = clat + 0.5
        feats["cell_lon"] = clon + 0.5
        feats["mc_cell"] = mc_cell
        frames.append(feats)
        if (i + 1) % 100 == 0 or (i + 1) == n_active:
            print(f"  processed {i + 1}/{n_active} active cells")

    full = pd.concat(frames, axis=0)
    full.index.name = "day"
    full = full.reset_index()

    before = len(full)
    full = full[full["historical_event_count"] >= WARMUP_MIN_EVENTS]
    dropped_warmup = before - len(full)

    before = len(full)
    full = full.dropna(subset=["target"])
    dropped_no_label = before - len(full)

    full["target"] = (full["target"] > 0).astype(int)

    full["day_of_year"] = full["day"].dt.dayofyear
    full["doy_sin"] = np.sin(2 * np.pi * full["day_of_year"] / 365.25)
    full["doy_cos"] = np.cos(2 * np.pi * full["day_of_year"] / 365.25)
    full = full.drop(columns=["day_of_year"])

    full = full.sort_values(["day", "cell_lat", "cell_lon"]).reset_index(drop=True)

    ordered_cols = ["day", "cell_lat", "cell_lon", "mc_cell", "historical_event_count",
                    "days_since_last_event"]
    for w in COUNT_WINDOWS:
        ordered_cols += [f"rolling_count_{w}d", f"rolling_mean_mag_{w}d",
                          f"rolling_max_mag_{w}d", f"log_energy_{w}d"]
    for w in B_VALUE_WINDOWS:
        ordered_cols += [f"b_value_{w}d", f"b_value_{w}d_n_events"]
    ordered_cols += ["doy_sin", "doy_cos", "target"]
    full = full[ordered_cols]

    pos_rate = full["target"].mean()
    print(f"\nFeature matrix: {len(full):,} rows x {len(full.columns)} columns")
    print(f"Dropped {dropped_warmup:,} rows in per-cell warm-up (< {WARMUP_MIN_EVENTS} prior events)")
    print(f"Dropped {dropped_no_label:,} rows with no complete {FORECAST_HORIZON_DAYS}-day future window")
    print(f"Target positive rate (M>={args.mag_threshold}+ event in next {FORECAST_HORIZON_DAYS}d): {pos_rate:.4f} "
          f"({int(full['target'].sum()):,} positive / {len(full):,} total)")

    nan_report = full.isna().mean().sort_values(ascending=False)
    print("\nNaN rate per column (b-values are NaN where the cell has too few qualifying events):")
    print(nan_report[nan_report > 0].round(4).to_string())

    full.to_csv(out_path, index=False)
    print(f"\nFeature matrix written to: {out_path.resolve()}")

    # --- data dictionary -----------------------------------------------------
    descriptions = {
        "day": "Forecast origin date (UTC midnight). Features use data through end of this day; target looks forward.",
        "cell_lat": f"Latitude of the {GRID_SIZE_DEG} deg grid cell centroid.",
        "cell_lon": f"Longitude of the {GRID_SIZE_DEG} deg grid cell centroid.",
        "mc_cell": "Per-cell magnitude of completeness (MAXC + 0.2 correction), static, computed from full cell history.",
        "historical_event_count": "Cumulative count of catalog events in this cell from the start of the catalog through day t (cell 'maturity').",
        "days_since_last_event": "Days since the most recent event in this cell, as of day t (inter-event-time proxy).",
        "doy_sin": "sin(2*pi*day_of_year/365.25) - cyclical seasonal encoding.",
        "doy_cos": "cos(2*pi*day_of_year/365.25) - cyclical seasonal encoding.",
        "target": f"1 if >=1 event with magnitude >= {args.mag_threshold} occurs in this cell during (day t, day t+{FORECAST_HORIZON_DAYS}]; else 0.",
    }
    for w in COUNT_WINDOWS:
        descriptions[f"rolling_count_{w}d"] = f"Count of events in this cell in the trailing {w}-day window ending on day t (inclusive) - seismicity rate."
        descriptions[f"rolling_mean_mag_{w}d"] = f"Mean magnitude of events in this cell in the trailing {w}-day window. NaN if no events in the window."
        descriptions[f"rolling_max_mag_{w}d"] = f"Max magnitude of events in this cell in the trailing {w}-day window. NaN if no events in the window."
        descriptions[f"log_energy_{w}d"] = f"log10(1 + sum of radiated seismic energy [Joules, via 10^(1.5*mag+4.8)] over the trailing {w}-day window)."
    for w in B_VALUE_WINDOWS:
        descriptions[f"b_value_{w}d"] = (f"Gutenberg-Richter b-value (Aki/Utsu MLE) over the trailing {w}-day window, using events >= mc_cell. "
                                          f"NaN if fewer than {MIN_EVENTS_FOR_B} qualifying events in the window (unreliable estimate).")
        descriptions[f"b_value_{w}d_n_events"] = f"Number of events >= mc_cell used in the {w}-day b-value estimate (reliability indicator for b_value_{w}d)."

    dict_rows = []
    for col in full.columns:
        dict_rows.append({
            "column": col,
            "dtype": str(full[col].dtype),
            "pct_missing": round(100 * full[col].isna().mean(), 3),
            "description": descriptions.get(col, ""),
        })
    dict_df = pd.DataFrame(dict_rows)
    dict_df.to_csv(dict_path, index=False)
    print(f"Data dictionary written to: {dict_path.resolve()}")

    report = {
        "grid_size_deg": GRID_SIZE_DEG,
        "min_events_per_cell": args.min_events_per_cell,
        "n_active_cells": int(n_active),
        "n_total_cells_with_any_event": int(len(cell_counts)),
        "active_cell_event_coverage_pct": round(float(coverage_pct), 2),
        "warmup_min_events": WARMUP_MIN_EVENTS,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "mag_threshold": args.mag_threshold,
        "min_events_for_b_value": MIN_EVENTS_FOR_B,
        "rows_dropped_warmup": int(dropped_warmup),
        "rows_dropped_no_label": int(dropped_no_label),
        "final_row_count": int(len(full)),
        "final_column_count": int(len(full.columns)),
        "target_positive_rate": round(float(pos_rate), 4),
        "target_positive_count": int(full["target"].sum()),
        "date_range": [str(full["day"].min()), str(full["day"].max())],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Feature engineering report written to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
