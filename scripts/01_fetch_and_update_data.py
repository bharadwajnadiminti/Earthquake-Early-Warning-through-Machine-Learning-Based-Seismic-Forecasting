#!/usr/bin/env python3
"""
01_fetch_and_update_data.py

Incrementally fetch USGS earthquake catalog data (any date range, any
magnitude) and UPSERT it into a single running CSV file — safe to re-run
any time, from any machine with internet access to earthquake.usgs.gov.

Why "upsert" and not just "append":
USGS revises events after publication (magnitude gets refined, review
status moves from "automatic" to "reviewed", etc.), and the same event
can be re-fetched if date ranges overlap. This script dedupes by the
event `id` and keeps whichever copy has the newer `updated` timestamp,
so running it repeatedly — even with overlapping windows — never creates
duplicate or stale rows.

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------

First run (backfill from a known start date):
    python 01_fetch_and_update_data.py --start 2025-01-01 \
        --output ../dataset/usgs_earthquake_catalog.csv

Every run after that — no args needed. It reads the existing file, finds
the latest event already in it, and fetches everything new since then:
    python 01_fetch_and_update_data.py --output ../dataset/usgs_earthquake_catalog.csv

Re-check/refill a specific window (e.g. pick up recent magnitude revisions):
    python 01_fetch_and_update_data.py --start 2026-09-01 --end 2026-09-11 \
        --output ../dataset/usgs_earthquake_catalog.csv

--------------------------------------------------------------------------
OPTIONS
--------------------------------------------------------------------------
--start            ISO date/datetime (UTC), e.g. 2025-01-01 or
                    2025-01-01T00:00:00. If omitted: auto-resumes from the
                    latest `time` already in --output, minus a small
                    overlap window (to catch late revisions near the old
                    boundary). Required on the very first run (no existing
                    file to resume from).
--end              ISO date/datetime (UTC). Default: right now.
--min-magnitude    Default: none -> ALL magnitudes.
--max-magnitude    Default: none.
--eventtype        Default: 'earthquake' (drops quarry blasts, explosions,
                    etc). Pass --eventtype '' to disable this filter.
--output           CSV file to create/update.
                    Default: ../dataset/usgs_earthquake_catalog.csv
                    (pick ONE stable filename and always point the script
                    at it — that's what lets auto-resume work).
--chunk-cap        Max events requested per API call before the date
                    range is auto-split in half. USGS's hard limit is
                    20000; default here is 8000 for safety margin.
                    Default: 8000
--overlap-minutes  Overlap window applied only when auto-resuming (no
                    --start given), to catch any event revised right
                    around the previous run's cutoff. Default: 60

Requires: pip install pandas requests
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/"
MAX_RETRIES = 4


def parse_dt(s: str) -> datetime:
    """Parse a user-supplied date/datetime string as UTC."""
    s = s.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return pd.to_datetime(s, utc=True).to_pydatetime()


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _get_with_retry(session, path, params):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(BASE_URL + path, params=params, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    request failed ({e}); retrying in {wait}s...", file=sys.stderr)
            time.sleep(wait)
    raise last_err


def get_count(session, start, end, min_mag, max_mag, eventtype):
    params = {"starttime": fmt_dt(start), "endtime": fmt_dt(end)}
    if min_mag is not None:
        params["minmagnitude"] = min_mag
    if max_mag is not None:
        params["maxmagnitude"] = max_mag
    if eventtype:
        params["eventtype"] = eventtype
    r = _get_with_retry(session, "count", params)
    return int(r.text.strip())


def get_csv_chunk(session, start, end, min_mag, max_mag, eventtype):
    params = {
        "format": "csv",
        "starttime": fmt_dt(start),
        "endtime": fmt_dt(end),
        "orderby": "time-asc",
    }
    if min_mag is not None:
        params["minmagnitude"] = min_mag
    if max_mag is not None:
        params["maxmagnitude"] = max_mag
    if eventtype:
        params["eventtype"] = eventtype
    r = _get_with_retry(session, "query", params)
    text = r.text.strip()
    if not text:
        return None
    return pd.read_csv(StringIO(text), dtype=str)


def fetch_range(session, start, end, min_mag, max_mag, eventtype, chunk_cap, frames, stats):
    if start >= end:
        return
    count = get_count(session, start, end, min_mag, max_mag, eventtype)
    stats["requests"] += 1
    if count == 0:
        return
    if count <= chunk_cap:
        df = get_csv_chunk(session, start, end, min_mag, max_mag, eventtype)
        stats["requests"] += 1
        n = len(df) if df is not None else 0
        if n:
            frames.append(df)
            stats["rows"] += n
        print(f"  {start.isoformat()} -> {end.isoformat()}: {n} rows")
    else:
        mid = start + (end - start) / 2
        fetch_range(session, start, mid, min_mag, max_mag, eventtype, chunk_cap, frames, stats)
        fetch_range(session, mid, end, min_mag, max_mag, eventtype, chunk_cap, frames, stats)


def main():
    p = argparse.ArgumentParser(
        description="Fetch & upsert USGS earthquake data into a running CSV catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start", type=str, default=None,
                   help="Start date/time (UTC). Omit to auto-resume from --output.")
    p.add_argument("--end", type=str, default=None, help="End date/time (UTC). Default: now.")
    p.add_argument("--min-magnitude", type=float, default=None)
    p.add_argument("--max-magnitude", type=float, default=None)
    p.add_argument("--eventtype", type=str, default="earthquake",
                   help="Default 'earthquake'. Pass '' to disable this filter.")
    p.add_argument("--output", type=str, default="../dataset/usgs_earthquake_catalog.csv")
    p.add_argument("--chunk-cap", type=int, default=8000)
    p.add_argument("--overlap-minutes", type=int, default=60)
    args = p.parse_args()

    out_path = Path(args.output)
    existing = None
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype=str)
        print(f"Loaded existing file: {out_path} ({len(existing)} rows)")

    end = parse_dt(args.end) if args.end else datetime.now(timezone.utc)

    if args.start:
        start = parse_dt(args.start)
    elif existing is not None and len(existing):
        last_time = pd.to_datetime(existing["time"], utc=True).max()
        start = last_time.to_pydatetime() - timedelta(minutes=args.overlap_minutes)
        print(f"No --start given; auto-resuming from {start.isoformat()} "
              f"(latest existing event minus {args.overlap_minutes}min overlap)")
    else:
        sys.exit(
            "No --start given and no existing --output file to resume from.\n"
            "Provide --start for the first run, e.g. --start 2025-01-01"
        )

    if start >= end:
        print("Nothing to do: start is not before end.")
        return

    eventtype = args.eventtype if args.eventtype != "" else None

    print(f"Fetching {start.isoformat()} -> {end.isoformat()} "
          f"(min_mag={args.min_magnitude}, max_mag={args.max_magnitude}, eventtype={eventtype})")

    session = requests.Session()
    frames = []
    stats = {"requests": 0, "rows": 0}
    fetch_range(session, start, end, args.min_magnitude, args.max_magnitude,
                eventtype, args.chunk_cap, frames, stats)

    print(f"Fetched {stats['rows']} rows in {stats['requests']} API calls.")

    if not frames and existing is None:
        print("No data fetched and no existing file — nothing to write.")
        return

    new_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined = pd.concat([existing, new_df], ignore_index=True) if existing is not None else new_df
    before = len(combined)

    # --- Upsert: keep the most recently revised copy of each event id ---
    combined["_updated_dt"] = pd.to_datetime(combined["updated"], utc=True, errors="coerce")
    combined = combined.sort_values("_updated_dt").drop_duplicates(subset="id", keep="last")
    combined = combined.drop(columns=["_updated_dt"])

    # --- Final chronological order (matters for time-series-aware training) ---
    combined["_time_dt"] = pd.to_datetime(combined["time"], utc=True, errors="coerce")
    combined = combined.sort_values("_time_dt").drop(columns=["_time_dt"])

    removed = before - len(combined)

    if out_path.exists():
        backup = out_path.with_suffix(out_path.suffix + ".bak")
        out_path.replace(backup)  # existing data is already safely loaded in memory above

    combined.to_csv(out_path, index=False)

    print(f"Upserted. {removed} duplicate/superseded rows resolved.")
    print(f"Final file: {out_path} -> {len(combined)} rows, "
          f"covering {combined['time'].min()} to {combined['time'].max()}")


if __name__ == "__main__":
    main()
