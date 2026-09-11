#!/usr/bin/env python3
"""
02_generate_profile_report.py

First-level EDA on your USGS earthquake catalog CSV using ydata-profiling
(the current name for what used to be called "pandas-profiling").

Produces a self-contained HTML report AND a JSON report: per-column
distributions, missing-value map, correlations, duplicate-row check,
alerts (skew, high cardinality, zeros, etc.) — everything you'd otherwise
write by hand as a first pass before feature engineering.

Note on formats: ydata-profiling's `to_file()` natively supports exactly
two output formats — HTML (rendered report) and JSON (machine-readable
dump of every stat/alert the HTML is built from). There isn't a native
PDF/CSV/Excel export built into the library itself. If you want a PDF,
the simplest route is opening the HTML in a browser and using
"Print -> Save as PDF", or running the HTML through a tool like
`weasyprint` / `wkhtmltopdf` afterwards — this script will tell you the
one-liner for that at the end if you pass --pdf.

--------------------------------------------------------------------------
SETUP (one-time)
--------------------------------------------------------------------------
    pip install ydata-profiling

    # only needed if you use --pdf:
    pip install weasyprint

--------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------
Default (reads ../dataset/usgs_earthquake_catalog.csv, minimal mode —
recommended first, since this dataset is 200k+ rows and full mode computes
expensive pairwise interactions that can take a long time at that size).
Writes both usgs_earthquake_profile.html and usgs_earthquake_profile.json:

    python 02_generate_profile_report.py

Custom input/output (extension is stripped and used as a shared base name
for both .html and .json):

    python 02_generate_profile_report.py \
        --input ../dataset/usgs_earthquake_catalog.csv \
        --output ../outputs/profiling/usgs_earthquake_profile.html

Full report (all interactions/correlations — slower, try on a sample first):

    python 02_generate_profile_report.py --mode full --sample 20000

Also render a PDF copy of the HTML report (requires weasyprint):

    python 02_generate_profile_report.py --pdf

--------------------------------------------------------------------------
OPTIONS
--------------------------------------------------------------------------
--input     Path to the catalog CSV. Default: ../dataset/usgs_earthquake_catalog.csv
--output    Base output path (extension is ignored/stripped; .html and
            .json are both written next to it).
            Default: ../outputs/profiling/usgs_earthquake_profile
--mode      'minimal' (fast, default — skips expensive interactions/
            correlations, still gives full per-column stats) or
            'full' (every section ydata-profiling can compute).
--sample    Optional: profile a random sample of N rows instead of the
            whole file (useful for a quick look at a huge file, or to
            afford --mode full). Full dataset is used if omitted.
--title     Report title. Default derived from the input filename.
--formats   Comma-separated list of formats to write. Choices: html,json.
            Default: html,json (both).
--pdf       Also render the HTML report to PDF via weasyprint (best-effort;
            skipped with a warning if weasyprint isn't installed).
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Numeric columns in the standard USGS catalog CSV export
NUMERIC_COLS = [
    "latitude", "longitude", "depth", "mag", "nst", "gap", "dmin", "rms",
    "horizontalError", "depthError", "magError", "magNst",
]
DATETIME_COLS = ["time", "updated"]


def load_catalog(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)

    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Everything else (magType, net, id, place, type, status,
    # locationSource, magSource) is left as string/categorical — that's
    # what ydata-profiling expects for categorical analysis.
    return df


def main():
    p = argparse.ArgumentParser(description="First-level EDA on the USGS earthquake catalog with ydata-profiling.")
    p.add_argument("--input", type=str, default="../dataset/usgs_earthquake_catalog.csv")
    p.add_argument("--output", type=str, default="../outputs/profiling/usgs_earthquake_profile",
                   help="Base path; .html/.json are appended. Any extension you pass is stripped.")
    p.add_argument("--mode", choices=["minimal", "full"], default="minimal")
    p.add_argument("--sample", type=int, default=None,
                   help="Profile a random sample of N rows instead of the full file.")
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--formats", type=str, default="html,json",
                   help="Comma-separated: html,json")
    p.add_argument("--pdf", action="store_true",
                   help="Also render the HTML report to PDF via weasyprint, if installed.")
    args = p.parse_args()

    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        sys.exit(
            "ydata-profiling is not installed in this environment.\n"
            "Run:  pip install ydata-profiling\n"
            "then re-run this script."
        )

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Input file not found: {in_path}")

    print(f"Loading {in_path} ...")
    df = load_catalog(in_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns.")
    if "time" in df.columns and df["time"].notna().any():
        print(f"Date range: {df['time'].min()} -> {df['time'].max()}")
    if "mag" in df.columns:
        print(f"Magnitude range: {df['mag'].min()} -> {df['mag'].max()} "
              f"(missing: {df['mag'].isna().sum():,})")

    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).sort_values("time")
        print(f"Sampled down to {len(df):,} rows (random_state=42) for the report.")

    title = args.title or f"USGS Earthquake Catalog — {in_path.name}"

    minimal = args.mode == "minimal"
    print(f"Generating {'minimal' if minimal else 'full'} profile report "
          f"({'fast, skips pairwise interactions/correlations' if minimal else 'includes all sections — can be slow on large data'}) ...")

    profile = ProfileReport(df, title=title, minimal=minimal)

    # Strip whatever extension was passed (if any) to get a shared base name
    out_base = Path(args.output)
    out_base = out_base.with_suffix("") if out_base.suffix else out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)

    requested_formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    unknown = requested_formats - {"html", "json"}
    if unknown:
        sys.exit(f"Unknown format(s) in --formats: {', '.join(sorted(unknown))}. Choices are: html, json")

    html_path = out_base.with_suffix(".html")
    json_path = out_base.with_suffix(".json")

    if "html" in requested_formats:
        profile.to_file(html_path)
        print(f"HTML report written to: {html_path.resolve()}")

    if "json" in requested_formats:
        profile.to_file(json_path)
        print(f"JSON report written to: {json_path.resolve()}")

    if args.pdf:
        if "html" not in requested_formats:
            print("--pdf requested but HTML wasn't generated (it's needed as the source); "
                  "add 'html' to --formats.")
        else:
            try:
                from weasyprint import HTML
                pdf_path = out_base.with_suffix(".pdf")
                HTML(str(html_path)).write_pdf(str(pdf_path))
                print(f"PDF report written to: {pdf_path.resolve()}")
            except ImportError:
                print(
                    "Skipped PDF: weasyprint isn't installed. Run "
                    "'pip install weasyprint' and re-run with --pdf, or just "
                    "open the HTML file in a browser and use Print -> Save as PDF."
                )

    print("Done. Open the .html report in any browser; the .json report is "
          "for programmatic access to the same stats/alerts.")


if __name__ == "__main__":
    main()