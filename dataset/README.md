# dataset/

Holds the project's single canonical input file:

| File | Size | Tracked in git? |
|---|---|---|
| `usgs_earthquake_catalog.csv` | ~46MB, grows each fetch | **No** — too large, and fully rebuildable |

## Rebuilding it

From a clean clone this directory is empty. Populate it with stage 01:

```bash
cd scripts
py -3.14 01_fetch_and_update_data.py --start 2025-01-01
```

That backfills the whole catalog from the USGS FDSN API (all magnitudes,
`eventtype=earthquake`). Every later run needs no arguments — it reads the
existing file, finds the most recent event in it, and fetches only what's
new, upserting by event `id` so re-runs never duplicate or leave stale rows:

```bash
py -3.14 01_fetch_and_update_data.py
```

Because USGS revises events after publication (magnitudes get refined,
review status changes), re-fetching an overlapping window is safe and is in
fact how you pick up those revisions.

Everything downstream is derived from this one file — see `outputs/README.md`.
