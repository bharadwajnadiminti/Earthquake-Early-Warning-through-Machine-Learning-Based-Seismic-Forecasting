"""
Shared fixtures for the stage 08 webapp test suite.

08_webapp/ isn't an installable package (app.py / inference.py are flat
modules, matching the rest of this project's scripts/ style), so tests
import them the same way app.py itself does: with 08_webapp/ on sys.path.
"""

import sys
from pathlib import Path

WEBAPP_DIR = Path(__file__).resolve().parent.parent
if str(WEBAPP_DIR) not in sys.path:
    sys.path.insert(0, str(WEBAPP_DIR))

import pandas as pd
import pytest

import inference  # noqa: E402 (must follow the sys.path fix-up above)


@pytest.fixture(scope="session", autouse=True)
def _warm_inference_cache():
    """inference.get_forecast() is expensive the first time it runs (~15s:
    rereads the full cleaned catalog and recomputes rolling-window features
    for every one of the 372 active cells) because it deliberately reuses
    the real scripts/04_feature_engineering.py code path rather than a
    cheaper stand-in - see inference.py's module docstring for why that
    matters. Running it once per test would make the suite take minutes;
    warming it once per session here means every other test just reads the
    already-populated cache, which is near-instant."""
    inference.get_forecast()


@pytest.fixture
def synthetic_catalog(tmp_path, monkeypatch):
    """A tiny, fully controlled 3-cell earthquake catalog, so tests of
    _compute_live_features's active-cell threshold logic can assert exact
    inclusion/exclusion instead of just "some real cell happened to
    qualify" - and run in milliseconds instead of paying for a ~15s read
    of the real 234k-row catalog. Points inference.PROCESSED_DIR at it for
    the duration of the test (monkeypatch auto-restores it after)."""
    start = pd.Timestamp("2024-01-01", tz="UTC")
    rows = []

    # Cell A: floor(10, 20) -> 50 events on 50 distinct days. Comfortably
    # at/above even the real pipeline's default MIN_EVENTS_PER_CELL (50).
    for d in range(50):
        rows.append({"time": (start + pd.Timedelta(days=d)).isoformat(),
                      "latitude": 10.5, "longitude": 20.5, "mag": 2.0})
    # Cell B: floor(5, 5) -> 3 events. Above a low explicit threshold (2)
    # but below the real pipeline's default (50).
    for d in (0, 10, 20):
        rows.append({"time": (start + pd.Timedelta(days=d)).isoformat(),
                      "latitude": 5.5, "longitude": 5.5, "mag": 2.0})
    # Cell C: floor(-5, -5) -> 1 event. Below every threshold used below.
    rows.append({"time": start.isoformat(), "latitude": -5.5, "longitude": -5.5, "mag": 2.0})

    df = pd.DataFrame(rows)
    processed_dir = tmp_path / "03_processed"
    processed_dir.mkdir()
    df.to_csv(processed_dir / "03_cleaned_catalog.csv", index=False)

    monkeypatch.setattr(inference, "PROCESSED_DIR", processed_dir)
    return processed_dir


@pytest.fixture
def flask_client():
    """Flask test client for app.py's routes. Imported lazily (not at
    module level) so conftest.py's sys.path fix-up above has already run
    by the time `import app` executes."""
    import app as app_module
    app_module.app.testing = True
    with app_module.app.test_client() as client:
        yield client
