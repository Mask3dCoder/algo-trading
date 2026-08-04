#!/usr/bin/env python3
"""HYPOTHESIS_001 — Options Data Backfill for IS period (2016-2021).

Downloads historical SPY options chain snapshots from Polygon.io and
caches them to data/options/SPY/{date}.parquet for use in the IS
calibration pass.

Run after setting POLYGON_API_KEY:
    export POLYGON_API_KEY=your_key_here
    python research/h001_opra_flow/backfill_is_data.py

Cost: Polygon Startup plan ($200/mo) or higher required for historical
      snapshot queries back to 2016.

Usage:
    python research/h001_opra_flow/backfill_is_data.py [--dry-run]
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

IS_START  = "2016-01-01"
IS_END    = "2021-05-31"   # up to the day before current cache starts
UNDERLYING = "SPY"


def _trading_days(start: date, end: date) -> list[date]:
    """Return all weekdays (Mon-Fri) between start and end inclusive."""
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon=0 … Fri=4
            days.append(current)
        current += timedelta(days=1)
    return days


def main() -> int:
    import pandas as pd

    dry_run = "--dry-run" in sys.argv

    # ── Check API key ──────────────────────────────────────────────────
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set.")
        print()
        print("  Set your Polygon.io API key:")
        print("    export POLYGON_API_KEY=your_key_here")
        print()
        print("  Required plan: Startup ($200/mo) or higher for")
        print("  historical options snapshot access.")
        print()
        print("  Alternative: Check if your existing cache was fetched")
        print("  from a different provider (Alpaca, CBOE, etc.).")
        return 1

    print("=" * 60)
    print("  HYPOTHESIS_001 — Options Data Backfill")
    print("  Period: %s -> %s" % (IS_START, IS_END))
    print("  Dry run: %s" % dry_run)
    print("=" * 60)

    # ── Count existing cache ───────────────────────────────────────────
    cache_dir = _ROOT / "data" / "options" / UNDERLYING
    existing = set()
    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            d = date.fromisoformat(f.stem)
            if d <= date(2021, 5, 31):
                existing.add(d)

    is_trading_days = _trading_days(date.fromisoformat(IS_START), date(2021, 5, 31))
    missing = [d for d in is_trading_days if d not in existing]
    already_cached = len(is_trading_days) - len(missing)

    print()
    print("Options cache status (%s -> %s):" % (IS_START, IS_END))
    print("  Trading days in IS period: %d" % len(is_trading_days))
    print("  Already cached:           %d" % already_cached)
    print("  Still missing:            %d" % len(missing))
    if missing:
        print("  First missing date:       %s" % missing[0])
        print("  Last missing date:       %s" % missing[-1])

    if dry_run:
        print()
        print("  DRY RUN — exiting without downloading.")
        print("  Remove --dry-run to begin backfill.")
        return 0

    if not missing:
        print()
        print("  All IS period data already cached.")
        print("  Run the IS pass: python research/h001_opra_flow/run_insample.py")
        return 0

    print()
    print("Estimated time at ~5 req/min: %d minutes" % (len(missing) * 12 / 60))

    # ── Create loader ─────────────────────────────────────────────────
    from quantcore.data.options_loader import PolygonOptionsDataLoader

    loader = PolygonOptionsDataLoader()

    # ── Download missing dates in batches of 100 with checkpointing ───
    BATCH = 100
    for batch_start in range(0, len(missing), BATCH):
        batch = missing[batch_start:batch_start + BATCH]
        fetched = 0
        for dt in batch:
            _log.info("Fetching %s (%d/%d)", dt, batch_start + fetched + 1, len(missing))
            df = loader._fetch_chain_snapshot(UNDERLYING, dt)
            if not df.empty:
                loader._save_chain(UNDERLYING, dt, df)
            # Rate limit: Polygon free tier = 5 req/min
            time.sleep(12.5)
            fetched += 1

        _log.info("Batch %d-%d done (%d/%d fetched)",
                  batch_start, batch_start + BATCH,
                  batch_start + fetched, len(missing))
        # Checkpoint: log progress
        remaining = [d for d in missing if d > batch[-1]]
        _log.info("Remaining: %d dates", len(remaining))

    # ── Final report ─────────────────────────────────────────────────
    final_existing = set()
    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            d = date.fromisoformat(f.stem)
            if date.fromisoformat(IS_START) <= d <= date(2021, 5, 31):
                final_existing.add(d)

    still_missing = set(missing) - final_existing
    print()
    print("=" * 60)
    print("  BACKFILL COMPLETE")
    print("=" * 60)
    print("  Dates added:     %d" % (len(final_existing) - len(existing)))
    print("  Final IS cache:   %d / %d" % (len(final_existing), len(is_trading_days)))
    print("  Still missing:    %d" % len(still_missing))
    if still_missing:
        print("  Missing dates:    %s ... %s" % (
            sorted(still_missing)[0], sorted(still_missing)[-1]))
    if len(still_missing) == 0:
        print()
        print("  IS data fully cached.")
        print("  Next: python research/h001_opra_flow/run_insample.py")
    print("=" * 60)
    return 0 if len(still_missing) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())