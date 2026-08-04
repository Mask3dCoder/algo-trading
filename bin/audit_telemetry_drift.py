#!/usr/bin/env python3
"""Automated Telemetry Audit & Drift Validator.

Parses ``reports/paper_trading/delta_telemetry.csv`` and computes
descriptive statistics over a rolling window.  Warns if delta spikes
breach the critical threshold too frequently, suggesting the rebalancing
frequency is too slow for live market microstructure.

Usage
-----
    python bin/audit_telemetry_drift.py
    python bin/audit_telemetry_drift.py --input reports/paper_trading/delta_telemetry.csv

Exit codes
----------
    0  : all within normal bounds
    1  : warning threshold exceeded
    2  : critical threshold exceeded (rebalancing frequency too slow)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_CSV = Path("reports/paper_trading/delta_telemetry.csv")
WARN_THRESHOLD = 5.0    # |net_delta|  > 5  → warning
CRIT_THRESHOLD = 15.0   # |net_delta|  > 15 → critical
MAX_CRIT_BREACHES = 3   # more than this → exit 2 (rebalance too slow)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_csv(path: Path) -> list[dict]:
    """Read delta_telemetry.csv, return list of row dicts."""
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def compute_window_stats(rows: list[dict], window: int = 20) -> dict:
    """Compute rolling-window statistics from the telemetry rows."""
    if not rows:
        return {"status": "no_data"}

    # Extract numeric delta drift values
    drifts: list[float] = []
    for r in rows:
        try:
            drifts.append(float(r.get("delta_drift", 0.0)))
        except (ValueError, TypeError):
            continue

    if not drifts:
        return {"status": "no_numeric_data"}

    arr = np.array(drifts)
    abs_arr = np.abs(arr)

    return {
        "status": "ok",
        "total_records": len(rows),
        "window_size": window,
        "mean_drift": round(float(np.mean(arr)), 4),
        "std_drift": round(float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0), 4),
        "max_abs_drift": round(float(np.max(abs_arr)), 4),
        "min_drift": round(float(np.min(arr)), 4),
        "max_drift": round(float(np.max(arr)), 4),
        "breaches_warning": int(np.sum(abs_arr > WARN_THRESHOLD)),
        "breaches_critical": int(np.sum(abs_arr > CRIT_THRESHOLD)),
        "pct_above_warning": round(float(np.mean(abs_arr > WARN_THRESHOLD) * 100), 2),
        "pct_above_critical": round(float(np.mean(abs_arr > CRIT_THRESHOLD) * 100), 2),
    }


def rolling_window_analysis(rows: list[dict], window: int = 20) -> list[dict]:
    """Compute statistics over each rolling window."""
    if len(rows) < window:
        return []

    results: list[dict] = []
    for i in range(len(rows) - window + 1):
        chunk = rows[i : i + window]
        stats = compute_window_stats(chunk, window=window)
        results.append({
            "window_start": i,
            "window_end": i + window - 1,
            **stats,
        })
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(stats: dict, window_stats: list[dict]) -> int:
    """Print audit report and return exit code."""
    print("=" * 60)
    print("Delta Telemetry Drift Audit Report")
    print("=" * 60)
    print()

    if stats["status"] != "ok":
        print(f"Status: {stats['status']}")
        return 1

    print(f"Total records analysed: {stats['total_records']}")
    print(f"Window size:            {stats['window_size']} snapshots")
    print()
    print("Overall statistics:")
    print(f"  Mean drift:            {stats['mean_drift']}")
    print(f"  Std deviation:         {stats['std_drift']}")
    print(f"  Min / Max drift:       {stats['min_drift']} / {stats['max_drift']}")
    print(f"  Max absolute drift:    {stats['max_abs_drift']}")
    print()
    print("Threshold breaches:")
    print(f"  Warning  (|drift| > {WARN_THRESHOLD}):  {stats['breaches_warning']} "
          f"({stats['pct_above_warning']}%)")
    print(f"  Critical (|drift| > {CRIT_THRESHOLD}): {stats['breaches_critical']} "
          f"({stats['pct_above_critical']}%)")
    print()

    # --- Interval evaluation ---
    exit_code = 0
    if stats["breaches_critical"] > MAX_CRIT_BREACHES:
        print("CRITICAL: Delta spikes exceeded critical threshold more than "
              f"{MAX_CRIT_BREACHES} times.")
        print("    -> The current 300-second rebalancing frequency is TOO SLOW")
        print("      to maintain delta neutrality against live market microstructure.")
        print("    -> Recommendation: DECREASE rebalancing interval to <= 60 seconds")
        print("      OR tighten delta-neutral tolerance.")
        exit_code = 2
    elif stats["breaches_warning"] > 0:
        print(f"WARNING: {stats['breaches_warning']} delta drift(s) exceeded "
              f"warning threshold ({WARN_THRESHOLD}).")
        print("    -> Consider reviewing rebalancing frequency or hedge slippage.")
        exit_code = 1
    else:
        print("[PASS] All delta drifts within normal bounds.")

    # Rolling window anomalies
    if window_stats:
        max_crit_in_window = max(
            (w.get("breaches_critical", 0) for w in window_stats), default=0
        )
        if max_crit_in_window > 0:
            print()
            print(f"Rolling-window max critical breaches: {max_crit_in_window}")

    print()
    print("=" * 60)

    return exit_code


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit delta telemetry for drift anomalies",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to delta_telemetry.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size (default: 20)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of human-readable report",
    )
    args = parser.parse_args(argv)

    rows = parse_csv(args.input)
    stats = compute_window_stats(rows, window=args.window)
    window_stats = rolling_window_analysis(rows, window=args.window)

    if args.json:
        output = {
            "summary": stats,
            "window_analysis": window_stats,
        }
        print(json.dumps(output, indent=2, default=str))
        return 0 if stats.get("breaches_critical", 0) <= MAX_CRIT_BREACHES else 2

    return print_report(stats, window_stats)


if __name__ == "__main__":
    raise SystemExit(main())
