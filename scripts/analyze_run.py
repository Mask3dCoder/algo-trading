#!/usr/bin/env python
"""Analyze paper trading run results from metrics CSV and log files.

Usage:
    python scripts/analyze_run.py --latest
    python scripts/analyze_run.py --metrics logs/paper_trade_metrics_20260427_120000.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402


def find_latest_metrics(log_dir: str = "logs") -> str | None:
    """Find the most recent metrics CSV file."""
    pattern = os.path.join(log_dir, "paper_trade_metrics_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def analyze_metrics(csv_path: str) -> dict:
    """Parse a metrics CSV and return summary statistics."""
    rows: list[dict] = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"error": "No data in metrics CSV"}

    timestamps = [r["timestamp"] for r in rows]
    total_pnls = [float(r["total_pnl"]) for r in rows if r.get("total_pnl")]
    mems = [float(r["memory_rss_mb"]) for r in rows if r.get("memory_rss_mb") and r["memory_rss_mb"] != "None"]
    data_ages = [float(r["last_data_age_seconds"]) for r in rows if r.get("last_data_age_seconds") and r["last_data_age_seconds"] != "None"]
    order_submitted = int(rows[-1].get("orders_submitted_count", 0)) if rows else 0
    order_filled = int(rows[-1].get("orders_filled_count", 0)) if rows else 0
    order_rejected = int(rows[-1].get("orders_rejected_count", 0)) if rows else 0
    ks_events = sum(1 for r in rows if r.get("kill_switch_active") == "True")

    # P&L metrics
    final_pnl = total_pnls[-1] if total_pnls else 0.0
    pnl_series = np.array(total_pnls)
    max_pnl = float(np.max(pnl_series)) if len(pnl_series) > 0 else 0.0
    min_pnl = float(np.min(pnl_series)) if len(pnl_series) > 0 else 0.0
    avg_pnl = float(np.mean(pnl_series)) if len(pnl_series) > 0 else 0.0

    # Drawdown from P&L series
    peak = np.maximum.accumulate(pnl_series)
    dd = pnl_series - peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0
    max_dd_pct = max_dd / 100000.0  # rough assuming 100k initial capital

    # Run duration
    if len(timestamps) >= 2:
        try:
            t0 = datetime.fromisoformat(timestamps[0])
            t1 = datetime.fromisoformat(timestamps[-1])
            duration_hours = (t1 - t0).total_seconds() / 3600
        except (ValueError, TypeError):
            duration_hours = 0.0
    else:
        duration_hours = 0.0

    # Winning vs losing periods (by P&L delta)
    pnl_deltas = np.diff(pnl_series) if len(pnl_series) > 1 else np.array([])
    winning_periods = int(np.sum(pnl_deltas > 0))
    losing_periods = int(np.sum(pnl_deltas < 0))
    win_rate = winning_periods / (winning_periods + losing_periods) if (winning_periods + losing_periods) > 0 else 0.0

    return {
        "csv_path": csv_path,
        "n_samples": len(rows),
        "first_timestamp": timestamps[0] if timestamps else "N/A",
        "last_timestamp": timestamps[-1] if timestamps else "N/A",
        "duration_hours": round(duration_hours, 1),
        "total_pnl": round(final_pnl, 2),
        "max_pnl": round(max_pnl, 2),
        "min_pnl": round(min_pnl, 2),
        "avg_pnl": round(avg_pnl, 2),
        "max_drawdown_from_peak": round(max_dd, 2),
        "orders_submitted": order_submitted,
        "orders_filled": order_filled,
        "orders_rejected": order_rejected,
        "fill_rate": round(order_filled / order_submitted, 3) if order_submitted > 0 else 0.0,
        "kill_switch_events": ks_events,
        "win_rate_periods": round(win_rate, 3),
        "memory_avg_mb": round(float(np.mean(mems)), 1) if mems else None,
        "memory_max_mb": round(float(np.max(mems)), 1) if mems else None,
        "data_age_avg_sec": round(float(np.mean(data_ages)), 1) if data_ages else None,
        "data_age_max_sec": round(float(np.max(data_ages)), 1) if data_ages else None,
    }


def find_latest_log(log_dir: str = "logs") -> str | None:
    """Find the most recent paper trade log file."""
    pattern = os.path.join(log_dir, "paper_trade_*.log")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def count_log_events(log_path: str) -> dict:
    """Count errors, warnings, and critical events in a log file."""
    counts = {"ERROR": 0, "CRITICAL": 0, "WARNING": 0, "INFO": 0, "total_lines": 0}
    with open(log_path, "r") as f:
        for line in f:
            counts["total_lines"] += 1
            for level in ("ERROR", "CRITICAL", "WARNING", "INFO"):
                if f"[{level}]" in line or f" {level} " in line:
                    counts[level] += 1
                    break
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze paper trading run results")
    parser.add_argument("--latest", action="store_true", help="Analyze the most recent run")
    parser.add_argument("--metrics", help="Path to specific metrics CSV file")
    args = parser.parse_args()

    csv_path = args.metrics
    if args.latest or not csv_path:
        csv_path = find_latest_metrics()
        if not csv_path:
            print("No metrics CSV files found in logs/")
            return 1

    log_path = find_latest_log()

    print("=" * 65)
    print("  QuantCore Paper Trading — Run Analysis")
    print("=" * 65)

    # Metrics analysis
    results = analyze_metrics(csv_path)
    if "error" in results:
        print(f"  Error: {results['error']}")
        return 1

    print(f"\n  Metrics source:  {results['csv_path']}")
    print(f"  Log source:      {log_path or 'not found'}")
    print(f"  Duration:        {results['duration_hours']:.1f} hours")
    print(f"  Data points:     {results['n_samples']}")
    print(f"  Period:          {results['first_timestamp']} -> {results['last_timestamp']}")
    print()
    print(f"  Total P&L:       ${results['total_pnl']:,.2f}")
    print(f"  Peak P&L:        ${results['max_pnl']:,.2f}")
    print(f"  Trough P&L:      ${results['min_pnl']:,.2f}")
    print(f"  Avg P&L:         ${results['avg_pnl']:,.2f}")
    print(f"  Max DD (from peak): ${results['max_drawdown_from_peak']:,.2f}")
    print(f"  Period win rate: {results['win_rate_periods']:.1%}")
    print()
    print(f"  Orders submitted: {results['orders_submitted']}")
    print(f"  Orders filled:    {results['orders_filled']}")
    print(f"  Orders rejected:  {results['orders_rejected']}")
    print(f"  Fill rate:        {results['fill_rate']:.1%}")
    print(f"  Kill switch events: {results['kill_switch_events']}")
    print()
    if results["memory_avg_mb"] is not None:
        print(f"  Memory (avg/max): {results['memory_avg_mb']}MB / {results['memory_max_mb']}MB")
    if results["data_age_avg_sec"] is not None:
        print(f"  Data age (avg/max): {results['data_age_avg_sec']}s / {results['data_age_max_sec']}s")

    # Log events
    if log_path:
        log_counts = count_log_events(log_path)
        print(f"\n  Log summary ({log_path}):")
        print(f"    Total lines: {log_counts['total_lines']}")
        print(f"    ERROR:   {log_counts['ERROR']}")
        print(f"    CRITICAL: {log_counts['CRITICAL']}")
        print(f"    WARNING: {log_counts['WARNING']}")
        print(f"    INFO:    {log_counts['INFO']}")

    print("\n" + "=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
