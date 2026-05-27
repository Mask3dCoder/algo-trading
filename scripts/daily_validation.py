"""Daily performance validation — compare live paper returns to OOS baseline.

Reads walk-forward OOS daily returns from reports/aggregator_results.json,
reads today's live return from reports/live_equity.csv, computes a z-score,
classifies as OK / WARNING / ALERT, and logs the result.

Usage:
    python scripts/daily_validation.py                # validate today
    python scripts/daily_validation.py --date 2026-05-28  # validate specific day
    python scripts/daily_validation.py --verbose      # show distribution stats
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
AGGREGATOR_JSON = ROOT / "reports" / "aggregator_results.json"
LIVE_EQUITY_CSV = ROOT / "reports" / "live_equity.csv"
VALIDATION_LOG = ROOT / "operations" / "validation_log.csv"
ALERTS_LOG = ROOT / "reports" / "alerts.log"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_oos_returns() -> np.ndarray:
    """Load combined OOS daily returns from aggregator results.

    Strategy-level returns are risk-parity weighted and summed.
    Returns a 1D numpy array of daily return fractions.
    """
    if not AGGREGATOR_JSON.exists():
        print(f"ERROR: {AGGREGATOR_JSON} not found. Run aggregator backtest first.")
        sys.exit(1)

    with open(AGGREGATOR_JSON) as f:
        data = json.load(f)

    individual = data.get("individual", {})
    if not individual:
        print("ERROR: No individual strategy results in aggregator JSON.")
        sys.exit(1)

    # Parse each strategy's daily_returns_raw (numpy array string)
    strategy_returns: dict[str, np.ndarray] = {}
    for name, info in individual.items():
        raw = info.get("daily_returns_raw", "")
        if not raw:
            continue
        try:
            arr = np.fromstring(raw.strip("[]\n"), sep=" ")
            strategy_returns[name] = arr
        except Exception as exc:
            print(f"WARNING: Could not parse returns for {name}: {exc}")

    if not strategy_returns:
        print("ERROR: No valid strategy returns found.")
        sys.exit(1)

    # Equal-weight combination (matches aggregator method)
    min_len = min(len(v) for v in strategy_returns.values())
    combined = np.zeros(min_len)
    n = len(strategy_returns)
    for arr in strategy_returns.values():
        combined += arr[:min_len] / n

    return combined


def load_live_return(target_date: str | None = None) -> tuple[str, float]:
    """Load the latest live return from live_equity.csv.

    Returns (date_str, daily_return_fraction).
    """
    if not LIVE_EQUITY_CSV.exists():
        print(f"ERROR: {LIVE_EQUITY_CSV} not found.")
        sys.exit(1)

    import csv

    rows: list[dict[str, str]] = []
    with open(LIVE_EQUITY_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        print("ERROR: live_equity.csv is empty.")
        sys.exit(1)

    if target_date:
        # Find rows matching target date
        matching = [r for r in rows if r["date"].startswith(target_date)]
        if not matching:
            print(f"ERROR: No equity data for {target_date}.")
            sys.exit(1)
        last = matching[-1]
    else:
        # Use the last non-zero-equity row
        last = rows[-1]
        for r in reversed(rows):
            try:
                eq = float(r.get("total_equity", 0))
                if eq > 0:
                    last = r
                    break
            except (ValueError, TypeError):
                continue

    date_str = last["date"][:10]
    try:
        ret = float(last.get("ret", 0.0))
    except (ValueError, TypeError):
        ret = 0.0

    return date_str, ret


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def compute_zscore(live_ret: float, oos_returns: np.ndarray) -> float:
    """Compute z-score of live return vs OOS distribution."""
    mu = np.mean(oos_returns)
    sigma = np.std(oos_returns)
    if sigma < 1e-10:
        return 0.0
    return (live_ret - mu) / sigma


def classify(z: float) -> str:
    """Classify z-score into OK / WARNING / ALERT."""
    az = abs(z)
    if az >= 3.0:
        return "ALERT"
    if az >= 2.0:
        return "WARNING"
    return "OK"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def append_validation_log(date_str: str, live_ret: float, z: float, status: str) -> None:
    """Append a row to operations/validation_log.csv."""
    VALIDATION_LOG.parent.mkdir(parents=True, exist_ok=True)

    write_header = not VALIDATION_LOG.exists()
    with open(VALIDATION_LOG, "a") as f:
        if write_header:
            f.write("date,live_return,z_score,status\n")
        f.write(f"{date_str},{live_ret:.6f},{z:.3f},{status}\n")


def append_alert(date_str: str, live_ret: float, z: float, status: str) -> None:
    """Append to reports/alerts.log when status is ALERT or WARNING."""
    ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().isoformat(timespec="seconds")
    with open(ALERTS_LOG, "a") as f:
        f.write(f"[{ts}] {status}: date={date_str} ret={live_ret:.4%} z={z:.2f}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily performance validation")
    parser.add_argument("--date", help="Validate specific date (YYYY-MM-DD)")
    parser.add_argument("--verbose", action="store_true", help="Show OOS distribution stats")
    args = parser.parse_args()

    # Load OOS baseline
    oos_returns = load_oos_returns()
    oos_mu = float(np.mean(oos_returns))
    oos_sigma = float(np.std(oos_returns))
    oos_n = len(oos_returns)

    if args.verbose:
        print(f"OOS distribution: n={oos_n}, mu={oos_mu:.6f}, sigma={oos_sigma:.6f}")
        print(f"  5th pct: {np.percentile(oos_returns, 5):.6f}")
        print(f"  95th pct: {np.percentile(oos_returns, 95):.6f}")

    # Load live return
    date_str, live_ret = load_live_return(args.date)

    # Compute z-score and classify
    z = compute_zscore(live_ret, oos_returns)
    status = classify(z)

    # One-line status
    print(f"{date_str} | ret={live_ret:+.4%} | z={z:+.2f} | {status}")

    # Log
    append_validation_log(date_str, live_ret, z, status)

    # Alert on WARNING or ALERT
    if status in ("WARNING", "ALERT"):
        append_alert(date_str, live_ret, z, status)
        if status == "ALERT":
            print(f"ALERT: z-score {z:.2f} exceeds +/-3 threshold. Check positions immediately.")

    # Exit code: 0 for OK, 1 for WARNING, 2 for ALERT
    if status == "ALERT":
        sys.exit(2)
    elif status == "WARNING":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
