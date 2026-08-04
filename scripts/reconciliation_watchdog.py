#!/usr/bin/env python3
# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
"""reconciliation_watchdog.py — Audit Mandate 2026-07-12 (Phase 5).

Real-time P&L integrity. Every 5 minutes:
  1. Fetch broker equity via Alpaca (account.equity).
  2. Read the last row of reports/live_equity.csv (the internal ledger).
  3. delta = broker_equity - ledger_equity.
  4. If abs(delta) > $5.00 -> append CRITICAL to reports/audit_alerts.log AND
     invoke scripts/emergency_kill.py.

This catches exactly the failure in reports/live_equity.csv: a flat $100,000
ledger while fills stream through Alpaca (ledger and broker diverge silently).

Usage:
    python scripts/reconciliation_watchdog.py            # loop, 5-min interval
    python scripts/reconciliation_watchdog.py --once     # single check (CI/test)
    python scripts/reconciliation_watchdog.py --interval 60
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LEDGER_CSV = REPO_ROOT / "reports" / "live_equity.csv"
ALERT_LOG = REPO_ROOT / "reports" / "audit_alerts.log"
EMERGENCY_KILL = REPO_ROOT / "scripts" / "emergency_kill.py"
TOLERANCE_USD = 5.0


def _alert(msg: str, critical: bool = False) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    level = "CRITICAL" if critical else "INFO"
    line = f"{stamp} | {level} | {msg}"
    print(line)
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERT_LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_ledger_equity() -> float | None:
    if not LEDGER_CSV.exists():
        _alert(f"Ledger file missing: {LEDGER_CSV}", critical=True)
        return None
    last = None
    with open(LEDGER_CSV, encoding="utf-8") as fh:
        for row in fh:
            row = row.strip()
            if row and not row.lower().startswith("date"):
                last = row
    if not last:
        return None
    try:
        # columns: date,cash,positions_value,total_equity,ret
        return float(last.split(",")[3])
    except (IndexError, ValueError):
        _alert(f"Malformed ledger row: {last!r}", critical=True)
        return None


def fetch_broker_equity() -> float | None:
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        _alert("ALPACA credentials not set — cannot fetch broker equity.", critical=False)
        return None
    try:
        from alpaca.trading.client import TradingClient

        acct = TradingClient(key, secret, paper=True).get_account()
        return float(acct.equity)
    except Exception as exc:  # pragma: no cover - env dependent
        _alert(f"Broker equity fetch failed: {exc}", critical=False)
        return None


def invoke_kill(arm: bool) -> None:
    cmd = [sys.executable, str(EMERGENCY_KILL)] + ([] if arm else ["--dry-run"])
    _alert(f"Invoking emergency kill ({'ARMED' if arm else 'dry-run'})", critical=True)
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        _alert(f"emergency_kill invocation error: {exc}", critical=True)


def check_once(arm_kill: bool) -> bool:
    """Return True if reconciled within tolerance, else False."""
    ledger = read_ledger_equity()
    broker = fetch_broker_equity()
    if ledger is None or broker is None:
        _alert(f"Reconciliation SKIPPED (ledger={ledger}, broker={broker}).")
        return False
    delta = broker - ledger
    if abs(delta) > TOLERANCE_USD:
        _alert(
            f"RECONCILIATION BREAK: broker=${broker:,.2f} ledger=${ledger:,.2f} "
            f"delta=${delta:,.2f} (> ${TOLERANCE_USD}). HALTING.",
            critical=True,
        )
        invoke_kill(arm_kill)
        return False
    _alert(f"Reconciled OK: broker=${broker:,.2f} ledger=${ledger:,.2f} delta=${delta:,.2f}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="P&L reconciliation watchdog.")
    ap.add_argument("--once", action="store_true", help="Single check then exit.")
    ap.add_argument("--interval", type=int, default=300, help="Seconds between checks.")
    args = ap.parse_args()

    arm_kill = os.environ.get("WATCHDOG_ARM_KILL") == "1"
    if args.once:
        return 0 if check_once(arm_kill) else 1

    _alert(f"Watchdog started (interval={args.interval}s, tolerance=${TOLERANCE_USD}).")
    while True:
        try:
            check_once(arm_kill)
        except Exception as exc:  # never let the watchdog die silently
            _alert(f"Watchdog iteration error: {exc}", critical=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
