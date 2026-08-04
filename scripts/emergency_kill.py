#!/usr/bin/env python3
# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
"""emergency_kill.py — Audit Mandate 2026-07-12 (Phase 1).

Hard stop:
  1. touch /tmp/KILL_SWITCH (the live loop polls this and halts).
  2. Cancel ALL open orders via Alpaca.
  3. Flatten ALL positions via Alpaca.
  4. Append everything to reports/emergency_stop.log.

Safety:
  * Targets the PAPER endpoint by default (paper-api.alpaca.markets).
  * ``--dry-run`` writes the kill switch + logs intent but sends no broker calls.
  * Never raises on broker error — logs and continues (a kill must not fail open).

Usage:
    python scripts/emergency_kill.py            # execute (paper)
    python scripts/emergency_kill.py --dry-run  # simulate, no broker calls
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

KILL_SWITCH_PATH = os.environ.get("QUANTCORE_KILL_SWITCH", "/tmp/KILL_SWITCH")
LOG_PATH = REPO_ROOT / "reports" / "emergency_stop.log"


def _log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    line = f"{stamp} | {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def trip_kill_switch(path: str = KILL_SWITCH_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8"):
        os.utime(path, None)
    _log(f"KILL SWITCH SET: {path}")


def _alpaca_client():
    """Return an authenticated Alpaca TradingClient, or None if unavailable."""
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        _log("ALPACA credentials not found in env — skipping broker flatten.")
        return None
    try:
        from alpaca.trading.client import TradingClient

        # paper=True forces the paper endpoint regardless of base_url.
        return TradingClient(key, secret, paper=True)
    except Exception as exc:  # pragma: no cover - env dependent
        _log(f"Failed to construct Alpaca TradingClient: {exc}")
        return None


def cancel_and_flatten(dry_run: bool) -> None:
    if dry_run:
        _log("DRY-RUN: would cancel all open orders and flatten all positions.")
        return
    client = _alpaca_client()
    if client is None:
        return
    # Cancel open orders.
    try:
        client.cancel_orders()
        _log("Cancelled ALL open orders.")
    except Exception as exc:
        _log(f"cancel_orders failed: {exc}")
    # Flatten positions.
    try:
        client.close_all_positions(cancel_orders=True)
        _log("Flattened ALL positions (close_all_positions).")
    except Exception as exc:
        _log(f"close_all_positions failed: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Emergency kill switch + flatten.")
    ap.add_argument("--dry-run", action="store_true", help="No broker calls; write kill switch + log only.")
    ap.add_argument("--kill-switch-path", default=KILL_SWITCH_PATH)
    args = ap.parse_args()

    _log("=" * 60)
    _log(f"EMERGENCY KILL INVOKED (dry_run={args.dry_run})")
    trip_kill_switch(args.kill_switch_path)
    cancel_and_flatten(args.dry_run)
    _log("EMERGENCY KILL COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
