#!/usr/bin/env python3
# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
"""live_migration_controller.py — Audit Mandate 2026-07-12 (Phase 7).

The 30-day paper->live bridge. Flips config/live.yaml from single_share to
dynamic_vol ONLY when BOTH hold:
  1. reports/GATE_VERDICT.json exists and gate_status == "PASS".
  2. reports/live_equity.csv has >= 30 distinct reconciled trading days.

Otherwise prints a blocking message with the number of days remaining.
"never trusts, always verifies": it re-reads the files every run; nothing is
cached, nothing is assumed.

Usage:
    python scripts/live_migration_controller.py            # check + maybe flip
    python scripts/live_migration_controller.py --check    # report only, no write
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CONFIG = REPO_ROOT / "config" / "live.yaml"
VERDICT = REPO_ROOT / "reports" / "GATE_VERDICT.json"
LEDGER = REPO_ROOT / "reports" / "live_equity.csv"
REQUIRED_DAYS = 30


def gate_passed() -> tuple[bool, str]:
    if not VERDICT.exists():
        return False, "GATE_VERDICT.json missing (run scripts/institutional_gate.py)"
    try:
        v = json.loads(VERDICT.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return False, f"GATE_VERDICT.json unreadable: {exc}"
    status = v.get("gate_status")
    return (status == "PASS"), f"gate_status={status}"


def reconciled_days() -> int:
    """Distinct trading days present in the ledger (proxy for reconciled days)."""
    if not LEDGER.exists():
        return 0
    days = set()
    for row in LEDGER.read_text(encoding="utf-8").splitlines():
        row = row.strip()
        if not row or row.lower().startswith("date"):
            continue
        days.add(row.split(",")[0])
    return len(days)


def flip_to_dynamic_vol() -> bool:
    """Flip scale_mode: single_share -> dynamic_vol in config/live.yaml.

    Text-level edit (comments/order preserved). Idempotent.
    """
    text = CONFIG.read_text(encoding="utf-8")
    if "dynamic_vol" in text and "single_share" not in text:
        print("Already migrated (scale_mode: dynamic_vol).")
        return True
    new = text.replace("scale_mode: single_share", "scale_mode: dynamic_vol")
    if new == text:
        print("No 'scale_mode: single_share' occurrence found to flip.")
        return False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    CONFIG.with_suffix(f".yaml.pre_migration_{stamp}.bak").write_text(text, encoding="utf-8")
    CONFIG.write_text(new, encoding="utf-8")
    print(f"MIGRATED: scale_mode -> dynamic_vol (backup .pre_migration_{stamp}.bak).")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="30-day paper->live migration controller.")
    ap.add_argument("--check", action="store_true", help="Report only; never modify config.")
    args = ap.parse_args()

    passed, gate_reason = gate_passed()
    days = reconciled_days()
    remaining = max(0, REQUIRED_DAYS - days)

    print("=" * 64)
    print("LIVE MIGRATION CONTROLLER (Phase 7)")
    print(f"  Gate: {'PASS' if passed else 'BLOCK'} ({gate_reason})")
    print(f"  Reconciled days: {days}/{REQUIRED_DAYS}")
    print("=" * 64)

    if not passed:
        print(f"Locked. Gate has not PASSED. ({gate_reason})")
        return 1
    if days < REQUIRED_DAYS:
        print(f"Locked. Need 30 reconciled days. Days remaining: {remaining}")
        return 1

    if args.check:
        print("READY to migrate (both conditions met). --check set: no changes written.")
        return 0

    return 0 if flip_to_dynamic_vol() else 1


if __name__ == "__main__":
    sys.exit(main())
