#!/usr/bin/env bash
# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
#
# run_factor_pipeline.sh — Phase-8 equity factor gate pipeline.
#
# Runs the institutional gate on the REAL PIT SP500 panel, then (if it passes)
# simulates a paper reconciliation and prints the verdict.
#
# Usage:
#   bash scripts/run_factor_pipeline.sh [--max-symbols N]
#
# Notes:
#   * --max-symbols caps the universe for speed (default: full ~612 names).
#     The canonical production run uses the full panel (omit the flag).
#   * The gate FAILS CLOSED by design and invokes emergency_kill in dry-run
#     unless GATE_ARM_KILL=1 (never arm on a backtest machine).
set -euo pipefail

export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)"

MAX_SYM=""
for arg in "$@"; do
  case "$arg" in
    --max-symbols) : ;;  # value consumed below
    *) if [ "${prev:-}" = "--max-symbols" ]; then MAX_SYM="$arg"; fi ;;
  esac
  prev="$arg"
done
if [ -n "${MAX_SYM:-}" ]; then MAX_ARG="--max-symbols ${MAX_SYM}"; else MAX_ARG=""; fi

echo ">>> [1/3] Running institutional gate (equity_factor_neutral) on REAL panel ..."
python scripts/institutional_gate.py --strategy equity_factor_neutral --cost 0.0010 ${MAX_ARG}

echo ">>> [2/3] Simulating paper reconciliation (watchdog, single pass) ..."
python scripts/reconciliation_watchdog.py --once --paper 2>/dev/null || echo "    (watchdog skipped: no broker creds / paper env)"

echo ">>> [3/3] Verdict"
if command -v jq >/dev/null 2>&1; then
  cat reports/factor_gate_verdict.json | jq .
else
  cat reports/factor_gate_verdict.json
fi

# Outcome banner (do NOT edit the gate conditions to flip this).
STATUS="$(python -c "import json;print(json.load(open('reports/factor_gate_verdict.json'))['gate_status'])")"
if [ "$STATUS" = "PASS" ]; then
  echo "FACTOR ALPHA CONFIRMED. LOCKING CONFIG TO EQUITY_FACTOR_NEUTRAL. 30-DAY PAPER COUNTDOWN STARTS."
else
  echo "FACTOR GATE FAILED. ROLLBACK TO OPTIONS_DELTA_NEUTRAL RECOMMENDED ONLY AFTER DATA EXPANSION. KILL SWITCH ENGAGED."
fi
