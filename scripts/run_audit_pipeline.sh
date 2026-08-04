#!/usr/bin/env bash
# EXECUTION OVERRIDE: This code is for an offline backtest/paper-trading environment. Ensure Alpaca keys are in .env.
#
# run_audit_pipeline.sh — Audit Mandate 2026-07-12 orchestration.
# Runs the QuantCore salvage pipeline in dependency order and STOPS on the
# first hard failure. The institutional gate is the arbiter: if it does not
# emit gate_status=PASS, no migration to dynamic sizing occurs.
#
#   Env toggles:
#     GATE_ARM_KILL=1      -> gate fires a REAL flatten on FAIL (default: dry-run)
#     GATE_NUM_TRIALS=N    -> override DSR trial count
#     WATCHDOG_ARM_KILL=1  -> watchdog fires a REAL flatten on break
#
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=.

echo "############################################################"
echo "# QUANTCORE AUDIT PIPELINE — $(date -u +%FT%TZ)"
echo "############################################################"

# 0. Load .env credentials if present (Alpaca / Telegram).
if [ -f .env ]; then set -a; . ./.env; set +a; fi

# 1. Unit tests for the Phase-4 routing guard (must pass before anything trades).
echo ">>> [1/5] Routing guard unit tests"
python -m pytest tests/test_option_routing.py -q

# 2. Institutional gate — the single source of truth (writes reports/GATE_VERDICT.json).
#    On FAIL it internally invokes scripts/emergency_kill.py. We let it run and
#    capture the verdict rather than aborting, so the migration step can report.
echo ">>> [2/5] Institutional gate (real data only)"
set +e
python scripts/institutional_gate.py
GATE_RC=$?
set -e
echo "    gate exit code: ${GATE_RC}"

# 3. Reconciliation watchdog — single pass (CI-safe). In production run without
#    --once so it loops every 5 minutes.
echo ">>> [3/5] Reconciliation watchdog (single pass)"
python scripts/reconciliation_watchdog.py --once || echo "    (watchdog reported a break/skip)"

# 4. Migration controller — will stay LOCKED until gate PASS + 30 reconciled days.
echo ">>> [4/5] Live migration controller (check only)"
python scripts/live_migration_controller.py --check || echo "    (migration locked — expected)"

# 5. Summary.
echo ">>> [5/5] Verdict"
python - <<'PY'
import json, pathlib
v = json.loads(pathlib.Path("reports/GATE_VERDICT.json").read_text())
print("  GATE_STATUS:", v.get("gate_status"))
for f in v.get("failures", []):
    print("   -", f)
PY

echo "############################################################"
echo "# PIPELINE COMPLETE. Deployment of dynamic sizing requires"
echo "# gate_status=PASS (currently gated by real-data coverage)."
echo "############################################################"
