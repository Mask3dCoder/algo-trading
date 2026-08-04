#!/usr/bin/env bash
# =============================================================================
# manage_silent_run.sh — Delta Neutral VRP 72-Hour Silent Run Operations
# =============================================================================
# Path:        bin/manage_silent_run.sh
# Purpose:     Automated deployment, health monitoring, and EOD audit for the
#              delta_neutral_vrp passive-tracking validation window.
# Usage:
#   ./bin/manage_silent_run.sh launch          # Phase 1: validate + start
#   ./bin/manage_silent_run.sh --status        # Phase 2: intraday dashboard
#   ./bin/manage_silent_run.sh --audit         # Phase 3: EOD telemetry audit
#   ./bin/manage_silent_run.sh stop            # graceful shutdown
#   ./bin/manage_silent_run.sh restart         # stop + launch
#   ./bin/manage_silent_run.sh validate        # config-only, no launch
#
# Exit codes:
#   0  — success / pass
#   1  — warning / degraded (not halted)
#   2  — critical failure / configuration mismatch / launch abort
#   3  — runner process not running
# =============================================================================

set -euo pipefail

# ── Paths (resolve relative to this script's directory) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${WORKSPACE_DIR}/config/paper_trading.yaml"
RUNNER_MODULE="quantcore.live.paper_trading.runner"
LOG_DIR="${WORKSPACE_DIR}/reports/paper_trading"
LOG_FILE="${LOG_DIR}/run.log"
PID_FILE="${WORKSPACE_DIR}/var/run/silent_run.pid"
AUDIT_SCRIPT="${WORKSPACE_DIR}/bin/audit_telemetry_drift.py"
PASSIVE_TRADES="${LOG_DIR}/passive_trades.json"
DELTA_TELEMETRY="${LOG_DIR}/delta_telemetry.csv"
PYTHON="${PYTHON:-python}"

# ── Derived paths (created on demand) ────────────────────────────────────────
VAR_RUN_DIR="${WORKSPACE_DIR}/var/run"

# ── Required config values ────────────────────────────────────────────────────
REQUIRED_PASSIVE_MODE="true"
REQUIRED_SCALE_MODE="CAPITAL_ALLOCATED"

# ── Audit thresholds ───────────────────────────────────────────────────────────
MAX_CRIT_BREACHES=3

# =============================================================================
# HELPERS
# =============================================================================

log()  { printf '%s [%-8s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "INFO"  "$*"; }
warn() { printf '%s [%-8s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "WARN"  "$*" >&2; }
err()  { printf '%s [%-8s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "ERROR" "$*" >&2; }
die()  { err "$@"; exit 2; }

# -----------------------------------------------------------------------------
# ensure_dirs — create working directories if missing
# -----------------------------------------------------------------------------
ensure_dirs() {
    mkdir -p "$LOG_DIR" "$VAR_RUN_DIR"
}

# -----------------------------------------------------------------------------
# get_pid — read PID from pidfile; empty string if not running
# -----------------------------------------------------------------------------
get_pid() {
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        cat "$PID_FILE"
    else
        echo ""
    fi
}

# -----------------------------------------------------------------------------
# is_running — returns 0 if the runner process is alive
# -----------------------------------------------------------------------------
is_running() {
    local pid
    pid=$(get_pid)
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# -----------------------------------------------------------------------------
# read_yaml_field — parse a top-level YAML key value via Python
#                  returns empty string on parse failure
# -----------------------------------------------------------------------------
read_yaml_field() {
    local key="$1"
    "$PYTHON" - "$key" <<'PYEOF'
import sys, yaml
key = sys.argv[1]
with open(sys.argv[2] if len(sys.argv) > 2 else "config/paper_trading.yaml") as f:
    data = yaml.safe_load(f) or {}
# Support dot-notation keys
for k in key.split("."):
    if isinstance(data, dict):
        data = data.get(k, {})
    else:
        print("")
        sys.exit(0)
v = data if data is not None else ""
print(v)
PYEOF
    # Pass key as first arg and config path as second
    "$PYTHON" -c "
import sys, yaml
key = '$key'
with open('$CONFIG_FILE') as f:
    data = yaml.safe_load(f) or {}
for k in key.split('.'):
    if isinstance(data, dict):
        data = data.get(k, {})
    else:
        print('')
        sys.exit(0)
v = data if data is not None else ''
print(v)
"
}

YAML_FIELD="${WORKSPACE_DIR}/bin/yaml_field.py"

# -----------------------------------------------------------------------------
# read_yaml_val — parse a dot-notation key from config/paper_trading.yaml
#                  prints empty string if not found
# -----------------------------------------------------------------------------
read_yaml_val() {
    local key="$1"
    "$PYTHON" "$YAML_FIELD" "$key" 2>/dev/null || echo ""
}

# =============================================================================
# PHASE 1 — PRE-FLIGHT CONFIG VALIDATION
# =============================================================================

validate_config() {
    log "──────────────────────────────────────────────────────────────"
    log "PHASE 1  |  Pre-flight configuration validation"
    log "──────────────────────────────────────────────────────────────"

    # ---------- file existence ----------
    if [[ ! -f "$CONFIG_FILE" ]]; then
        die "Config file not found: $CONFIG_FILE"
    fi
    log "  config file  : $CONFIG_FILE"

    # ---------- parse fields ----------
    local passive_mode scale_mode log_interval
    passive_mode="$(read_yaml_val 'execution.passive_tracking_mode')"
    scale_mode="$(read_yaml_val 'execution.scale_mode')"
    log_interval="$(read_yaml_val 'delta_telemetry.log_interval_seconds')"

    log "  passive_tracking_mode : $passive_mode  (required: $REQUIRED_PASSIVE_MODE)"
    log "  scale_mode            : $scale_mode     (required: $REQUIRED_SCALE_MODE)"
    log "  log_interval_seconds   : $log_interval  (telemetry snapshot cadence)"

    local issues=0

    # Check 1 — passive_tracking_mode
    if [[ "${passive_mode,,}" != "${REQUIRED_PASSIVE_MODE,,}" ]]; then
        err "FAIL: execution.passive_tracking_mode must be '${REQUIRED_PASSIVE_MODE}'"
        err "       Found: '$passive_mode'"
        err "       ABORT — will not launch with broker execution enabled."
        issues=$((issues + 1))
    else
        log "  [OK] passive_tracking_mode = true"
    fi

    # Check 2 — scale_mode must be CAPITAL_ALLOCATED for realistic validation
    if [[ "$scale_mode" != "$REQUIRED_SCALE_MODE" ]]; then
        err "FAIL: execution.scale_mode must be '$REQUIRED_SCALE_MODE'"
        err "       Found: '$scale_mode'"
        err "       ABORT — single-share mode will not generate sufficient"
        err "       P&L / risk to evaluate PositionSizer or RiskController."
        issues=$((issues + 1))
    else
        log "  [OK] scale_mode = CAPITAL_ALLOCATED"
    fi

    if [[ $issues -gt 0 ]]; then
        die "Configuration validation failed with $issues error(s)."
    fi

    log "  [OK] All pre-flight checks passed."
    return 0
}

# =============================================================================
# PHASE 1 — DAEMONIZED LAUNCH
# =============================================================================

do_launch() {
    log "──────────────────────────────────────────────────────────────"
    log "PHASE 1  |  Launching silent run in background"
    log "──────────────────────────────────────────────────────────────"

    ensure_dirs

    # Abort if already running
    if is_running; then
        local pid
        pid=$(get_pid)
        warn "Runner already running (PID $pid). Stop it first."
        warn "Hint: $SCRIPT_DIR/manage_silent_run.sh stop"
        exit 2
    fi

    # Run pre-flight validation first (non-interactive — fail fast)
    validate_config || exit 2

    log "  starting runner  : $RUNNER_MODULE"
    log "  log file         : $LOG_FILE"
    log "  pid file         : $PID_FILE"

    # Launch in background — nohup + setsid for full session independence
    cd "$WORKSPACE_DIR"
    nohup "$PYTHON" -m "$RUNNER_MODULE" \
        --config "$CONFIG_FILE" \
        >> "$LOG_FILE" 2>&1 \
        &
    local pid=$!
    echo "$pid" > "$PID_FILE"

    log "  background PID   : $pid"
    log "  waiting 5s for initialisation..."

    # Post-launch telemetry hook
    sleep 5

    # Poll for log activity (up to 20s) to confirm runner started
    local waited=0
    while [[ $waited -lt 20 ]]; do
        if [[ -s "$LOG_FILE" ]]; then
            break
        fi
        sleep 2
        waited=$((waited + 2))
    done

    # Verify no initial errors in run.log
    local init_errors
    init_errors=$(grep -cE "(Traceback|Error|FATAL|KILLED|UNEXPECTED)" "$LOG_FILE" 2>/dev/null || true)
    if [[ "${init_errors:-0}" -gt 0 ]]; then
        err "FAIL: $init_errors error(s) found in run.log during startup."
        err "  Review: tail -50 $LOG_FILE"
        rm -f "$PID_FILE"
        exit 2
    fi

    # Verify runner process is still alive
    if ! is_running; then
        err "FAIL: Runner process exited unexpectedly."
        err "  Review: tail -50 $LOG_FILE"
        rm -f "$PID_FILE"
        exit 2
    fi

    # Verify delta_telemetry.csv has headers
    if [[ -f "$DELTA_TELEMETRY" ]]; then
        local has_header
        has_header=$(head -1 "$DELTA_TELEMETRY" 2>/dev/null | grep -c "delta_drift" || true)
        if [[ "$has_header" -eq 0 ]]; then
            warn "WARNING: delta_telemetry.csv exists but headers missing."
        else
            log "  [OK] delta_telemetry.csv initialised"
        fi
    else
        log "  [NOTE] delta_telemetry.csv not yet written (runner initialising)"
    fi

    log "──────────────────────────────────────────────────────────────"
    log "LAUNCH SUCCESS — Silent run active (PID $pid)"
    log "  Monitor : $SCRIPT_DIR/manage_silent_run.sh --status"
    log "  Audit   : $SCRIPT_DIR/manage_silent_run.sh --audit"
    log "  Stop    : $SCRIPT_DIR/manage_silent_run.sh stop"
    log "──────────────────────────────────────────────────────────────"
    return 0
}

# =============================================================================
# PHASE 2 — INTRADAY HEALTH CHECK DASHBOARD
# =============================================================================

do_status() {
    log "──────────────────────────────────────────────────────────────"
    log "PHASE 2  |  Silent Run Health Check Dashboard"
    log "──────────────────────────────────────────────────────────────"
    echo ""

    local pid
    pid=$(get_pid)

    # ---- 2a: Process Status ----
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  2a. PROCESS STATUS                                           │"
    echo "├──────────────────────────────────────────────────────────────┤"
    if [[ -z "$pid" ]]; then
        echo "│  [DOWN] PID file empty or runner not running                 │"
        echo "│  Run './manage_silent_run.sh launch' to start.                │"
        echo "└──────────────────────────────────────────────────────────────┘"
        echo ""
        echo "┌──────────────────────────────────────────────────────────────┐"
        echo "│  OVERALL STATUS: NOT RUNNING (exit 3)                         │"
        echo "└──────────────────────────────────────────────────────────────┘"
        return 3
    elif ! kill -0 "$pid" 2>/dev/null; then
        echo "│  [DOWN] PID $pid — process dead (stale pidfile)              │"
        rm -f "$PID_FILE"
        echo "│  (pidfile removed)                                           │"
        echo "│  Run './manage_silent_run.sh launch' to restart.             │"
        echo "└──────────────────────────────────────────────────────────────┘"
        echo ""
        echo "┌──────────────────────────────────────────────────────────────┐"
        echo "│  OVERALL STATUS: NOT RUNNING (exit 3)                         │"
        echo "└──────────────────────────────────────────────────────────────┘"
        return 3
    else
        echo "│  [UP]   Running — PID $pid"
        # RSS in KB
        local rss
        rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo "N/A")
        if [[ "$rss" != "N/A" ]]; then
            local rss_mb
            rss_mb=$(awk "BEGIN {printf \"%.1f\", $rss/1024}")
            echo "│  RSS    : ${rss_mb} MB  (resident memory)"
        else
            echo "│  RSS    : (ps unavailable on this platform)"
        fi
    fi
    echo "└──────────────────────────────────────────────────────────────┘"
    echo ""

    # ---- 2b: Connection Heartbeat ----
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  2b. CONNECTION HEARTBEAT  (last 5 events)                   │"
    echo "├──────────────────────────────────────────────────────────────┤"
    local conn_events
    conn_events=$(grep -E "(DEGRADED|DISCONNECTED|RECONNECT|HEARTBEAT|CONNECTED)" \
        "$LOG_FILE" 2>/dev/null | tail -5 || true)
    if [[ -z "$conn_events" ]]; then
        echo "│  (no connection events found in log yet)                     │"
    else
        echo "$conn_events" | while IFS= read -r line; do
            local flag=""
            if   echo "$line" | grep -q "DISCONNECTED"; then flag="[DISCONNECTED]"
            elif echo "$line" | grep -q "DEGRADED";     then flag="[DEGRADED]"
            elif echo "$line" | grep -q "RECONNECT";    then flag="[RECONNECT]"
            elif echo "$line" | grep -q "CONNECTED";     then flag="[CONNECTED]"
            fi
            printf "│  %s\n" "$line"
        done
    fi
    echo "└──────────────────────────────────────────────────────────────┘"
    echo ""

    # ---- 2c: Risk Boundary Tracker ----
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  2c. RISK BOUNDARY TRACKER                                   │"
    echo "├──────────────────────────────────────────────────────────────┤"
    local risk_events
    risk_events=$(grep -E "(MAX_DD|SHARPE_HALT|KILL|HALT|CRITICAL|FLATTEN)" \
        "$LOG_FILE" 2>/dev/null | tail -10 || true)
    if [[ -z "$risk_events" ]]; then
        echo "│  (no risk events — all within normal bounds)               │"
    else
        echo "$risk_events" | while IFS= read -r line; do
            local severity="INFO  "
            if   echo "$line" | grep -qE "MAX_DD_KILL|CRITICAL|FLATTEN_ALL"; then severity="CRIT  "
            elif echo "$line" | grep -qE "SHARPE_HALT|HALT";                 then severity="WARN  "
            fi
            printf "│  [%s] %s\n" "$severity" "$line"
        done
    fi
    echo "└──────────────────────────────────────────────────────────────┘"
    echo ""

    # ---- 2d: Passive Trade Volume ----
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  2d. PASSIVE TRADE VOLUME                                    │"
    echo "├──────────────────────────────────────────────────────────────┤"
    if [[ ! -f "$PASSIVE_TRADES" ]]; then
        echo "│  (passive_trades.json not created yet — runner init pending) │"
    else
        local lines
        lines=$(wc -l < "$PASSIVE_TRADES" 2>/dev/null || echo "0")
        echo "│  File  : $PASSIVE_TRADES"
        echo "│  Lines : $lines"
        # Last trade summary
        local last_ts last_ticker
        last_ts=$(grep -oP '"timestamp":\s*"\K[^"]+' "$PASSIVE_TRADES" 2>/dev/null | tail -1 || true)
        last_ticker=$(grep -oP '"ticker":\s*"\K[^"]+' "$PASSIVE_TRADES" 2>/dev/null | tail -1 || true)
        if [[ -n "$last_ts" ]]; then
            echo "│  Last  : $last_ticker @ $last_ts"
        fi
    fi
    echo "└──────────────────────────────────────────────────────────────┘"
    echo ""

    # ---- Summary Banner ----
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  OVERALL STATUS: RUNNING                                       │"
    echo "│  Next: ./manage_silent_run.sh --audit  (at EOD)              │"
    echo "└──────────────────────────────────────────────────────────────┘"

    return 0
}

# =============================================================================
# PHASE 3 — END-OF-DAY TELEMETRY AUDIT
# =============================================================================

do_audit() {
    log "──────────────────────────────────────────────────────────────"
    log "PHASE 3  |  End-of-Day Telemetry Audit"
    log "──────────────────────────────────────────────────────────────"

    # Guard: runner must be running
    if ! is_running; then
        local pid
        pid=$(get_pid)
        if [[ -z "$pid" ]]; then
            err "Runner not running — nothing to audit."
        else
            err "Runner process $pid is dead."
        fi
        err "Start with: $SCRIPT_DIR/manage_silent_run.sh launch"
        return 3
    fi

    # Guard: telemetry file must exist
    if [[ ! -f "$DELTA_TELEMETRY" ]]; then
        err "delta_telemetry.csv not found at: $DELTA_TELEMETRY"
        err "Audit requires at least one log interval (300s) of telemetry."
        return 2
    fi

    local records
    records=$(awk 'END {print NR-1}' "$DELTA_TELEMETRY" 2>/dev/null || echo "0")
    log "  telemetry file   : $DELTA_TELEMETRY ($records snapshots)"

    # Run audit script and capture output + exit code
    local audit_out audit_exit
    audit_out=$(python "$AUDIT_SCRIPT" \
        --input "$DELTA_TELEMETRY" \
        --window 20 2>&1)
    audit_exit=$?

    # Pretty-print audit output
    echo ""
    echo "$audit_out"
    echo ""

    local pid
    pid=$(get_pid)

    case $audit_exit in
        0)
            echo "┌──────────────────────────────────────────────────────────────┐"
            echo "│  OVERALL STATUS: PASS  (exit 0)                              │"
            echo "│  Delta drift within bounds. Maintain current configurations. │"
            echo "└──────────────────────────────────────────────────────────────┘"
            return 0
            ;;
        1)
            echo "┌──────────────────────────────────────────────────────────────┐"
            echo "│  OVERALL STATUS: WARNING  (exit 1)                            │"
            echo "│  Elevated drift. Consider tightening rebalancing frequency   │"
            echo "│  after the 72-hour window completes.                         │"
            echo "└──────────────────────────────────────────────────────────────┘"
            return 1
            ;;
        2)
            echo "┌──────────────────────────────────────────────────────────────┐"
            echo "│  OVERALL STATUS: CRITICAL FAILURE  (exit 2)                  │"
            echo "│  Delta neutral parameters violated >${MAX_CRIT_BREACHES} times.          │"
            echo "│  Automated response: killing runner PID $pid to prevent     │"
            echo "│  further telemetry corruption.                              │"
            echo "│                                                              │"
            echo "│  REQUIRED ACTIONS BEFORE RESTART:                            │"
            echo "│  1. Remove the stale PID file: rm $PID_FILE                  │"
            echo "│  2. In config/paper_trading.yaml set:                        │"
            echo "│       delta_telemetry.log_interval_seconds: 30              │"
            echo "│  3. Then restart: ./manage_silent_run.sh restart            │"
            echo "│  4. The 72-hour clock restarts from new launch time.         │"
            echo "└──────────────────────────────────────────────────────────────┘"

            # Kill the background runner
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                log "Terminating runner PID $pid..."
                kill "$pid" 2>/dev/null || true
                sleep 2
                # Double-kill if still alive
                if kill -0 "$pid" 2>/dev/null; then
                    kill -9 "$pid" 2>/dev/null || true
                fi
                log "Runner PID $pid terminated."
            fi
            rm -f "$PID_FILE"
            return 2
            ;;
        *)
            err "Unexpected audit exit code: $audit_exit"
            return 2
            ;;
    esac
}

# =============================================================================
# STOP / RESTART
# =============================================================================

do_stop() {
    local pid
    pid=$(get_pid)
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        warn "Runner not running. Nothing to stop."
        rm -f "$PID_FILE"
        return 0
    fi

    log "Sending SIGTERM to runner PID $pid..."
    kill "$pid" 2>/dev/null || true
    sleep 3

    if kill -0 "$pid" 2>/dev/null; then
        log "Runner did not terminate cleanly — sending SIGKILL..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    rm -f "$PID_FILE"
    log "Runner stopped. PID file removed."
}

do_restart() {
    log "Restarting silent run..."
    do_stop
    sleep 2
    do_launch
}

# =============================================================================
# USAGE
# =============================================================================

usage() {
    cat <<'EOF'
Usage: ./manage_silent_run.sh <command>

Commands:
  launch         Run pre-flight validation and start the silent run
                 in the background. Writes PID to var/run/silent_run.pid.
  --status       Print intraday health-check dashboard:
                   2a. Process status + RSS memory
                   2b. Connection heartbeat (last 5 events)
                   2c. Risk boundary tracker
                   2d. Passive trade volume
  --audit        Run end-of-day telemetry audit.
                   Exit 0: PASS    — maintain settings
                   Exit 1: WARNING — review rebalance configs
                   Exit 2: CRITICAL — kill runner + alert
  stop           Gracefully stop the background runner.
  restart        Stop + launch (full cycle restart).
  validate       Config-only check, no launch.

Pre-flight enforces two hard gates:
  1. passive_tracking_mode = true   (abort if false — no live broker)
  2. scale_mode           = CAPITAL_ALLOCATED
       (abort if single_share — insufficient P&L/risk for validation)

Exit codes:
  0  Success / PASS
  1  Warning (elevated drift)
  2  Critical failure / config abort / audit failure
  3  Runner not running
EOF
}

# =============================================================================
# MAIN DISPATCH
# =============================================================================

main() {
    ensure_dirs

    local cmd="${1:-}"

    case "$cmd" in
        launch)     do_launch ;;
        --status)   do_status ;;
        --audit)    do_audit  ;;
        stop)       do_stop   ;;
        restart)    do_restart ;;
        validate)   validate_config ;;
        help|-h|--help) usage; exit 0 ;;
        *)          usage; exit 2 ;;
    esac
}

main "$@"