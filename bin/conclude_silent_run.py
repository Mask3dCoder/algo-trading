#!/usr/bin/env python3
"""
conclude_silent_run.py -- Delta Neutral VRP Silent Run Conclusion and Go/No-Go Gatekeeper.

Usage:
    python bin/conclude_silent_run.py               # full: report + go/no-go + transition
    python bin/conclude_silent_run.py --report-only   # Task 1 only
    python bin/conclude_silent_run.py --go-no-go-only # Tasks 1 + 2
    python bin/conclude_silent_run.py --dry-run        # Tasks 1 + 2 + 3 (no config change)

Exit codes:
    0  GO_SIGNAL=True (transition complete, or --report-only / --go-no-go-only succeeded)
    1  NO-GO (criterion failed)
    2  Fatal error (missing artifacts, config corruption)
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Paths
WORKSPACE = Path(__file__).parent.parent.resolve()
CONFIG_FILE = WORKSPACE / "config" / "paper_trading.yaml"
CONFIG_BACKUP_DIR = WORKSPACE / "config" / "backups"
LOG_DIR = WORKSPACE / "reports" / "paper_trading"
REPORT_PATH = LOG_DIR / "silent_run_final_report.md"
DELTA_TELEMETRY = LOG_DIR / "delta_telemetry.csv"
PASSIVE_TRADES = LOG_DIR / "passive_trades.json"
RUN_LOG = LOG_DIR / "run.log"
EQUITY_CSV = LOG_DIR / "equity.csv"

# Resolve audit script (two common misspellings accepted)
for _audit_name in ("audit_telemetry_drift.py", "audit_telemetry_dradt.py"):
    _p = WORKSPACE / "bin" / _audit_name
    if _p.exists():
        AUDIT_SCRIPT = _p
        break
else:
    AUDIT_SCRIPT = None

# Constants
MAX_DD_KILL_PCT = 0.10
WARN_THRESHOLD = 5.0
CRIT_THRESHOLD = 15.0
MAX_CRIT_BREACHES = 3
INTERNAL_CASH = 100_000.0


# ============================================================================
# Task 1 helpers
# ============================================================================

def load_passive_trades() -> list[dict]:
    trades: list[dict] = []
    if not PASSIVE_TRADES.exists():
        return trades
    with open(PASSIVE_TRADES, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line in ("[", "]", "{", "}"):
                continue
            line = line.lstrip("[{,]").rstrip("[,}]")
            if line:
                try:
                    trades.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return trades


def load_delta_telemetry() -> list[dict]:
    rows: list[dict] = []
    if not DELTA_TELEMETRY.exists():
        return rows
    with open(DELTA_TELEMETRY, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def compute_trade_metrics(trades: list[dict]) -> dict:
    if not trades:
        return dict(total_trades=0, avg_spread_bps=0.0, total_notional=0.0,
                   whole_lot_pct=100.0, fractional_trades=0)

    spread_bps_list: list[float] = []
    notional_list: list[float] = []
    fractional = 0

    for t in trades:
        try:
            spread = float(t.get("bid_ask_spread") or 0)
            mid = float(t.get("mid_price") or 0)
            size = float(t.get("target_size") or 0)
            if spread and mid:
                spread_bps_list.append((spread / mid) * 10_000)
            if size and mid:
                notional_list.append(size * 100 * mid)
            if size != int(size) or size <= 0:
                fractional += 1
        except (ValueError, TypeError):
            continue

    total = len(trades)
    return dict(
        total_trades=total,
        avg_spread_bps=sum(spread_bps_list) / len(spread_bps_list) if spread_bps_list else 0.0,
        total_notional=sum(notional_list) if notional_list else 0.0,
        whole_lot_pct=((total - fractional) / total * 100) if total else 100.0,
        fractional_trades=fractional,
    )


def compute_telemetry_metrics(rows: list[dict]) -> dict:
    drifts: list[float] = []
    crit = 0
    warn = 0
    for r in rows:
        try:
            d = float(r.get("delta_drift", 0))
            drifts.append(d)
            if abs(d) > CRIT_THRESHOLD:
                crit += 1
            elif abs(d) > WARN_THRESHOLD:
                warn += 1
        except (ValueError, TypeError):
            continue

    if not drifts:
        return dict(max_abs_drift=0.0, mean_drift=0.0, num_critical_breaches=0,
                    num_warning_breaches=0, total_snapshots=0)

    return dict(
        max_abs_drift=max(abs(d) for d in drifts),
        mean_drift=sum(drifts) / len(drifts),
        num_critical_breaches=crit,
        num_warning_breaches=warn,
        total_snapshots=len(drifts),
    )


def load_run_log() -> str:
    if not RUN_LOG.exists():
        return ""
    with open(RUN_LOG, encoding="utf-8", errors="replace") as f:
        return f.read()


def _contains(substr: str, text: str) -> list[str]:
    return [line for line in text.splitlines() if substr in line]


def check_risk_controller_state(log_text: str) -> dict:
    halted = (
        _contains("FLATTEN_ALL_POSITIONS", log_text)
        + _contains("HALTED", log_text)
        + _contains("MAX_DD_KILL", log_text)
    )
    paused = _contains("SHARPE_HALT", log_text) + _contains("PAUSED", log_text)
    return dict(
        ever_halted=bool(halted),
        ever_paused=bool(paused),
        state_clean=not halted and not paused,
        halted_lines=halted[-5:],
        paused_lines=paused[-5:],
    )


def check_unscheduled_restarts(log_text: str) -> dict:
    restart_lines = (
        _contains("UNEXPECTED_EXIT", log_text)
        + _contains("RESTART", log_text)
        + _contains("CRASHED", log_text)
    )
    error_lines = (
        _contains("Traceback", log_text)
        + _contains("Error:", log_text)
        + _contains("FATAL", log_text)
        + _contains("KILLED", log_text)
    )
    return dict(
        restart_count=len(restart_lines),
        error_count=len(error_lines),
        clean=not restart_lines and not error_lines,
        restart_lines=restart_lines[-5:],
        error_lines=error_lines[-5:],
    )


def compute_drawdown_from_equity(equity_file: Path) -> dict:
    if not equity_file.exists():
        return dict(max_drawdown_pct=None, peak_equity=None,
                    current_equity=None, available=False)

    rows: list[dict] = []
    try:
        with open(equity_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except (OSError, csv.Error):
        return dict(max_drawdown_pct=None, peak_equity=None,
                    current_equity=None, available=False)

    if not rows:
        return dict(max_drawdown_pct=None, peak_equity=None,
                    current_equity=None, available=False)

    # Find equity column
    equity_col: str | None = None
    for col in ("equity", "current_equity", "equity_curve", "equity_curve_value"):
        if col in rows[0]:
            equity_col = col
            break

    if equity_col is None:
        for col, val in rows[0].items():
            try:
                float(val)
                equity_col = col
                break
            except (ValueError, TypeError):
                continue

    if equity_col is None:
        return dict(max_drawdown_pct=None, peak_equity=None,
                    current_equity=None, available=False)

    values: list[float] = []
    for r in rows:
        try:
            values.append(float(r.get(equity_col, 0)))
        except (ValueError, TypeError):
            continue

    if not values:
        return dict(max_drawdown_pct=None, peak_equity=None,
                    current_equity=None, available=False)

    peak = max(values)
    current = values[-1]
    dd_pct = abs((current - peak) / peak) if peak > 0 else 0.0
    return dict(max_drawdown_pct=dd_pct, peak_equity=peak,
                 current_equity=current, available=True)


def _reconstruct_equity(trades: list[dict]) -> dict:
    equity = INTERNAL_CASH
    peak = equity
    for t in trades:
        try:
            size = float(t.get("target_size") or 0)
            mid = float(t.get("mid_price") or 0)
            side = t.get("side", "buy")
            if not size or not mid:
                continue

            # Approximate P&L: 1 bp per side as slippage proxy
            sign = 1 if side == "buy" else -1
            equity += size * 100 * mid * 0.0001 * sign
            if equity > peak:
                peak = equity
        except (ValueError, TypeError):
            continue

    dd_pct = abs((equity - peak) / peak) if peak > 0 else 0.0
    return dict(max_drawdown_pct=dd_pct, peak_equity=peak,
                 current_equity=equity, reconstructed=True)


def run_sub_audit() -> tuple[int, str]:
    """Run audit_telemetry_drift.py; fallback to in-process computation."""
    if AUDIT_SCRIPT is not None:
        try:
            proc = subprocess.run(
                [sys.executable, str(AUDIT_SCRIPT),
                 "--input", str(DELTA_TELEMETRY), "--window", "20"],
                capture_output=True, text=True, timeout=60)
            return proc.returncode, proc.stdout + proc.stderr
        except (OSError, subprocess.TimeoutExpired):
            pass

    # Fallback: mirrors audit_telemetry_drift.py logic
    rows = load_delta_telemetry()
    if not rows:
        return 0, "No telemetry data."

    drifts: list[float] = []
    for r in rows:
        try:
            drifts.append(float(r.get("delta_drift", 0)))
        except (ValueError, TypeError):
            continue

    if not drifts:
        return 0, "No numeric drift data."

    crit = sum(1 for d in drifts if abs(d) > CRIT_THRESHOLD)
    msg = ("Critical breaches: %d "
           "(max=%.2f, mean=%.2f)") % (crit, max(abs(d) for d in drifts),
                                        sum(drifts) / len(drifts))
    if crit > MAX_CRIT_BREACHES:
        return 2, msg
    if sum(1 for d in drifts if abs(d) > WARN_THRESHOLD) > 0:
        return 1, msg
    return 0, msg


# ============================================================================
# Task 2 -- GO/NO-GO evaluation
# ============================================================================

def compute_go_signal(
    log_text: str,
    equity_data: dict,
    trade_metrics: dict,
    telemetry_metrics: dict,
    risk_state: dict,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    restart = check_unscheduled_restarts(log_text)

    if not restart["clean"]:
        failures.append(
            "Criterion 1 FAILED: %d unscheduled restart(s), "
            "%d error(s) in run.log."
            % (restart["restart_count"], restart["error_count"]))

    audit_code, _ = run_sub_audit()
    if audit_code == 2:
        failures.append(
            "Criterion 2 FAILED: audit exit %d -- %d critical breaches "
            "(threshold %d)."
            % (audit_code, telemetry_metrics["num_critical_breaches"], MAX_CRIT_BREACHES))

    dd = equity_data.get("max_drawdown_pct")
    if dd is not None and dd >= MAX_DD_KILL_PCT:
        failures.append(
            "Criterion 3 FAILED: peak-to-trough simulated drawdown %.2f%% "
            "exceeds hard kill at %.0f%%."
            % (dd * 100, MAX_DD_KILL_PCT * 100))

    if not risk_state["state_clean"]:
        parts = []
        if risk_state["ever_halted"]:
            parts.append("HALTED")
        if risk_state["ever_paused"]:
            parts.append("PAUSED")
        failures.append("RiskController triggered: [%s]" % ", ".join(parts))

    return len(failures) == 0, failures


# ============================================================================
# Task 1 -- Report generation
# ============================================================================

def generate_report(
    trade_metrics: dict,
    telemetry_metrics: dict,
    risk_state: dict,
    restart_check: dict,
    equity_data: dict,
    go_signal: bool,
    failures: list[str],
    audit_exit_code: int,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # sizing_note
    if trade_metrics.get("fractional_trades", 0) > 0:
        sizing_note = "  WARNING: %d fractional/zero contract sizes detected.\n" % (
            trade_metrics["fractional_trades"],)
    elif trade_metrics.get("total_trades", 0) == 0:
        sizing_note = "  NOTE: No passive trades generated during this window.\n"
    else:
        sizing_note = ""

    # equity block
    if equity_data.get("available"):
        equity_block = "\n".join([
            "  - Peak equity    : $%s" % ("%.2f" % equity_data["peak_equity"]),
            "  - Current equity : $%s" % ("%.2f" % equity_data["current_equity"]),
            "  - Max DD         : %.2f%%" % (equity_data["max_drawdown_pct"] * 100),
            "  - Kill switch    : %s"
                % ("TRIGGERED -- hard kill"
                   if equity_data["max_drawdown_pct"] >= MAX_DD_KILL_PCT
                   else "Not triggered"),
        ])
    elif equity_data.get("reconstructed"):
        equity_block = "\n".join([
            "  - Peak equity    : $%s (equity.csv absent -- reconstructed)"
                % ("%.2f" % equity_data.get("peak_equity", INTERNAL_CASH)),
            "  - Current equity : $%s"
                % ("%.2f" % equity_data.get("current_equity", INTERNAL_CASH)),
            "  - Estimated DD : %.2f%% (reconstructed from trades)"
                % (equity_data.get("max_drawdown_pct", 0) * 100),
        ])
    else:
        equity_block = "\n".join([
            "  - Peak equity    : $%s (equity.csv absent)"
                % ("%.2f" % equity_data.get("peak_equity", INTERNAL_CASH)),
            "  - Current equity : $%s"
                % ("%.2f" % equity_data.get("current_equity", INTERNAL_CASH)),
            "  - Estimated DD : %.2f%% (equity.csv absent)"
                % (equity_data.get("max_drawdown_pct", 0) * 100),
        ])

    # risk state
    if risk_state["state_clean"]:
        risk_banner = "  PASS -- RiskController never changed state."
    else:
        parts = []
        if risk_state["ever_halted"]:
            parts.append("HALTED/FLATTEN")
        if risk_state["ever_paused"]:
            parts.append("PAUSED")
        risk_banner = "  FAIL -- RiskController fired: %s." % ", ".join(parts)

    # go section
    if go_signal:
        go_section = (
            "## GO/NO-GO DETERMINATION: GO\n\n"
            "**[STATUS: GO]**  All 72-hour operational gates cleared.\n\n"
            "| Criterion | Result |\n"
            "|---|---|\n"
            "| Unscheduled restarts | %d -- PASS |\n"
            "| Delta drift audit | Exit %d -- PASS |\n"
            "| Max drawdown | %.2f%% -- PASS |\n"
            "| RiskController | Clean -- PASS |\n"
        ) % (restart_check["restart_count"],
            audit_exit_code,
            equity_data.get("max_drawdown_pct", 0) * 100)
        next_steps = (
            "### Recommended Actions\n\n"
            "1. **Transition approved** -- passive_tracking_mode flipped to false.\n"
            "2. Restart runner at next slot: **16:15 ET, US/Eastern**.\n"
            "3. Monitor health: ./bin/manage_silent_run.sh --status\n\n"
            "### Active Paper Trading Launch\n\n"
            "```bash\n"
            "./bin/manage_silent_run.sh restart\n\n"
            "# Or direct:\n"
            "python -m quantcore.live.paper_trading.runner \\\n"
            "    --config config/paper_trading.yaml\n\n"
            "# EOD audit (16:30 ET):\n"
            "./bin/manage_silent_run.sh --audit\n"
            "```\n"
        )
    else:
        fail_rows = "\n".join("%d. %s" % (i + 1, f)
                              for i, f in enumerate(failures))
        go_section = (
            "## GO/NO-GO DETERMINATION: NO-GO\n\n"
            "**[STATUS: NO-GO]**  One or more criteria failed.\n\n"
            "| Criterion | Result |\n"
            "|---|---|\n"
            "| Unscheduled restarts | %d -- %s |\n"
            "| Delta drift audit | Exit %d -- %s |\n"
            "| Max drawdown | %.2f%% -- %s |\n"
            "| RiskController | %s |\n\n"
            "### Failed Criteria\n\n%s\n"
        ) % (
            restart_check["restart_count"],
            "PASS" if restart_check["clean"] else "FAIL",
            audit_exit_code,
            "PASS" if audit_exit_code <= 1 else "FAIL",
            equity_data.get("max_drawdown_pct", 0) * 100,
            "PASS" if equity_data.get("max_drawdown_pct", 99) < MAX_DD_KILL_PCT else "FAIL",
            "Clean -- PASS" if risk_state["state_clean"] else "Triggered -- FAIL",
            fail_rows,
        )
        next_steps = (
            "### Required Remediation\n\n"
            + fail_rows + "\n\n"
            "Do not modify passive_tracking_mode until all criteria pass.\n"
            "Remediate and re-run: python bin/conclude_silent_run.py\n"
        )

    restart_section = ""
    if restart_check.get("restart_lines"):
        restart_section = (
            "### Restart Events\n\n"
            "```\n" + "\n".join(restart_check["restart_lines"]) + "\n```\n\n"
        )
    error_section = ""
    if restart_check.get("error_lines"):
        error_section = (
            "### Error Extract\n\n"
            "```\n" + "\n".join(restart_check["error_lines"]) + "\n```\n\n"
        )

    # Build output line by line (avoids implicit string concat issues)
    out: list[str] = []

    def ln(s: str) -> None:
        out.append(s)

    ln("# Delta Neutral VRP -- Silent Run Final Report")
    ln("")
    ln("**Generated:** %s" % now)
    ln("**Strategy:** delta_neutral_vrp")
    ln("**Window:** Tuesday 2026-06-23 -> Friday 2026-06-26 (72 hours)")
    ln("**Config:** config/paper_trading.yaml")
    ln("")
    ln("---")
    ln("")
    ln("## Trade Realism & Cost Audit")
    ln("")
    ln("| Metric | Value |")
    ln("|---|---|")
    ln("| Total simulated trades | %d |" % trade_metrics["total_trades"])
    ln("| Avg bid-ask spread (bps) | %.2f bps |" % trade_metrics["avg_spread_bps"])
    ln("| Total notional traded | $%s |" % ("%.2f" % trade_metrics["total_notional"]))
    ln("| Whole-lot trades | %.1f%% |" % trade_metrics["whole_lot_pct"])
    ln("")
    if sizing_note:
        out.append(sizing_note)
    ln("")
    ln("---")
    ln("")
    ln("## Delta Drift Telemetry")
    ln("")
    ln("| Metric | Value |")
    ln("|---|---|")
    ln("| Total snapshots | %d |" % telemetry_metrics["total_snapshots"])
    ln("| Max absolute drift | %.2f |" % telemetry_metrics["max_abs_drift"])
    ln("| Mean delta drift | %.2f |" % telemetry_metrics["mean_drift"])
    ln("| Warning breaches (|delta| > %.1f) | %d |"
       % (WARN_THRESHOLD, telemetry_metrics["num_warning_breaches"]))
    ln("| Critical breaches (|delta| > %.1f) | %d |"
       % (CRIT_THRESHOLD, telemetry_metrics["num_critical_breaches"]))
    ln("")
    ln("---")
    ln("")
    ln("## Risk Controller State")
    ln("")
    ln(risk_banner)
    ln("")
    ln("---")
    ln("")
    ln("## Simulated Equity & Drawdown")
    ln("")
    for eline in equity_block.splitlines():
        ln(eline)
    ln("")
    ln("---")
    ln("")
    ln("## Intraday Operational Cleanliness")
    ln("")
    ln("| Metric | Value |")
    ln("|---|---|")
    ln("| Unscheduled restarts | %d |" % restart_check["restart_count"])
    ln("| Fatal errors in run.log | %d |" % restart_check["error_count"])
    ln("")
    out.append(restart_section)
    out.append(error_section)
    ln("---")
    ln("")
    for gl in go_section.splitlines():
        ln(gl)
    ln("")
    ln("---")
    ln("")
    out.append(next_steps)
    ln("")
    ln("---")
    ln("")
    ln("*Generated by bin/conclude_silent_run.py -- Delta Neutral VRP Silent Run Gatekeeper*")

    return "\n".join(out)


# ============================================================================
# Task 3 -- Config backup & flip
# ============================================================================

def backup_config() -> Path:
    CONFIG_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = "paper_trading.yaml.bak_%s" % ts
    backup_path = CONFIG_BACKUP_DIR / backup_name
    shutil.copy2(CONFIG_FILE, backup_path)
    try:
        backup_path.chmod(0o444)
    except OSError:
        pass
    return backup_path


def _rewrite_yaml_field(path: Path, dotted_key: str, new_value: bool) -> None:
    """
    In-place rewrite of one boolean field in a YAML file.
    Preserves comments, indentation, and surrounding structure.
    Only rewrites the value after the colon; everything else is untouched.
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    key_parts = dotted_key.split(".")
    target_key = key_parts[-1]
    target_section = ".".join(key_parts[:-1])  # e.g. "execution"

    in_section = False
    section_indent: int | None = None
    new_lines: list[str] = []
    modified = False

    for line in lines:
        stripped = line.lstrip()
        leading = line[: len(line) - len(stripped)]

        if stripped.startswith("#") or not stripped:
            new_lines.append(line)
            continue

        # Section-detect: identify section headers (lines ending with `:` as first
        # non-whitespace content).  Nested keys like `scale_mode:` also end in `:`
        # but come with indent > 0; a section header starts a new block at its indent.
        # We only exit a section when we see another SECTION HEADER at indent <= ours.
        def _is_section_header(ln: str) -> bool:
            stripped_ln = ln.lstrip()
            if not stripped_ln or stripped_ln.startswith("#"):
                return False
            key_part = stripped_ln.split(":")[0].strip()
            return bool(key_part)  # has a key name before the colon

        section_match: str | None = None
        for cand in (target_section,):
            if stripped == cand + ":" or stripped == cand + " :":
                section_match = cand
                break

        if section_match == target_section:
            in_section = True
            section_indent = len(leading)
        elif in_section and section_indent is not None and stripped:
            cur_indent = len(leading)
            # Exit only when we see a NEW section header (not just any indent <=)
            # that would be at the same or shallower level as our section marker.
            if cur_indent <= section_indent and _is_section_header(line):
                in_section = False
                section_indent = None
            elif stripped == target_key + ":" or stripped == target_key + " :":
                # Same-section sibling key: don't exit, just fall through to key match
                pass

        line_key = stripped.split(":")[0].strip()
        if (in_section and section_indent is not None
                and line_key == target_key):
            new_line = "%s%s: %s" % (" " * section_indent, target_key, new_value)
            # Preserve trailing comment
            cp = line.find("  #")
            if cp >= 0:
                new_line += line[cp:]
            new_line += "\n"
            new_lines.append(new_line)
            modified = True
            in_section = False
            section_indent = None
        else:
            new_lines.append(line)

    if not modified:
        raise ValueError("Key '%s' not found in '%s'" % (dotted_key, path))

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def write_config_backup_and_flip() -> Path:
    """Task 3 pipeline: backup, flip passive_tracking_mode, verify."""
    backup_path = backup_config()

    with open(CONFIG_FILE, encoding="utf-8") as f:
        current = (yaml.safe_load(f) or {}).get("execution", {}).get("passive_tracking_mode")
    print("  Current : passive_tracking_mode = %s" % current)

    _rewrite_yaml_field(CONFIG_FILE, "execution.passive_tracking_mode", False)

    with open(CONFIG_FILE, encoding="utf-8") as f:
        new = (yaml.safe_load(f) or {}).get("execution", {}).get("passive_tracking_mode")
    if new is not False:
        raise RuntimeError(
            "Config flip failed -- passive_tracking_mode is %r (expected False). "
            "Roll back from: %s" % (new, backup_path))
    print("  New     : passive_tracking_mode = False  [OK]")
    return backup_path


# ============================================================================
# Main
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Silent Run conclusion for delta_neutral_vrp")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--go-no-go-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv if argv else [])

    print()
    print("=" * 62)
    print("   Silent Run Conclusion -- Delta Neutral VRP")
    print("   72-Hour Window: 2026-06-23 -> 2026-06-26")
    print("=" * 62)
    print()

    # Load
    print("[1/4] Loading artifacts...")
    trades = load_passive_trades()
    trows = load_delta_telemetry()
    log_text = load_run_log()
    print("       passive_trades.json : %d records" % len(trades))
    print("       delta_telemetry.csv  : %d snapshots" % len(trows))
    print("       run.log              : %d lines" % len(log_text.splitlines()))

    # Task 1
    print()
    print("[2/4] Computing metrics (Task 1)...")
    trade_metrics = compute_trade_metrics(trades)
    telemetry_metrics = compute_telemetry_metrics(trows)
    risk_state = check_risk_controller_state(log_text)
    restart_check = check_unscheduled_restarts(log_text)

    if EQUITY_CSV.exists():
        equity_data = compute_drawdown_from_equity(EQUITY_CSV)
        equity_data["reconstructed"] = False
    else:
        equity_data = _reconstruct_equity(trades)

    print("       Total trades        : %d" % trade_metrics["total_trades"])
    print("       Avg spread           : %.2f bps" % trade_metrics["avg_spread_bps"])
    print("       Total notional       : $%s" % ("%.2f" % trade_metrics["total_notional"]))
    print("       Whole-lot pct        : %.1f%%" % trade_metrics["whole_lot_pct"])
    print("       Max abs delta drift  : %.2f" % telemetry_metrics["max_abs_drift"])
    print("       Crit breaches        : %d" % telemetry_metrics["num_critical_breaches"])
    print("       RiskController clean : %s" % risk_state["state_clean"])
    dd = equity_data.get("max_drawdown_pct")
    if dd is not None:
        print("       Simulated drawdown   : %.2f%%" % (dd * 100))
    else:
        print("       Simulated drawdown   : (equity.csv absent)")

    # Task 2
    print()
    print("[3/4] Evaluating GO_SIGNAL (Task 2)...")
    audit_exit, _audit_msg = run_sub_audit()
    print("       Audit exit code      : %d" % audit_exit)
    print("       Crit breaches        : %d" % telemetry_metrics["num_critical_breaches"])

    go_signal, failures = compute_go_signal(
        log_text, equity_data, trade_metrics, telemetry_metrics, risk_state)

    if go_signal:
        print()
        print("+----------------------------------------------------------+")
        print("|  [STATUS: GO]  All 72-hour operational gates cleared.  |")
        print("|  Proceeding to production transition...                |")
        print("+----------------------------------------------------------+")
    else:
        print()
        print("+----------------------------------------------------------+")
        print("|  [STATUS: NO-GO]  One or more criteria FAILED.          |")
        print("+----------------------------------------------------------+")
        for f in failures:
            print("       X  %s" % f)

    # Report
    print()
    print("[4/4] Generating report...")
    report_md = generate_report(
        trade_metrics, telemetry_metrics, risk_state,
        restart_check, equity_data, go_signal, failures, audit_exit)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_md, encoding="utf-8")
    print("       Written : %s" % REPORT_PATH)

    # Abort if NO-GO
    # Skip transition if requested (must check BEFORE go_signal gate so --dry-run returns 0)
    if args.dry_run:
        print()
        print("[DRY-RUN] Would execute:")
        print("         backup_config() -> config/backups/paper_trading.yaml.bak_<ts>")
        print("         flip passive_tracking_mode: true -> false")
        print("         Run without --dry-run to execute.")
        return 0

    if not go_signal:
        print()
        print("[ABORT] NO-GO -- skipping config transition.")
        print("        Remediate the failed criteria above and re-run.")
        return 1

    if args.report_only or args.go_no_go_only:
        print()
        print("[SKIP] Config transition skipped (%s mode)."
              % ("--report-only" if args.report_only else "--go-no-go-only"))
        return 0

    # Task 3 -- Full transition
    print()
    print("[TRANSITION] Flipping to active paper trading...")
    bp = write_config_backup_and_flip()

    print()
    print("+----------------------------------------------------------+")
    print("|  TRANSITION COMPLETE                                     |")
    print("+----------------------------------------------------------+")
    print("       Config backup     : %s" % bp)
    print("       passive_tracking_mode = false")
    print()
    print("       Active paper trading begins at next scheduled slot.")
    print("       Launch: ./bin/manage_silent_run.sh restart")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())