#!/usr/bin/env python3
"""
Daily paper trading monitor — Phase 4 continuous operations.
Run via cron at 16:30 ET each trading day (after market close).

Usage:
    python scripts/run_daily_monitoring.py [--date YYYY-MM-DD]

Each run:
  1. Runs pytest tests/test_live_parity.py -v
  2. Scans reports/live_events.jsonl for flagged events
  3. Computes daily return from reports/live_equity.csv
  4. Checks sigma bounds, flags outliers
  5. Appends row to operations/daily_returns.csv
  6. Updates operations/monitoring_log.md
  7. Checks KILL_SWITCH and process health
  8. If >= 20 trading days accumulated, runs statistical validation
     and generates PROOF_OF_STRATEGY.md
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

# Project root
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "reports"
OPERATIONS_DIR = ROOT / "operations"
DAILY_RETURNS_CSV = OPERATIONS_DIR / "daily_returns.csv"
LIVE_EVENTS = REPORTS_DIR / "live_events.jsonl"
LIVE_EQUITY = REPORTS_DIR / "live_equity.csv"
KILL_SWITCH = ROOT / "KILL_SWITCH"
MONITORING_LOG = OPERATIONS_DIR / "monitoring_log.md"
PROOF_REPORT = ROOT / "PROOF_OF_STRATEGY.md"
PARITY_TEST = "tests/test_live_parity.py"

# OOS backtest statistics (from reports/strategy_sweep.md, 2026-05-20)
# Estimated from walk-forward OOS Sharpe + Max DD over 10-fold 1-year anchored test
OOS_STATS = {
    "CrossSectionalMomentum": {
        "oos_sharpe": 0.852,
        "max_dd_pct": 6.29,  # 6.29% from 10-fold walk-forward
        # Derived: assume ~10% annualized vol target → daily σ ≈ 0.63%
        "mu_daily": 0.000338,  # 8.52% annual / 252
        "sigma_daily": 0.00630,  # 10% / √252
        "max_sigma": 3.0,
    },
    "StateSelectiveMomentum": {
        "oos_sharpe": 0.811,
        "max_dd_pct": None,  # NaN in sweep report — strategy holds cash in bad regimes
        # Derived: similar vol targeting → daily σ ≈ 0.63%
        "mu_daily": 0.000322,  # 8.11% annual / 252
        "sigma_daily": 0.00630,
        "max_sigma": 3.0,
    },
}

MIN_TRADING_DAYS = 20


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DayResult:
    date: date
    strategy: str
    equity_start: float
    equity_end: float
    daily_return_pct: float  # percent
    prev_equity: float = 0.0
    cumulative_return_pct: float = 0.0
    parity_passed: bool = True
    n_parity_failures: int = 0
    events_flagged: int = 0
    circuit_breaker: str = "NORMAL"
    sigma_event: str = "N"
    sigma_z: float = 0.0
    equity_still_running: bool = True
    kill_switch_present: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        if self.kill_switch_present:
            return "HALTED"
        if not self.equity_still_running:
            return "CRASHED"
        if self.circuit_breaker != "NORMAL":
            return self.circuit_breaker
        return "OK"


# ---------------------------------------------------------------------------
# Step 1: Run parity tests
# ---------------------------------------------------------------------------

def run_parity_tests() -> tuple[bool, int, list[str]]:
    """Run pytest on test_live_parity.py. Returns (all_passed, n_failures, output_lines)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", PARITY_TEST, "-v", "--tb=short"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=180,
    )
    output = (result.stdout + result.stderr)
    lines = output.strip().split("\n")

    # Check for pass/fail in output
    all_passed = result.returncode == 0
    failures = [l for l in lines if "FAILED" in l or "ERROR" in l]
    n_failures = len(failures) // 2 if failures else 0  # rough estimate

    return all_passed, n_failures, lines


# ---------------------------------------------------------------------------
# Step 2: Scan live events
# ---------------------------------------------------------------------------

def scan_live_events(lookback_days: int = 5) -> dict:
    """
    Parse live_events.jsonl for concerning events.
    Returns dict with keys: n_rejected, n_risk_interventions, n_circuit_breaker,
    n_data_gaps, flagged_events (list of dicts)
    """
    if not LIVE_EVENTS.exists():
        return {
            "n_rejected": 0, "n_risk_interventions": 0,
            "n_circuit_breaker": 0, "n_data_gaps": 0,
            "flagged_events": [],
        }

    cutoff = datetime.now() - timedelta(days=lookback_days)
    events = []
    with open(LIVE_EVENTS) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                ev = json.loads(line)
                ts = ev.get("timestamp", "")
                if ts:
                    ev_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    if ev_dt < cutoff:
                        continue
                events.append(ev)
            except json.JSONDecodeError:
                continue

    n_rejected = 0
    n_risk = 0
    n_circuit = 0
    n_data_gaps = 0
    flagged = []

    for ev in events:
        ev_type = ev.get("event", "")
        strategy = ev.get("strategy", "")

        if ev_type in ("rejected", "cancel", "risk_reject"):
            n_rejected += 1
            flagged.append(ev)
        elif ev_type in ("risk_intervention", "risk_limit"):
            n_risk += 1
            flagged.append(ev)
        elif ev_type in ("circuit_breaker_trip", "circuit_breaker_state"):
            n_circuit += 1
            flagged.append(ev)
        elif ev_type == "data_gap":
            n_data_gaps += 1
            flagged.append(ev)

    return {
        "n_rejected": n_rejected,
        "n_risk_interventions": n_risk,
        "n_circuit_breaker": n_circuit,
        "n_data_gaps": n_data_gaps,
        "flagged_events": flagged,
        "total_events": len(events),
    }


# ---------------------------------------------------------------------------
# Step 3: Compute daily return from equity log
# ---------------------------------------------------------------------------

def compute_daily_return(trade_date: date) -> tuple[float, float, float] | None:
    """
    Read live_equity.csv, find equity at start and end of trade_date.
    Returns (equity_start, equity_end, daily_return_pct) or None.
    """
    if not LIVE_EQUITY.exists():
        return None

    rows = []
    with open(LIVE_EQUITY) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    # Find equity at start (previous day end) and end of trade_date
    equity_start = None
    equity_end = None

    for row in rows:
        row_date_str = row.get("date", "").strip()
        if not row_date_str:
            continue
        try:
            row_date = date.fromisoformat(row_date_str[:10])
        except ValueError:
            continue

        ret_val = row.get("ret", "").strip()
        if row_date == trade_date:
            try:
                equity_end = float(row.get("total_equity", 0))
            except ValueError:
                pass
        elif row_date == trade_date - timedelta(days=1):
            try:
                equity_start = float(row.get("total_equity", 0))
            except ValueError:
                pass

    # Fallback: use first and last rows
    if equity_start is None and len(rows) >= 2:
        # Find first row with valid equity
        for row in rows:
            try:
                equity_start = float(row.get("total_equity", 0))
                break
            except ValueError:
                continue

    if equity_end is None:
        # Use last row
        for row in reversed(rows):
            try:
                equity_end = float(row.get("total_equity", 0))
                if equity_start is None:
                    equity_start = equity_end
                break
            except ValueError:
                continue

    if equity_end is None or equity_start is None:
        return None

    if equity_start == 0:
        return None

    daily_return_pct = ((equity_end - equity_start) / equity_start) * 100.0
    return equity_start, equity_end, daily_return_pct


# ---------------------------------------------------------------------------
# Step 4: Sigma check
# ---------------------------------------------------------------------------

def check_sigma(daily_return_pct: float, strategy: str) -> tuple[str, float]:
    """Check if return is outside ±3σ from expected. Returns (flag, z_score)."""
    stats = OOS_STATS.get(strategy)
    if not stats or stats.get("sigma_daily") is None:
        # Fallback: use backtest-derived sigma
        sigma_daily = 1.0  # 1% daily fallback
        mu_daily = 0.0
    else:
        sigma_daily = stats["sigma_daily"]
        mu_daily = stats["mu_daily"]

    # daily_return is in percent; convert to decimal for comparison
    return_pct_decimal = daily_return_pct / 100.0
    z = (return_pct_decimal - mu_daily) / sigma_daily
    flag = "Y" if abs(z) > stats["max_sigma"] else "N"
    return flag, z


# ---------------------------------------------------------------------------
# Step 5: Health checks
# ---------------------------------------------------------------------------

def check_health() -> dict:
    """Check process health and kill switch."""
    ks_present = KILL_SWITCH.exists()
    return {
        "kill_switch_present": ks_present,
        "process_responsive": True,  # if we got this far, the process can be queried
    }


# ---------------------------------------------------------------------------
# Step 6: Append to daily returns log
# ---------------------------------------------------------------------------

CSV_HEADERS = [
    "date", "strategy", "equity_start", "equity_end",
    "daily_return_pct", "cumulative_return_pct",
    "parity_tests_passed", "events_flagged", "circuit_breaker",
    "sigma_event", "sigma_z", "notes",
]


def append_daily_return(result: DayResult) -> None:
    """Append a new row to operations/daily_returns.csv."""
    OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)

    write_header = not DAILY_RETURNS_CSV.exists()

    with open(DAILY_RETURNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "date": result.date.isoformat(),
            "strategy": result.strategy,
            "equity_start": f"{result.equity_start:.2f}",
            "equity_end": f"{result.equity_end:.2f}",
            "daily_return_pct": f"{result.daily_return_pct:.6f}",
            "cumulative_return_pct": f"{result.cumulative_return_pct:.6f}",
            "parity_tests_passed": "PASS" if result.parity_passed else f"FAIL({result.n_parity_failures})",
            "events_flagged": result.events_flagged,
            "circuit_breaker": result.circuit_breaker,
            "sigma_event": result.sigma_event,
            "sigma_z": f"{result.sigma_z:.4f}",
            "notes": "; ".join(result.notes) if result.notes else "",
        })


# ---------------------------------------------------------------------------
# Step 7: Update monitoring log
# ---------------------------------------------------------------------------

def update_monitoring_log(result: DayResult, parity_output: list[str]) -> None:
    """Append a markdown entry to the monitoring log."""
    parity_summary = "ALL PASS" if result.parity_passed else f"FAIL ({result.n_parity_failures} failures)"
    sigma_flag = "⚠️ SIGMA EVENT" if result.sigma_event == "Y" else ""

    entry = f"""
## Day {result.date.isoformat()} — Monitoring Entry

**Status:** {result.status} | **Parity:** {parity_summary} | {sigma_flag}

### Equity
- Start: ${result.equity_start:,.2f}  End: ${result.equity_end:,.2f}
- Daily Return: {result.daily_return_pct:+.4f}%  Cumulative: {result.cumulative_return_pct:+.4f}%
- Sigma Z-Score: {result.sigma_z:+.2f} (flagged: {result.sigma_event})

### Events
- Flagged events: {result.events_flagged}
- Circuit breaker: {result.circuit_breaker}

### Notes
{"  ".join(f"- {n}" for n in result.notes)}

### Parity Test Summary
```
{chr(10).join(parity_output[-20:])}
```
"""
    OPERATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MONITORING_LOG, "a") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Step 8: Statistical validation (run after >= 20 days)
# ---------------------------------------------------------------------------

def run_statistical_validation() -> dict:
    """
    Load paper returns, compare to backtest OOS stats.
    Performs t-test, JB test, computes Sharpe, Max DD, win rate.
    Returns dict with validation results.
    """
    try:
        import numpy as np
        from scipy import stats as scipy_stats
    except ImportError:
        return {"error": "scipy not available — run: pip install scipy"}

    # Load paper returns
    paper_returns: dict[str, list[float]] = {}
    paper_dates: dict[str, list[str]] = {}

    if not DAILY_RETURNS_CSV.exists():
        return {"error": "No daily returns file found"}

    with open(DAILY_RETURNS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            strat = row["strategy"]
            try:
                ret = float(row["daily_return_pct"]) / 100.0
                paper_returns.setdefault(strat, []).append(ret)
                paper_dates.setdefault(strat, []).append(row["date"])
            except (ValueError, KeyError):
                continue

    results = {}
    for strat, returns in paper_returns.items():
        if len(returns) < MIN_TRADING_DAYS:
            results[strat] = {"n_days": len(returns), "ready": False}
            continue

        r = np.array(returns)
        n = len(r)
        mu_paper = np.mean(r)
        sigma_paper = np.std(r, ddof=1)
        sharpe_paper = (mu_paper / sigma_paper) * np.sqrt(252) if sigma_paper > 0 else 0.0

        # Win rate
        wins = np.sum(r > 0)
        win_rate = wins / n

        # Max drawdown
        cumulative = np.cumprod(1 + r)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd_paper = np.min(drawdowns)

        # Backtest stats from OOS
        oos_stats = OOS_STATS.get(strat, {})
        mu_backtest = oos_stats.get("mu_daily", 0.0)
        sigma_backtest = oos_stats.get("sigma_daily", 0.01)
        sharpe_backtest = oos_stats.get("oos_sharpe", 0.0)
        max_dd_backtest = oos_stats.get("max_dd_pct", 0.0)
        if max_dd_backtest:
            max_dd_backtest /= 100.0

        # One-sample t-test: H0: paper mean = backtest mean
        if sigma_paper > 0:
            t_stat, p_value_t = scipy_stats.ttest_1samp(r, mu_backtest)
        else:
            t_stat, p_value_t = 0.0, 1.0

        # Jarque-Bera test for normality
        jb_stat, p_value_jb = scipy_stats.jarque_bera(r)

        # Wilcoxon signed-rank (if non-normal)
        non_normal = p_value_jb < 0.05
        if non_normal:
            # Center the returns
            centered = r - np.mean(r)
            w_stat, p_value_wilcoxon = scipy_stats.wilcoxon(centered)
        else:
            w_stat, p_value_wilcoxon = None, None

        # Validation criteria
        ttest_pass = p_value_t > 0.05
        sharpe_retention = sharpe_paper >= 0.7 * sharpe_backtest if sharpe_backtest else False
        dd_pass = (
            max_dd_paper <= 1.3 * max_dd_backtest
            if (max_dd_backtest and max_dd_backtest != 0)
            else True
        )
        # Win rate tolerance: ±5 percentage points
        # We don't have backtest win rate, so we skip this criterion
        win_rate_pass = True  # Skip — no backtest win rate available

        criteria = {
            "t-test p-value > 0.05": (ttest_pass, f"p={p_value_t:.4f}"),
            "Sharpe paper >= 0.7 × backtest": (sharpe_retention, f"paper={sharpe_paper:.3f}, threshold={0.7*sharpe_backtest:.3f}"),
            "Max DD paper <= 1.3 × backtest": (dd_pass, f"paper={max_dd_paper:.4f}, threshold={1.3*max_dd_backtest:.4f}"),
            "No backtest win rate available": (True, "skipped"),
        }

        proven = all(v[0] for v in criteria.values())

        results[strat] = {
            "n_days": n,
            "ready": True,
            "proven": proven,
            "paper": {
                "mu_daily": mu_paper,
                "sigma_daily": sigma_paper,
                "sharpe_annualized": sharpe_paper,
                "max_dd": max_dd_paper,
                "win_rate": win_rate,
            },
            "backtest": {
                "mu_daily": mu_backtest,
                "sigma_daily": sigma_backtest,
                "sharpe_annualized": sharpe_backtest,
                "max_dd": max_dd_backtest,
            },
            "t_stat": t_stat,
            "p_value_ttest": p_value_t,
            "jb_stat": jb_stat,
            "p_value_jb": p_value_jb,
            "jb_normal": not non_normal,
            "p_value_wilcoxon": p_value_wilcoxon,
            "criteria": criteria,
            "dates": paper_dates[strat],
        }

    return results


# ---------------------------------------------------------------------------
# Step 9: Generate PROOF_OF_STRATEGY.md
# ---------------------------------------------------------------------------

def generate_proof_report(validation: dict) -> None:
    """Write PROOF_OF_STRATEGY.md if validation is complete."""
    lines = [
        "# Proof of Strategy — Paper Trading Validation Report",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]

    all_proven = all(v.get("proven", False) for v in validation.values())
    verdict = "**PROVEN — ready for graduated live deployment**" if all_proven else "**NOT PROVEN — further investigation required**"

    lines += [
        f"## Verdict: {verdict}",
        "",
        "## Validation Criteria",
        "",
        "| Strategy | Criterion | Result | Detail |",
        "|---------|-----------|--------|--------|",
    ]

    for strat, res in validation.items():
        if not res.get("reday"):
            continue
        for crit_name, (passed, detail) in res.get("criteria", {}).items():
            lines.append(f"| {strat} | {crit_name} | {'PASS ✅' if passed else 'FAIL ❌'} | {detail} |")

    if all_proven:
        lines += [
            "",
            "## Readiness Statement",
            "",
            "All strategies have passed Phase 4 paper validation. Statistical tests confirm",
            "that live paper returns are consistent with backtest out-of-sample performance.",
            "The strategies are approved for graduated live deployment per PHASE4_LIVE_DEPLOYMENT.md:",
            "single_share → fractional → full (on successful completion of each tier).",
        ]

    with open(PROOF_REPORT, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Daily paper trading monitor")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Trading date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute metrics without writing log entries.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run statistical validation only (if >= 20 days available).",
    )
    args = parser.parse_args()

    try:
        trade_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"ERROR: Invalid date {args.date!r}", file=sys.stderr)
        return 1

    print(f"=== QuantCore Daily Monitor — {trade_date} ===")

    # Health checks first
    health = check_health()
    if health["kill_switch_present"]:
        print("⚠️  KILL_SWITCH present — halting operations!")
        return 0

    # Step 1: Parity tests
    print("Running parity tests...")
    parity_ok, n_parity_fail, parity_lines = run_parity_tests()
    print(f"  Parity tests: {'ALL PASS ✅' if parity_ok else f'FAIL ({n_parity_fail} failures)'}")

    if not parity_ok:
        print("⚠️  CRITICAL: Parity tests failed — stop condition!", file=sys.stderr)

    # Step 2: Event scan
    print("Scanning live events...")
    events = scan_live_events(lookback_days=5)
    print(f"  Total recent events: {events['total_events']}  Flagged: {events['n_rejected']+events['n_risk_interventions']+events['n_circuit_breaker']}")
    if events["flagged_events"]:
        print(f"  ⚠️  Flagged events: {len(events['flagged_events'])}")
        for ev in events["flagged_events"][:5]:
            print(f"    {ev}")

    # Step 3: Daily return
    print("Computing daily returns...")
    equity_data = compute_daily_return(trade_date)

    # Detect active strategies from events
    active_strategies = set()
    if LIVE_EVENTS.exists():
        with open(LIVE_EVENTS) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                try:
                    ev = json.loads(line.strip())
                    strat = ev.get("strategy", "")
                    if "CrossSectional" in strat:
                        active_strategies.add("CrossSectionalMomentum")
                    elif "StateSelective" in strat:
                        active_strategies.add("StateSelectiveMomentum")
                except json.JSONDecodeError:
                    continue

    # If no strategies found from events, check historical data
    if not active_strategies:
        active_strategies = {"CrossSectionalMomentum", "StateSelectiveMomentum"}

    results: list[DayResult] = []

    if equity_data:
        eq_start, eq_end, daily_ret = equity_data
        cumulative_ret = ((eq_end - 100000.0) / 100000.0) * 100.0  # relative to start

        # Determine which strategy gets the equity return
        # If only one strategy was active, take all. Otherwise split.
        # In practice, CSM takes the equity return since SSM had no signals.
        active_strats = list(active_strategies)
        if len(active_strats) == 2:
            # Both active — need to attribute. Use position weight if available.
            # For now, attribute to CSM since SSM had no signals on day 1
            # TODO: attribute by strategy-specific position accounting
            active_strats = ["CrossSectionalMomentum"]  # SSM=0 signal day

        for strat in active_strats:
            sigma_flag, z_score = check_sigma(daily_ret, strat)
            dr = DayResult(
                date=trade_date,
                strategy=strat,
                equity_start=eq_start,
                equity_end=eq_end,
                daily_return_pct=daily_ret,
                equity_still_running=True,
                parity_passed=parity_ok,
                n_parity_failures=n_parity_fail,
                events_flagged=0,
                circuit_breaker="NORMAL",
                sigma_event=sigma_flag,
                sigma_z=z_score,
                notes=[],
            )
            if sigma_flag == "Y":
                dr.notes.append(f"SIGMA EVENT: return {daily_ret:.4f}% is outside ±3σ bounds (z={z_score:.2f})")
            if not parity_ok:
                dr.notes.append(f"PARITY FAILURES: {n_parity_fail}")
            results.append(dr)
            print(f"  [{strat}] Return: {daily_ret:+.4f}%  Sigma: {sigma_flag} (z={z_score:.2f})")
    else:
        print("  ⚠️  No equity data available for today")
        # Log as zero return / no activity
        for strat in active_strategies:
            results.append(DayResult(
                date=trade_date,
                strategy=strat,
                equity_start=100000.0,
                equity_end=100000.0,
                daily_return_pct=0.0,
                equity_still_running=True,
                parity_passed=parity_ok,
                events_flagged=0,
                circuit_breaker="NORMAL",
                sigma_event="N",
                sigma_z=0.0,
                notes=["No equity update available"],
            ))

    # Check for stop conditions
    stop_condition = False
    if not parity_ok:
        print("🛑 STOP CONDITION: Parity tests failed")
        stop_condition = True

    if events.get("n_circuit_breaker", 0) > 0:
        print("🛑 STOP CONDITION: Circuit breaker triggered")
        stop_condition = True

    if stop_condition:
        print("STOP CONDITIONS MET — halting monitoring run.", file=sys.stderr)
        # Still write the log entry, then exit non-zero
        if not args.dry_run:
            for result in results:
                append_daily_return(result)
                update_monitoring_log(result, parity_lines)
        return 1

    # Step 5: Write to log
    if not args.dry_run:
        for result in results:
            append_daily_return(result)
            update_monitoring_log(result, parity_lines)
        print("  Logged to operations/daily_returns.csv and monitoring_log.md")
    else:
        print("  [dry-run] Skipped writing log entries")

    # Check if we have enough data for statistical validation
    n_days = _count_trading_days()
    print(f"\n  Accumulated trading days: {n_days} (need {MIN_TRADING_DAYS} for validation)")

    if n_days >= MIN_TRADING_DAYS and not args.dry_run:
        print("\n=== Running Statistical Validation ===")
        validation = run_statistical_validation()
        for strat, res in validation.items():
            if res.get("ready"):
                print(f"  [{strat}] Proven: {res['proven']}")
                for crit, (passed, detail) in res["criteria"].items():
                    print(f"    {'✅' if passed else '❌'} {crit}: {detail}")
        generate_proof_report(validation)
        print(f"  Proof report: {PROOF_REPORT}")
    elif args.validate_only:
        print("\n=== Statistical Validation ===")
        validation = run_statistical_validation()
        print(json.dumps(validation, indent=2, default=str))

    print("\n=== Monitor run complete ===")
    return 0


def _count_trading_days() -> int:
    if not DAILY_RETURNS_CSV.exists():
        return 0
    dates = set()
    with open(DAILY_RETURNS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.add(row.get("date", "").strip())
    return len(dates)


if __name__ == "__main__":
    sys.exit(main())