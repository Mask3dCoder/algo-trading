#!/usr/bin/env python
"""Automated daily monitoring for Delta Neutral VRP paper trading.

Run after each market close (or schedule via cron / Task Scheduler):

    python scripts/monitor_paper_trading.py

Reads:
    - reports/live_equity.csv          (volume-mounted from container)
    - reports/live_events.jsonl        (trade log)

Writes:
    - reports/monitoring_log.csv       (appended each run)
    - stdout one-line summary

Alerts:
    - Telegram (via ALPACA / TELEGRAM env vars in .env)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EQUITY_PATH = PROJECT_ROOT / "reports" / "live_equity.csv"
EVENTS_PATH = PROJECT_ROOT / "reports" / "live_events.jsonl"
MONITOR_LOG = PROJECT_ROOT / "reports" / "monitoring_log.csv"

# ---------------------------------------------------------------------------
# Backtest expectations (from reports/delta_neutral_realistic_backtest.json)
# ---------------------------------------------------------------------------
EXPECTED = {
    "sharpe": 1.30,
    "sharpe_ok_range": (0.5, 1.5),       # acceptable for paper
    "sharpe_warn_range": (0.3, 0.5),     # warning zone
    "sharpe_critical": 0.0,              # below zero = critical
    "max_dd_pct": 15.4,
    "max_dd_warn_pct": 20.0,
    "max_dd_critical_pct": 25.0,
    "daily_loss_critical_pct": 2.0,
    "win_rate_pct": 78.9,
    "profit_factor": 4.4,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_equity() -> pd.DataFrame:
    """Load the live equity curve CSV.

    Returns an empty DataFrame with the correct columns if the file is
    missing or empty — callers must handle ``len(df) == 0``.
    """
    cols = ["date", "cash", "positions_value", "total_equity", "ret"]
    if not EQUITY_PATH.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(EQUITY_PATH, parse_dates=["date"])
    if df.empty:
        return df
    # Deduplicate by date (container restarts can append duplicates)
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return df


def load_events() -> list[dict]:
    """Load trade events from the JSONL event log."""
    if not EVENTS_PATH.exists():
        return []
    events = []
    with open(EVENTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute performance metrics from the equity curve.

    Returns a dict with all metrics.  Values that require at least N data
    points return ``float('nan')`` when the requirement is not met.
    """
    nan = float("nan")
    if df.empty or len(df) < 1:
        return {
            "n_days": 0,
            "current_equity": 0.0,
            "daily_pnl": 0.0,
            "cumulative_return_pct": nan,
            "daily_returns": pd.Series(dtype=float),
            "rolling_sharpe_21d": nan,
            "max_drawdown_pct": nan,
            "current_drawdown_pct": nan,
            "profit_factor": nan,
            "daily_loss_pct": nan,
            "win_rate_daily_pct": nan,
        }

    equity = df["total_equity"].values.astype(float)
    n = len(equity)
    current_equity = float(equity[-1])

    # Daily returns
    if n >= 2:
        daily_ret = pd.Series(equity).pct_change().dropna()
    else:
        daily_ret = pd.Series(dtype=float)

    # Cumulative return
    initial = float(equity[0])
    cum_ret = (current_equity / initial - 1) * 100 if initial > 0 else nan

    # Daily P&L
    daily_pnl = float(equity[-1] - equity[-2]) if n >= 2 else 0.0
    daily_loss_pct = abs(daily_pnl / current_equity * 100) if current_equity > 0 else 0.0

    # Rolling 21-day Sharpe (annualised)
    rolling_sharpe = nan
    if len(daily_ret) >= 21:
        roll_mean = daily_ret.rolling(21).mean()
        roll_std = daily_ret.rolling(21).std()
        rolling_sharpe = float((roll_mean.iloc[-1] / roll_std.iloc[-1]) * np.sqrt(252))
    elif len(daily_ret) >= 2:
        # Use available data if less than 21 days
        rolling_sharpe = float(
            (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
        ) if daily_ret.std() > 0 else nan

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100  # percentage
    max_dd = float(np.min(dd))
    current_dd = float(dd[-1])

    # Profit factor (sum positive P&L / |sum negative P&L|)
    if n >= 2:
        pnl_changes = np.diff(equity)
        pos_sum = float(np.sum(pnl_changes[pnl_changes > 0]))
        neg_sum = float(np.abs(np.sum(pnl_changes[pnl_changes < 0])))
        pf = pos_sum / neg_sum if neg_sum > 0 else (
            float("inf") if pos_sum > 0 else nan
        )
    else:
        pf = nan

    # Win rate (from daily P&L)
    if n >= 2:
        wins = int(np.sum(pnl_changes > 0))
        total = len(pnl_changes)
        win_rate = wins / total * 100 if total > 0 else nan
    else:
        win_rate = nan

    return {
        "n_days": n,
        "current_equity": current_equity,
        "daily_pnl": daily_pnl,
        "cumulative_return_pct": cum_ret,
        "daily_returns": daily_ret,
        "rolling_sharpe_21d": rolling_sharpe,
        "max_drawdown_pct": max_dd,
        "current_drawdown_pct": current_dd,
        "profit_factor": pf,
        "daily_loss_pct": daily_loss_pct,
        "win_rate_daily_pct": win_rate,
    }


# ---------------------------------------------------------------------------
# Status evaluation
# ---------------------------------------------------------------------------

def evaluate_status(metrics: dict) -> tuple[str, list[str]]:
    """Compare live metrics to backtest expectations.

    Returns
    -------
    status : str
        "OK", "WARNING", or "CRITICAL".
    reasons : list[str]
        Human-readable explanations for any non-OK status.
    """
    reasons: list[str] = []
    worst = "OK"  # tracks the most severe status

    def escalate(level: str) -> None:
        nonlocal worst
        severity = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
        if severity.get(level, 0) > severity.get(worst, 0):
            worst = level

    # --- Sharpe ---
    sharpe = metrics["rolling_sharpe_21d"]
    if not np.isnan(sharpe):
        if sharpe < EXPECTED["sharpe_critical"]:
            escalate("CRITICAL")
            reasons.append(
                f"Sharpe {sharpe:.2f} < 0 (expected {EXPECTED['sharpe']:.2f})"
            )
        elif sharpe < EXPECTED["sharpe_warn_range"][0]:
            escalate("WARNING")
            reasons.append(
                f"Sharpe {sharpe:.2f} below warning threshold "
                f"({EXPECTED['sharpe_warn_range'][0]:.1f})"
            )

    # --- Max drawdown ---
    max_dd = metrics["max_drawdown_pct"]
    if not np.isnan(max_dd):
        if abs(max_dd) > EXPECTED["max_dd_critical_pct"]:
            escalate("CRITICAL")
            reasons.append(
                f"MaxDD {max_dd:.1f}% exceeds critical limit "
                f"({EXPECTED['max_dd_critical_pct']:.0f}%)"
            )
        elif abs(max_dd) > EXPECTED["max_dd_warn_pct"]:
            escalate("WARNING")
            reasons.append(
                f"MaxDD {max_dd:.1f}% exceeds warning limit "
                f"({EXPECTED['max_dd_warn_pct']:.0f}%)"
            )

    # --- Daily loss (single-day spike) ---
    daily_loss = metrics["daily_loss_pct"]
    if not np.isnan(daily_loss):
        if daily_loss > EXPECTED["daily_loss_critical_pct"]:
            escalate("CRITICAL")
            reasons.append(
                f"Daily loss {daily_loss:.2f}% exceeds "
                f"{EXPECTED['daily_loss_critical_pct']:.0f}% limit"
            )

    if not reasons:
        reasons.append("All metrics within expected ranges")

    return worst, reasons


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def send_telegram_alert(
    status: str,
    reasons: list[str],
    metrics: dict,
) -> None:
    """Send a Telegram alert using the same bot credentials as the live runner.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from environment.
    """
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    if not bot_token or not chat_id:
        return

    sharpe = metrics["rolling_sharpe_21d"]
    sharpe_str = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"
    pnl_sign = "+" if metrics["daily_pnl"] >= 0 else ""

    emoji = "⚠️" if status == "WARNING" else "🚨"
    message = (
        f"{emoji} *QuantCore Paper Trading — {status}*\n"
        f"\n"
        f"Strategy: Delta Neutral VRP\n"
        f"Equity: ${metrics['current_equity']:,.2f}\n"
        f"Daily P&L: {pnl_sign}${metrics['daily_pnl']:,.2f}\n"
        f"Drawdown: {metrics['current_drawdown_pct']:.2f}%\n"
        f"Rolling Sharpe (21d): {sharpe_str}\n"
        f"\n"
        f"Issues:\n"
    )
    for r in reasons:
        message += f"  • {r}\n"

    try:
        import urllib.request

        url = (
            f"https://api.telegram.org/bot{bot_token}/sendMessage"
        )
        payload = json.dumps({
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "Markdown",
        }).encode()
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status == 200:
                print(f"  Telegram alert sent ({status})")
    except Exception as exc:
        print(f"  Telegram alert failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def format_status_line(
    status: str,
    metrics: dict,
) -> str:
    """Format the one-line summary for stdout."""
    dt = datetime.now().strftime("%Y-%m-%d")
    eq = metrics["current_equity"]
    pnl = metrics["daily_pnl"]
    pnl_sign = "+" if pnl >= 0 else ""
    dd = metrics["current_drawdown_pct"]
    dd_str = f"{dd:.1f}%" if not np.isnan(dd) else "0.0%"
    sharpe = metrics["rolling_sharpe_21d"]
    sharpe_str = f"{sharpe:.2f}" if not np.isnan(sharpe) else "N/A"

    return (
        f"[{status}] {dt} | "
        f"Equity: ${eq:,.0f} | "
        f"Daily P&L: {pnl_sign}${pnl:,.0f} | "
        f"Drawdown: {dd_str} | "
        f"Sharpe: {sharpe_str}"
    )


def append_monitor_log(status: str, metrics: dict) -> None:
    """Append a row to the monitoring log CSV."""
    MONITOR_LOG.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "equity": round(metrics["current_equity"], 2),
        "daily_return_pct": round(metrics["daily_pnl"] / metrics["current_equity"] * 100, 4)
        if metrics["current_equity"] > 0 else 0.0,
        "cumulative_return_pct": round(metrics["cumulative_return_pct"], 4)
        if not np.isnan(metrics["cumulative_return_pct"]) else 0.0,
        "drawdown_pct": round(metrics["current_drawdown_pct"], 4)
        if not np.isnan(metrics["current_drawdown_pct"]) else 0.0,
        "max_drawdown_pct": round(metrics["max_drawdown_pct"], 4)
        if not np.isnan(metrics["max_drawdown_pct"]) else 0.0,
        "rolling_sharpe_21d": round(metrics["rolling_sharpe_21d"], 4)
        if not np.isnan(metrics["rolling_sharpe_21d"]) else None,
        "profit_factor": round(metrics["profit_factor"], 4)
        if not np.isnan(metrics["profit_factor"]) else None,
        "n_days": metrics["n_days"],
    }

    write_header = not MONITOR_LOG.exists()
    df = pd.DataFrame([row])
    df.to_csv(MONITOR_LOG, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    # Load data
    equity_df = load_equity()
    events = load_events()

    # Compute metrics
    metrics = compute_metrics(equity_df)

    # Evaluate status
    status, reasons = evaluate_status(metrics)

    # Print summary
    print(format_status_line(status, metrics))

    # Log to CSV
    append_monitor_log(status, metrics)

    # Alert if not OK
    if status != "OK":
        send_telegram_alert(status, reasons, metrics)
        # Also print detailed reasons
        for r in reasons:
            print(f"  → {r}")
    else:
        print(f"  All metrics OK (backtest target: Sharpe {EXPECTED['sharpe']:.2f}, "
              f"MaxDD {EXPECTED['max_dd_pct']:.1f}%)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
