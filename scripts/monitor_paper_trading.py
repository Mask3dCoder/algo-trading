#!/usr/bin/env python
"""Automated daily monitoring for QuantCore multi-strategy paper trading.

Run after each market close (or schedule via cron / Task Scheduler)::

    python scripts/monitor_paper_trading.py

Reads:
    - config/live.yaml                    strategy & broker config (default)
    - config/paper.yaml                   strategy & broker config (--paper)
    - reports/live_equity.csv             portfolio equity curve (live default)
    - reports/paper_equity.csv            portfolio equity curve (--paper)
    - reports/live_events.jsonl           trade / fill event log (live default)
    - reports/paper_events.jsonl          trade / fill event log (--paper)
    - reports/{strategy}_backtest.json    backtest targets (optional)

Writes:
    - reports/monitoring_log.csv          appended each live run (default)
    - reports/paper_monitoring_log.csv    appended each paper run (--paper)
    - reports/strategy_attribution.csv    attribution snapshot (live default)
    - reports/paper_attribution.csv       attribution snapshot (--paper)

Alerts:
    - Telegram (via TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID env vars)

Exit codes:
    0   healthy (or insufficient data — no action needed)
    1   warnings present
    2   critical alerts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "live.yaml"
PAPER_CONFIG_PATH = PROJECT_ROOT / "config" / "paper.yaml"
LIVE_REPORTS_DIR = PROJECT_ROOT / "reports"
PAPER_REPORTS_DIR = PROJECT_ROOT / "reports"
LIVE_FILL_WINDOW_DAYS = 5


@dataclass(frozen=True)
class MonitorPaths:
    """Resolved config and report paths for live or paper monitoring."""

    config_path: Path
    reports_dir: Path
    equity_path: Path
    events_path: Path
    monitor_log: Path
    attribution_output: Path


def resolve_paths(config_arg: str | None = None, paper: bool = False) -> MonitorPaths:
    """Resolve paths for the requested monitoring mode."""
    if config_arg:
        config_path = Path(config_arg)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
    elif paper:
        config_path = PAPER_CONFIG_PATH
    else:
        config_path = DEFAULT_CONFIG_PATH

    is_paper = paper or config_path.name == "paper.yaml"
    reports_dir = PAPER_REPORTS_DIR if is_paper else LIVE_REPORTS_DIR

    if is_paper:
        return MonitorPaths(
            config_path=config_path,
            reports_dir=reports_dir,
            equity_path=reports_dir / "paper_equity.csv",
            events_path=reports_dir / "paper_events.jsonl",
            monitor_log=reports_dir / "paper_monitoring_log.csv",
            attribution_output=reports_dir / "paper_attribution.csv",
        )

    return MonitorPaths(
        config_path=config_path,
        reports_dir=reports_dir,
        equity_path=reports_dir / "live_equity.csv",
        events_path=reports_dir / "live_events.jsonl",
        monitor_log=reports_dir / "monitoring_log.csv",
        attribution_output=reports_dir / "strategy_attribution.csv",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_pnl(pnl: float) -> str:
    """Format a P&L value as a signed dollar string, e.g. ``+$12.40``."""
    sign = "+" if pnl >= 0 else ""
    return f"{sign}${pnl:,.2f}"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: Path) -> list[dict]:
    """Load enabled strategies from a config file.

    Returns a list of strategy dicts (``name``, ``weight``, etc.).
    """
    if not path.exists():
        print(f"  WARNING: {path} not found — no strategy config loaded")
        return []

    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    strategies = cfg.get("strategies", {})
    # ``strategies`` can be a list or a dict in YAML; normalise to list
    if isinstance(strategies, dict):
        strategies = list(strategies.values())

    enabled = [s for s in strategies if s.get("enabled", False)]
    return enabled


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_equity(path: Path) -> pd.DataFrame:
    """Load an equity curve CSV and return a clean DataFrame.

    Returns an empty DataFrame with columns
    ``["date", "cash", "positions_value", "total_equity", "ret"]``
    (portfolio) or ``["date", "equity", "daily_pnl", "weight", "status"]``
    (per-strategy) when the file is missing or empty. Callers must handle
    ``len(df) == 0``.
    """
    if not path.exists():
        return pd.DataFrame(), 0

    df = pd.read_csv(path)
    if df.empty:
        return df, 0

    # Consistent date parsing — the CSV may have mixed formats
    # (e.g. "2026-05-29 00:00:00" and "2026-05-29" in the same column).
    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    # Drop rows where date parsing failed
    df = df.dropna(subset=["date"])

    # Deduplicate by calendar date (container restarts can append duplicate rows)
    df["_cal_date"] = df["date"].dt.date
    df = df.drop_duplicates(subset=["_cal_date"], keep="last").reset_index(drop=True)
    n_trading_days = df["_cal_date"].nunique()

    return df, n_trading_days


def load_events(path: Path) -> list[dict]:
    """Load trade events from the JSONL event log."""
    if not path.exists():
        return []
    events: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _event_timestamp(event: dict) -> Any:
    """Parse an event timestamp, returning NaT when unavailable or invalid."""
    raw_ts = event.get("timestamp")
    if raw_ts is None:
        return pd.NaT
    try:
        return pd.to_datetime(raw_ts, utc=True, errors="coerce")
    except (TypeError, ValueError):
        return pd.NaT


def check_live_strategy_fill_activity(
    strategies: list[dict],
    events: list[dict],
    window_days: int = LIVE_FILL_WINDOW_DAYS,
) -> list[str]:
    """Return alerts for LIVE strategies with zero fills in the rolling window."""
    if not strategies:
        return []

    cutoff = datetime.now(tz=UTC) - pd.Timedelta(days=window_days)
    fill_events = [
        event
        for event in events
        if event.get("event") in {"fill", "close"} and event.get("strategy")
    ]
    alerts: list[str] = []

    for strategy in strategies:
        name = str(strategy.get("name", "")).strip()
        if not name:
            continue
        recent_fills = [
            event
            for event in fill_events
            if event.get("strategy") == name and _event_timestamp(event) >= cutoff
        ]
        if not recent_fills:
            alerts.append(f"{name}: zero fills in rolling {window_days}-day window")

    return alerts


def _combine_status(current: str, candidate: str) -> str:
    """Return the more severe status."""
    severity = {"OK": 0, "INSUFFICIENT_DATA": 1, "WARNING": 2, "CRITICAL": 3}
    return candidate if severity.get(candidate, 0) > severity.get(current, 0) else current


def load_backtest_targets(strategy_name: str) -> dict | None:
    """Find and load backtest targets for *strategy_name*.

    Tries candidate filenames in order:

    1. ``reports/{strategy_name}_backtest.json``
    2. Known aliases (e.g. ``delta_neutral_realistic_backtest.json``)

    Returns a dict with ``sharpe``, ``max_dd_pct``, ``win_rate_pct``,
    ``profit_factor``, … or ``None`` if no file is found or the file is
    older than 30 days.
    """
    today = datetime.now(tz=UTC)

    # --- Candidate filenames ---
    candidates: list[str] = [f"{strategy_name}_backtest.json"]

    # Known legacy / alias names
    _ALIASES: dict[str, list[str]] = {
        "options_delta_neutral": ["delta_neutral_realistic_backtest.json"],
    }
    candidates.extend(_ALIASES.get(strategy_name, []))

    for stem in candidates:
        path = PROJECT_ROOT / "reports" / stem
        if not path.exists():
            continue

        # Check freshness
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        age_days = (today - mtime).days
        if age_days > 30:
            print(
                f"  WARNING: Backtest file {stem} is {age_days} days old "
                f"(>30 days) — skipping comparison"
            )
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

    # --- Flatten nested paths ---
    # The file may have a ``walk_forward`` or ``direct_oos`` section.
        bt: dict[str, float] = {}
        wf = data.get("walk_forward") or {}
        doos = data.get("direct_oos") or {}

        bt["sharpe"] = wf.get("oos_sharpe") or doos.get("oos_sharpe") or 0.0
        bt["max_dd_pct"] = abs(
            wf.get("max_drawdown_pct") or doos.get("max_drawdown_pct") or 0.0
        )
        bt["win_rate_pct"] = doos.get("win_rate_pct", 0.0)
        bt["profit_factor"] = doos.get("profit_factor", 0.0)
        bt["n_trades"] = doos.get("n_trades", 0)

        return bt

    return None


def compute_strategy_metrics(
    strategy_name: str,
    events: list[dict],
    equity_path: Path,
) -> dict:
    """Compute summary metrics for a single strategy from its event log.

    Returns ``n_orders``, ``n_fills``, ``last_pnl`` (from the per-strategy
    equity CSV, *not* from the events).
    """
    strat_events = [e for e in events if e.get("strategy") == strategy_name]
    n_orders = len([e for e in strat_events if e.get("event") == "submit"])
    n_fills = len([e for e in strat_events if e.get("event") in ("fill", "close")])

    # Read per-strategy equity CSV for latest P&L
    strat_equity_path = equity_path.parent / f"{equity_path.stem}_{strategy_name}.csv"
    last_pnl = 0.0
    last_equity = 0.0
    status = "unknown"
    if strat_equity_path.exists():
        try:
            sedf = pd.read_csv(strat_equity_path)
            if not sedf.empty:
                last_row = sedf.iloc[-1]
                last_pnl = float(last_row.get("daily_pnl", 0.0))
                last_equity = float(last_row.get("equity", 0.0))
                status = str(last_row.get("status", "unknown"))
        except Exception:
            pass

    return {
        "n_orders": n_orders,
        "n_fills": n_fills,
        "last_pnl": last_pnl,
        "last_equity": last_equity,
        "status": status,
    }


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_metrics(df: pd.DataFrame) -> dict:
    """Compute performance metrics from the portfolio equity curve.

    Returns a dict with all metrics.  Values that require at least N data
    points return ``float('nan')`` when the requirement is not met.
    """
    nan = float("nan")
    if df.empty or len(df) < 1:
        return {
            "n_days": 0,
            "n_unique_days": 0,
            "current_equity": 0.0,
            "daily_pnl": 0.0,
            "cumulative_return_pct": nan,
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
    daily_loss_pct = (
        abs(daily_pnl / current_equity * 100) if current_equity > 0 else 0.0
    )

    # Rolling 21-day Sharpe (annualised)
    rolling_sharpe = nan
    if len(daily_ret) >= 21:
        roll_mean = daily_ret.rolling(21).mean()
        roll_std = daily_ret.rolling(21).std()
        rolling_sharpe = float(
            (roll_mean.iloc[-1] / roll_std.iloc[-1]) * np.sqrt(252)
        )
    elif len(daily_ret) >= 2:
        # Use available data if less than 21 days
        rolling_sharpe = (
            float((daily_ret.mean() / daily_ret.std()) * np.sqrt(252))
            if daily_ret.std() > 0
            else nan
        )

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
        pf = (
            pos_sum / neg_sum
            if neg_sum > 0
            else (float("inf") if pos_sum > 0 else nan)
        )
    else:
        pf = nan

    # Win rate (from daily P&L)
    win_rate = nan
    if n >= 2:
        pnl_changes = np.diff(equity)
        wins = int(np.sum(pnl_changes > 0))
        total = len(pnl_changes)
        win_rate = wins / total * 100 if total > 0 else nan

    return {
        "n_days": n,
        "n_unique_days": df["_cal_date"].nunique(),
        "current_equity": current_equity,
        "daily_pnl": daily_pnl,
        "cumulative_return_pct": cum_ret,
        "rolling_sharpe_21d": rolling_sharpe,
        "max_drawdown_pct": max_dd,
        "current_drawdown_pct": current_dd,
        "profit_factor": pf,
        "daily_loss_pct": daily_loss_pct,
        "win_rate_daily_pct": win_rate,
    }


# ---------------------------------------------------------------------------
# Fill-event tracking
# ---------------------------------------------------------------------------

def check_fill_events(events: list[dict]) -> dict:
    """Check for recent fill events and return a summary dict.

    Returns
    -------
    total_fills : int
    recent_fills : int  (within the last 3 calendar days)
    warning : bool
    message : str
    """
    fills = [e for e in events if e.get("event") in ("fill", "close")]
    total = len(fills)

    if total == 0:
        return {
            "total_fills": 0,
            "recent_fills": 0,
            "warning": True,
            "message": "WARNING: No fill events detected at all. Orders may not be executing.",
        }

    # Count fills whose timestamp is within the last 3 calendar days
    now = datetime.now(tz=UTC)
    recent = [
        e
        for e in fills
        if "timestamp" in e
        and (now - pd.Timestamp(e["timestamp"]).to_pydatetime()).days <= 3
    ]
    n_recent = len(recent)

    if n_recent == 0:
        return {
            "total_fills": total,
            "recent_fills": 0,
            "warning": True,
            "message": "WARNING: No fill events detected in the last 3 trading days. Orders may not be executing.",
        }

    return {
        "total_fills": total,
        "recent_fills": n_recent,
        "warning": False,
        "message": f"{total} fill events total, {n_recent} in the last 3 days",
    }


# ---------------------------------------------------------------------------
# Status evaluation
# ---------------------------------------------------------------------------

def evaluate_status(metrics: dict, backtest: dict | None) -> tuple[str, list[str]]:
    """Compare live metrics to backtest expectations.

    Returns
    -------
    status : str
        ``"OK"``, ``"WARNING"``, or ``"CRITICAL"``.
    reasons : list[str]
        Human-readable explanations for any non-OK status.
    """
    reasons: list[str] = []
    worst = "OK"

    def escalate(level: str) -> None:
        nonlocal worst
        severity = {"OK": 0, "WARNING": 1, "CRITICAL": 2}
        if severity.get(level, 0) > severity.get(worst, 0):
            worst = level

    if backtest is None:
        return worst, ["No backtest targets loaded — skipping comparison"]

    # --- Sharpe ---
    sharpe = metrics["rolling_sharpe_21d"]
    expected_sharpe = backtest.get("sharpe", 1.0)
    if not np.isnan(sharpe):
        if sharpe < 0.0:
            escalate("CRITICAL")
            reasons.append(
                f"Sharpe {sharpe:.2f} < 0 (expected {expected_sharpe:.2f})"
            )
        elif sharpe < 0.3:
            escalate("WARNING")
            reasons.append(
                f"Sharpe {sharpe:.2f} below 0.3 warning threshold "
                f"(expected {expected_sharpe:.2f})"
            )

    # --- Max drawdown ---
    max_dd = metrics["max_drawdown_pct"]
    expected_dd = backtest.get("max_dd_pct", 15.0)
    if not np.isnan(max_dd):
        if abs(max_dd) > 25.0:
            escalate("CRITICAL")
            reasons.append(
                f"MaxDD {max_dd:.1f}% exceeds critical limit (25%)"
            )
        elif abs(max_dd) > 20.0:
            escalate("WARNING")
            reasons.append(
                f"MaxDD {max_dd:.1f}% exceeds warning limit (20%)"
            )

    # --- Daily loss ---
    daily_loss = metrics["daily_loss_pct"]
    if not np.isnan(daily_loss) and daily_loss > 2.0:
        escalate("CRITICAL")
        reasons.append(
            f"Daily loss {daily_loss:.2f}% exceeds 2% limit"
        )

    if not reasons and backtest is not None:
        reasons.append(
            f"All metrics OK (backtest target: Sharpe {expected_sharpe:.2f}, "
            f"MaxDD {expected_dd:.1f}%)"
        )

    return worst, reasons


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------

def send_telegram_alert(
    status: str,
    reasons: list[str],
    metrics: dict,
    fill_info: dict,
) -> None:
    """Send a Telegram alert via the same bot credentials as the live runner.

    Reads ``TELEGRAM_BOT_TOKEN`` and ``TELEGRAM_CHAT_ID`` from the environment.
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
        f"Equity: ${metrics['current_equity']:,.2f}\n"
        f"Daily P&L: {pnl_sign}${metrics['daily_pnl']:,.2f}\n"
        f"Drawdown: {metrics['current_drawdown_pct']:.2f}%\n"
        f"Rolling Sharpe (21d): {sharpe_str}\n"
        f"Fills (recent/total): {fill_info['recent_fills']}/{fill_info['total_fills']}\n"
        f"\n"
        f"Issues:\n"
    )
    for r in reasons:
        message += f"  • {r}\n"

    try:
        import urllib.request

        url = "https://api.telegram.org/bot" + bot_token + "/sendMessage"
        payload = json.dumps(
            {
                "chat_id": chat_id,
                "text": message,
                "parse_mode": "Markdown",
            }
        ).encode()
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

def format_status_line(status: str, metrics: dict) -> str:
    """Format the one-line portfolio summary for stdout."""
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


def format_strategy_line(name: str, m: dict, bt: dict | None) -> str:
    """Format a one-line per-strategy status."""
    pnl_str = _fmt_pnl(m["last_pnl"])
    eq_str = f"${m['last_equity']:,.2f}"

    parts = [
        f"{name}: eq={eq_str}",
        f"orders={m['n_orders']}",
        f"fills={m['n_fills']}",
        f"last_pnl={pnl_str}",
        f"status={m['status']}",
    ]

    if bt is not None:
        parts.append(f"bt_target_sharpe={bt.get('sharpe', '?'):.2f}")

    return "  " + " | ".join(parts)


def append_monitor_log(
    status: str,
    metrics: dict,
    fill_info: dict,
    strategy_metrics: list[dict],
    monitor_log: Path,
) -> None:
    """Append a row to the monitoring log CSV."""
    monitor_log.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "equity": round(metrics["current_equity"], 2),
        "daily_return_pct": round(
            metrics["daily_pnl"] / metrics["current_equity"] * 100, 4
        )
        if metrics["current_equity"] > 0
        else 0.0,
        "cumulative_return_pct": round(metrics["cumulative_return_pct"], 4)
        if not np.isnan(metrics["cumulative_return_pct"])
        else 0.0,
        "drawdown_pct": round(metrics["current_drawdown_pct"], 4)
        if not np.isnan(metrics["current_drawdown_pct"])
        else 0.0,
        "max_drawdown_pct": round(metrics["max_drawdown_pct"], 4)
        if not np.isnan(metrics["max_drawdown_pct"])
        else 0.0,
        "rolling_sharpe_21d": round(metrics["rolling_sharpe_21d"], 4)
        if not np.isnan(metrics["rolling_sharpe_21d"])
        else None,
        "profit_factor": round(metrics["profit_factor"], 4)
        if not np.isnan(metrics["profit_factor"])
        else None,
        "n_days": metrics["n_days"],
        "n_unique_days": metrics["n_unique_days"],
        "total_fills": fill_info["total_fills"],
        "recent_fills": fill_info["recent_fills"],
    }

    write_header = not monitor_log.exists()
    df_log = pd.DataFrame([row])
    df_log.to_csv(monitor_log, mode="a", header=write_header, index=False)


def write_attribution_snapshot(
    strategy_metrics: list[dict],
    metrics: dict,
    attribution_output: Path,
) -> None:
    """Write a same-day attribution snapshot for correlation monitoring."""
    attribution_output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "date",
        "strategy",
        "ret",
        "pnl",
        "allocation_weight",
        "status",
    ]
    rows: list[dict] = []
    if metrics.get("n_days", 0) > 0 and strategy_metrics:
        daily_return = metrics["daily_pnl"] / metrics["current_equity"] if metrics.get("current_equity", 0) else 0.0
        for item in strategy_metrics:
            weight = float(item.get("allocation_weight", 0.0) or 0.0)
            rows.append(
                {
                    "date": datetime.now().date().isoformat(),
                    "strategy": item["name"],
                    "ret": round(daily_return, 8),
                    "pnl": round(float(item.get("metrics", {}).get("last_pnl", 0.0)), 4),
                    "allocation_weight": round(weight, 8),
                    "status": item.get("metrics", {}).get("status", "unknown"),
                }
            )
    pd.DataFrame(rows, columns=columns).to_csv(attribution_output, index=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily monitoring for QuantCore paper/live trading")
    parser.add_argument("--paper", action="store_true", help="Use config/paper.yaml and reports/paper_* paths")
    parser.add_argument("--config", type=Path, help="Config path override; defaults to config/live.yaml")
    parser.add_argument("--events", type=Path, help="Event log JSONL path override")
    parser.add_argument("--equity", type=Path, help="Portfolio equity CSV path override")
    parser.add_argument("--monitor-log", type=Path, help="Monitoring log CSV path override")
    parser.add_argument(
        "--attribution-output",
        type=Path,
        help="Strategy attribution CSV path override",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    paths = resolve_paths(str(args.config) if args.config else None, paper=args.paper)
    if args.events:
        paths = MonitorPaths(**{**paths.__dict__, "events_path": args.events})
    if args.equity:
        paths = MonitorPaths(**{**paths.__dict__, "equity_path": args.equity})
    if args.monitor_log:
        paths = MonitorPaths(**{**paths.__dict__, "monitor_log": args.monitor_log})
    if args.attribution_output:
        paths = MonitorPaths(**{**paths.__dict__, "attribution_output": args.attribution_output})

    # ---- 1. Load config ----
    strategies = load_config(paths.config_path)
    if not strategies:
        print(f"WARNING: No enabled strategies found in {paths.config_path}")

    # ---- 2. Load portfolio equity ----
    equity_df, n_trading_days = load_equity(paths.equity_path)

    # ---- 3. Load events ----
    events = load_events(paths.events_path)

    # ---- 4. Compute portfolio metrics ----
    metrics = compute_metrics(equity_df)

    # ---- 5. Check fill events ----
    fill_info = check_fill_events(events)
    fill_alerts = [] if args.paper else check_live_strategy_fill_activity(strategies, events)

    # ---- 6. Per-strategy status ----
    strategy_metrics_list: list[dict] = []
    for s in strategies:
        name = s["name"]
        allocation_weight = float(s.get("allocation_weight", s.get("weight", 0.0)) or 0.0)
        m = compute_strategy_metrics(name, events, paths.equity_path)
        bt = load_backtest_targets(name)
        strategy_metrics_list.append({
            "name": name,
            "metrics": m,
            "backtest": bt,
            "allocation_weight": allocation_weight,
        })
        print(format_strategy_line(name, m, bt))

    print()  # blank line

    # ---- 7. Print portfolio summary ----
    if n_trading_days < 5:
        print(
            f"INSUFFICIENT DATA — need at least 5 trading days for meaningful "
            f"metrics. Current: {n_trading_days} day(s)."
        )
        print(f"  Equity: ${metrics['current_equity']:,.2f}")
        print(f"  Daily P&L: {_fmt_pnl(metrics['daily_pnl'])}")
        status = "INSUFFICIENT_DATA"
        reasons = []
    else:
        # Load portfolio-level backtest targets (use first strategy's targets)
        bt_portfolio = None
        if strategies:
            bt_portfolio = load_backtest_targets(strategies[0]["name"])
        status, reasons = evaluate_status(metrics, bt_portfolio)
        if fill_alerts:
            status = _combine_status(status, "WARNING")
            reasons.extend(fill_alerts)
        print(format_status_line(status, metrics))
        for r in reasons:
            print(f"  {r}")

    print()  # blank line

    # ---- 8. Fill-event summary ----
    print(f"  Fills: {fill_info['recent_fills']} recent / {fill_info['total_fills']} total")
    if fill_info["warning"]:
        print(f"  {fill_info['message']}")
    for alert in fill_alerts:
        print(f"  FILL ALERT: {alert}")

    # ---- 9. Log to CSV and attribution snapshot ----
    append_monitor_log(status, metrics, fill_info, strategy_metrics_list, paths.monitor_log)
    write_attribution_snapshot(strategy_metrics_list, metrics, paths.attribution_output)

    # ---- 10. Alert if not OK ----
    if status not in ("OK", "INSUFFICIENT_DATA"):
        send_telegram_alert(status, reasons, metrics, fill_info)

    # Determine exit code
    if status == "CRITICAL":
        return 2
    elif status == "WARNING":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
