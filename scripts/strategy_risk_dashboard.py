#!/usr/bin/env python3
"""Console risk dashboard for Phase 4 live/paper strategy validation.

Reads the existing production/paper event and equity logs and prints:
  - approximate open positions by strategy
  - daily return and cumulative return
  - Sharpe and max drawdown per strategy
  - strategies breaching the 20% Phase 4 drawdown kill-switch
  - optional daily P&L attribution CSV export

The script intentionally uses only logged/current-day information. It does not
fetch market data, run event studies, or inspect future validation windows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_EVENTS = Path("reports/live_events.jsonl")
DEFAULT_EQUITY = Path("reports/live_equity.csv")
DEFAULT_PAPER_EVENTS = Path("reports/paper_events.jsonl")
DEFAULT_PAPER_EQUITY = Path("reports/paper_equity.csv")
DEFAULT_ATTRIBUTION = Path("reports/strategy_attribution.csv")
DEFAULT_PAPER_ATTRIBUTION = Path("reports/paper_attribution.csv")
DEFAULT_KILL_SWITCH_LOG = Path("reports/kill_switch_events.jsonl")
DEFAULT_PAPER_KILL_SWITCH_LOG = Path("reports/paper_kill_switch_events.jsonl")
DEFAULT_LIVE_ATTRIBUTION_ARCHIVE_DIR = Path("reports/live")
DEFAULT_PAPER_ATTRIBUTION_ARCHIVE_DIR = Path("reports/paper")
MAX_DRAWDOWN_PCT = -0.20


def load_events(path: str | Path) -> pd.DataFrame:
    events_path = Path(path)
    if not events_path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with events_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "timestamp" in df:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "trade_date" in df:
        df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
    return df


def load_equity(path: str | Path) -> pd.DataFrame:
    equity_path = Path(path)
    if not equity_path.exists():
        return pd.DataFrame(columns=["date", "total_equity", "ret"])
    df = pd.read_csv(equity_path)
    date_col = "date" if "date" in df.columns else "timestamp"
    if date_col not in df.columns:
        return pd.DataFrame(columns=["date", "total_equity", "ret"])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).dropna(subset=[date_col])
    df["date"] = df[date_col].dt.date
    if "total_equity" not in df.columns:
        equity_col = "portfolio_value" if "portfolio_value" in df.columns else "equity"
        if equity_col in df.columns:
            df["total_equity"] = df[equity_col].astype(float)
    if "total_equity" not in df.columns or df["total_equity"].empty:
        return pd.DataFrame(columns=["date", "total_equity", "ret"])
    df["total_equity"] = df["total_equity"].astype(float)
    if "ret" not in df.columns:
        df["ret"] = df["total_equity"].pct_change().fillna(0.0)
    else:
        df["ret"] = df["ret"].astype(float).fillna(0.0)
    return df[["date", "total_equity", "ret"]]


def _side_sign(side: str) -> float:
    return 1.0 if str(side).lower() in {"buy", "SELL_TO_OPEN"} else -1.0


def infer_strategy_weights(events: pd.DataFrame) -> dict[str, float]:
    if events.empty or "strategy" not in events.columns:
        return {}
    counts = events.groupby("strategy")["notional"].apply(lambda s: float(s.abs().sum()))
    total = float(counts.sum())
    if total <= 0:
        return {str(k): 1.0 / len(counts) for k in counts.index}
    return {str(k): float(v / total) for k, v in counts.items()}


def open_positions(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty or "event" not in events.columns:
        return pd.DataFrame(columns=["strategy", "symbol", "side", "qty", "notional"])
    fills = events[events["event"].eq("fill")].copy()
    if fills.empty:
        return pd.DataFrame(columns=["strategy", "symbol", "side", "qty", "notional"])
    fills["signed_qty"] = fills.apply(
        lambda r: float(r.get("qty", 0.0)) * _side_sign(str(r.get("side", ""))), axis=1
    )
    fills["signed_notional"] = fills.apply(
        lambda r: float(r.get("notional", 0.0)) * _side_sign(str(r.get("side", ""))), axis=1
    )
    grouped = (
        fills.groupby(["strategy", "symbol"], dropna=False)
        .agg(qty=("signed_qty", "sum"), notional=("signed_notional", "sum"))
        .reset_index()
    )
    grouped["side"] = np.where(grouped["qty"] >= 0, "long", "short")
    grouped["qty"] = grouped["qty"].abs()
    grouped["notional"] = grouped["notional"].abs()
    return grouped[grouped["qty"] > 1e-9].sort_values(["strategy", "symbol"])


def strategy_daily_attribution(
    events: pd.DataFrame,
    equity: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["date", "strategy", "ret", "pnl"])
    if weights is None:
        weights = infer_strategy_weights(events)
    if not weights:
        return pd.DataFrame(columns=["date", "strategy", "ret", "pnl"])

    rows: list[dict[str, Any]] = []
    for _, row in equity.iterrows():
        equity_value = float(row.get("total_equity", 0.0) or 0.0)
        daily_ret = float(row.get("ret", 0.0) or 0.0)
        for strategy, weight in weights.items():
            rows.append(
                {
                    "date": row["date"],
                    "strategy": strategy,
                    "ret": daily_ret,
                    "pnl": equity_value * daily_ret * float(weight),
                    "allocation_weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def write_attribution_csv(attribution: pd.DataFrame, path: str | Path) -> None:
    """Persist dashboard attribution rows for later review."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attribution.to_csv(output_path, index=False)


def _archive_date() -> str:
    return dt.date.today().isoformat()


def archive_attribution_csv(
    attribution: pd.DataFrame,
    base_path: str | Path,
    archive_dir: str | Path,
) -> Path:
    """Write a timestamped attribution copy without replacing the default export."""
    base = Path(base_path)
    archive_path = Path(archive_dir) / f"{base.stem}_{_archive_date()}{base.suffix}"
    write_attribution_csv(attribution, archive_path)
    return archive_path


def _read_existing_kill_switch_log(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            strategy = str(record.get("strategy", ""))
            as_of = str(record.get("as_of", ""))
            if strategy and as_of:
                keys.add(f"{strategy}|{as_of}")
    return keys


def append_kill_switch_events(metrics: pd.DataFrame, path: str | Path) -> int:
    """Append structured kill-switch breach events, avoiding duplicate daily entries."""
    if metrics.empty:
        return 0
    breaches = metrics[metrics["kill_switch"]].copy()
    if breaches.empty:
        return 0

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_existing_kill_switch_log(log_path)
    appended = 0
    as_of = _archive_date()

    with log_path.open("a", encoding="utf-8") as f:
        for _, row in breaches.iterrows():
            key = f"{row['strategy']}|{as_of}"
            if key in existing:
                continue
            event = {
                "event": "kill_switch_breach",
                "source": "strategy_risk_dashboard",
                "as_of": as_of,
                "strategy": str(row["strategy"]),
                "max_drawdown": float(row["max_drawdown"]),
                "reason": "max_drawdown <= -20% threshold",
                "kill_switch": True,
            }
            f.write(json.dumps(event, sort_keys=True) + "\n")
            existing.add(key)
            appended += 1
    return appended


def strategy_metrics(events: pd.DataFrame, equity: pd.DataFrame) -> pd.DataFrame:
    attribution = strategy_daily_attribution(events, equity)
    if attribution.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "daily_return",
                "cumulative_return",
                "sharpe",
                "max_drawdown",
                "kill_switch",
            ]
        )

    rows: list[dict[str, Any]] = []
    for strategy, group in attribution.groupby("strategy"):
        group = group.sort_values("date")
        returns = group["ret"].to_numpy(dtype=float)
        pnls = group["pnl"].to_numpy(dtype=float)
        if len(returns) == 0:
            continue
        equity_curve = 1.0 + np.cumsum(returns)
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = equity_curve / peak - 1.0
        max_dd = float(np.min(drawdowns)) if len(drawdowns) else 0.0
        std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        sharpe = float(np.mean(returns) / std * np.sqrt(252)) if std > 0 else 0.0
        kill_switch = bool(max_dd <= MAX_DRAWDOWN_PCT + 1e-12)
        rows.append(
            {
                "strategy": strategy,
                "daily_return": float(np.mean(returns)) if len(returns) else 0.0,
                "cumulative_return": float(equity_curve[-1] - 1.0) if len(equity_curve) else 0.0,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "kill_switch": kill_switch,
                "cumulative_pnl": float(np.sum(pnls)),
            }
        )
    return pd.DataFrame(rows).sort_values("strategy")


def print_report(
    events_path: Path,
    equity_path: Path,
    attribution_output: Path | None = None,
    attribution_archive_dir: Path | None = None,
    kill_switch_log: Path | None = None,
) -> None:
    events = load_events(events_path)
    equity = load_equity(equity_path)
    weights = infer_strategy_weights(events)
    positions = open_positions(events)
    attribution = strategy_daily_attribution(events, equity)
    metrics = strategy_metrics(events, equity)
    attribution_archive_output = None
    if attribution_output is not None:
        write_attribution_csv(attribution, attribution_output)
        if attribution_archive_dir is not None:
            attribution_archive_output = archive_attribution_csv(attribution, attribution_output, attribution_archive_dir)

    print(f"Strategy risk dashboard — events={events_path}, equity={equity_path}")
    print(f"Events loaded: {len(events):,}; equity rows: {len(equity):,}")
    print(f"Inferred allocation weights: {weights or 'n/a'}")
    if attribution_output is not None:
        print(f"Attribution export: {attribution_output} ({len(attribution):,} rows)")
    if (
        attribution_archive_output is not None
        and Path(attribution_archive_output).resolve() != Path(attribution_output).resolve()
    ):
        print(f"Attribution archive: {attribution_archive_output} ({len(attribution):,} rows)")
    if kill_switch_log is not None:
        appended = append_kill_switch_events(metrics, kill_switch_log)
        print(f"Kill-switch log: {kill_switch_log} ({appended} new events)")
    print()

    print("Open positions by strategy:")
    if positions.empty:
        print("  none")
    else:
        print(positions.to_string(index=False))
    print()

    print("Strategy performance:")
    if metrics.empty:
        print("  no equity data available")
    else:
        print(metrics.to_string(index=False))
    print()

    breached = metrics[metrics["kill_switch"]] if not metrics.empty else metrics
    if breached.empty:
        print("Kill-switch status: no strategy breached 20% drawdown.")
    else:
        print("Kill-switch breaches:")
        print(breached[["strategy", "max_drawdown"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4 strategy risk dashboard")
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--equity", type=Path, default=DEFAULT_EQUITY)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--paper", action="store_true", help="Use paper default log paths")
    mode.add_argument("--live", action="store_true", help="Use live default log paths")
    parser.add_argument(
        "--attribution-output",
        type=Path,
        default=None,
        help="CSV path for daily strategy attribution export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events_path = DEFAULT_PAPER_EVENTS if args.paper else args.events
    equity_path = DEFAULT_PAPER_EQUITY if args.paper else args.equity
    attribution_output = args.attribution_output
    kill_switch_log = DEFAULT_PAPER_KILL_SWITCH_LOG if args.paper else DEFAULT_KILL_SWITCH_LOG
    if attribution_output is None:
        attribution_output = DEFAULT_PAPER_ATTRIBUTION if args.paper else DEFAULT_ATTRIBUTION
    archive_dir = DEFAULT_PAPER_ATTRIBUTION_ARCHIVE_DIR if args.paper else DEFAULT_LIVE_ATTRIBUTION_ARCHIVE_DIR
    print_report(events_path, equity_path, attribution_output, archive_dir, kill_switch_log)


if __name__ == "__main__":
    main()
