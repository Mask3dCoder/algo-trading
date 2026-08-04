"""Evaluate Simulation Alpha — Terminal Report for Historical Backtest Harness.

Reads the four telemetry artifacts produced by PaperTradingRunner and
emits a formatted summary of Sharpe, drawdown, per-year win-rate, TCA
costs, and kill-switch audit.

Usage
-----
    python bin/evaluate_simulation_alpha.py [--csv-root PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants & paths
# --------------------------------------------------------------------------
DEFAULT_CSV_ROOT = Path("reports/paper_trading")

DAILY_LOG_COLS = [
    "date",
    "equity",
    "peak_equity",
    "current_drawdown",
    "max_drawdown",
    "realized_pnl",
    "expected_pnl",
    "pnl_drift_pct",
    "daily_return",
    "rolling_sharpe_1m",
    "rolling_sharpe_3m",
    "slippage_bps",
    "slippage_dollars",
    "win_rate_all",
    "win_rate_3m",
    "win_rate_drift_alert",
    "trades_today",
    "open_positions",
    "net_delta",
    "vix_close",
    "macro_exit_triggered",
    "profit_factor",
]

TCA_COLS = [
    "timestamp", "order_id", "symbol", "side", "asset_type",
    "contracts", "shares", "arrival_price", "fill_price", "mid_price",
    "slippage_bps", "expected_slippage_bps", "realized_slippage_bps",
    "commission", "legging_risk_bps", "grade",
    "is_partial", "is_sweep", "notes",
]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load_daily_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[ERROR] daily_log.csv not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def _load_tca(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    # TCA file may have comment lines (# rotation markers)
    df = pd.read_csv(path, comment="#")
    if df.empty:
        return None
    return df


def _load_risk_state(path: Path) -> dict:
    if not path.exists():
        return {"state": "unknown"}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"[WARN] Could not parse risk state: {exc}", file=sys.stderr)
        return {"state": "unknown"}


def _compute_sharpe(equity_series: pd.Series) -> float:  # noqa: D401
    rets = equity_series.pct_change().dropna()
    if rets.empty or rets.std(ddof=1) == 0:
        return float("nan")
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))


def _fmt_num(x: float | None, pct: bool = False, decimals: int = 2) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    fmt = f"{{:,.{decimals}f}}"
    if pct:
        return f"{fmt.format(x * 100)}%"
    return fmt.format(x)


def _print_divider(width: int = 70) -> None:
    print("=" * width)


# --------------------------------------------------------------------------
# Report engine
# --------------------------------------------------------------------------

def generate_report(csv_root: Path) -> None:
    daily_path = csv_root / "daily_log.csv"
    tca_path = csv_root / "tca_performance.csv"
    state_path = csv_root / "state" / "risk_controller_state.json"

    daily = _load_daily_log(daily_path)
    tca = _load_tca(tca_path)
    risk_state = _load_risk_state(state_path)

    daily["date"] = pd.to_datetime(daily["date"])
    daily.set_index("date", inplace=True)

    # -- 1. Aggregate returns / drawdown / Sharpe -----------------------------
    equity = daily["equity"]
    start_equity = equity.iloc[0]
    end_equity = equity.iloc[-1]
    total_return = (end_equity - start_equity) / start_equity
    sharpe = _compute_sharpe(equity)
    max_dd = daily["max_drawdown"].min() / 100  # stored as percent

    # Per-year win rate (days with realized_pnl > 0)
    daily["year"] = daily.index.year
    yr_win_rates = {}
    for yr, grp in daily.groupby("year"):
        wins = (grp["realized_pnl"] > 0).sum()
        total_days = max(1, len(grp))
        yr_win_rates[yr] = wins / total_days

    # -- 2. TCA aggregates -----------------------------------------------------
    tca_total_slippage_dollars = 0.0
    tca_total_commission = 0.0
    tca_avg_slippage_bps = 0.0
    tca_avg_expected_bps = 0.0
    if tca is not None and not tca.empty:
        # slippage_bps * arrival_price * (contracts or shares) / 1e4
        tca["notional"] = tca["arrival_price"] * tca["contracts"] * 100  # option contracts
        tca["slippage_dollars"] = tca["slippage_bps"] * tca["notional"] / 1e4
        tca_total_slippage_dollars = tca["slippage_dollars"].sum()
        tca_total_commission = tca["commission"].sum()
        tca_avg_slippage_bps = tca["slippage_bps"].mean()
        tca_avg_expected_bps = tca["expected_slippage_bps"].mean()

    # -- 3. Kill-switch audit -------------------------------------------------
    killed = risk_state.get("state") in ("halted", "paused")
    kill_reason = None
    if killed:
        macro_date = risk_state.get("macro_trigger_date")
        sharpe_date = risk_state.get("last_sharpe_check")
        if macro_date:
            kill_reason = f"Macro volatility exit on {macro_date}"
        else:
            kill_reason = f"Sharpe halt (last check {sharpe_date})"

    # -- 4. Render ------------------------------------------------------------
    _print_divider()
    print("  QUANTCORE - ZERO-DATA HISTORICAL SIMULATION REPORT")
    _print_divider()

    print(f"\n[DAILY LOG]  --> {daily_path.resolve()}")
    print(f"   Trading days : {len(daily)}")
    print(f"   Start equity : ${_fmt_num(start_equity)}")
    print(f"   End equity   : ${_fmt_num(end_equity)}")
    print(f"   Total return : {_fmt_num(total_return, pct=True)}")
    print(f"   Sharpe (ann) : {_fmt_num(sharpe)}")
    print(f"   Max drawdown : {_fmt_num(max_dd, pct=True)}")

    print(f"\n[PER-YEAR] WIN RATE (days with P&L > 0)")
    for yr, rate in sorted(yr_win_rates.items()):
        print(f"   {yr} : {_fmt_num(rate, pct=True)}")

    print(f"\n[TCA]  --> {tca_path.resolve() if tca is not None else 'N/A'}")
    if tca is not None:
        print(f"   Avg slippage bps    : {_fmt_num(tca_avg_slippage_bps)} bps")
        print(f"   Avg expected bps    : {_fmt_num(tca_avg_expected_bps)} bps")
        print(f"   Total slippage ($)  : ${_fmt_num(tca_total_slippage_dollars)}")
        print(f"   Total commission ($): ${_fmt_num(tca_total_commission)}")
        tca_n = len(tca)
        print(f"   Fill count          : {tca_n}")
        if tca_total_slippage_dollars > 0:
            print("   [WARN] Non-zero slippage -- verify harness config (slippage_bps=0)")
        else:
            print("   [OK] Zero slippage -- mid-price fills confirmed")
    else:
        print("   No TCA data found.")

    print(f"\n[KILL-SWITCH AUDIT]  --> {state_path.resolve()}")
    print(f"   Terminal state : {risk_state.get('state', 'unknown')}")
    if killed:
        print(f"   [HALTED] -- {kill_reason}")
    else:
        print("   [ACTIVE] -- no kill triggered during simulation window")
    print(f"   Peak equity observed : {_fmt_num(risk_state.get('peak_equity'))}")
    print(f"   Daily returns tracked : {risk_state.get('daily_returns_count', 'N/A')}")

    _print_divider()
    print("  Simulation complete. Review reports/paper_trading/ for raw data.")
    _print_divider()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate historical simulation results")
    ap.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT,
                    help="Root directory containing daily_log.csv, tca_performance.csv, etc.")
    args = ap.parse_args()
    generate_report(args.csv_root)


if __name__ == "__main__":
    main()
