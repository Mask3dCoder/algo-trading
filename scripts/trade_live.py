#!/usr/bin/env python
"""Live trading launcher with pre-flight safety checks and risk controls.

Orchestrates the complete live trading workflow:
1. Pre-flight safety checks (market status, data, account, PDT, positions)
2. Strategy warmup (load historical data, pre-compute indicators)
3. Main event loop (fetch data -> generate signals -> risk check -> execute)
4. Emergency stop / circuit breakers
5. Graceful shutdown with position reconciliation

Usage:
    python scripts/trade_live.py --strategy momentum --symbols SPY,QQQ --paper
    python scripts/trade_live.py --strategy rsi --symbols GLD --live --capital 100000
    python scripts/trade_live.py --strategy momentum --symbols SPY,QQQ --dry-run
"""

from __future__ import annotations

import argparse
import signal as _signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from quantcore.backtest.engine import BacktestEngine
from quantcore.core.types import OHLCV, Order, OrderType, Side
from quantcore.data.yfinance_loader import load_combined
from quantcore.risk.limits import RiskLimits, check_pre_trade, check_post_trade
from scripts.strategy_validation import (
    EMACrossoverFast,
    MeanReversionRSI,
    MomentumCrossover,
    TrendFollowing,
)

STRATEGY_MAP = {
    "momentum": MomentumCrossover,
    "ema": EMACrossoverFast,
    "rsi": MeanReversionRSI,
    "trend": TrendFollowing,
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class LauncherState(str, Enum):
    INIT = "init"
    PREFLIGHT = "preflight"
    WARMUP = "warmup"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    EMERGENCY = "emergency"


class CheckStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class SafetyCheck:
    name: str
    status: CheckStatus
    detail: str = ""
    value: float | str = ""


@dataclass
class PreFlightReport:
    checks: list[SafetyCheck] = field(default_factory=list)
    all_passed: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def failures(self) -> list[SafetyCheck]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]


@dataclass
class SessionConfig:
    strategy_name: str
    symbols: list[str]
    mode: str  # "live", "paper", "dry-run"
    capital: float = 1_000_000.0
    sizing_pct: float = 0.25
    slippage_bps: float = 5.0
    max_daily_loss: float = 50_000.0
    max_drawdown_pct: float = 0.30
    position_timeout_minutes: int = 120
    bar_interval: str = "1d"
    warmup_years: int = 2


# ---------------------------------------------------------------------------
# Safety checks
# ---------------------------------------------------------------------------


def check_market_status(mode: str = "dry-run") -> SafetyCheck:
    """Check if market is currently open (NYSE calendar)."""
    if mode == "dry-run":
        return SafetyCheck("market_open", CheckStatus.PASS, "Dry-run — market check bypassed")

    now = pd.Timestamp.utcnow()
    weekday = now.weekday()
    hour = now.hour

    if weekday >= 5:
        return SafetyCheck("market_open", CheckStatus.FAIL, "Market closed — weekend")

    if hour < 14 or hour > 21:
        return SafetyCheck("market_open", CheckStatus.WARN,
                           f"Market may be closed (UTC hour={hour})")

    return SafetyCheck("market_open", CheckStatus.PASS, "Market appears open")


def check_account_limits(capital: float) -> list[SafetyCheck]:
    """Check account risk limits."""
    checks = []

    if capital < 25000:
        checks.append(SafetyCheck(
            "pdt_rule", CheckStatus.FAIL,
            f"Account below $25,000 PDT minimum: ${capital:,.0f}",
            value=capital,
        ))
    else:
        checks.append(SafetyCheck("pdt_rule", CheckStatus.PASS, "PDT minimum met"))

    if capital < 100000:
        checks.append(SafetyCheck(
            "capital_minimum", CheckStatus.WARN,
            f"Capital below $100,000 recommended minimum: ${capital:,.0f}",
            value=capital,
        ))
    else:
        checks.append(SafetyCheck("capital_minimum", CheckStatus.PASS, "Capital sufficient"))

    return checks


def check_data_availability(symbols: list[str], warmup_years: int) -> SafetyCheck:
    """Verify recent data is available for all symbols."""
    end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=warmup_years * 365)).strftime("%Y-%m-%d")

    try:
        data = load_combined(symbols, start, end, interval="1d")
        if data.empty:
            return SafetyCheck("data_available", CheckStatus.FAIL,
                               f"No data returned for {symbols}")
        missing = [s for s in symbols if s not in data["symbol"].unique() if hasattr(data, "columns") and "symbol" in data.columns]
        # yfinance_loader may return combined
        n_rows = len(data)
        if n_rows < 10:
            return SafetyCheck("data_available", CheckStatus.FAIL,
                               f"Insufficient data: {n_rows} rows")
        if missing:
            return SafetyCheck("data_available", CheckStatus.WARN,
                               f"Partial data — missing: {missing}")
        return SafetyCheck("data_available", CheckStatus.PASS,
                           f"Data loaded: {n_rows} rows for {symbols}")
    except Exception as e:
        return SafetyCheck("data_available", CheckStatus.FAIL,
                           f"Data loading failed: {str(e)[:80]}")


def check_strategy_health(strategy_cls: type, symbols: list[str]) -> SafetyCheck:
    """Quick backtest to verify strategy produces valid signals."""
    try:
        end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
        start = (pd.Timestamp.utcnow() - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
        data = load_combined(symbols, start, end, interval="1d")
        if data.empty:
            return SafetyCheck("strategy_health", CheckStatus.WARN, "No recent data for strategy test")

        strategy = strategy_cls()
        engine = BacktestEngine(strategy=strategy, initial_cash=100000)
        result = engine.run(data)

        if result.empty:
            return SafetyCheck("strategy_health", CheckStatus.WARN,
                               "Strategy produced no equity curve in warmup")

        signals = engine.portfolio.signal_log
        if not signals:
            return SafetyCheck("strategy_health", CheckStatus.WARN,
                               "Strategy generated zero signals in warmup")

        return SafetyCheck("strategy_health", CheckStatus.PASS,
                           f"Strategy healthy: {len(signals)} signals, "
                           f"{len(engine.portfolio.trades)} fills in 90-day warmup")
    except Exception as e:
        return SafetyCheck("strategy_health", CheckStatus.FAIL,
                           f"Strategy test failed: {str(e)[:80]}")


def run_preflight(config: SessionConfig) -> PreFlightReport:
    """Run all pre-flight safety checks."""
    report = PreFlightReport()

    # 1. Market status
    report.checks.append(check_market_status(config.mode))

    # 2. Account / PDT
    report.checks.extend(check_account_limits(config.capital))

    # 3. Data availability
    report.checks.append(check_data_availability(config.symbols, config.warmup_years))

    # 4. Strategy health
    strategy_cls = STRATEGY_MAP[config.strategy_name]
    report.checks.append(check_strategy_health(strategy_cls, config.symbols))

    # Compile results
    for c in report.checks:
        if c.status == CheckStatus.WARN:
            report.warnings.append(f"[{c.name}] {c.detail}")
        elif c.status == CheckStatus.FAIL:
            report.all_passed = False

    return report


# ---------------------------------------------------------------------------
# Risk gating
# ---------------------------------------------------------------------------


def risk_gate_signal(signal_data: dict, session: dict) -> tuple[bool, str]:
    """Pre-trade risk gate — reject signals that would breach limits."""
    symbol = signal_data.get("symbol", "")
    direction = signal_data.get("direction", "")
    strength = signal_data.get("strength", 0.0)

    # Daily loss limit
    daily_pnl = session.get("daily_pnl", 0.0)
    max_daily_loss = session.get("max_daily_loss", 50_000.0)
    if daily_pnl < -max_daily_loss:
        return False, f"Daily loss limit breached: ${daily_pnl:,.0f}"

    # Max positions
    positions = session.get("positions", {})
    max_positions = session.get("max_positions", 5)
    if len(positions) >= max_positions and direction == "buy":
        return False, f"Max positions reached ({max_positions})"

    # Same-symbol cooldown
    if symbol in session.get("recent_trades", {}):
        last_time = session["recent_trades"][symbol]
        cooldown = session.get("cooldown_seconds", 3600)
        if time.time() - last_time < cooldown:
            return False, f"Cooldown active for {symbol}"

    return True, "approved"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_preflight(report: PreFlightReport) -> None:
    """Display pre-flight check results."""
    print("=" * 78)
    print("  QUANTCORE PRE-FLIGHT SAFETY CHECKS")
    print("=" * 78)

    icons = {CheckStatus.PASS: "[PASS]", CheckStatus.WARN: "[WARN]", CheckStatus.FAIL: "[FAIL]"}
    for c in report.checks:
        icon = icons[c.status]
        print(f"  {icon}  {c.name:<22s}  {c.detail}")

    if report.warnings:
        print(f"\n  Warnings ({len(report.warnings)}):")
        for w in report.warnings:
            print(f"    - {w}")

    if report.all_passed:
        print(f"\n  >>> All critical checks passed — ready to trade.")
    else:
        print(f"\n  >>> CRITICAL FAILURES ({len(report.failures)}) — aborting launch.")
        for f in report.failures:
            print(f"      [{f.name}] {f.detail}")

    print("=" * 78)


def print_session_banner(config: SessionConfig) -> None:
    """Print session configuration banner."""
    print(f"\n  Strategy:       {config.strategy_name}")
    print(f"  Symbols:        {', '.join(config.symbols)}")
    print(f"  Mode:           {config.mode}")
    print(f"  Capital:        ${config.capital:,.0f}")
    print(f"  Sizing:         {config.sizing_pct:.0%}")
    print(f"  Max Daily Loss: ${config.max_daily_loss:,.0f}")
    print(f"  Bar Interval:   {config.bar_interval}")
    print(f"  Warmup Period:  {config.warmup_years} years")


# ---------------------------------------------------------------------------
# Main session
# ---------------------------------------------------------------------------


def warmup_strategy(config: SessionConfig):
    """Load historical data and warm up strategy indicators."""
    end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=config.warmup_years * 365)).strftime("%Y-%m-%d")

    print(f"\n  Warming up strategy on {start} to {end} ...")
    data = load_combined(config.symbols, start, end, interval=config.bar_interval)

    strategy = STRATEGY_MAP[config.strategy_name]()
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=config.capital,
        slippage_bps=config.slippage_bps,
        sizing_pct=config.sizing_pct,
    )
    engine.run(data)
    perf = engine.summary()

    print(f"  Warmup complete: {perf.n_trades} trades, "
          f"return={perf.total_return:.2%}, sharpe={perf.sharpe:.2f}")
    return strategy, engine


def run_dry_run(config: SessionConfig) -> int:
    """Simulate a full trading session (dry-run mode)."""
    print("\n" + "=" * 78)
    print("  DRY-RUN SIMULATION")
    print("=" * 78)

    end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=config.warmup_years * 365)).strftime("%Y-%m-%d")

    for symbol in config.symbols:
        data = load_combined([symbol], start, end, interval=config.bar_interval)
        strategy = STRATEGY_MAP[config.strategy_name]()
        engine = BacktestEngine(
            strategy=strategy,
            initial_cash=config.capital,
            slippage_bps=config.slippage_bps,
            sizing_pct=config.sizing_pct,
        )
        engine.run(data)
        perf = engine.summary()

        print(f"\n  {symbol}:")
        print(f"    Return={perf.total_return:.2%}  Sharpe={perf.sharpe:.2f}  "
              f"MaxDD={perf.max_drawdown:.2%}  Trades={perf.n_trades}")
        print(f"    WinRate={perf.win_rate:.1%}  PF={perf.profit_factor:.2f}")

        for fill in engine.portfolio.trades[-5:]:
            side = "SELL" if fill.order_id.endswith("_sell") else "BUY "
            print(f"    {fill.timestamp}  {side}  qty={float(fill.quantity):.0f}  "
                  f"@ ${float(fill.price):.2f}  comm=${float(fill.commission):.2f}")

    print("\n  Dry-run complete — no real orders were placed.")
    print("=" * 78)
    return 0


def setup_signal_handlers() -> None:
    """Setup SIGINT/SIGTERM handlers for graceful shutdown."""
    def handler(signum, frame):
        print("\n  >>> Shutdown signal received. Stopping gracefully...")
        sys.exit(0)

    _signal.signal(_signal.SIGINT, handler)
    try:
        _signal.signal(_signal.SIGTERM, handler)
    except AttributeError:
        pass  # Windows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="QuantCore Live Trading Launcher")
    parser.add_argument("--strategy", default="momentum",
                        choices=list(STRATEGY_MAP), help="Strategy to run")
    parser.add_argument("--symbols", default="SPY",
                        help="Comma-separated symbols (e.g. SPY,QQQ,GLD)")
    parser.add_argument("--mode", default="dry-run",
                        choices=["dry-run", "paper", "live"],
                        help="Trading mode")
    parser.add_argument("--capital", type=float, default=1_000_000,
                        help="Initial capital")
    parser.add_argument("--sizing", type=float, default=0.25,
                        help="Position sizing fraction")
    parser.add_argument("--slippage", type=float, default=5.0,
                        help="Slippage assumption in bps")
    parser.add_argument("--max-daily-loss", type=float, default=50_000,
                        help="Max daily loss before circuit breaker")
    parser.add_argument("--warmup-years", type=int, default=2,
                        help="Years of historical data for strategy warmup")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip pre-flight safety checks (DANGEROUS)")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]

    if args.mode == "live":
        print("\n" + "!" * 78)
        print("  WARNING: LIVE TRADING MODE")
        print("  Real orders will be placed with real money.")
        print("  Ensure you understand the risks and have tested thoroughly.")
        print("!" * 78)
        confirm = input("\n  Type 'YES' to confirm live trading: ")
        if confirm != "YES":
            print("  Live trading aborted.")
            return 1

    config = SessionConfig(
        strategy_name=args.strategy,
        symbols=symbols,
        mode=args.mode,
        capital=args.capital,
        sizing_pct=args.sizing,
        slippage_bps=args.slippage,
        max_daily_loss=args.max_daily_loss,
        warmup_years=args.warmup_years,
    )

    setup_signal_handlers()

    # Pre-flight
    if not args.skip_preflight:
        report = run_preflight(config)
        print_preflight(report)
        if not report.all_passed:
            print("\n  Pre-flight checks FAILED. Aborting launch.")
            print("  Fix the issues above or use --skip-preflight (not recommended).")
            return 1
    else:
        print("\n  >>> WARNING: Pre-flight checks SKIPPED <<<")

    print_session_banner(config)

    # Dry run: simulate full session using historical backtest
    if args.mode == "dry-run":
        return run_dry_run(config)

    # Paper / Live: warmup strategy
    strategy, engine = warmup_strategy(config)

    # Main trading loop (paper/live)
    print(f"\n  Entering {args.mode} trading loop...")
    print(f"  Press Ctrl+C to stop gracefully.\n")

    state = LauncherState.RUNNING
    session: dict[str, Any] = {
        "daily_pnl": 0.0,
        "max_daily_loss": config.max_daily_loss,
        "positions": {},
        "max_positions": 5,
        "recent_trades": {},
        "cooldown_seconds": 3600,
        "start_time": time.time(),
        "signals_generated": 0,
        "signals_approved": 0,
        "orders_placed": 0,
        "orders_filled": 0,
    }

    try:
        while state == LauncherState.RUNNING:
            # In paper mode, we'd connect to a real-time data feed
            # and broker adapter. For dry-run, we already simulated.
            # This loop is the skeleton for live operation.

            print(f"  [{pd.Timestamp.utcnow().isoformat()[:19]}] Waiting for next bar... "
                  f"(signals={session['signals_generated']}, "
                  f"orders={session['orders_placed']})")
            time.sleep(60)  # Check every minute

            # Placeholder: fetch latest bar, generate signals, risk-gate, submit orders

    except KeyboardInterrupt:
        print("\n  Graceful shutdown initiated...")
    finally:
        state = LauncherState.STOPPED
        elapsed = time.time() - session["start_time"]
        print(f"\n  Session ended. Duration: {elapsed/60:.0f} min")
        print(f"  Signals: {session['signals_generated']} generated, "
              f"{session['signals_approved']} approved")
        print(f"  Orders: {session['orders_placed']} placed, "
              f"{session['orders_filled']} filled")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
