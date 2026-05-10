#!/usr/bin/env python
"""Backtest demo — real-data momentum crossover on SPY using yfinance.

Usage:
    python scripts/backtest_demo.py
    python scripts/backtest_demo.py --symbols SPY,QQQ --start 2023-01-01 --end 2024-12-31
    python scripts/backtest_demo.py --symbols SPY --start 2022-01-01 --end 2024-12-31 --plot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure quantcore is importable from the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantcore.backtest.engine import (  # noqa: E402
    OHLCV,
    BacktestEngine,
    Fill,
    PortfolioState,
    SignalData,
)
from quantcore.data.yfinance_loader import load_combined  # noqa: E402

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple momentum crossover strategy
# ---------------------------------------------------------------------------


class MomentumCrossoverStrategy:
    """Buy when fast SMA crosses above slow SMA; sell when it crosses below.

    Implements the Strategy Protocol expected by BacktestEngine.
    """

    def __init__(self, fast_period: int = 50, slow_period: int = 200) -> None:
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prices: dict[str, list[float]] = {}
        self._fast_sma: float = 0.0
        self._slow_sma: float = 0.0
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self._in_position: bool = False
        self._warm: bool = False

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        price = float(bar.close)
        sym = bar.symbol

        self._prices.setdefault(sym, []).append(price)
        prices = self._prices[sym]

        if len(prices) < self.slow_period:
            return []

        self._fast_sma = float(np.mean(prices[-self.fast_period:]))
        self._slow_sma = float(np.mean(prices[-self.slow_period:]))

        signals: list[SignalData] = []

        if not self._warm:
            self._prev_fast = self._fast_sma
            self._prev_slow = self._slow_sma
            self._warm = True
            # Enter initial position if already in bullish configuration
            if self._fast_sma > self._slow_sma and not self._in_position:
                signals.append(SignalData(
                    symbol=sym,
                    direction="long",
                    strength=1.0,
                    strategy_id="momentum_crossover",
                ))
                self._in_position = True
            return signals

        # Golden cross: fast crosses above slow
        if self._prev_fast <= self._prev_slow and self._fast_sma > self._slow_sma:
            if not self._in_position:
                signals.append(SignalData(
                    symbol=sym,
                    direction="long",
                    strength=1.0,
                    strategy_id="momentum_crossover",
                ))
                self._in_position = True
        # Death cross: fast crosses below slow
        elif self._prev_fast >= self._prev_slow and self._fast_sma < self._slow_sma:
            if self._in_position:
                signals.append(SignalData(
                    symbol=sym,
                    direction="short",
                    strength=1.0,
                    strategy_id="momentum_crossover",
                ))
                self._in_position = False

        self._prev_fast = self._fast_sma
        self._prev_slow = self._slow_sma
        return signals

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        pass


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def compute_buy_and_hold(data: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    """Compute a buy-and-hold equity curve for the first symbol in the data."""
    sym = data["symbol"].iloc[0]
    sym_data = data[data["symbol"] == sym].sort_values("timestamp")
    if sym_data.empty:
        return pd.DataFrame()

    first_close = float(sym_data["close"].iloc[0])
    shares = initial_cash / first_close

    equity = sym_data[["timestamp"]].copy()
    equity["equity"] = sym_data["close"].astype(float) * shares
    return equity


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="QuantCore backtest demo")
    parser.add_argument("--symbols", default="SPY", help="Comma-separated symbols")
    parser.add_argument("--start", default="2023-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2024-12-31", help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Initial capital")
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps")
    parser.add_argument("--fast", type=int, default=50, help="Fast SMA period")
    parser.add_argument("--slow", type=int, default=200, help="Slow SMA period")
    parser.add_argument("--sizing", type=float, default=0.95, help="Fraction of cash to deploy per signal (0-1)")
    parser.add_argument("--framework", default=None,
                        help="Framework strategy to use (e.g., equity_mean_reversion, equity_momentum)")
    parser.add_argument("--fw-params", default="{}",
                        help="JSON dict of strategy params to override defaults")
    parser.add_argument("--plot", action="store_true", help="Save equity curve plot")
    parser.add_argument("--output", default="scripts/output", help="Output directory for plot")
    parser.add_argument("--interval", default="1d",
                        help="Bar interval: 1d, 1h, 15m, 5m, 1m (default: 1d)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 62)
    print("  QuantCore Backtest Demo")
    print("=" * 62)
    print(f"  Symbols:    {', '.join(symbols)}")
    print(f"  Period:     {args.start} -> {args.end}")
    print(f"  Capital:    ${args.capital:,.0f}")
    print(f"  Slippage:   {args.slippage} bps")
    if args.framework:
        print(f"  Strategy:   {args.framework} (framework)")
    else:
        print(f"  Strategy:   SMA({args.fast}/{args.slow}) crossover")
    print(f"  Sizing:     {args.sizing:.0%} per signal")
    print(f"  Interval:   {args.interval}")
    print("=" * 62)

    # ---- Load data ----
    print("\nLoading data via yfinance...")
    data = load_combined(symbols, args.start, args.end, interval=args.interval)
    if data.empty:
        print("ERROR: No data loaded. Check your symbols and date range.")
        sys.exit(1)

    n_bars = len(data)
    n_days = data["timestamp"].nunique()
    print(f"Loaded {n_bars:,} bars across {n_days} trading days")
    print(f"Date range: {data['timestamp'].min()} -> {data['timestamp'].max()}")

    # ---- Create strategy ----
    if args.framework:
        from quantcore.backtest.adapter import FrameworkStrategyAdapter
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.registry import create

        # Import strategy modules so @register decorators fire
        import quantcore.strategies.equity.momentum  # noqa: F811
        import quantcore.strategies.equity.mean_reversion  # noqa: F811
        import quantcore.strategies.equity.trend_following  # noqa: F811
        import quantcore.strategies.equity.stat_arb  # noqa: F811
        import quantcore.strategies.equity.intraday_mean_reversion  # noqa: F811
        import quantcore.strategies.equity.intraday_momentum  # noqa: F811
        import quantcore.strategies.equity.intraday_trend  # noqa: F811
        import quantcore.strategies.equity.intraday_breakout  # noqa: F811
        import quantcore.strategies.macro.cta_trend  # noqa: F811
        import quantcore.strategies.macro.carry  # noqa: F811
        import quantcore.strategies.macro.risk_parity  # noqa: F811
        import quantcore.strategies.options.delta_neutral  # noqa: F811
        import quantcore.strategies.options.earnings_vol  # noqa: F811
        import quantcore.strategies.options.iron_condor  # noqa: F811
        import quantcore.strategies.options.tail_hedge  # noqa: F811

        if args.framework.startswith("macro_"):
            stype = StrategyType.MACRO
        elif args.framework.startswith("options_"):
            stype = StrategyType.OPTIONS
        else:
            stype = StrategyType.EQUITY

        import json
        fw_params = json.loads(args.fw_params)
        config = StrategyConfig(
            name=args.framework,
            strategy_type=stype,
            max_position_notional=args.capital,
            risk_budget_pct=0.02,
            params=fw_params,
        )
        fw = create(args.framework, config)
        strategy = FrameworkStrategyAdapter(fw, symbols)
    else:
        strategy = MomentumCrossoverStrategy(fast_period=args.fast, slow_period=args.slow)

    # ---- Run backtest ----
    print("\nRunning backtest...")
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=args.capital,
        slippage_bps=args.slippage,
        sizing_pct=args.sizing,
    )
    equity_curve = engine.run(data)

    if equity_curve.empty:
        print("ERROR: Backtest produced no results.")
        sys.exit(1)

    # ---- Performance summary ----
    report = engine.summary()
    print("\n" + "-" * 62)
    print("  Performance Summary")
    print("-" * 62)
    print(f"  Total Return:   {report.total_return:>+8.2%}")
    print(f"  Annual Return:  {report.annual_return:>+8.2%}")
    print(f"  Sharpe Ratio:   {report.sharpe:>8.2f}")
    print(f"  Sortino Ratio:  {report.sortino:>8.2f}")
    print(f"  Calmar Ratio:   {report.calmar:>8.2f}")
    print(f"  Max Drawdown:   {report.max_drawdown:>8.2%}")
    print(f"  Win Rate:       {report.win_rate:>8.2%}")
    print(f"  Profit Factor:  {report.profit_factor:>8.2f}")
    print(f"  # Trades:       {report.n_trades:>8d}")
    print("-" * 62)

    # Compare to buy-and-hold
    final_equity = float(equity_curve["equity"].iloc[-1])
    strategy_return = (final_equity - args.capital) / args.capital

    bh = compute_buy_and_hold(data, args.capital)
    if not bh.empty:
        bh_final = float(bh["equity"].iloc[-1])
        bh_return = (bh_final - args.capital) / args.capital
        print(f"\n  Strategy Return: {strategy_return:+.2%}")
        print(f"  Buy & Hold:      {bh_return:+.2%}")
        print(f"  Alpha:           {strategy_return - bh_return:+.2%}")

    # ---- Plot ----
    if args.plot:
        _plot_equity(equity_curve, bh, symbols, args)


def _plot_equity(
    equity_curve: pd.DataFrame,
    benchmark: pd.DataFrame,
    symbols: list[str],
    args: argparse.Namespace,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plot")
        return

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    strategy_label = args.framework.replace("_", " ").title() if args.framework else "SMA Crossover"
    ax.plot(
        equity_curve["timestamp"],
        equity_curve["equity"],
        label=f"Strategy ({strategy_label})",
        linewidth=1.5,
        color="#1f77b4",
    )

    if not benchmark.empty:
        ax.plot(
            benchmark["timestamp"],
            benchmark["equity"],
            label=f"Buy & Hold ({symbols[0]})",
            linewidth=1.0,
            color="#7f7f7f",
            alpha=0.7,
            linestyle="--",
        )

    ax.set_title(f"QuantCore Backtest -- {', '.join(symbols)} ({args.start} -> {args.end})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    fig.tight_layout()
    out_path = output_dir / "backtest_equity.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nEquity curve saved to: {out_path}")


if __name__ == "__main__":
    main()
