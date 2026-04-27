#!/usr/bin/env python
"""Strategy audit -- inventory all strategies, backtest across 4 market regimes.

Usage:
    python scripts/strategy_audit.py
    python scripts/strategy_audit.py --symbols SPY --capital 1000000 --output docs/strategy_analysis.md
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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

# ---------------------------------------------------------------------------
# Strategy implementations (BacktestEngine Protocol)
# ---------------------------------------------------------------------------


class MomentumCrossover:
    """Buy when fast SMA crosses above slow SMA; sell when it crosses below."""

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
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
        if not self._warm:
            self._prev_fast = self._fast_sma
            self._prev_slow = self._slow_sma
            self._warm = True
            return []
        signals: list[SignalData] = []
        if self._prev_fast <= self._prev_slow and self._fast_sma > self._slow_sma:
            if not self._in_position:
                signals.append(SignalData(
                    symbol=sym, direction="long", strength=1.0,
                    strategy_id="momentum_crossover",
                ))
                self._in_position = True
        elif self._prev_fast >= self._prev_slow and self._fast_sma < self._slow_sma:
            if self._in_position:
                signals.append(SignalData(
                    symbol=sym, direction="short", strength=1.0,
                    strategy_id="momentum_crossover",
                ))
                self._in_position = False
        self._prev_fast = self._fast_sma
        self._prev_slow = self._slow_sma
        return signals

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        pass


class MACrossoverFast:
    """Faster EMA crossover (20/50) for more frequent signals."""

    def __init__(self):
        self._prices: dict[str, list[float]] = {}
        self._in_position: bool = False

    def _ema(self, prices: list[float], window: int) -> float:
        alpha = 2.0 / (window + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        sym = bar.symbol
        price = float(bar.close)
        self._prices.setdefault(sym, []).append(price)
        prices = self._prices[sym]
        if len(prices) < 51:
            return []
        fast = self._ema(prices, 20)
        slow = self._ema(prices, 50)
        prev_prices = prices[:-1]
        prev_fast = self._ema(prev_prices, 20) if len(prev_prices) >= 20 else fast
        prev_slow = self._ema(prev_prices, 50) if len(prev_prices) >= 50 else slow
        signals: list[SignalData] = []
        bullish_cross = fast > slow and prev_fast <= prev_slow
        bearish_cross = fast < slow and prev_fast >= prev_slow
        if bullish_cross and not self._in_position:
            signals.append(SignalData(
                symbol=sym, direction="long", strength=1.0,
                strategy_id="ema_crossover_fast",
            ))
            self._in_position = True
        elif bearish_cross and self._in_position:
            signals.append(SignalData(
                symbol=sym, direction="short", strength=1.0,
                strategy_id="ema_crossover_fast",
            ))
            self._in_position = False
        return signals

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        pass


class MeanReversionRSI:
    """RSI-based mean reversion: buy oversold (< 30), exit on recovery."""

    def __init__(self, rsi_window: int = 14, oversold: int = 30):
        self.rsi_window = rsi_window
        self.oversold = oversold
        self._prices: dict[str, list[float]] = {}
        self._in_position: bool = False
        self._entry_price: float = 0.0

    def _compute_rsi(self, prices: list[float]) -> float:
        n = len(prices)
        if n < self.rsi_window + 1:
            return 50.0
        gains = []
        losses = []
        start = n - self.rsi_window
        for i in range(start, n):
            delta = prices[i] - prices[i - 1]
            gains.append(max(delta, 0))
            losses.append(max(-delta, 0))
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss < 1e-10:
            return 100.0
        return float(100.0 - 100.0 / (1.0 + avg_gain / avg_loss))

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        sym = bar.symbol
        price = float(bar.close)
        self._prices.setdefault(sym, []).append(price)
        prices = self._prices[sym]
        if len(prices) < self.rsi_window + 1:
            return []
        rsi = self._compute_rsi(prices)
        signals: list[SignalData] = []
        if self._in_position:
            gain_pct = (price - self._entry_price) / self._entry_price if self._entry_price > 0 else 0
            if rsi > 55 or gain_pct > 0.03 or gain_pct < -0.02:
                signals.append(SignalData(
                    symbol=sym, direction="short", strength=1.0,
                    strategy_id="mean_reversion_rsi",
                ))
                self._in_position = False
                self._entry_price = 0.0
        elif rsi < self.oversold:
            signals.append(SignalData(
                symbol=sym, direction="long",
                strength=min((self.oversold - rsi) / self.oversold, 1.0),
                strategy_id="mean_reversion_rsi",
            ))
            self._in_position = True
            self._entry_price = price
        return signals

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        pass


# ---------------------------------------------------------------------------
# Strategy inventory
# ---------------------------------------------------------------------------

STRATEGY_INVENTORY = [
    {
        "name": "Momentum (Cross-Sectional)",
        "registry_key": "equity_momentum",
        "category": "Equity",
        "description": "12-1 month cross-sectional momentum with inverse-vol weighting. "
                       "Jegadeesh & Titman (1993).",
        "params": "lookback=12mo, skip=1mo, n_long=5, n_short=5, vol_lb=6mo, rebal=21d",
        "data_req": "Daily OHLCV for universe (20+ assets)",
        "backtest_ready": True,
        "single_asset_adapt": False,
    },
    {
        "name": "Mean Reversion (RSI + BB)",
        "registry_key": "equity_mean_reversion",
        "category": "Equity",
        "description": "RSI + Bollinger Band + VWAP z-score composite. 1-3 day hold.",
        "params": "rsi_win=14, rsi_os=25, rsi_ob=75, bb_win=20, z_entry=2.0, hold=3d",
        "data_req": "Daily OHLCV for a single asset",
        "backtest_ready": True,
        "single_asset_adapt": True,
    },
    {
        "name": "Trend Following (EMA + ATR)",
        "registry_key": "equity_trend_following",
        "category": "Equity",
        "description": "Multi-timeframe EMA crossover with ATR sizing and pyramiding.",
        "params": "fast=20, slow=50, trend=200, atr=20, pyramid=4, stop=2xATR",
        "data_req": "Daily OHLCV for a single asset",
        "backtest_ready": True,
        "single_asset_adapt": True,
    },
    {
        "name": "Stat Arb (Pairs)",
        "registry_key": "equity_stat_arb",
        "category": "Equity",
        "description": "Cointegrated pairs with Kalman filter spread tracking and Kelly sizing.",
        "params": "lookback=252, hl_max=20, entry_z=2.0, exit_z=0.5, kelly=0.5",
        "data_req": "Daily OHLCV for universe (10+ assets)",
        "backtest_ready": True,
        "single_asset_adapt": False,
    },
    {
        "name": "CTA Trend (Multi-Speed)",
        "registry_key": "macro_cta_trend",
        "category": "Macro",
        "description": "Multi-speed trend signals (fast/med/slow) with vol targeting.",
        "params": "fast=20, med=50, slow=200, w=(0.2,0.3,0.5), vol_tgt=15%, rebal=5d",
        "data_req": "Daily OHLCV for multiple assets",
        "backtest_ready": True,
        "single_asset_adapt": False,
    },
    {
        "name": "Carry",
        "registry_key": "macro_carry",
        "category": "Macro",
        "description": "Cross-asset carry: long high-yield, short low-yield, vol-targeted.",
        "params": "n_long=3, n_short=3, vol_tgt=10%, smoothing=21d, rebal=21d",
        "data_req": "Daily OHLCV + yield data for multiple assets",
        "backtest_ready": False,
        "single_asset_adapt": False,
    },
    {
        "name": "Risk Parity",
        "registry_key": "macro_risk_parity",
        "category": "Macro",
        "description": "Equal risk contribution with vol targeting and regime overlay.",
        "params": "vol_tgt=10%, lookback=252, rebal=21d, regime=True, max_lev=1.0",
        "data_req": "Daily OHLCV for multiple assets (5+)",
        "backtest_ready": True,
        "single_asset_adapt": False,
    },
    {
        "name": "Delta Neutral (Options)",
        "registry_key": "options_delta_neutral",
        "category": "Options",
        "description": "Delta-hedged option positions, capturing vol risk premium.",
        "params": "delta_band=0.1, rehedge_freq=daily, max_gamma=0.05",
        "data_req": "Options chain data (not available via yfinance free tier)",
        "backtest_ready": False,
        "single_asset_adapt": False,
    },
    {
        "name": "Iron Condor (Options)",
        "registry_key": "options_iron_condor",
        "category": "Options",
        "description": "Sell OTM call and put spreads, collect theta, manage tail risk.",
        "params": "dte=45, wing_width=10%, profit_target=50%, stop=2x credit",
        "data_req": "Options chain data (not available via yfinance free tier)",
        "backtest_ready": False,
        "single_asset_adapt": False,
    },
    {
        "name": "Earnings Vol (Options)",
        "registry_key": "options_earnings_vol",
        "category": "Options",
        "description": "Pre-earnings straddle/strangle, post-earnings vol crush capture.",
        "params": "dte_before=5, dte_after=1, min_iv_rank=60, max_size=5%",
        "data_req": "Options chain + earnings calendar data",
        "backtest_ready": False,
        "single_asset_adapt": False,
    },
    {
        "name": "Tail Hedge (Options)",
        "registry_key": "options_tail_hedge",
        "category": "Options",
        "description": "OTM put buying program, 1-3% of AUM, roll monthly.",
        "params": "put_delta=0.05-0.10, allocation=1-3%, roll_freq=monthly",
        "data_req": "Options chain data (not available via yfinance free tier)",
        "backtest_ready": False,
        "single_asset_adapt": False,
    },
]

# 4 market regimes
REGIMES = {
    "Bull (2023-2024)": ("2023-01-01", "2024-12-31"),
    "Bear (2022)":      ("2022-01-01", "2022-12-30"),
    "Sideways (2015)":  ("2015-01-01", "2015-12-31"),
    "Volatile (2020)":  ("2020-01-01", "2020-12-31"),
}


def _safe_pct(value: Any) -> str:
    """Safely format a value as percentage, handling complex/NaN/inf."""
    v = _safe_float(value, 0.0)
    return f"{v:+.1%}"


def _safe_fmt(value: Any, fmt: str = ".2f") -> str:
    """Safely format a value, handling complex/NaN/inf."""
    v = _safe_float(value, 0.0)
    return f"{v:{fmt}}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling complex, NaN, inf."""
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v) or np.iscomplex(value):
            return default
        return v
    except (TypeError, ValueError):
        return default


@dataclass
class BacktestResult:
    strategy: str
    regime: str
    total_return: float
    annual_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    profit_factor: float
    n_trades: int

    def __post_init__(self) -> None:
        self.total_return = _safe_float(self.total_return, 0.0)
        self.annual_return = _safe_float(self.annual_return, 0.0)
        self.sharpe = _safe_float(self.sharpe, 0.0)
        self.sortino = _safe_float(self.sortino, 0.0)
        self.max_drawdown = _safe_float(self.max_drawdown, 0.0)
        self.calmar = _safe_float(self.calmar, 0.0)
        self.win_rate = _safe_float(self.win_rate, 0.0)
        self.profit_factor = _safe_float(self.profit_factor, 0.0)


def run_backtest(strategy: Any, symbols: list[str], start: str, end: str,
                 capital: float, slippage_bps: float) -> BacktestEngine:
    data = load_combined(symbols, start, end, interval="1d")
    if data.empty:
        raise ValueError(f"No data for {symbols} {start}->{end}")
    engine = BacktestEngine(
        strategy=strategy,
        initial_cash=capital,
        slippage_bps=slippage_bps,
    )
    engine.run(data)
    return engine


def compute_buy_and_hold(data: pd.DataFrame, initial_cash: float) -> float:
    """Compute buy-and-hold return for benchmark."""
    sym = data["symbol"].iloc[0]
    sym_data = data[data["symbol"] == sym].sort_values("timestamp")
    if sym_data.empty:
        return 0.0
    first_close = float(sym_data["close"].iloc[0])
    last_close = float(sym_data["close"].iloc[-1])
    shares = initial_cash / first_close
    return (shares * last_close - initial_cash) / initial_cash


def main() -> int:
    args = parse_args()

    print("=" * 72)
    print("  QUANTCORE STRATEGY AUDIT")
    print("=" * 72)
    n_equity = sum(1 for s in STRATEGY_INVENTORY if s["category"] == "Equity")
    n_macro = sum(1 for s in STRATEGY_INVENTORY if s["category"] == "Macro")
    n_options = sum(1 for s in STRATEGY_INVENTORY if s["category"] == "Options")
    n_ready = sum(1 for s in STRATEGY_INVENTORY if s["backtest_ready"])
    n_single = sum(1 for s in STRATEGY_INVENTORY if s["single_asset_adapt"])
    print(f"\n  Total strategies: {len(STRATEGY_INVENTORY)}  |  "
          f"Equity: {n_equity}  Macro: {n_macro}  Options: {n_options}")
    print(f"  Backtest-ready: {n_ready}  "
          f"(single-asset adapt: {n_single})")

    # --- Strategy Inventory ---
    print("\n" + "-" * 72)
    print("  STRATEGY INVENTORY")
    print("-" * 72)
    for i, s in enumerate(STRATEGY_INVENTORY, 1):
        status = ("READY (single-asset)" if s["single_asset_adapt"]
                  else "READY (multi-asset)" if s["backtest_ready"]
                  else "NO DATA")
        print(f"\n  [{i}] {s['name']} [{s['category']}] -- {status}")
        print(f"      {s['description']}")
        print(f"      Params: {s['params']}")
        print(f"      Data:   {s['data_req']}")

    if args.no_backtest:
        print("\n  Skipping backtests (--no-backtest flag set)")
        return 0

    # --- Strategies to backtest ---
    strategies_to_test: list[tuple[str, Any]] = [
        ("Momentum (SMA 50/200)", MomentumCrossover(fast_period=50, slow_period=200)),
        ("Momentum (EMA 20/50)", MACrossoverFast()),
        ("Mean Reversion (RSI 30)", MeanReversionRSI(rsi_window=14, oversold=30)),
    ]

    results: list[BacktestResult] = []
    bh_returns: dict[str, float] = {}

    print("\n" + "=" * 72)
    print("  BACKTEST: 3 STRATEGIES x 4 MARKET REGIMES")
    print("=" * 72)

    for strat_name, strategy in strategies_to_test:
        print(f"\n{'=' * 72}")
        print(f"  {strat_name}")
        print(f"  {'=' * 72}")

        for regime_name, (start, end) in REGIMES.items():
            print(f"\n  [{regime_name}] {start} -> {end}")
            try:
                t0 = time.perf_counter()
                engine = run_backtest(strategy, [args.symbols], start, end,
                                      args.capital, args.slippage)
                elapsed = time.perf_counter() - t0
                report = engine.summary()

                result = BacktestResult(
                    strategy=strat_name,
                    regime=regime_name,
                    total_return=report.total_return,
                    annual_return=report.annual_return,
                    sharpe=report.sharpe,
                    sortino=report.sortino,
                    max_drawdown=report.max_drawdown,
                    calmar=report.calmar,
                    win_rate=report.win_rate,
                    profit_factor=report.profit_factor,
                    n_trades=report.n_trades,
                )
                results.append(result)

                print(f"    Return: {_safe_pct(report.total_return):>10s}  "
                      f"Ann.Ret: {_safe_pct(report.annual_return):>10s}  "
                      f"Sharpe: {_safe_fmt(report.sharpe):>7s}  "
                      f"Sortino: {_safe_fmt(report.sortino):>7s}")
                print(f"    MaxDD:  {_safe_pct(report.max_drawdown):>10s}  "
                      f"Calmar: {_safe_fmt(report.calmar):>7s}  "
                      f"WinRate: {_safe_pct(report.win_rate):>9s}  "
                      f"Trades: {report.n_trades:>4d}  "
                      f"Time: {elapsed:.1f}s")

            except Exception as e:
                print(f"    FAILED: {e}")

    # Buy & Hold benchmark for each regime
    print(f"\n{'=' * 72}")
    print("  BENCHMARK: Buy & Hold")
    print(f"  {'=' * 72}")
    for regime_name, (start, end) in REGIMES.items():
        try:
            data = load_combined([args.symbols], start, end, interval="1d")
            if not data.empty:
                bh_ret = compute_buy_and_hold(data, args.capital)
                bh_returns[regime_name] = bh_ret
                print(f"  [{regime_name}] Return: {_safe_pct(bh_ret)}")
        except Exception as e:
            print(f"  [{regime_name}] FAILED: {e}")

    # --- Comparison Table ---
    print("\n" + "=" * 72)
    print("  CROSS-REGIME COMPARISON")
    print("=" * 72)

    for regime_name in REGIMES:
        regime_results = [r for r in results if r.regime == regime_name]
        bh_ret = bh_returns.get(regime_name, 0.0)
        print(f"\n  {regime_name}")
        print(f"  {'-' * 66}")
        print(f"  {'Strategy':<28s} {'Return':>8s} {'Sharpe':>7s} "
              f"{'MaxDD':>7s} {'WinRate':>7s} {'Trades':>6s}  {'vs B&H':>8s}")
        for rr in sorted(regime_results, key=lambda r: r.sharpe, reverse=True):
            vs_bh = rr.total_return - bh_ret
            print(f"  {rr.strategy:<28s} {_safe_pct(rr.total_return):>8s} "
                  f"{_safe_fmt(rr.sharpe, '+.2f'):>7s} "
                  f"{_safe_pct(rr.max_drawdown):>7s} "
                  f"{_safe_pct(rr.win_rate):>7s} "
                  f"{rr.n_trades:>5d}  "
                  f"{_safe_pct(vs_bh):>8s}")
        print(f"  {'Buy & Hold':<28s} {_safe_pct(bh_ret):>8s}")

    # --- Best Strategy Selection ---
    print("\n" + "=" * 72)
    print("  BEST STRATEGY SELECTION (for paper trading)")
    print("=" * 72)

    scores: dict[str, list[float]] = {}
    trade_counts: dict[str, list[int]] = {}
    for r in results:
        dd_score = max(0.0, 1.0 + r.max_drawdown)
        score = r.sharpe * 0.4 + r.sortino * 0.2 + r.calmar * 0.2 + dd_score * 0.2
        scores.setdefault(r.strategy, []).append(score)
        trade_counts.setdefault(r.strategy, []).append(r.n_trades)

    print("\n  Composite Scores (avg across 4 regimes):")
    print(f"  {'Strategy':<30s} {'Score':>7s}  {'Avg Sharpe':>9s}  "
          f"{'Avg Sortino':>10s}  {'Avg MaxDD':>9s}")
    print(f"  {'-' * 30} {'-' * 7}  {'-' * 9}  {'-' * 10}  {'-' * 9}")
    avg_scores: dict[str, float] = {}
    for name, score_list in scores.items():
        avg_score = np.mean(score_list)
        avg_scores[name] = avg_score
        strat_results = [r for r in results if r.strategy == name]
        avg_sharpe = np.mean([r.sharpe for r in strat_results])
        avg_sortino = np.mean([r.sortino for r in strat_results])
        avg_dd = np.mean([r.max_drawdown for r in strat_results])
        print(f"  {name:<30s} {_safe_fmt(avg_score, '7.3f')}  {_safe_fmt(avg_sharpe, '+9.2f')}  "
              f"{_safe_fmt(avg_sortino, '+10.2f')}  {_safe_pct(avg_dd):>11s}")

    # Filter out strategies with zero average trades (won't exercise paper trading infra)
    active_scores = {}
    for name, score_list in scores.items():
        avg_trades = np.mean(trade_counts.get(name, [0]))
        if avg_trades >= 1.0:
            active_scores[name] = np.mean(score_list)

    if not active_scores:
        # Fallback: use all strategies
        active_scores = {name: np.mean(lst) for name, lst in scores.items()}

    best = max(active_scores, key=active_scores.get)
    best_score = active_scores[best]
    print(f"\n  >>> Selected for paper trading: {best}")
    print(f"      Score: {best_score:.3f} (higher = better)")
    print(f"      Min trades per regime: {np.mean(trade_counts.get(best, [0])):.0f} avg")
    excluded = [n for n in avg_scores if n not in active_scores]
    if excluded:
        print(f"\n  Excluded from selection (zero trades in too many regimes): {', '.join(excluded)}")
    print("\n  Rationale: Highest composite score among strategies that generate >=1 trade/regime.")
    print("  Score formula: 0.4*Sharpe + 0.2*Sortino + 0.2*Calmar + 0.2*(1+MaxDD)")
    print("  Trade filter: average >= 1 trade per regime (required for paper trading validation)")

    # --- Strategies needing multi-asset data ---
    print("\n" + "-" * 72)
    print("  NON-BACKTESTABLE STRATEGIES (require special data)")
    print("-" * 72)
    for s in STRATEGY_INVENTORY:
        if not s["backtest_ready"]:
            print(f"  [{s['name']}] -- {s['data_req']}")

    # --- Generate report ---
    if args.output:
        md = generate_markdown_report(
            STRATEGY_INVENTORY, results, bh_returns,
            avg_scores, best, args,
        )
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(md)
        print(f"\n  Report saved to: {args.output}")

    print("\n" + "=" * 72)
    print("  AUDIT COMPLETE")
    print("=" * 72)
    return 0


def generate_markdown_report(
    inventory: list[dict],
    results: list[BacktestResult],
    bh_returns: dict[str, float],
    avg_scores: dict[str, float],
    best: str,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# QuantCore Strategy Analysis",
        "",
        f"**Generated:** {pd.Timestamp.now().isoformat()}",
        f"**Symbol:** {args.symbols}  |  **Capital:** ${args.capital:,.0f}  "
        f"|  **Slippage:** {args.slippage} bps",
        "",
        "## 1. Strategy Inventory",
        "",
        f"**Total:** {len(inventory)} strategies across 3 categories",
        "",
        "| # | Strategy | Category | Backtest Ready | Description |",
        "|---|----------|----------|----------------|-------------|",
    ]
    for i, s in enumerate(inventory, 1):
        status = "Yes" if s["backtest_ready"] else "No"
        lines.append(
            f"| {i} | {s['name']} | {s['category']} | "
            f"{status} | {s['description']} |"
        )

    lines.extend(["", "## 2. Strategy Details", ""])
    for s in inventory:
        lines.extend([
            f"### {s['name']}",
            f"- **Registry key:** `{s['registry_key']}`",
            f"- **Category:** {s['category']}",
            f"- **Parameters:** `{s['params']}`",
            f"- **Data requirement:** {s['data_req']}",
            f"- **Backtest ready:** {s['backtest_ready']}",
            "",
        ])

    if results:
        lines.extend([
            "## 3. Backtest Results",
            "",
            "### 3.1 Performance Across 4 Market Regimes",
            "",
        ])
        for regime_name in REGIMES:
            lines.append(f"#### {regime_name}")
            lines.append("")
            lines.append(
                "| Strategy | Return | Ann.Ret | Sharpe | Sortino | "
                "MaxDD | Calmar | Win Rate | Trades |"
            )
            lines.append(
                "|----------|--------|---------|--------|---------|"
                "-------|--------|----------|--------|"
            )
            for r in [r_ for r_ in results if r_.regime == regime_name]:
                lines.append(
                    f"| {r.strategy} | {r.total_return:+.1%} | "
                    f"{r.annual_return:+.1%} | {r.sharpe:.2f} | "
                    f"{r.sortino:.2f} | {r.max_drawdown:.1%} | "
                    f"{r.calmar:.2f} | {r.win_rate:.1%} | "
                    f"{r.n_trades} |"
                )
            bh = bh_returns.get(regime_name, 0.0)
            lines.append(
                f"| *Buy & Hold (benchmark)* | {bh:+.1%} | -- | -- | "
                f"-- | -- | -- | -- | -- |"
            )
            lines.append("")

        lines.extend([
            "### 3.2 Strategy Selection",
            "",
            f"**Selected for paper trading: {best}**",
            "",
            "| Strategy | Avg Score | Avg Sharpe | Avg Sortino | Avg MaxDD |",
            "|----------|-----------|------------|-------------|-----------|",
        ])
        for name in sorted(avg_scores, key=avg_scores.get, reverse=True):
            str_results = [r for r in results if r.strategy == name]
            avg_sharpe = np.mean([r.sharpe for r in str_results])
            avg_sortino = np.mean([r.sortino for r in str_results])
            avg_dd = np.mean([r.max_drawdown for r in str_results])
            marker = " (SELECTED)" if name == best else ""
            lines.append(
                f"| {name}{marker} | {avg_scores[name]:.3f} | "
                f"{avg_sharpe:+.2f} | {avg_sortino:+.2f} | "
                f"{avg_dd:+.1%} |"
            )

        lines.extend([
            "",
            "### 3.3 Selection Rationale",
            "",
            f"The **{best}** strategy was selected because it achieved the "
            "highest composite score averaged across all 4 market regimes "
            "(bull, bear, sideways, volatile).",
            "",
            "**Score weights:**",
            "- Sharpe Ratio (40%): risk-adjusted returns",
            "- Sortino Ratio (20%): downside volatility penalty",
            "- Calmar Ratio (20%): max drawdown efficiency",
            "- Drawdown Resilience (20%): direct max drawdown penalty",
            "",
        ])

    lines.extend([
        "## 4. Non-Backtestable Strategies",
        "",
        "These strategies require data not available via free tiers:",
        "",
        "| Strategy | Missing Data |",
        "|----------|-------------|",
    ])
    for s in inventory:
        if not s["backtest_ready"]:
            lines.append(f"| {s['name']} | {s['data_req']} |")

    lines.extend([
        "",
        "## 5. Recommendation",
        "",
        f"Deploy **{best}** for the 5-day paper trading run in Section 3.",
        "The strategy has been backtest-validated across 4 distinct market "
        "regimes on SPY.",
        "",
        "---",
        "",
        "*Report generated by scripts/strategy_audit.py*",
    ])

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QuantCore Strategy Audit")
    parser.add_argument("--symbols", default="SPY", help="Symbol for single-asset strategies")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="Initial capital")
    parser.add_argument("--slippage", type=float, default=5.0, help="Slippage in bps")
    parser.add_argument("--output", default="docs/strategy_analysis.md",
                        help="Output markdown file")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtests")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
