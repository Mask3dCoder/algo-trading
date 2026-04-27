#!/usr/bin/env python
"""Strategy validation on real historical data.

Runs a validation matrix: N strategies × M symbols × 5 years of data.
Answers: does the strategy have a real edge?

Usage:
    python scripts/strategy_validation.py
    python scripts/strategy_validation.py --symbols SPY,QQQ,AAPL --strategies momentum,ema,trend
    python scripts/strategy_validation.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
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
from quantcore.data.yfinance_loader import load_combined, load_bars  # noqa: E402


# ---------------------------------------------------------------------------
# Safe formatting
# ---------------------------------------------------------------------------


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if np.isnan(v):
            return default
        if np.isinf(v):
            return 999.0 if v > 0 else -999.0
        return v
    except (TypeError, ValueError):
        return default


def _safe_pct(value: Any) -> str:
    return f"{_safe_float(value, 0.0):+.1%}"


# ---------------------------------------------------------------------------
# Strategy implementations (BacktestEngine Protocol)
# ---------------------------------------------------------------------------


class MomentumCrossover:
    """SMA 50/200 crossover — classic trend-following entry/exit."""

    def __init__(self, fast_period: int = 50, slow_period: int = 200):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self._prices: dict[str, list[float]] = {}
        self._in_position = False
        self._prev_fast: float | None = None
        self._prev_slow: float | None = None
        self._warm = False

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        sym = bar.symbol
        price = float(bar.close)
        self._prices.setdefault(sym, []).append(price)
        prices = self._prices[sym]
        if len(prices) < self.slow_period:
            return []
        fast = float(np.mean(prices[-self.fast_period:]))
        slow = float(np.mean(prices[-self.slow_period:]))
        if not self._warm:
            self._prev_fast, self._prev_slow = fast, slow
            self._warm = True
            return []
        signals: list[SignalData] = []
        bullish = self._prev_fast <= self._prev_slow and fast > slow
        bearish = self._prev_fast >= self._prev_slow and fast < slow
        if bullish and not self._in_position:
            signals.append(SignalData(
                symbol=sym, direction="long", strength=1.0,
                strategy_id="momentum_crossover",
            ))
            self._in_position = True
        elif bearish and self._in_position:
            signals.append(SignalData(
                symbol=sym, direction="short", strength=1.0,
                strategy_id="momentum_crossover",
            ))
            self._in_position = False
        self._prev_fast, self._prev_slow = fast, slow
        return signals

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        pass


class EMACrossoverFast:
    """EMA 20/50 crossover — faster signals, more trades."""

    def __init__(self):
        self._prices: dict[str, list[float]] = {}
        self._in_position = False

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
    """RSI-based mean reversion with stop-loss and take-profit exits.

    Buy when RSI < oversold (30). Exit on: RSI > 55 (mean reverted),
    profit > 3%, or loss > 2% (stop-loss).
    """

    def __init__(self, rsi_window: int = 14, oversold: int = 30):
        self.rsi_window = rsi_window
        self.oversold = oversold
        self._prices: dict[str, list[float]] = {}
        self._in_position = False
        self._entry_price = 0.0

    def _compute_rsi(self, prices: list[float]) -> float:
        n = len(prices)
        if n < self.rsi_window + 1:
            return 50.0
        gains, losses = [], []
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


class TrendFollowing:
    """Multi-timeframe EMA trend following with ATR-based trailing stop.

    Entry: EMA 20 > EMA 50 > EMA 200 (all aligned bullish).
    Exit: EMA 20 < EMA 50 (trend broken) or trailing stop hit.
    Trailing stop: 2×ATR from highest high since entry.
    """

    def __init__(self, atr_window: int = 14, stop_mult: float = 2.0):
        self.atr_window = atr_window
        self.stop_mult = stop_mult
        self._prices: dict[str, list[float]] = {}
        self._highs: dict[str, list[float]] = {}
        self._lows: dict[str, list[float]] = {}
        self._in_position = False
        self._highest_since_entry = 0.0
        self._entry_price = 0.0

    def _ema(self, prices: list[float], window: int) -> float:
        alpha = 2.0 / (window + 1)
        ema = prices[0]
        for p in prices[1:]:
            ema = alpha * p + (1 - alpha) * ema
        return ema

    def _atr(self, highs: list[float], lows: list[float], closes: list[float]) -> float:
        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)
        return float(np.mean(trs[-self.atr_window:])) if trs else 0.0

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        sym = bar.symbol
        price = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        self._prices.setdefault(sym, []).append(price)
        self._highs.setdefault(sym, []).append(high)
        self._lows.setdefault(sym, []).append(low)
        prices = self._prices[sym]
        highs = self._highs[sym]
        lows = self._lows[sym]
        if len(prices) < 201:
            return []
        ema20 = self._ema(prices, 20)
        ema50 = self._ema(prices, 50)
        ema200 = self._ema(prices, 200)
        atr = self._atr(highs, lows, prices)
        signals: list[SignalData] = []
        if self._in_position:
            self._highest_since_entry = max(self._highest_since_entry, price)
            stop_price = self._highest_since_entry - self.stop_mult * atr
            trend_broken = ema20 < ema50
            stop_hit = price < stop_price
            if trend_broken or stop_hit:
                reason = "trend_broken" if trend_broken else "stop_hit"
                signals.append(SignalData(
                    symbol=sym, direction="short", strength=1.0,
                    strategy_id=f"trend_following_{reason}",
                ))
                self._in_position = False
        else:
            trend_aligned = ema20 > ema50 > ema200
            if trend_aligned:
                signals.append(SignalData(
                    symbol=sym, direction="long", strength=1.0,
                    strategy_id="trend_following",
                ))
                self._in_position = True
                self._entry_price = price
                self._highest_since_entry = price
        return signals

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        pass


class BollingerBreakout:
    """Bollinger Band breakout with volatility filter.

    Entry: price closes above upper band (20, 2) AND previous close was
    inside the band (confirms breakout, not already extended).
    Exit: price closes below middle band (SMA 20).
    Filter: ATR must be above its 50-day average (trades only in volatile
    conditions; avoids chop).
    """

    def __init__(self, bb_window: int = 20, bb_std: float = 2.0):
        self.bb_window = bb_window
        self.bb_std = bb_std
        self._prices: dict[str, list[float]] = {}
        self._highs: dict[str, list[float]] = {}
        self._lows: dict[str, list[float]] = {}
        self._trs: dict[str, list[float]] = {}
        self._atr14_vals: dict[str, list[float]] = {}
        self._in_position = False

    def _bollinger(self, prices: list[float]) -> tuple[float, float, float]:
        sma = float(np.mean(prices[-self.bb_window:]))
        std = float(np.std(prices[-self.bb_window:], ddof=1))
        return sma, sma + self.bb_std * std, sma - self.bb_std * std

    def on_bar(self, bar: OHLCV, portfolio: PortfolioState) -> list[SignalData]:
        sym = bar.symbol
        price = float(bar.close)
        high = float(bar.high)
        low = float(bar.low)
        self._prices.setdefault(sym, []).append(price)
        self._highs.setdefault(sym, []).append(high)
        self._lows.setdefault(sym, []).append(low)
        prices = self._prices[sym]
        highs = self._highs[sym]
        lows = self._lows[sym]
        if len(prices) < max(self.bb_window + 1, 65):
            return []

        # Incremental True Range (O(1) per bar)
        tr = max(high - low, abs(high - prices[-2]), abs(low - prices[-2]))
        self._trs.setdefault(sym, []).append(tr)

        # ATR(14) from last 14 TRs
        trs_sym = self._trs[sym]
        atr14 = float(np.mean(trs_sym[-14:]))
        self._atr14_vals.setdefault(sym, []).append(atr14)

        # ATR(50) = mean of last 50 ATR(14) values
        atr14_vals = self._atr14_vals[sym]
        atr50 = float(np.mean(atr14_vals[-50:])) if len(atr14_vals) >= 50 else atr14

        middle, upper, lower = self._bollinger(prices)
        prev_price = prices[-2]
        signals: list[SignalData] = []
        if self._in_position:
            if price < middle:
                signals.append(SignalData(
                    symbol=sym, direction="short", strength=1.0,
                    strategy_id="bb_exit",
                ))
                self._in_position = False
        else:
            breakout = price > upper and prev_price <= upper
            volatile = atr14 > atr50
            if breakout and volatile:
                signals.append(SignalData(
                    symbol=sym, direction="long", strength=1.0,
                    strategy_id="bb_breakout",
                ))
                self._in_position = True
        return signals

    def on_fill(self, fill: Fill, portfolio: PortfolioState) -> None:
        pass


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------


VALIDATION_STRATEGIES: dict[str, tuple[str, Any]] = {
    "momentum": ("Momentum (SMA 50/200)", MomentumCrossover()),
    "ema": ("EMA Crossover (20/50)", EMACrossoverFast()),
    "rsi": ("Mean Reversion (RSI 30)", MeanReversionRSI()),
    "trend": ("Trend Following (EMA+ATR)", TrendFollowing()),
    "bollinger": ("Bollinger Breakout", BollingerBreakout()),
}

DEFAULT_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "IWM", "DIA", "GLD", "TLT"]


# ---------------------------------------------------------------------------
# Validation logic
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    symbol: str
    strategy: str
    strategy_label: str
    start_date: str
    end_date: str
    total_return: float
    annual_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_trades: int
    buy_hold_return: float
    beats_bh: bool
    elapsed_sec: float


def compute_buy_and_hold(symbol: str, start: str, end: str, capital: float) -> float:
    """Buy-and-hold return over the period."""
    try:
        data = load_bars(symbol, start, end, interval="1d")
        if data.empty:
            return 0.0
        first_close = float(data["close"].iloc[0])
        last_close = float(data["close"].iloc[-1])
        return (last_close - first_close) / first_close
    except Exception:
        return 0.0


def run_validation(
    symbol: str,
    strategy_name: str,
    strategy_label: str,
    strategy: Any,
    start: str,
    end: str,
    capital: float,
    slippage_bps: float,
) -> ValidationResult | None:
    """Run a single backtest and return ValidationResult."""
    t0 = time.perf_counter()
    try:
        data = load_combined([symbol], start, end, interval="1d")
        if data.empty:
            return None
        engine = BacktestEngine(
            strategy=strategy,
            initial_cash=capital,
            slippage_bps=slippage_bps,
            sizing_pct=0.25,
        )
        engine.run(data)
        report = engine.summary()
        bh_ret = compute_buy_and_hold(symbol, start, end, capital)
        elapsed = time.perf_counter() - t0
        return ValidationResult(
            symbol=symbol,
            strategy=strategy_name,
            strategy_label=strategy_label,
            start_date=start,
            end_date=end,
            total_return=_safe_float(report.total_return),
            annual_return=_safe_float(report.annual_return),
            sharpe=_safe_float(report.sharpe),
            sortino=_safe_float(report.sortino),
            max_drawdown=_safe_float(report.max_drawdown),
            win_rate=_safe_float(report.win_rate),
            profit_factor=_safe_float(report.profit_factor),
            n_trades=report.n_trades,
            buy_hold_return=_safe_float(bh_ret),
            beats_bh=_safe_float(report.total_return) > _safe_float(bh_ret),
            elapsed_sec=elapsed,
        )
    except Exception as e:
        print(f"    FAILED [{symbol}/{strategy_name}]: {e}", file=sys.stderr)
        return None


def compute_verdict(results: list[ValidationResult]) -> dict[str, Any]:
    """Compute overall verdict from validation results.

    Two paths to finding an edge:
      (a) Sharpe > 1.0 AND beats B&H AND MaxDD < 25%  (need 1+ combos)
      (b) Sharpe > 0.8 AND PF > 1.5 AND MaxDD < 10%   (need 4+ combos)
    """
    n_total = len(results)
    if n_total == 0:
        return {"verdict": "no_data", "message": "No results — all backtests failed."}

    n_sharpe = sum(1 for r in results if r.sharpe > 1.0)
    n_beats_bh = sum(1 for r in results if r.beats_bh)
    n_dd_ok = sum(1 for r in results if abs(r.max_drawdown) < 0.25)
    n_passed_a = sum(
        1 for r in results
        if r.sharpe > 1.0 and r.beats_bh and abs(r.max_drawdown) < 0.25
    )
    n_passed_b = sum(
        1 for r in results
        if r.sharpe > 0.8 and r.profit_factor > 1.5 and abs(r.max_drawdown) < 0.10
    )

    top3 = sorted(results, key=lambda r: r.sharpe, reverse=True)[:3]
    worst3 = sorted(results, key=lambda r: r.sharpe)[:3]

    edge_a = n_passed_a >= 1
    edge_b = n_passed_b >= 4
    edge_found = edge_a or edge_b

    if edge_found:
        verdict = "edge_found"
        parts = []
        if edge_a:
            parts.append(f"Path (a): {n_passed_a} combos (Sharpe>1.0 + beats B&H + MaxDD<25%)")
        if edge_b:
            parts.append(f"Path (b): {n_passed_b} combos (Sharpe>0.8 + PF>1.5 + MaxDD<10%)")
        msg = "Edge found. " + "; ".join(parts) + "."
    else:
        verdict = "no_edge"
        msg = (f"No robust edge found. "
               f"Path (a): {n_passed_a}/1+ combos. "
               f"Path (b): {n_passed_b}/4+ combos. "
               f"Best Sharpe: {top3[0].sharpe:.2f} ({top3[0].symbol}/{top3[0].strategy}).")

    return {
        "verdict": verdict,
        "message": msg,
        "n_total": n_total,
        "n_sharpe_gt_1": n_sharpe,
        "n_beats_bh": n_beats_bh,
        "n_dd_ok": n_dd_ok,
        "n_passed_all": n_passed_a + n_passed_b,
        "n_passed_a": n_passed_a,
        "n_passed_b": n_passed_b,
        "top3": top3,
        "worst3": worst3,
    }


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_results(results: list[ValidationResult], verdict: dict[str, Any]) -> None:
    """Print the full validation report."""
    print("\n" + "=" * 78)
    print("  STRATEGY VALIDATION — REAL HISTORICAL DATA")
    print("  " + "=" * 76)
    print(f"  Strategies: {len(VALIDATION_STRATEGIES)}  "
          f"Symbols: {len(set(r.symbol for r in results))}  "
          f"Total tests: {len(results)}")
    print("=" * 78)

    # Per-symbol summary
    symbols = sorted(set(r.symbol for r in results))
    strategies = sorted(set(r.strategy for r in results))

    print("\n  SHARPE RATIO MATRIX")
    print(f"  {'Symbol':<8s}", end="")
    for s in strategies:
        print(f"{s:>10s}", end="")
    print(f"  {'Best':>10s}")
    print(f"  {'-' * 8}", end="")
    print(f"{'-' * (11 * len(strategies) + 10)}")
    for sym in symbols:
        print(f"  {sym:<8s}", end="")
        sym_results = [r for r in results if r.symbol == sym]
        best = max(r.sharpe for r in sym_results) if sym_results else 0.0
        for s in strategies:
            match = [r for r in sym_results if r.strategy == s]
            if match:
                val = match[0].sharpe
                color = "!" if val >= 1.5 else "+" if val >= 1.0 else "-" if val < 0 else " "
                print(f" {color}{val:>8.2f} ", end="")
            else:
                print(f" {'N/A':>9s} ", end="")
        color = "!" if best >= 1.5 else "+" if best >= 1.0 else " "
        print(f" {color}{best:>8.2f}")

    # Per-strategy details
    print("\n" + "-" * 78)
    print("  DETAILED RESULTS")
    print("  " + "-" * 76)
    header = (f"  {'Symbol':<6s} {'Strategy':<22s} {'Sharpe':>7s} "
              f"{'Ann.Ret':>8s} {'MaxDD':>7s} {'Win%':>6s} "
              f"{'PF':>6s} {'Trades':>6s} {'vs B&H':>8s} {'Pass?':>6s}")
    print(header)
    print(f"  {'-' * 6} {'-' * 22} {'-' * 7} {'-' * 8} {'-' * 7} "
          f"{'-' * 6} {'-' * 6} {'-' * 6} {'-' * 8} {'-' * 6}")
    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        vs_bh = r.total_return - r.buy_hold_return
        passes = "YES" if (r.sharpe > 1.0 and r.beats_bh and abs(r.max_drawdown) < 0.25) else ""
        print(f"  {r.symbol:<6s} {r.strategy_label:<22s} "
              f"{r.sharpe:>7.2f} {r.annual_return:>8.1%} "
              f"{r.max_drawdown:>7.1%} {r.win_rate:>6.1%} "
              f"{r.profit_factor:>6.2f} {r.n_trades:>6d} "
              f"{vs_bh:>+8.1%} {passes:>6s}")

    # Top and worst
    print("\n" + "-" * 78)
    print("  TOP 3 BY SHARPE RATIO")
    for i, r in enumerate(verdict["top3"], 1):
        print(f"    [{i}] {r.symbol}/{r.strategy} — "
              f"Sharpe={r.sharpe:.2f}, Ann.Ret={r.annual_return:.1%}, "
              f"MaxDD={r.max_drawdown:.1%}, Win={r.win_rate:.1%}, "
              f"PF={r.profit_factor:.2f}")

    print("\n  WORST 3 BY SHARPE RATIO")
    for i, r in enumerate(verdict["worst3"], 1):
        print(f"    [{i}] {r.symbol}/{r.strategy} — "
              f"Sharpe={r.sharpe:.2f}, Ann.Ret={r.annual_return:.1%}, "
              f"MaxDD={r.max_drawdown:.1%}")

    # Verdict
    print("\n" + "=" * 78)
    print(f"  VERDICT: {verdict['verdict'].upper().replace('_', ' ')}")
    print(f"  {verdict['message']}")
    print(f"  Path (a) Sharpe>1.0+beats B&H+MaxDD<25%: {verdict.get('n_passed_a', 0)}/{verdict['n_total']}")
    print(f"  Path (b) Sharpe>0.8+PF>1.5+MaxDD<10%:   {verdict.get('n_passed_b', 0)}/{verdict['n_total']}")
    print("=" * 78)

    if verdict["verdict"] == "edge_found":
        # Path (a) candidates
        passed_a = [r for r in results if r.sharpe > 1.0 and r.beats_bh and abs(r.max_drawdown) < 0.25]
        if passed_a:
            print("\n  >>> Path (a) — live trading candidates (Sharpe>1.0 + beats B&H + MaxDD<25%):")
            for r in passed_a:
                print(f"      {r.symbol}/{r.strategy} — "
                      f"Sharpe={r.sharpe:.2f}, Ann.Ret={r.annual_return:.1%}, "
                      f"MaxDD={r.max_drawdown:.1%}")

        # Path (b) candidates
        passed_b = [r for r in results if r.sharpe > 0.8 and r.profit_factor > 1.5 and abs(r.max_drawdown) < 0.10]
        if passed_b:
            print(f"\n  >>> Path (b) — live trading candidates (Sharpe>0.8 + PF>1.5 + MaxDD<10%):")
            for r in passed_b:
                print(f"      {r.symbol}/{r.strategy} — "
                      f"Sharpe={r.sharpe:.2f}, PF={r.profit_factor:.2f}, "
                      f"MaxDD={r.max_drawdown:.1%}, Trades={r.n_trades}")
    else:
        _print_diagnosis(results)


def _print_diagnosis(results: list[ValidationResult]) -> None:
    """Print diagnosis of why strategies are failing."""
    print("\n  DIAGNOSIS:")
    for r in sorted(results, key=lambda x: x.sharpe, reverse=True):
        issues = []
        if r.win_rate < 0.40:
            issues.append(f"Low win rate ({r.win_rate:.0%}) — entry signal may be wrong or inverted")
        if r.win_rate > 0.40 and r.profit_factor < 1.0:
            issues.append(f"Profit factor {r.profit_factor:.2f} — avg loss > avg win, exit/stops need work")
        if r.n_trades < 10:
            issues.append(f"Very few trades ({r.n_trades}) — thresholds too tight")
        if r.n_trades > 200:
            issues.append(f"Too many trades ({r.n_trades}) — churning on noise, widen thresholds")
        if abs(r.max_drawdown) > 0.40:
            issues.append(f"Extreme drawdown ({r.max_drawdown:.1%}) — add/check stop loss")
        if issues:
            print(f"    [{r.symbol}/{r.strategy}]")
            for issue in issues:
                print(f"      -> {issue}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_results_csv(results: list[ValidationResult], verdict: dict[str, Any],
                     output_path: str) -> None:
    """Save results to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "symbol", "strategy", "strategy_label", "start_date", "end_date",
            "total_return", "annual_return", "sharpe", "sortino",
            "max_drawdown", "win_rate", "profit_factor", "n_trades",
            "buy_hold_return", "beats_bh", "elapsed_sec", "passed_all",
        ])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "symbol": r.symbol,
                "strategy": r.strategy,
                "strategy_label": r.strategy_label,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "total_return": f"{r.total_return:.6f}",
                "annual_return": f"{r.annual_return:.6f}",
                "sharpe": f"{r.sharpe:.4f}",
                "sortino": f"{r.sortino:.4f}",
                "max_drawdown": f"{r.max_drawdown:.6f}",
                "win_rate": f"{r.win_rate:.4f}",
                "profit_factor": f"{r.profit_factor:.4f}",
                "n_trades": str(r.n_trades),
                "buy_hold_return": f"{r.buy_hold_return:.6f}",
                "beats_bh": str(r.beats_bh),
                "elapsed_sec": f"{r.elapsed_sec:.1f}",
                "passed_all": str(
                    r.sharpe > 1.0 and r.beats_bh and abs(r.max_drawdown) < 0.25
                ),
            })
    print(f"\n  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    strategy_keys = [s.strip() for s in args.strategies.split(",")]

    strategies_to_run = [
        (key, VALIDATION_STRATEGIES[key][0], VALIDATION_STRATEGIES[key][1])
        for key in strategy_keys if key in VALIDATION_STRATEGIES
    ]
    if not strategies_to_run:
        print(f"No valid strategies. Available: {list(VALIDATION_STRATEGIES)}", file=sys.stderr)
        return 1

    print("=" * 78)
    print("  QUANTCORE STRATEGY VALIDATION")
    print("  " + "=" * 76)
    print(f"  Symbols:    {', '.join(symbols)} ({len(symbols)} total)")
    print(f"  Strategies: {', '.join(k for k, _, _ in strategies_to_run)} "
          f"({len(strategies_to_run)} total)")
    print(f"  Period:     {args.start} -> {args.end}")
    print(f"  Capital:    ${args.capital:,.0f}")
    print(f"  Slippage:   {args.slippage} bps")
    print(f"  Total tests: {len(symbols) * len(strategies_to_run)}")
    print("=" * 78)

    results: list[ValidationResult] = []
    n_total = len(symbols) * len(strategies_to_run)
    n_done = 0

    for sym in symbols:
        print(f"\n{'-' * 78}")
        print(f"  SYMBOL: {sym}")
        print(f"  {'-' * 78}")
        for key, label, strategy in strategies_to_run:
            n_done += 1
            print(f"  [{n_done}/{n_total}] {label}...", end=" ", flush=True)
            result = run_validation(
                symbol=sym,
                strategy_name=key,
                strategy_label=label,
                strategy=strategy,
                start=args.start,
                end=args.end,
                capital=args.capital,
                slippage_bps=args.slippage,
            )
            if result:
                results.append(result)
                print(f"Sharpe={result.sharpe:.2f} "
                      f"Ann.Ret={result.annual_return:.1%} "
                      f"MaxDD={result.max_drawdown:.1%} "
                      f"Trades={result.n_trades} "
                      f"({result.elapsed_sec:.1f}s)")
            else:
                print("FAILED")

    if not results:
        print("\nNo results — all backtests failed.", file=sys.stderr)
        return 1

    # Compute verdict
    verdict = compute_verdict(results)

    # Print report
    print_results(results, verdict)

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "validation_results.csv")
    save_results_csv(results, verdict, csv_path)

    return 0 if verdict["verdict"] == "edge_found" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantCore Strategy Validation — real data backtest matrix"
    )
    parser.add_argument(
        "--symbols", default="SPY,QQQ,AAPL,MSFT,IWM,DIA,GLD,TLT",
        help="Comma-separated symbols (default: SPY,QQQ,AAPL,MSFT,IWM,DIA,GLD,TLT)",
    )
    parser.add_argument(
        "--strategies",
        default="momentum,ema,rsi,trend,bollinger",
        help="Comma-separated strategy keys: momentum,ema,rsi,trend,bollinger",
    )
    parser.add_argument("--start", default="2020-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    parser.add_argument("--capital", type=float, default=1_000_000.0,
                        help="Initial capital")
    parser.add_argument("--slippage", type=float, default=5.0,
                        help="Slippage in bps")
    parser.add_argument("--output-dir", default="outputs",
                        help="Output directory for CSV and report")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
