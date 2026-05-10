"""Phase 2.3: Walk-Forward Validation.

Tests whether the alignment filter's edge is stable across time and not
regime-dependent.

Uses rolling windows:
- Train: 2 years (not used for tuning - just for warm-up)
- Test: 6 months
- Roll forward by 6 months

NO parameter tuning - critical for testing temporal stability.

For each window, runs:
- Baseline (No Filter)
- Alignment Filter
- Random Skip Control (same skip % as alignment)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

# Add repo root to path
REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: Import macro strategies to trigger registration
import quantcore.strategies.macro  # noqa: F401
from quantcore.strategies.registry import get as get_strategy_class
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.risk.metrics import sharpe_ratio
from scripts.real_data_loader import load_universe


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_start: str
    window_end: str
    baseline_sharpe: float = 0.0
    alignment_sharpe: float = 0.0
    random_sharpe: float = 0.0
    baseline_trades: int = 0
    alignment_trades: int = 0
    random_trades: int = 0
    baseline_return: float = 0.0
    alignment_return: float = 0.0
    random_return: float = 0.0
    baseline_dd: float = 0.0
    alignment_dd: float = 0.0
    random_dd: float = 0.0


@dataclass
class Trade:
    """Record of a single trade."""
    symbol: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: int
    size: float
    pnl: float = 0.0


# =============================================================================
# Backtest Simulator (reused from previous phases)
# =============================================================================

class WalkForwardBacktester:
    """Backtest for a specific window of data."""

    def __init__(self, close_prices: np.ndarray, symbols: list[str],
                 start_bar: int, end_bar: int,
                 initial_capital: float = 1_000_000):
        self.close = close_prices
        self.symbols = symbols
        self.start_bar = start_bar
        self.end_bar = end_bar
        self.n_periods = end_bar - start_bar
        self.n_assets = close_prices.shape[1]
        self.initial_capital = initial_capital
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        # State
        self.positions = {}
        self.trades: list[Trade] = []

    def _get_position(self, symbol: str) -> Optional[dict]:
        return self.positions.get(symbol)

    def _open_position(self, signal, bar: int, price: float):
        """Open a new position."""
        symbol = signal.symbol
        direction = signal.direction
        size = signal.size

        self.positions[symbol] = {
            'size': size,
            'entry_price': price,
            'entry_bar': bar,
            'direction': direction,
        }

    def _close_position(self, symbol: str, bar: int, price: float) -> Trade:
        """Close an existing position and record the trade."""
        pos = self.positions.pop(symbol)
        direction = pos['direction']
        entry_price = pos['entry_price']
        size = pos['size']

        if direction == 1:  # Long
            pnl = (price - entry_price) / entry_price * size
        else:  # Short
            pnl = (entry_price - price) / entry_price * size

        trade = Trade(
            symbol=symbol,
            entry_bar=pos['entry_bar'],
            exit_bar=bar,
            entry_price=entry_price,
            exit_price=price,
            direction=direction,
            size=size,
            pnl=pnl,
        )
        return trade

    def run(self, strategy, cost_bps: float = 0) -> tuple[list[float], list[Trade]]:
        """Run backtest over the window. Returns (returns, trades)."""
        self.trades = []
        equity = self.initial_capital
        returns = []

        for bar in range(self.start_bar, self.end_bar):
            # Create data view (include warm-up period for calculation)
            warm_up_start = max(0, bar - 300)  # Include enough history
            data = {
                "close": self.close[warm_up_start:bar+1],
                "symbols": self.symbols,
            }

            strategy.set_timestamp(bar)
            signals = strategy.on_data(data)

            for signal in signals:
                symbol = signal.symbol
                direction = signal.direction
                current_pos = self._get_position(symbol)
                price = self.close[bar, self.symbol_to_idx[symbol]]

                if current_pos is None:
                    self._open_position(signal, bar, price)
                elif current_pos['direction'] != direction:
                    trade = self._close_position(symbol, bar, price)
                    self.trades.append(trade)
                    self._open_position(signal, bar, price)
                # else: same direction, do nothing

            # Close positions at end of window
            if bar == self.end_bar - 1:
                for symbol in list(self.positions.keys()):
                    price = self.close[bar, self.symbol_to_idx[symbol]]
                    trade = self._close_position(symbol, bar, price)
                    self.trades.append(trade)

        # Calculate returns from trades
        trade_by_bar = {}
        for trade in self.trades:
            if trade.exit_bar not in trade_by_bar:
                trade_by_bar[trade.exit_bar] = []
            trade_by_bar[trade.exit_bar].append(trade)

        for bar in range(self.start_bar + 1, self.end_bar):
            bar_pnl = sum(t.pnl for t in trade_by_bar.get(bar, []))
            if equity > 0:
                ret = bar_pnl / equity
                returns.append(ret)
            equity += bar_pnl

        return returns, self.trades


class RandomSkipWFBacktester(WalkForwardBacktester):
    """Walk-forward backtester with random skip."""

    def run_with_random_skip(self, strategy, skip_pct: float,
                              random_seed: int = 42,
                              cost_bps: float = 0) -> tuple[list[float], list[Trade]]:
        """Run backtest with random trade skipping."""
        rng = np.random.RandomState(random_seed)

        self.trades = []
        equity = self.initial_capital
        returns = []

        for bar in range(self.start_bar, self.end_bar):
            warm_up_start = max(0, bar - 300)
            data = {
                "close": self.close[warm_up_start:bar+1],
                "symbols": self.symbols,
            }

            strategy.set_timestamp(bar)
            signals = strategy.on_data(data)

            for signal in signals:
                # Randomly skip
                if rng.random() < skip_pct:
                    continue

                symbol = signal.symbol
                direction = signal.direction
                current_pos = self._get_position(symbol)
                price = self.close[bar, self.symbol_to_idx[symbol]]

                if current_pos is None:
                    self._open_position(signal, bar, price)
                elif current_pos['direction'] != direction:
                    trade = self._close_position(symbol, bar, price)
                    self.trades.append(trade)
                    self._open_position(signal, bar, price)

            if bar == self.end_bar - 1:
                for symbol in list(self.positions.keys()):
                    price = self.close[bar, self.symbol_to_idx[symbol]]
                    trade = self._close_position(symbol, bar, price)
                    self.trades.append(trade)

        # Calculate returns
        trade_by_bar = {}
        for trade in self.trades:
            if trade.exit_bar not in trade_by_bar:
                trade_by_bar[trade.exit_bar] = []
            trade_by_bar[trade.exit_bar].append(trade)

        for bar in range(self.start_bar + 1, self.end_bar):
            bar_pnl = sum(t.pnl for t in trade_by_bar.get(bar, []))
            if equity > 0:
                ret = bar_pnl / equity
                returns.append(ret)
            equity += bar_pnl

        return returns, self.trades


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_window_metrics(returns: list[float], trades: list[Trade]) -> dict:
    """Calculate metrics for a single window."""
    if len(returns) < 2 or len(trades) == 0:
        return {
            'sharpe': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'n_trades': 0,
        }

    returns_arr = np.array(returns)

    # Sharpe ratio
    sharpe = sharpe_ratio(returns_arr)

    # Total return (approximate from returns)
    total_return = np.prod(1 + returns_arr) - 1

    # Max drawdown
    equity = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'n_trades': len(trades),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_walk_forward_validation():
    """Run the full walk-forward validation experiment."""
    print("=" * 70)
    print("PHASE 2.3: WALK-FORWARD VALIDATION")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_universe('cta', '2018-01-01', '2024-12-31', use_cache=True)

    # Pivot to wide format
    print("Pivoting to wide format...")
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close')
    wide_df = wide_df.dropna()
    close_prices = wide_df.values
    symbols = [col for col in wide_df.columns]
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    print(f"Data shape: {close_prices.shape}")
    print(f"Date range: {wide_df.index[0]} to {wide_df.index[-1]}")

    # Get strategy class
    strategy_cls = get_strategy_class("macro_cta_trend")

    # Define walk-forward windows
    # Train: 2 years (504 trading days), Test: 6 months (126 trading days)
    # Roll forward by 6 months
    TRAIN_DAYS = 504
    TEST_DAYS = 126
    ROLL_DAYS = 126

    n_periods = close_prices.shape[0]

    # Calculate windows
    windows = []
    train_start = 200  # Start after warm-up
    while train_start + TRAIN_DAYS + TEST_DAYS <= n_periods:
        train_end = train_start + TRAIN_DAYS
        test_start = train_end
        test_end = test_start + TEST_DAYS

        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'window_label': f"{wide_df.index[test_start].strftime('%Y-%m')} to {wide_df.index[test_end-1].strftime('%Y-%m')}"
        })

        train_start += ROLL_DAYS

    print(f"\nNumber of walk-forward windows: {len(windows)}")
    for w in windows:
        print(f"  {w['window_label']}")

    # First, determine skip percentage from a full run with alignment filter
    print("\n" + "=" * 70)
    print("PRE-RUN: Determine skip percentage")
    print("=" * 70)

    config_pre = StrategyConfig(
        name="alignment_pre",
        strategy_type=StrategyType.MACRO,
        params={
            "alignment_filter": True,
            "min_signal_strength": 0.0,
        },
        max_position_notional=1_000_000,
    )
    strategy_pre = strategy_cls(config_pre)

    # Run on full dataset to get skip percentage
    backtester_pre = WalkForwardBacktester(
        close_prices, symbols, 200, n_periods
    )
    returns_pre, trades_pre = backtester_pre.run(strategy_pre)
    n_skipped = strategy_pre.get_skipped_opportunities()
    n_executed = len(trades_pre)
    n_total = n_executed + n_skipped
    skip_pct = n_skipped / n_total if n_total > 0 else 0

    print(f"Skip percentage: {skip_pct:.1%}")

    # Run walk-forward validation
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)

    window_results = []

    for i, window in enumerate(windows):
        test_start = window['test_start']
        test_end = window['test_end']
        window_label = window['window_label']

        print(f"\n--- Window {i+1}: {window_label} ---")

        # ---------------------------------------------------------------------
        # Run 1: Baseline (No Filter)
        # ---------------------------------------------------------------------
        config_baseline = StrategyConfig(
            name="baseline",
            strategy_type=StrategyType.MACRO,
            params={
                "alignment_filter": False,
                "min_signal_strength": 0.0,
            },
            max_position_notional=1_000_000,
        )
        strategy_baseline = strategy_cls(config_baseline)
        backtester_baseline = WalkForwardBacktester(
            close_prices, symbols, test_start, test_end
        )
        returns_baseline, trades_baseline = backtester_baseline.run(strategy_baseline)
        metrics_baseline = calculate_window_metrics(returns_baseline, trades_baseline)

        # ---------------------------------------------------------------------
        # Run 2: Alignment Filter
        # ---------------------------------------------------------------------
        config_alignment = StrategyConfig(
            name="alignment",
            strategy_type=StrategyType.MACRO,
            params={
                "alignment_filter": True,
                "min_signal_strength": 0.0,
            },
            max_position_notional=1_000_000,
        )
        strategy_alignment = strategy_cls(config_alignment)
        backtester_alignment = WalkForwardBacktester(
            close_prices, symbols, test_start, test_end
        )
        returns_alignment, trades_alignment = backtester_alignment.run(strategy_alignment)
        metrics_alignment = calculate_window_metrics(returns_alignment, trades_alignment)

        # ---------------------------------------------------------------------
        # Run 3: Random Skip Control
        # ---------------------------------------------------------------------
        config_random = StrategyConfig(
            name="random_skip",
            strategy_type=StrategyType.MACRO,
            params={
                "alignment_filter": False,
                "min_signal_strength": 0.0,
            },
            max_position_notional=1_000_000,
        )
        strategy_random = strategy_cls(config_random)
        backtester_random = RandomSkipWFBacktester(
            close_prices, symbols, test_start, test_end
        )
        returns_random, trades_random = backtester_random.run_with_random_skip(
            strategy_random, skip_pct, random_seed=42
        )
        metrics_random = calculate_window_metrics(returns_random, trades_random)

        # Store results
        result = WindowResult(
            window_start=window_label.split(' to ')[0],
            window_end=window_label.split(' to ')[1],
            baseline_sharpe=metrics_baseline['sharpe'],
            alignment_sharpe=metrics_alignment['sharpe'],
            random_sharpe=metrics_random['sharpe'],
            baseline_trades=metrics_baseline['n_trades'],
            alignment_trades=metrics_alignment['n_trades'],
            random_trades=metrics_random['n_trades'],
            baseline_return=metrics_baseline['total_return'],
            alignment_return=metrics_alignment['total_return'],
            random_return=metrics_random['total_return'],
            baseline_dd=metrics_baseline['max_drawdown'],
            alignment_dd=metrics_alignment['max_drawdown'],
            random_dd=metrics_random['max_drawdown'],
        )
        window_results.append(result)

        print(f"  Baseline:   Sharpe={metrics_baseline['sharpe']:.4f}, Trades={metrics_baseline['n_trades']}")
        print(f"  Alignment:   Sharpe={metrics_alignment['sharpe']:.4f}, Trades={metrics_alignment['n_trades']}")
        print(f"  Random Skip: Sharpe={metrics_random['sharpe']:.4f}, Trades={metrics_random['n_trades']}")

    # =========================================================================
    # Output Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS TABLE")
    print("=" * 70)

    header = f"{'Window':<20} {'Base SR':>8} {'Align SR':>9} {'Rand SR':>8} {'Align>Base':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in window_results:
        window_label = f"{r.window_start} to {r.window_end}"
        align_beats_base = r.alignment_sharpe > r.baseline_sharpe
        print(f"{window_label:<20} {r.baseline_sharpe:>8.4f} {r.alignment_sharpe:>9.4f} "
              f"{r.random_sharpe:>8.4f} {'Yes' if align_beats_base else 'No':>10}")

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    baseline_sharpes = [r.baseline_sharpe for r in window_results]
    alignment_sharpes = [r.alignment_sharpe for r in window_results]
    random_sharpes = [r.random_sharpe for r in window_results]

    print(f"\n{'Metric':<25} {'Baseline':>10} {'Alignment':>10} {'Random':>10}")
    print("-" * 55)

    print(f"{'Mean Sharpe':<25} {np.mean(baseline_sharpes):>10.4f} "
          f"{np.mean(alignment_sharpes):>10.4f} {np.mean(random_sharpes):>10.4f}")

    print(f"{'Std Dev Sharpe':<25} {np.std(baseline_sharpes):>10.4f} "
          f"{np.std(alignment_sharpes):>10.4f} {np.std(random_sharpes):>10.4f}")

    print(f"{'Min Sharpe':<25} {np.min(baseline_sharpes):>10.4f} "
          f"{np.min(alignment_sharpes):>10.4f} {np.min(random_sharpes):>10.4f}")

    print(f"{'Max Sharpe':<25} {np.max(baseline_sharpes):>10.4f} "
          f"{np.max(alignment_sharpes):>10.4f} {np.max(random_sharpes):>10.4f}")

    # % of periods alignment beats baseline
    n_beats = sum(1 for r in window_results if r.alignment_sharpe > r.baseline_sharpe)
    pct_beats = n_beats / len(window_results) * 100 if window_results else 0

    print(f"\nAlignment beats Baseline in {n_beats}/{len(window_results)} periods ({pct_beats:.1f}%)")

    # % of periods with positive Sharpe
    n_pos_align = sum(1 for r in window_results if r.alignment_sharpe > 0)
    pct_pos = n_pos_align / len(window_results) * 100 if window_results else 0

    print(f"Alignment has positive Sharpe in {n_pos_align}/{len(window_results)} periods ({pct_pos:.1f}%)")

    # Alignment beats random
    n_beats_random = sum(1 for r in window_results if r.alignment_sharpe > r.random_sharpe)
    pct_beats_random = n_beats_random / len(window_results) * 100 if window_results else 0

    print(f"Alignment beats Random Skip in {n_beats_random}/{len(window_results)} periods ({pct_beats_random:.1f}%)")

    # =========================================================================
    # Decision Framework
    # =========================================================================
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)

    # Criteria
    stable_edge = pct_beats >= 60  # Beats baseline in >60% of periods
    consistent_positive = pct_pos >= 70  # Positive Sharpe in >70% of periods
    beats_random_consistently = pct_beats_random >= 60

    # Check for regime dependency (large variance)
    align_std = np.std(alignment_sharpes)
    low_variance = align_std < 0.5  # Reasonable threshold

    print(f"\n* Alignment beats Baseline in >60% periods: {stable_edge} ({pct_beats:.1f}%)")
    print(f"* Alignment Sharpe positive in >70% periods: {consistent_positive} ({pct_pos:.1f}%)")
    print(f"* Alignment beats Random Skip in >60% periods: {beats_random_consistently} ({pct_beats_random:.1f}%)")
    print(f"* Low variance (std < 0.5): {low_variance} (std={align_std:.4f})")

    if stable_edge and consistent_positive and beats_random_consistently:
        print("\n>>> STRONG / REAL EDGE - Stable across time!")
        print("    - Proceed to combine filters (alignment + strength)")
    elif stable_edge and not consistent_positive:
        print("\n>>> MARGINAL EDGE - Some instability")
        print("    - Consider regime conditioning")
    else:
        print("\n>>> FRAGILE EDGE / OVERFIT")
        print("    - Edge may be regime-dependent")
        print("    - Investigate specific periods of underperformance")

    return window_results


if __name__ == "__main__":
    results = run_walk_forward_validation()
