"""Alignment Filter Validation Experiment.

Tests whether fast/slow alignment filter provides real predictive edge
or just improves performance by reducing trade frequency.

Three configurations:
1. Baseline (no filter)
2. Alignment filter (only trade when fast/slow agree)
3. Random skip control (skip same % of trades randomly)

Outputs comparison tables and diagnostics.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

# Add repo root to path
REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: Import macro strategies to trigger registration
import quantcore.strategies.macro  # noqa: F401
from quantcore.strategies.registry import get as get_strategy_class
from quantcore.strategies.base import StrategyConfig, StrategyType, TradeSignal
from quantcore.risk.metrics import sharpe_ratio
from scripts.real_data_loader import load_universe


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Trade:
    """Record of a single trade."""
    symbol: str
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: int  # 1 for long, -1 for short
    size: float
    pnl: float = 0.0
    was_skipped: bool = False  # True if this trade was skipped (for simulation)
    fast_signal: float = 0.0
    slow_signal: float = 0.0


@dataclass
class RunResults:
    """Results from a single backtest run."""
    name: str
    equity_curve: list[float]
    returns: list[float]
    trades: list[Trade]
    n_signals_total: int
    n_trades_executed: int


# =============================================================================
# Simple Backtest Simulator
# =============================================================================

class SimpleBacktester:
    """Simple backtest that tracks positions and trade-level PnL."""

    def __init__(self, close_prices: np.ndarray, symbols: list[str],
                 initial_capital: float = 1_000_000):
        self.close = close_prices  # Shape: (n_periods, n_assets)
        self.symbols = symbols
        self.n_periods, self.n_assets = close_prices.shape
        self.initial_capital = initial_capital
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        # State
        self.positions = {}  # symbol -> {'size': float, 'entry_price': float, 'entry_bar': int, 'direction': int}
        self.trades: list[Trade] = []

    def _get_position(self, symbol: str) -> Optional[dict]:
        return self.positions.get(symbol)

    def _open_position(self, signal: TradeSignal, bar: int, price: float):
        """Open a new position based on signal."""
        symbol = signal.symbol
        direction = signal.direction  # 1 or -1
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

        # Calculate PnL
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

    def run(self, strategy, bar_data_fn) -> RunResults:
        """Run backtest with the given strategy."""
        n_signals_total = 0
        n_trades_executed = 0
        self.trades = []

        for bar in range(200, self.n_periods):  # Start after warm-up
            # Get data for current bar
            data = bar_data_fn(bar)

            # Set timestamp for strategy
            strategy.set_timestamp(bar)

            # Generate signals
            signals = strategy.on_data(data)

            for signal in signals:
                n_signals_total += 1
                symbol = signal.symbol
                direction = signal.direction
                current_pos = self._get_position(symbol)

                # Execute the trade
                price = self.close[bar, self.symbol_to_idx[symbol]]

                if current_pos is None:
                    # New position
                    self._open_position(signal, bar, price)
                    n_trades_executed += 1
                elif current_pos['direction'] != direction:
                    # Close old position, open new one
                    trade = self._close_position(symbol, bar, price)
                    self.trades.append(trade)
                    n_trades_executed += 1

                    # Open new position
                    self._open_position(signal, bar, price)
                # else: same direction, do nothing (pyramiding disabled)

            # Close positions at end of backtest
            if bar == self.n_periods - 1:
                for symbol in list(self.positions.keys()):
                    price = self.close[bar, self.symbol_to_idx[symbol]]
                    trade = self._close_position(symbol, bar, price)
                    self.trades.append(trade)

        # Calculate equity curve and returns
        equity = self.initial_capital
        equity_curve = [equity]
        returns = []

        # Reconstruct equity curve from trades
        trade_by_bar = {}
        for trade in self.trades:
            if trade.exit_bar not in trade_by_bar:
                trade_by_bar[trade.exit_bar] = []
            trade_by_bar[trade.exit_bar].append(trade)

        for bar in range(1, self.n_periods):
            bar_pnl = sum(t.pnl for t in trade_by_bar.get(bar, []))
            equity += bar_pnl
            equity_curve.append(equity)
            if bar > 0 and equity_curve[-2] > 0:
                ret = bar_pnl / equity_curve[-2]
                returns.append(ret)

        return RunResults(
            name=strategy.config.name,
            equity_curve=equity_curve,
            returns=returns,
            trades=self.trades,
            n_signals_total=n_signals_total,
            n_trades_executed=n_trades_executed,
        )


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_metrics(results: RunResults) -> dict:
    """Calculate performance metrics from backtest results."""
    returns = np.array(results.returns)
    trades = results.trades

    if len(returns) < 2:
        return {
            'sharpe': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'n_trades': 0,
            'avg_pnl': 0.0,
            'win_rate': 0.0,
            'whipsaw_rate': 0.0,
        }

    # Sharpe ratio
    sharpe = sharpe_ratio(returns)

    # Total return
    total_return = (results.equity_curve[-1] / results.equity_curve[0] - 1)

    # Max drawdown
    equity = np.array(results.equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    # Trade metrics
    n_trades = len(trades)
    if n_trades > 0:
        avg_pnl = np.mean([t.pnl for t in trades])
        win_rate = sum(1 for t in trades if t.pnl > 0) / n_trades
        # Whipsaw: exit within 5 bars
        whipsaw_rate = sum(1 for t in trades if t.exit_bar - t.entry_bar <= 5) / n_trades
    else:
        avg_pnl = 0.0
        win_rate = 0.0
        whipsaw_rate = 0.0

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
        'whipsaw_rate': whipsaw_rate,
    }


def simulate_skipped_trades(strategy, close_prices, symbol_to_idx, n_periods, symbols) -> list[Trade]:
    """Simulate PnL for skipped trades from strategy's internal tracking.
    
    For proper simulation, we need to track when the strategy would have
    exited these positions. Since we don't have that data directly,
    we'll use a 5-bar holding period for whipsaw calculation.
    """
    skipped_signals = strategy.get_skipped_signals()
    simulated = []

    for sig in skipped_signals:
        symbol = sig['asset']
        if symbol not in symbol_to_idx:
            continue

        # Estimate entry bar from timestamp
        # The timestamp is stored as _current_timestamp in the strategy
        # We need to find which bar this trade would have occurred at
        entry_bar = sig.get('_bar_index', 200)
        direction = sig['direction']
        entry_price = sig['entry_price']

        # For proper exit simulation, we'd need to run the strategy without
        # the alignment filter and track when these positions would close.
        # For now, use a realistic holding period based on rebalance_freq (default 5)
        # and add some randomness to simulate realistic exits.
        
        # Use 5 bars for whipsaw calculation (standard definition)
        # This is conservative - most skipped trades would be whipsaws
        exit_bar = min(entry_bar + 5, n_periods - 1)  # 5 bars = whipsaw threshold
        exit_price = close_prices[exit_bar, symbol_to_idx[symbol]]

        size = sig.get('position', 100000)

        if direction == 1:  # Long
            pnl = (exit_price - entry_price) / entry_price * size
        else:  # Short
            pnl = (entry_price - exit_price) / entry_price * size

        t = Trade(
            symbol=symbol,
            entry_bar=entry_bar,
            exit_bar=exit_bar,
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            size=size,
            pnl=pnl,
            was_skipped=True,
            fast_signal=sig.get('fast_signal', 0.0),
            slow_signal=sig.get('slow_signal', 0.0),
        )
        simulated.append(t)

    return simulated


# =============================================================================
# Main Experiment
# =============================================================================

def run_alignment_validation():
    """Run the full alignment filter validation experiment."""
    print("=" * 70)
    print("ALIGNMENT FILTER VALIDATION EXPERIMENT")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = load_universe('cta', '2018-01-01', '2024-12-31', use_cache=True)

    # Pivot to wide format
    print("Pivoting to wide format...")
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close')
    wide_df = wide_df.dropna()
    close_prices = wide_df.values
    symbols = [col for col in wide_df.columns]  # Keep as is, don't add 'close_' prefix
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    print(f"Data shape: {close_prices.shape}")
    print(f"Symbols: {symbols}")

    # Get strategy class
    strategy_cls = get_strategy_class("macro_cta_trend")

    # Bar data function
    def get_bar_data(bar: int) -> dict:
        return {
            "close": close_prices[:bar+1],  # Use all data up to current bar
            "symbols": symbols,
        }

    # =========================================================================
    # Run 1: Baseline (No Filter)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUN 1: BASELINE (No Filter)")
    print("=" * 70)

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
    backtester_baseline = SimpleBacktester(close_prices, symbols)
    results_baseline = backtester_baseline.run(strategy_baseline, get_bar_data)

    metrics_baseline = calculate_metrics(results_baseline)
    print(f"Sharpe: {metrics_baseline['sharpe']:.4f}")
    print(f"Total Return: {metrics_baseline['total_return']:.2%}")
    print(f"Max Drawdown: {metrics_baseline['max_drawdown']:.2%}")
    print(f"Trades: {metrics_baseline['n_trades']}")
    print(f"Avg PnL: {metrics_baseline['avg_pnl']:.2f}")
    print(f"Win Rate: {metrics_baseline['win_rate']:.2%}")
    print(f"Whipsaw Rate: {metrics_baseline['whipsaw_rate']:.2%}")

    # =========================================================================
    # Run 2: Alignment Filter
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUN 2: ALIGNMENT FILTER")
    print("=" * 70)

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
    backtester_alignment = SimpleBacktester(close_prices, symbols)
    results_alignment = backtester_alignment.run(strategy_alignment, get_bar_data)

    metrics_alignment = calculate_metrics(results_alignment)
    print(f"Sharpe: {metrics_alignment['sharpe']:.4f}")
    print(f"Total Return: {metrics_alignment['total_return']:.2%}")
    print(f"Max Drawdown: {metrics_alignment['max_drawdown']:.2%}")
    print(f"Trades: {metrics_alignment['n_trades']}")
    print(f"Avg PnL: {metrics_alignment['avg_pnl']:.2f}")
    print(f"Win Rate: {metrics_alignment['win_rate']:.2%}")
    print(f"Whipsaw Rate: {metrics_alignment['whipsaw_rate']:.2%}")

    # Get skipped trade count from strategy
    n_skipped_alignment = strategy_alignment.get_skipped_opportunities()
    print(f"Skipped trades (from strategy): {n_skipped_alignment}")

    # Calculate skip percentage for random control
    n_total = results_alignment.n_trades_executed + n_skipped_alignment
    skip_pct = n_skipped_alignment / n_total if n_total > 0 else 0
    print(f"Skip percentage: {skip_pct:.2%}")

    # Simulate skipped trades for diagnostics (using 5-bar exit for whipsaw)
    skipped_signals = strategy_alignment.get_skipped_signals()
    skipped_trades_sim = simulate_skipped_trades(
        strategy_alignment, close_prices, symbol_to_idx, len(close_prices), symbols
    )

    # Alignment diagnostics - compare executed vs skipped
    all_opps = strategy_alignment.get_all_opportunities()
    
    executed_pnls = []
    skipped_pnls = []
    
    # We need to simulate PnL for both executed and skipped trades
    # For executed trades, use actual PnL from results_alignment.trades
    # For skipped trades, simulate with 5-bar holding (whipsaw threshold)
    
    executed_pnls = [t.pnl for t in results_alignment.trades]
    
    # For skipped trades, simulate PnL
    # Use the skipped_signals which has entry info
    skipped_pnls = []
    for sig in skipped_signals:
        symbol = sig['asset']
        if symbol not in symbol_to_idx:
            continue
        entry_price = sig['entry_price']
        direction = sig['direction']
        entry_bar = sig.get('_bar_index', 200)
        
        # Simulate 5-bar holding (whipsaw definition)
        exit_bar = min(entry_bar + 5, len(close_prices) - 1)
        exit_price = close_prices[exit_bar, symbol_to_idx[symbol]]
        size = sig.get('position', 100000)
        
        if direction == 1:
            pnl = (exit_price - entry_price) / entry_price * size
        else:
            pnl = (entry_price - exit_price) / entry_price * size
        skipped_pnls.append(pnl)
    
    diag = {
        'aligned_avg_pnl': np.mean(executed_pnls) if executed_pnls else 0.0,
        'misaligned_avg_pnl': np.mean(skipped_pnls) if skipped_pnls else 0.0,
        'n_aligned': len(results_alignment.trades),
        'n_misaligned': len(skipped_pnls),
    }
    
    print(f"\nAlignment Diagnostics:")
    print(f"  Aligned (executed) trades: {diag['n_aligned']}")
    print(f"  Misaligned (skipped) trades: {diag['n_misaligned']}")
    print(f"  Aligned Avg PnL: {diag['aligned_avg_pnl']:.2f}")
    print(f"  Misaligned Avg PnL: {diag['misaligned_avg_pnl']:.2f}")

    # =========================================================================
    # Run 3: Random Skip Control
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUN 3: RANDOM SKIP CONTROL")
    print("=" * 70)
    print(f"(Skipping {skip_pct:.1%} of trades randomly with seed=42)")

    # For random skip, we need to run the strategy and randomly skip trades
    # This requires modifying the backtester to support random skipping
    # Let's create a modified version

    class RandomSkipBacktester(SimpleBacktester):
        def run_with_random_skip(self, strategy, bar_data_fn, skip_pct: float, random_seed: int = 42) -> RunResults:
            """Run backtest with random trade skipping."""
            rng = np.random.RandomState(random_seed)

            n_signals_total = 0
            n_trades_executed = 0
            self.trades = []

            for bar in range(200, self.n_periods):  # Start after warm-up
                data = bar_data_fn(bar)
                strategy.set_timestamp(bar)
                signals = strategy.on_data(data)

                for signal in signals:
                    n_signals_total += 1

                    # Randomly skip
                    if rng.random() < skip_pct:
                        continue

                    symbol = signal.symbol
                    direction = signal.direction
                    current_pos = self._get_position(symbol)

                    price = self.close[bar, self.symbol_to_idx[symbol]]

                    if current_pos is None:
                        self._open_position(signal, bar, price)
                        n_trades_executed += 1
                    elif current_pos['direction'] != direction:
                        trade = self._close_position(symbol, bar, price)
                        self.trades.append(trade)
                        n_trades_executed += 1
                        self._open_position(signal, bar, price)
                    elif bar == self.n_periods - 1:
                        for symbol in list(self.positions.keys()):
                            price = self.close[bar, self.symbol_to_idx[symbol]]
                            trade = self._close_position(symbol, bar, price)
                            self.trades.append(trade)

            # Calculate equity curve and returns (same as parent class)
            equity = self.initial_capital
            equity_curve = [equity]
            returns = []

            trade_by_bar = {}
            for trade in self.trades:
                if trade.exit_bar not in trade_by_bar:
                    trade_by_bar[trade.exit_bar] = []
                trade_by_bar[trade.exit_bar].append(trade)

            for bar in range(1, self.n_periods):
                bar_pnl = sum(t.pnl for t in trade_by_bar.get(bar, []))
                equity += bar_pnl
                equity_curve.append(equity)
                if bar > 0 and equity_curve[-2] > 0:
                    ret = bar_pnl / equity_curve[-2]
                    returns.append(ret)

            return RunResults(
                name=strategy.config.name,
                equity_curve=equity_curve,
                returns=returns,
                trades=self.trades,
                n_signals_total=n_signals_total,
                n_trades_executed=n_trades_executed,
            )

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
    backtester_random = RandomSkipBacktester(close_prices, symbols)
    results_random = backtester_random.run_with_random_skip(
        strategy_random, get_bar_data, skip_pct, random_seed=42
    )

    metrics_random = calculate_metrics(results_random)
    print(f"Sharpe: {metrics_random['sharpe']:.4f}")
    print(f"Total Return: {metrics_random['total_return']:.2%}")
    print(f"Max Drawdown: {metrics_random['max_drawdown']:.2%}")
    print(f"Trades: {metrics_random['n_trades']}")
    print(f"Avg PnL: {metrics_random['avg_pnl']:.2f}")
    print(f"Win Rate: {metrics_random['win_rate']:.2%}")
    print(f"Whipsaw Rate: {metrics_random['whipsaw_rate']:.2%}")

    # =========================================================================
    # Comparison Tables
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    header = f"{'Strategy':<15} {'Sharpe':>8} {'Trades':>7} {'Avg PnL':>9} {'Whipsaw%':>9} {'Max DD':>9}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, metrics in [("Baseline", metrics_baseline),
                          ("Alignment", metrics_alignment),
                          ("Random Skip", metrics_random)]:
        print(f"{name:<15} {metrics['sharpe']:>8.4f} {metrics['n_trades']:>7} "
              f"{metrics['avg_pnl']:>9.2f} {metrics['whipsaw_rate']:>8.2%} "
              f"{metrics['max_drawdown']:>8.2%}")

    # Diagnostic table
    print("\n" + "=" * 70)
    print("DIAGNOSTIC TABLE (Alignment Filter Run)")
    print("=" * 70)

    print(f"\n{'Trade Type':<20} {'Avg PnL':>9}")
    print("-" * 30)
    print(f"{'Aligned (executed)':<20} {diag['aligned_avg_pnl']:>9.2f}")
    print(f"{'Misaligned (skip)':<20} {diag['misaligned_avg_pnl']:>9.2f}")

    # =========================================================================
    # Decision Framework
    # =========================================================================
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)

    # Check criteria (using reliable metrics only)
    # Whipsaw comparison removed - unreliable for simulated skipped trades
    sharpe_improvement = metrics_alignment['sharpe'] > metrics_baseline['sharpe']
    beats_random = metrics_alignment['sharpe'] > metrics_random['sharpe']
    avg_pnl_improvement = diag['aligned_avg_pnl'] > diag['misaligned_avg_pnl']
    
    # Whipsaw: compare alignment vs baseline (both use actual trades)
    whipsaw_reduction = metrics_alignment['whipsaw_rate'] < metrics_baseline['whipsaw_rate']
    
    print(f"\n* Sharpe improves vs baseline: {sharpe_improvement}")
    print(f"* Beats random skipping: {beats_random}")
    print(f"* Avg PnL improves (aligned > misaligned): {avg_pnl_improvement}")
    print(f"* Whipsaw decreases (alignment vs baseline): {whipsaw_reduction}")

    if sharpe_improvement and beats_random and avg_pnl_improvement:
        print("\n>>> STRONG SIGNAL - Keep & Expand alignment filter!")
    elif sharpe_improvement and not beats_random:
        print("\n>>> WEAK SIGNAL - Improvement may just be from trading less")
    else:
        print("\n>>> NO SIGNAL / MISLEADING - Reject alignment hypothesis")

    return {
        'baseline': metrics_baseline,
        'alignment': metrics_alignment,
        'random': metrics_random,
        'diagnostics': diag,
    }


if __name__ == "__main__":
    results = run_alignment_validation()
