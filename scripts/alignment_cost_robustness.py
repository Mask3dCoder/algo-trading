"""Phase 2.2: Transaction Cost Robustness Test.

Verifies that the alignment filter's edge is real and tradable by testing
performance under realistic transaction costs and slippage.

Re-runs the same 3 configurations at different cost levels:
- Baseline (No Filter)
- Alignment Filter
- Random Skip Control

Cost levels tested: 0 bps, 10 bps, 25 bps, 50 bps
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

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
    entry_cost: float = 0.0
    exit_cost: float = 0.0


@dataclass
class RunResults:
    """Results from a single backtest run."""
    name: str
    cost_bps: float
    equity_curve: list[float]
    returns: list[float]
    trades: list[Trade]
    n_trades_executed: int


# =============================================================================
# Backtest Simulator with Transaction Costs
# =============================================================================

class CostAwareBacktester:
    """Backtest that tracks positions, trade-level PnL, and transaction costs."""

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

    def _get_position(self, symbol: str) -> dict | None:
        return self.positions.get(symbol)

    def _open_position(self, signal, bar: int, price: float, cost_bps: float) -> float:
        """Open a new position and return the cost."""
        symbol = signal.symbol
        direction = signal.direction
        size = signal.size

        # Calculate entry cost
        notional = size  # This is already the notional value
        entry_cost = notional * (cost_bps / 10000)

        self.positions[symbol] = {
            'size': size,
            'entry_price': price,
            'entry_bar': bar,
            'direction': direction,
        }

        return entry_cost

    def _close_position(self, symbol: str, bar: int, price: float, cost_bps: float) -> Trade:
        """Close an existing position, record trade with costs."""
        pos = self.positions.pop(symbol)
        direction = pos['direction']
        entry_price = pos['entry_price']
        size = pos['size']

        # Calculate gross PnL
        if direction == 1:  # Long
            gross_pnl = (price - entry_price) / entry_price * size
        else:  # Short
            gross_pnl = (entry_price - price) / entry_price * size

        # Calculate exit cost
        notional = size
        exit_cost = notional * (cost_bps / 10000)

        # Net PnL after costs
        net_pnl = gross_pnl - exit_cost

        trade = Trade(
            symbol=symbol,
            entry_bar=pos['entry_bar'],
            exit_bar=bar,
            entry_price=entry_price,
            exit_price=price,
            direction=direction,
            size=size,
            pnl=net_pnl,
            entry_cost=exit_cost,  # Actually entry cost - will fix
            exit_cost=exit_cost,
        )
        return trade, exit_cost

    def run(self, strategy, bar_data_fn, cost_bps: float = 0) -> RunResults:
        """Run backtest with the given strategy and transaction costs."""
        n_trades_executed = 0
        self.trades = []
        total_entry_costs = 0.0
        total_exit_costs = 0.0

        for bar in range(200, self.n_periods):  # Start after warm-up
            data = bar_data_fn(bar)
            strategy.set_timestamp(bar)
            signals = strategy.on_data(data)

            for signal in signals:
                symbol = signal.symbol
                direction = signal.direction
                current_pos = self._get_position(symbol)
                price = self.close[bar, self.symbol_to_idx[symbol]]

                if current_pos is None:
                    # New position - pay entry cost
                    entry_cost = self._open_position(signal, bar, price, cost_bps)
                    total_entry_costs += entry_cost
                    n_trades_executed += 1
                elif current_pos['direction'] != direction:
                    # Close old position (pay exit cost), open new one (pay entry cost)
                    trade, exit_cost = self._close_position(symbol, bar, price, cost_bps)
                    self.trades.append(trade)
                    total_exit_costs += exit_cost
                    n_trades_executed += 1

                    # Open new position - pay entry cost
                    entry_cost = self._open_position(signal, bar, price, cost_bps)
                    total_entry_costs += entry_cost
                # else: same direction, do nothing

            # Close positions at end of backtest
            if bar == self.n_periods - 1:
                for symbol in list(self.positions.keys()):
                    price = self.close[bar, self.symbol_to_idx[symbol]]
                    trade, exit_cost = self._close_position(symbol, bar, price, cost_bps)
                    self.trades.append(trade)
                    total_exit_costs += exit_cost

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
            cost_bps=cost_bps,
            equity_curve=equity_curve,
            returns=returns,
            trades=self.trades,
            n_trades_executed=n_trades_executed,
        )


class RandomSkipBacktester(CostAwareBacktester):
    """Backtest with random trade skipping."""

    def run_with_random_skip(self, strategy, bar_data_fn, cost_bps: float,
                              skip_pct: float, random_seed: int = 42) -> RunResults:
        """Run backtest with random trade skipping."""
        rng = np.random.RandomState(random_seed)

        n_trades_executed = 0
        self.trades = []

        for bar in range(200, self.n_periods):
            data = bar_data_fn(bar)
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
                    self._open_position(signal, bar, price, cost_bps)
                    n_trades_executed += 1
                elif current_pos['direction'] != direction:
                    trade, _ = self._close_position(symbol, bar, price, cost_bps)
                    self.trades.append(trade)
                    n_trades_executed += 1
                    self._open_position(signal, bar, price, cost_bps)

            if bar == self.n_periods - 1:
                for symbol in list(self.positions.keys()):
                    price = self.close[bar, self.symbol_to_idx[symbol]]
                    trade, _ = self._close_position(symbol, bar, price, cost_bps)
                    self.trades.append(trade)

        # Calculate equity curve (same as parent)
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
            cost_bps=cost_bps,
            equity_curve=equity_curve,
            returns=returns,
            trades=self.trades,
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
    else:
        avg_pnl = 0.0
        win_rate = 0.0

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'n_trades': n_trades,
        'avg_pnl': avg_pnl,
        'win_rate': win_rate,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_cost_robustness_test():
    """Run the full transaction cost robustness test."""
    print("=" * 70)
    print("PHASE 2.2: TRANSACTION COST ROBUSTNESS TEST")
    print("=" * 70)

    # Load data once
    print("\nLoading data...")
    df = load_universe('cta', '2018-01-01', '2024-12-31', use_cache=True)

    print("Pivoting to wide format...")
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close')
    wide_df = wide_df.dropna()
    close_prices = wide_df.values
    symbols = [col for col in wide_df.columns]
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    print(f"Data shape: {close_prices.shape}")
    print(f"Symbols: {symbols}")

    # Get strategy class
    strategy_cls = get_strategy_class("macro_cta_trend")

    # Bar data function
    def get_bar_data(bar: int) -> dict:
        return {
            "close": close_prices[:bar+1],
            "symbols": symbols,
        }

    # Cost levels to test (in basis points)
    cost_levels = [0, 10, 25, 50]

    # Store all results
    all_results = []

    # First, run alignment filter to get skip percentage
    print("\n" + "=" * 70)
    print("PRE-RUN: Determine skip percentage for Random Skip Control")
    print("=" * 70)

    config_alignment_pre = StrategyConfig(
        name="alignment_pre",
        strategy_type=StrategyType.MACRO,
        params={
            "alignment_filter": True,
            "min_signal_strength": 0.0,
        },
        max_position_notional=1_000_000,
    )
    strategy_alignment_pre = strategy_cls(config_alignment_pre)
    backtester_pre = CostAwareBacktester(close_prices, symbols)
    backtester_pre.run(strategy_alignment_pre, get_bar_data, cost_bps=0)

    n_skipped = strategy_alignment_pre.get_skipped_opportunities()
    n_executed = len(backtester_pre.trades)
    n_total = n_executed + n_skipped
    skip_pct = n_skipped / n_total if n_total > 0 else 0

    print(f"Alignment filter skips: {n_skipped} out of {n_total} ({skip_pct:.1%})")

    # Run experiments for each cost level
    for cost_bps in cost_levels:
        print("\n" + "=" * 70)
        print(f"COST LEVEL: {cost_bps} bps (commission + slippage)")
        print("=" * 70)

        # ---------------------------------------------------------------------
        # Run 1: Baseline (No Filter)
        # ---------------------------------------------------------------------
        print(f"\n--- Baseline (No Filter) @ {cost_bps} bps ---")

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
        backtester_baseline = CostAwareBacktester(close_prices, symbols)
        results_baseline = backtester_baseline.run(strategy_baseline, get_bar_data, cost_bps=cost_bps)

        metrics_baseline = calculate_metrics(results_baseline)
        print(f"Sharpe: {metrics_baseline['sharpe']:.4f}, Avg PnL: {metrics_baseline['avg_pnl']:.2f}, Trades: {metrics_baseline['n_trades']}")

        all_results.append({
            'strategy': 'Baseline',
            'cost_bps': cost_bps,
            'metrics': metrics_baseline,
        })

        # ---------------------------------------------------------------------
        # Run 2: Alignment Filter
        # ---------------------------------------------------------------------
        print(f"\n--- Alignment Filter @ {cost_bps} bps ---")

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
        backtester_alignment = CostAwareBacktester(close_prices, symbols)
        results_alignment = backtester_alignment.run(strategy_alignment, get_bar_data, cost_bps=cost_bps)

        metrics_alignment = calculate_metrics(results_alignment)
        print(f"Sharpe: {metrics_alignment['sharpe']:.4f}, Avg PnL: {metrics_alignment['avg_pnl']:.2f}, Trades: {metrics_alignment['n_trades']}")

        all_results.append({
            'strategy': 'Alignment',
            'cost_bps': cost_bps,
            'metrics': metrics_alignment,
        })

        # ---------------------------------------------------------------------
        # Run 3: Random Skip Control
        # ---------------------------------------------------------------------
        print(f"\n--- Random Skip Control @ {cost_bps} bps ---")

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
            strategy_random, get_bar_data, cost_bps=cost_bps,
            skip_pct=skip_pct, random_seed=42
        )

        metrics_random = calculate_metrics(results_random)
        print(f"Sharpe: {metrics_random['sharpe']:.4f}, Avg PnL: {metrics_random['avg_pnl']:.2f}, Trades: {metrics_random['n_trades']}")

        all_results.append({
            'strategy': 'Random Skip',
            'cost_bps': cost_bps,
            'metrics': metrics_random,
        })

    # =========================================================================
    # Output Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS TABLE: TRANSACTION COST ROBUSTNESS")
    print("=" * 70)

    header = f"{'Strategy':<15} {'Cost bps':>8} {'Sharpe':>8} {'Trades':>7} {'Avg PnL':>10} {'Max DD':>9}"
    print(f"\n{header}")
    print("-" * len(header))

    for result in all_results:
        m = result['metrics']
        print(f"{result['strategy']:<15} {result['cost_bps']:>8} {m['sharpe']:>8.4f} {m['n_trades']:>7} "
              f"{m['avg_pnl']:>10.2f} {m['max_drawdown']:>8.2%}")

    # =========================================================================
    # Decision Framework
    # =========================================================================
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)

    # Get metrics at each cost level
    def get_metrics(strategy: str, cost_bps: int) -> dict:
        for r in all_results:
            if r['strategy'] == strategy and r['cost_bps'] == cost_bps:
                return r['metrics']
        return {}

    # Check at 10 bps (low cost)
    align_10 = get_metrics('Alignment', 10)
    baseline_10 = get_metrics('Baseline', 10)

    # Check at 25 bps (moderate cost)
    align_25 = get_metrics('Alignment', 25)

    # Check at 50 bps (stress test)
    align_50 = get_metrics('Alignment', 50)

    print(f"\nAt 10 bps:")
    print(f"  Alignment Sharpe: {align_10.get('sharpe', 0):.4f}")
    print(f"  Baseline Sharpe: {baseline_10.get('sharpe', 0):.4f}")
    print(f"  Alignment beats Baseline: {align_10.get('sharpe', 0) > baseline_10.get('sharpe', 0)}")

    print(f"\nAt 25 bps (moderate cost):")
    print(f"  Alignment Sharpe: {align_25.get('sharpe', 0):.4f}")
    print(f"  Alignment Sharpe positive: {align_25.get('sharpe', 0) > 0}")

    print(f"\nAt 50 bps (stress test):")
    print(f"  Alignment Sharpe: {align_50.get('sharpe', 0):.4f}")
    print(f"  Alignment Sharpe positive: {align_50.get('sharpe', 0) > 0}")

    # Decision
    robust_10 = align_10.get('sharpe', 0) > 0 and align_10.get('sharpe', 0) > baseline_10.get('sharpe', 0)
    robust_25 = align_25.get('sharpe', 0) > 0
    robust_50 = align_50.get('sharpe', 0) > 0

    print(f"\n" + "=" * 70)
    if robust_10 and robust_25:
        print(">>> ROBUST EDGE (Proceed to combine filters)")
        print(f"    - Alignment outperforms baseline at 10 bps")
        print(f"    - Edge remains positive at 25 bps")
        if robust_50:
            print(f"    - Edge survives stress test at 50 bps")
    elif robust_10 and not robust_25:
        print(">>> MARGINAL EDGE (Caution)")
        print(f"    - Edge positive at 10 bps but not at 25 bps")
        print(f"    - Consider optimizing for lower costs or higher frequency")
    else:
        print(">>> FRAGILE / NOT TRADABLE")
        print(f"    - Edge does not survive realistic costs")
        print(f"    - Reject or rethink signal")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = run_cost_robustness_test()
