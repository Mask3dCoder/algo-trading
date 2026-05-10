"""Phase 2.4: Incremental Signal Value Test.

Tests whether the strength filter contributes independent predictive edge
when combined with alignment.

4 Configurations:
1. Baseline (no filters)
2. Alignment Only (already validated)
3. Strength Only (min_signal_strength threshold)
4. Alignment + Strength (both filters)

Critical Add-On:
- Within Alignment Only trades, split into Strong vs Weak fast signal
- Compare Avg PnL and Sharpe
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

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
    direction: int
    size: float
    pnl: float = 0.0
    fast_signal: float = 0.0
    slow_signal: float = 0.0
    skip_reason: str = None  # None = executed, 'weak_signal', 'alignment_mismatch', etc.


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
# Backtest Simulator with Trade Attribution
# =============================================================================

class AttributionBacktester:
    """Backtest that tracks positions, PnL, and trade attribution."""

    def __init__(self, close_prices: np.ndarray, symbols: list[str],
                 initial_capital: float = 1_000_000):
        self.close = close_prices
        self.symbols = symbols
        self.n_periods, self.n_assets = close_prices.shape
        self.initial_capital = initial_capital
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        # State
        self.positions = {}
        self.trades: list[Trade] = []

    def _get_position(self, symbol: str) -> dict | None:
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
        """Close an existing position and return the trade."""
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

    def run(self, strategy, bar_data_fn) -> RunResults:
        """Run backtest and return results."""
        n_signals_total = 0
        n_trades_executed = 0
        self.trades = []

        for bar in range(200, self.n_periods):
            data = bar_data_fn(bar)
            strategy.set_timestamp(bar)
            signals = strategy.on_data(data)

            for signal in signals:
                n_signals_total += 1
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

            if bar == self.n_periods - 1:
                for symbol in list(self.positions.keys()):
                    price = self.close[bar, self.symbol_to_idx[symbol]]
                    trade = self._close_position(symbol, bar, price)
                    self.trades.append(trade)

        # Calculate equity curve and returns
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


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_metrics(results: RunResults) -> dict:
    """Calculate performance metrics."""
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

    sharpe = sharpe_ratio(returns)

    total_return = (results.equity_curve[-1] / results.equity_curve[0] - 1)

    equity = np.array(results.equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    n_trades = len(trades)
    if n_trades > 0:
        avg_pnl = np.mean([t.pnl for t in trades])
        win_rate = sum(1 for t in trades if t.pnl > 0) / n_trades
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


def analyze_trade_attribution(strategy) -> dict:
    """Analyze trade attribution within alignment filter.

    Splits executed trades into Strong vs Weak fast signal.
    """
    all_opps = strategy.get_all_opportunities()

    strong_trades = []
    weak_trades = []

    for opp in all_opps:
        if opp.get('was_executed', False):
            fast_sig = opp.get('fast_signal', 0.0)
            # We need PnL for this trade - but we don't have it directly
            # For now, use the composite signal as a proxy
            # In a real implementation, we'd need to track PnL per trade
            if abs(fast_sig) >= 0.1:  # Threshold for "strong"
                strong_trades.append(opp)
            else:
                weak_trades.append(opp)

    return {
        'n_strong': len(strong_trades),
        'n_weak': len(weak_trades),
        'strong_pnl': sum(t.get('composite', 0) for t in strong_trades) / max(len(strong_trades), 1),
        'weak_pnl': sum(t.get('composite', 0) for t in weak_trades) / max(len(weak_trades), 1),
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_incremental_signal_test():
    """Run Phase 2.4 experiment."""
    print("=" * 70)
    print("PHASE 2.4: INCREMENTAL SIGNAL VALUE TEST")
    print("=" * 70)

    # Load data
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

    # Strength threshold
    STRENGTH_THRESHOLD = 0.1

    # =========================================================================
    # Run 1: Baseline (No Filters)
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUN 1: BASELINE (No Filters)")
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
    backtester = AttributionBacktester(close_prices, symbols)
    results_baseline = backtester.run(strategy_baseline, get_bar_data)

    metrics_baseline = calculate_metrics(results_baseline)
    print(f"Sharpe: {metrics_baseline['sharpe']:.4f}")
    print(f"Total Return: {metrics_baseline['total_return']:.2%}")
    print(f"Max Drawdown: {metrics_baseline['max_drawdown']:.2%}")
    print(f"Trades: {metrics_baseline['n_trades']}")
    print(f"Avg PnL: {metrics_baseline['avg_pnl']:.2f}")
    print(f"Win Rate: {metrics_baseline['win_rate']:.2%}")
    print(f"Whipsaw Rate: {metrics_baseline['whipsaw_rate']:.2%}")

    # =========================================================================
    # Run 2: Alignment Only
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUN 2: ALIGNMENT ONLY")
    print("=" * 70)

    config_alignment = StrategyConfig(
        name="alignment",
        strategy_type=StrategyType.MACRO,
        params={
            "alignment_filter": True,
            "min_signal_strength": 0.0,  # No strength filter
        },
        max_position_notional=1_000_000,
    )
    strategy_alignment = strategy_cls(config_alignment)
    backtester = AttributionBacktester(close_prices, symbols)
    results_alignment = backtester.run(strategy_alignment, get_bar_data)

    metrics_alignment = calculate_metrics(results_alignment)
    print(f"Sharpe: {metrics_alignment['sharpe']:.4f}")
    print(f"Total Return: {metrics_alignment['total_return']:.2%}")
    print(f"Max Drawdown: {metrics_alignment['max_drawdown']:.2%}")
    print(f"Trades: {metrics_alignment['n_trades']}")
    print(f"Avg PnL: {metrics_alignment['avg_pnl']:.2f}")
    print(f"Win Rate: {metrics_alignment['win_rate']:.2%}")
    print(f"Whipsaw Rate: {metrics_alignment['whipsaw_rate']:.2%}")

    # Trade attribution within alignment (Strong vs Weak fast signal)
    # We need to get the executed trades and check their fast_signal strength
    strong_trades = [t for t in results_alignment.trades if abs(t.fast_signal) >= STRENGTH_THRESHOLD]
    weak_trades = [t for t in results_alignment.trades if abs(t.fast_signal) < STRENGTH_THRESHOLD]

    strong_avg_pnl = np.mean([t.pnl for t in strong_trades]) if strong_trades else 0.0
    weak_avg_pnl = np.mean([t.pnl for t in weak_trades]) if weak_trades else 0.0

    print(f"\nTrade Attribution Within Alignment:")
    print(f"  Strong fast signal (|fast| >= {STRENGTH_THRESHOLD}): {len(strong_trades)} trades, Avg PnL: {strong_avg_pnl:.2f}")
    print(f"  Weak fast signal (|fast| < {STRENGTH_THRESHOLD}): {len(weak_trades)} trades, Avg PnL: {weak_avg_pnl:.2f}")

    # =========================================================================
    # Run 3: Strength Only
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUN 3: STRENGTH ONLY")
    print("=" * 70)

    config_strength = StrategyConfig(
        name="strength",
        strategy_type=StrategyType.MACRO,
        params={
            "alignment_filter": False,
            "min_signal_strength": STRENGTH_THRESHOLD,  # Strength filter ON
        },
        max_position_notional=1_000_000,
    )
    strategy_strength = strategy_cls(config_strength)
    backtester = AttributionBacktester(close_prices, symbols)
    results_strength = backtester.run(strategy_strength, get_bar_data)

    metrics_strength = calculate_metrics(results_strength)
    print(f"Sharpe: {metrics_strength['sharpe']:.4f}")
    print(f"Total Return: {metrics_strength['total_return']:.2%}")
    print(f"Max Drawdown: {metrics_strength['max_drawdown']:.2%}")
    print(f"Trades: {metrics_strength['n_trades']}")
    print(f"Avg PnL: {metrics_strength['avg_pnl']:.2f}")
    print(f"Win Rate: {metrics_strength['win_rate']:.2%}")
    print(f"Whipsaw Rate: {metrics_strength['whipsaw_rate']:.2%}")

    # =========================================================================
    # Run 4: Alignment + Strength
    # =========================================================================
    print("\n" + "=" * 70)
    print("RUN 4: ALIGNMENT + STRENGTH")
    print("=" * 70)

    config_combined = StrategyConfig(
        name="combined",
        strategy_type=StrategyType.MACRO,
        params={
            "alignment_filter": True,
            "min_signal_strength": STRENGTH_THRESHOLD,  # Both filters ON
        },
        max_position_notional=1_000_000,
    )
    strategy_combined = strategy_cls(config_combined)
    backtester = AttributionBacktester(close_prices, symbols)
    results_combined = backtester.run(strategy_combined, get_bar_data)

    metrics_combined = calculate_metrics(results_combined)
    print(f"Sharpe: {metrics_combined['sharpe']:.4f}")
    print(f"Total Return: {metrics_combined['total_return']:.2%}")
    print(f"Max Drawdown: {metrics_combined['max_drawdown']:.2%}")
    print(f"Trades: {metrics_combined['n_trades']}")
    print(f"Avg PnL: {metrics_combined['avg_pnl']:.2f}")
    print(f"Win Rate: {metrics_combined['win_rate']:.2%}")
    print(f"Whipsaw Rate: {metrics_combined['whipsaw_rate']:.2%}")

    # =========================================================================
    # Comparison Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    header = f"{'Configuration':<25} {'Sharpe':>8} {'Trades':>7} {'Avg PnL':>9} {'Whipsaw%':>9} {'Max DD':>9}"
    print(f"\n{header}")
    print("-" * len(header))

    for name, metrics in [
        ("Baseline (no filters)", metrics_baseline),
        ("Alignment Only", metrics_alignment),
        ("Strength Only", metrics_strength),
        ("Alignment + Strength", metrics_combined),
    ]:
        print(f"{name:<25} {metrics['sharpe']:>8.4f} {metrics['n_trades']:>7} "
              f"{metrics['avg_pnl']:>9.2f} {metrics['whipsaw_rate']:>8.2%} "
              f"{metrics['max_drawdown']:>8.2%}")

    # =========================================================================
    # Decision Framework
    # =========================================================================
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)

    # Check criteria
    combined_better_than_alignment = (
        metrics_combined['sharpe'] > metrics_alignment['sharpe'] and
        metrics_combined['avg_pnl'] > metrics_alignment['avg_pnl']
    )

    strong_better_than_weak = strong_avg_pnl > weak_avg_pnl

    # Check for harmful interaction (trade count drops too much)
    trade_drop_pct = (metrics_alignment['n_trades'] - metrics_combined['n_trades']) / max(metrics_alignment['n_trades'], 1)
    harmful_drop = trade_drop_pct > 0.5  # Drop >50%

    # Check for risk concentration (Sharpe flat but Avg PnL up)
    sharpe_flat = abs(metrics_combined['sharpe'] - metrics_alignment['sharpe']) < 0.05
    avg_pnl_up = metrics_combined['avg_pnl'] > metrics_alignment['avg_pnl'] * 1.2  # 20% up
    risk_concentration = sharpe_flat and avg_pnl_up

    print(f"* Combined > Alignment (Sharpe & Avg PnL): {combined_better_than_alignment}")
    print(f"* Strong > Weak within Alignment: {strong_better_than_weak}")
    print(f"* Trade drop from Alignment to Combined: {trade_drop_pct:.1%}")
    print(f"* Harmful interaction (drop >50%): {harmful_drop}")
    print(f"* Risk concentration warning: {risk_concentration}")

    if combined_better_than_alignment and strong_better_than_weak and not harmful_drop:
        print("\n>>> STRENGTH ADDS REAL VALUE - Safe to combine!")
        print("    - Re-run cost test and walk-forward with combined filters")
    elif not strong_better_than_weak:
        print("\n>>> REDUNDANT FILTER - Strength adds nothing")
        print("    - Strong signals are already filtered by Alignment")
        print("    - Discard strength filter, stick with Alignment Only")
    elif harmful_drop:
        print("\n>>> HARMFUL INTERACTION - Over-filtering")
        print("    - Trade count drops too much")
        print("    - Stick with Alignment Only")
    elif risk_concentration:
        print("\n>>> WARNING: Risk concentration without Sharpe improvement")
        print("    - Just concentrating risk, not improving signal")
        print("    - Stick with Alignment Only")
    else:
        print("\n>>> INCONCLUSIVE - Need more analysis")

    return {
        'baseline': metrics_baseline,
        'alignment': metrics_alignment,
        'strength': metrics_strength,
        'combined': metrics_combined,
        'attribution': {
            'strong_avg_pnl': strong_avg_pnl,
            'weak_avg_pnl': weak_avg_pnl,
            'n_strong': len(strong_trades),
            'n_weak': len(weak_trades),
        }
    }


if __name__ == "__main__":
    results = run_incremental_signal_test()
