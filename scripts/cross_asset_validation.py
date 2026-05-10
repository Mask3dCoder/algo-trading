"""Phase 2.5: Cross-Asset Validation.

Tests whether the alignment edge is a general market phenomenon
or specific to the current dataset.

Tests on:
1. Original CTA universe (reference)
2. Risk Parity universe (different application)
3. Asset class subsets (Equity, FX, Commodity, Bonds)
4. New downloads (if available)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: Import macro strategies to trigger registration
import quantcore.strategies.macro  # noqa: F401
from quantcore.strategies.registry import get as get_strategy_class
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.risk.metrics import sharpe_ratio
from scripts.real_data_loader import load_universe, UNIVERSES


# =============================================================================
# Backtest Simulator (reused from Phase 2.4)
# =============================================================================

class CrossAssetBacktester:
    """Backtest for cross-asset validation."""

    def __init__(self, close_prices: np.ndarray, symbols: list[str]):
        self.close = close_prices
        self.symbols = symbols
        self.n_periods, self.n_assets = close_prices.shape
        self.symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    def run(self, strategy) -> tuple[list[float], list[dict]]:
        """Run backtest. Returns (daily_returns, trades)."""
        positions = {}
        trades = []
        equity = 1_000_000
        daily_returns = []
        trade_pnl_by_bar = {}

        for bar in range(200, self.n_periods):
            warm_up_start = max(0, bar - 300)
            data = {
                "close": self.close[warm_up_start:bar+1],
                "symbols": self.symbols,
            }

            strategy.set_timestamp(bar)
            signals = strategy.on_data(data)

            for signal in signals:
                symbol = signal.symbol
                direction = signal.direction
                price = self.close[bar, self.symbol_to_idx[symbol]]

                if symbol not in positions:
                    positions[symbol] = {
                        'direction': direction,
                        'entry_price': price,
                        'entry_bar': bar,
                    }
                elif positions[symbol]['direction'] != direction:
                    pos = positions[symbol]
                    entry_price = pos['entry_price']
                    entry_direction = pos['direction']
                    size = signal.size

                    if entry_direction == 1:
                        pnl = (price - entry_price) / entry_price * size
                    else:
                        pnl = (entry_price - price) / entry_price * size

                    trades.append({
                        'symbol': symbol,
                        'pnl': pnl,
                        'bar': bar,
                    })

                    if bar not in trade_pnl_by_bar:
                        trade_pnl_by_bar[bar] = 0.0
                    trade_pnl_by_bar[bar] += pnl

                    positions[symbol] = {
                        'direction': direction,
                        'entry_price': price,
                        'entry_bar': bar,
                    }

            if bar == self.n_periods - 1:
                for symbol, pos in list(positions.items()):
                    price = self.close[bar, self.symbol_to_idx[symbol]]
                    entry_price = pos['entry_price']
                    entry_direction = pos['direction']
                    size = 100000

                    if entry_direction == 1:
                        pnl = (price - entry_price) / entry_price * size
                    else:
                        pnl = (entry_price - price) / entry_price * size

                    trades.append({
                        'symbol': symbol,
                        'pnl': pnl,
                        'bar': bar,
                    })

                    if bar not in trade_pnl_by_bar:
                        trade_pnl_by_bar[bar] = 0.0
                    trade_pnl_by_bar[bar] += pnl

            bar_pnl = trade_pnl_by_bar.get(bar, 0.0)
            if equity > 0:
                ret = bar_pnl / equity
                daily_returns.append(ret)
            equity += bar_pnl

        return daily_returns, trades


def calculate_metrics(returns: list[float], trades: list[dict]) -> dict:
    """Calculate performance metrics."""
    if len(returns) < 2:
        return {
            'sharpe': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'n_trades': len(trades),
            'avg_pnl': 0.0,
        }

    returns_arr = np.array(returns)
    sharpe = sharpe_ratio(returns_arr)

    # Total return
    total_return = np.prod(1 + returns_arr) - 1

    # Max drawdown
    equity = np.cumprod(1 + returns_arr)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(np.min(drawdown))

    # Avg PnL
    pnls = [t['pnl'] for t in trades]
    avg_pnl = np.mean(pnls) if pnls else 0.0

    return {
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'n_trades': len(trades),
        'avg_pnl': avg_pnl,
    }


# =============================================================================
# Main Experiment
# =============================================================================

def run_cross_asset_validation():
    """Run Phase 2.5 cross-asset validation."""
    print("=" * 70)
    print("PHASE 2.5: CROSS-ASSET VALIDATION")
    print("=" * 70)

    # Get strategy class
    strategy_cls = get_strategy_class("macro_cta_trend")

    # Define test configurations
    # NO parameter changes - same alignment filter for all
    def get_alignment_config(name: str) -> StrategyConfig:
        return StrategyConfig(
            name=name,
            strategy_type=StrategyType.MACRO,
            params={
                "alignment_filter": True,
                "min_signal_strength": 0.0,  # No strength filter
            },
            max_position_notional=1_000_000,
        )

    results = []

    # =========================================================================
    # Test 1: Original CTA Universe (Reference)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: ORIGINAL CTA UNIVERSE (Reference)")
    print("=" * 70)

    df = load_universe('cta', '2018-01-01', '2024-12-31', use_cache=True)
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close').dropna()
    close_prices = wide_df.values
    symbols = [col for col in wide_df.columns]

    print(f"Symbols: {symbols}")
    print(f"Data shape: {close_prices.shape}")

    config = get_alignment_config("cta_reference")
    strategy = strategy_cls(config)
    backtester = CrossAssetBacktester(close_prices, symbols)
    returns, trades = backtester.run(strategy)
    metrics = calculate_metrics(returns, trades)

    results.append({
        'asset_class': 'CTA (reference)',
        'symbols': ', '.join(symbols),
        'sharpe': metrics['sharpe'],
        'trades': metrics['n_trades'],
        'avg_pnl': metrics['avg_pnl'],
        'max_dd': metrics['max_drawdown'],
    })

    print(f"Sharpe: {metrics['sharpe']:.4f}")
    print(f"Trades: {metrics['n_trades']}")
    print(f"Avg PnL: {metrics['avg_pnl']:.2f}")
    print(f"Max DD: {metrics['max_drawdown']:.2%}")

    # =========================================================================
    # Test 2: Risk Parity Universe
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: RISK PARITY UNIVERSE")
    print("=" * 70)

    df = load_universe('rp', '2018-01-01', '2024-12-31', use_cache=True)
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close').dropna()
    close_prices = wide_df.values
    symbols = [col for col in wide_df.columns]

    print(f"Symbols: {symbols}")
    print(f"Data shape: {close_prices.shape}")

    config = get_alignment_config("rp_universe")
    strategy = strategy_cls(config)
    backtester = CrossAssetBacktester(close_prices, symbols)
    returns, trades = backtester.run(strategy)
    metrics = calculate_metrics(returns, trades)

    results.append({
        'asset_class': 'Risk Parity',
        'symbols': ', '.join(symbols),
        'sharpe': metrics['sharpe'],
        'trades': metrics['n_trades'],
        'avg_pnl': metrics['avg_pnl'],
        'max_dd': metrics['max_drawdown'],
    })

    print(f"Sharpe: {metrics['sharpe']:.4f}")
    print(f"Trades: {metrics['n_trades']}")
    print(f"Avg PnL: {metrics['avg_pnl']:.2f}")
    print(f"Max DD: {metrics['max_drawdown']:.2%}")

    # =========================================================================
    # Test 3: Asset Class Subsets (from CTA data)
    # =========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: ASSET CLASS SUBSETS")
    print("=" * 70)

    # Define asset class groupings
    asset_groups = {
        'Equity (SPY+EFA)': ['SPY', 'EFA'],
        'Bonds (TLT+IEF)': ['TLT', 'IEF'],
        'Commodities (GLD+DBC)': ['GLD', 'DBC'],
        'FX (FXE)': ['FXE'],
    }

    df = load_universe('cta', '2018-01-01', '2024-12-31', use_cache=True)

    for group_name, group_symbols in asset_groups.items():
        print(f"\n--- {group_name} ---")
        print(f"Symbols: {group_symbols}")

        # Filter data for this group
        group_df = df[df['symbol'].isin(group_symbols)]
        if group_df.empty:
            print("No data available, skipping...")
            continue

        wide_df = group_df.pivot(index='timestamp', columns='symbol', values='close').dropna()
        if wide_df.empty:
            print("No overlapping data, skipping...")
            continue

        close_prices = wide_df.values
        symbols = [col for col in wide_df.columns]

        print(f"Data shape: {close_prices.shape}")

        config = get_alignment_config(group_name.replace(' ', '_'))
        strategy = strategy_cls(config)
        backtester = CrossAssetBacktester(close_prices, symbols)
        returns, trades = backtester.run(strategy)
        metrics = calculate_metrics(returns, trades)

        results.append({
            'asset_class': group_name,
            'symbols': ', '.join(symbols),
            'sharpe': metrics['sharpe'],
            'trades': metrics['n_trades'],
            'avg_pnl': metrics['avg_pnl'],
            'max_dd': metrics['max_drawdown'],
        })

        print(f"Sharpe: {metrics['sharpe']:.4f}")
        print(f"Trades: {metrics['n_trades']}")
        print(f"Avg PnL: {metrics['avg_pnl']:.2f}")
        print(f"Max DD: {metrics['max_drawdown']:.2%}")

    # =========================================================================
    # Output Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("CROSS-ASSET VALIDATION RESULTS")
    print("=" * 70)

    header = f"{'Asset Class':<25} {'Sharpe':>8} {'Trades':>7} {'Avg PnL':>9} {'Max DD':>9}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        print(f"{r['asset_class']:<25} {r['sharpe']:>8.4f} {r['trades']:>7} "
              f"{r['avg_pnl']:>9.2f} {r['max_dd']:>8.2%}")

    # =========================================================================
    # Interpretation Framework
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION FRAMEWORK")
    print("=" * 70)

    # Check if alignment works across asset classes
    sharpes = [r['sharpe'] for r in results if r['sharpe'] > 0]
    n_positive = sum(1 for r in results if r['sharpe'] > 0)
    n_total = len(results)

    print(f"\nTotal tests: {n_total}")
    print(f"Tests with positive Sharpe: {n_positive}/{n_total} ({n_positive/n_total*100:.1f}%)")

    if len(sharpes) > 0:
        mean_sharpe = np.mean(sharpes)
        print(f"Mean Sharpe (positive only): {mean_sharpe:.4f}")
        print(f"Min Sharpe: {min(r['sharpe'] for r in results):.4f}")
        print(f"Max Sharpe: {max(r['sharpe'] for r in results):.4f}")

    # Decision
    pct_positive = n_positive / n_total * 100 if n_total > 0 else 0
    strong_edge = pct_positive >= 80  # >=80% of tests positive
    consistent = all(r['sharpe'] > 0 for r in results if 'reference' not in r['asset_class'].lower())

    print(f"\n* Positive Sharpe in >80% tests: {pct_positive >= 80}% ({pct_positive:.1f}%)")
    print(f"* Consistent across non-reference: {consistent}")

    if strong_edge:
        print("\n>>> STRONG / REAL EDGE (Structural)")
        print("    - Alignment filter works across asset classes")
        print("    - This is a generalizable market truth!")
        print("    - Proceed to portfolio construction (multi-asset)")
    elif pct_positive >= 60:
        print("\n>>> PARTIAL ROBUSTNESS")
        print("    - Works well in some asset classes")
        print("    - Edge is conditional, not universal")
        print("    - Consider regime filtering for weaker assets")
    else:
        print("\n>>> OVERFIT / LUCKY DATASET")
        print("    - Only works in original universe")
        print("    - Edge may be specific to those assets/timeframe")
        print("    - Re-evaluate before expanding")

    return results


if __name__ == "__main__":
    results = run_cross_asset_validation()
