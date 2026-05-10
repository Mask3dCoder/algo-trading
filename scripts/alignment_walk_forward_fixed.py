"""Phase 2.3: Walk-Forward Validation (Fixed).

Tests whether the alignment filter's edge is stable across time.
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
from scripts.real_data_loader import load_universe


# =============================================================================
# Simplified Walk-Forward Backtester
# =============================================================================

def run_window(strategy, close_prices, symbols, symbol_to_idx,
              start_bar: int, end_bar: int) -> tuple[list[float], int]:
    """Run a single walk-forward window. Returns (returns, n_trades)."""
    positions = {}  # symbol -> {'direction': int, 'entry_price': float, 'entry_bar': int}
    trades = []
    equity = 1_000_000
    returns = []

    # The strategy needs warm-up data
    warm_up_start = max(0, start_bar - 300)  # 300 bars warm-up

    for bar in range(start_bar, end_bar):
        # Prepare data with warm-up history
        data = {
            "close": close_prices[warm_up_start:bar+1],
            "symbols": symbols,
        }

        strategy.set_timestamp(bar)
        signals = strategy.on_data(data)

        for signal in signals:
            symbol = signal.symbol
            direction = signal.direction
            price = close_prices[bar, symbol_to_idx[symbol]]

            if symbol not in positions:
                # New position
                positions[symbol] = {
                    'direction': direction,
                    'entry_price': price,
                    'entry_bar': bar,
                }
            elif positions[symbol]['direction'] != direction:
                # Close old position, open new one
                pos = positions[symbol]
                entry_price = pos['entry_price']
                entry_direction = pos['direction']
                size = signal.size

                # Calculate PnL
                if entry_direction == 1:  # Long
                    pnl = (price - entry_price) / entry_price * size
                else:  # Short
                    pnl = (entry_price - price) / entry_price * size

                trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'bar': bar,
                })

                # Update equity and returns
                if equity > 0:
                    ret = pnl / equity
                    returns.append(ret)
                equity += pnl

                # Open new position
                positions[symbol] = {
                    'direction': direction,
                    'entry_price': price,
                    'entry_bar': bar,
                }
            # else: same direction, do nothing

        # Close all positions at end of window
        if bar == end_bar - 1:
            for symbol, pos in list(positions.items()):
                price = close_prices[bar, symbol_to_idx[symbol]]
                entry_price = pos['entry_price']
                entry_direction = pos['direction']
                size = 100000  # Approximate

                if entry_direction == 1:
                    pnl = (price - entry_price) / entry_price * size
                else:
                    pnl = (entry_price - price) / entry_price * size

                trades.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'bar': bar,
                })

                if equity > 0:
                    ret = pnl / equity
                    returns.append(ret)
                equity += pnl

    return returns, len(trades)


def calculate_sharpe(returns: list[float]) -> float:
    """Calculate Sharpe ratio from returns."""
    if len(returns) < 2:
        return 0.0
    return sharpe_ratio(np.array(returns))


# =============================================================================
# Main Experiment
# =============================================================================

def run_walk_forward_validation():
    """Run the full walk-forward validation experiment."""
    print("=" * 70)
    print("PHASE 2.3: WALK-FORWARD VALIDATION (FIXED)")
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
    print(f"Date range: {wide_df.index[0].date()} to {wide_df.index[-1].date()}")

    # Get strategy class
    strategy_cls = get_strategy_class("macro_cta_trend")

    # Define walk-forward windows
    # Test: 6 months (126 trading days)
    # Roll forward by 6 months
    TEST_DAYS = 126
    ROLL_DAYS = 126

    n_periods = close_prices.shape[0]

    # Calculate windows (test periods only)
    windows = []
    # Start after warm-up (200 bars for slow window) + 2 years training
    test_start = 200 + 504  # After 2 years of data
    while test_start + TEST_DAYS <= n_periods:
        test_end = test_start + TEST_DAYS
        window_label = f"{wide_df.index[test_start].strftime('%Y-%m')} to {wide_df.index[test_end-1].strftime('%Y-%m')}"
        windows.append({
            'start': test_start,
            'end': test_end,
            'label': window_label,
        })
        test_start += ROLL_DAYS

    print(f"\nNumber of walk-forward windows: {len(windows)}")
    for w in windows:
        print(f"  {w['label']}")

    # First, determine skip percentage from a full run
    print("\n" + "=" * 70)
    print("PRE-RUN: Determine skip percentage")
    print("=" * 70)

    config_pre = StrategyConfig(
        name="alignment_pre",
        strategy_type=StrategyType.MACRO,
        params={"alignment_filter": True, "min_signal_strength": 0.0},
        max_position_notional=1_000_000,
    )
    strategy_pre = strategy_cls(config_pre)

    # Run on full dataset
    returns_pre, trades_pre = run_window(
        strategy_pre, close_prices, symbols, symbol_to_idx, 200, n_periods
    )

    n_skipped = strategy_pre.get_skipped_opportunities()
    skip_pct = n_skipped / (trades_pre + n_skipped) if (trades_pre + n_skipped) > 0 else 0
    print(f"Skip percentage: {skip_pct:.1%}")

    # Run walk-forward validation
    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 70)

    results = []

    for i, window in enumerate(windows):
        start = window['start']
        end = window['end']
        label = window['label']

        print(f"\n--- Window {i+1}: {label} ---")

        # Reset strategies for each window
        # Baseline
        config_base = StrategyConfig(
            name="baseline",
            strategy_type=StrategyType.MACRO,
            params={"alignment_filter": False, "min_signal_strength": 0.0},
            max_position_notional=1_000_000,
        )
        strategy_base = strategy_cls(config_base)
        returns_base, trades_base = run_window(
            strategy_base, close_prices, symbols, symbol_to_idx, start, end
        )
        sharpe_base = calculate_sharpe(returns_base)
        print(f"  Baseline:   Sharpe={sharpe_base:.4f}, Trades={trades_base}")

        # Alignment
        config_align = StrategyConfig(
            name="alignment",
            strategy_type=StrategyType.MACRO,
            params={"alignment_filter": True, "min_signal_strength": 0.0},
            max_position_notional=1_000_000,
        )
        strategy_align = strategy_cls(config_align)
        returns_align, trades_align = run_window(
            strategy_align, close_prices, symbols, symbol_to_idx, start, end
        )
        sharpe_align = calculate_sharpe(returns_align)
        n_skipped_align = strategy_align.get_skipped_opportunities()
        print(f"  Alignment:   Sharpe={sharpe_align:.4f}, Trades={trades_align}, Skipped={n_skipped_align}")

        # Random Skip (approximation - using reduced trade count)
        # For now, approximate by reducing baseline trades
        sharpe_random = sharpe_base * (1 - skip_pct)  # Rough approximation
        trades_random = int(trades_base * (1 - skip_pct))
        print(f"  Random Skip: Sharpe~{sharpe_random:.4f}, Trades~{trades_random}")

        results.append({
            'label': label,
            'sharpe_base': sharpe_base,
            'sharpe_align': sharpe_align,
            'sharpe_random': sharpe_random,
            'trades_base': trades_base,
            'trades_align': trades_align,
            'trades_random': trades_random,
            'align_beats_base': sharpe_align > sharpe_base,
        })

    # =========================================================================
    # Output Table
    # =========================================================================
    print("\n" + "=" * 70)
    print("WALK-FORWARD RESULTS TABLE")
    print("=" * 70)

    header = f"{'Window':<25} {'Base SR':>8} {'Align SR':>9} {'Rand SR':>8} {'Align>Base':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for r in results:
        print(f"{r['label']:<25} {r['sharpe_base']:>8.4f} {r['sharpe_align']:>9.4f} "
              f"{r['sharpe_random']:>8.4f} {'Yes' if r['align_beats_base'] else 'No':>10}")

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    sharpes_base = [r['sharpe_base'] for r in results]
    sharpes_align = [r['sharpe_align'] for r in results]

    print(f"\n{'Metric':<25} {'Baseline':>10} {'Alignment':>10}")
    print("-" * 45)

    print(f"{'Mean Sharpe':<25} {np.mean(sharpes_base):>10.4f} {np.mean(sharpes_align):>10.4f}")
    print(f"{'Std Dev Sharpe':<25} {np.std(sharpes_base):>10.4f} {np.std(sharpes_align):>10.4f}")
    print(f"{'Min Sharpe':<25} {np.min(sharpes_base):>10.4f} {np.min(sharpes_align):>10.4f}")
    print(f"{'Max Sharpe':<25} {np.max(sharpes_base):>10.4f} {np.max(sharpes_align):>10.4f}")

    n_beats = sum(1 for r in results if r['align_beats_base'])
    pct_beats = n_beats / len(results) * 100 if results else 0

    n_pos_align = sum(1 for r in results if r['sharpe_align'] > 0)
    pct_pos = n_pos_align / len(results) * 100 if results else 0

    print(f"\nAlignment beats Baseline in {n_beats}/{len(results)} periods ({pct_beats:.1f}%)")
    print(f"Alignment has positive Sharpe in {n_pos_align}/{len(results)} periods ({pct_pos:.1f}%)")

    # =========================================================================
    # Decision Framework
    # =========================================================================
    print("\n" + "=" * 70)
    print("DECISION FRAMEWORK")
    print("=" * 70)

    stable_edge = pct_beats >= 60
    consistent_positive = pct_pos >= 70
    align_std = np.std(sharpes_align)
    low_variance = align_std < 0.5

    print(f"\n* Alignment beats Baseline in >60% periods: {stable_edge} ({pct_beats:.1f}%)")
    print(f"* Alignment Sharpe positive in >70% periods: {consistent_positive} ({pct_pos:.1f}%)")
    print(f"* Low variance (std < 0.5): {low_variance} (std={align_std:.4f})")

    if stable_edge and consistent_positive:
        print("\n>>> STRONG / REAL EDGE - Stable across time!")
        print("    - Proceed to combine filters (alignment + strength)")
    elif stable_edge:
        print("\n>>> MARGINAL EDGE - Some instability")
        print("    - Consider regime conditioning")
    else:
        print("\n>>> FRAGILE EDGE / OVERFIT")
        print("    - Edge may be regime-dependent")

    return results


if __name__ == "__main__":
    results = run_walk_forward_validation()
