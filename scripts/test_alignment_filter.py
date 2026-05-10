"""Test alignment filter for CTA trend strategy with real data."""
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
from scripts.real_data_loader import load_universe

def test_alignment_filter():
    print("=" * 70)
    print("Phase 2.2: Alignment Filter Test")
    print("Strategy: macro_cta_trend")
    print("=" * 70)

    # Load real data (long format: one row per symbol per timestamp)
    print("\nLoading data...")
    df = load_universe('cta', '2018-01-01', '2024-12-31', use_cache=True)
    
    # Pivot to wide format: rows=timestamps, columns=symbols, values=close price
    print("Pivoting data to wide format...")
    wide_df = df.pivot(index='timestamp', columns='symbol', values='close')
    wide_df.columns = [f'close_{col}' for col in wide_df.columns]  # Rename columns to close_*
    wide_df = wide_df.dropna()  # Drop rows with missing data
    
    print(f"Wide DataFrame shape: {wide_df.shape}")
    print(f"Symbols: {[col.replace('close_', '') for col in wide_df.columns]}")
    
    # Prepare data in format expected by strategy
    close_prices = wide_df.values  # Shape: (n_periods, n_assets)
    symbols = [col.replace('close_', '') for col in wide_df.columns]
    
    # Convert to dict format expected by on_data
    data = {
        "close": close_prices,
        "symbols": symbols,
    }

    # Test 1: Alignment filter OFF (baseline)
    print("\n" + "-" * 50)
    print("Test 1: Alignment Filter OFF (baseline)")
    print("-" * 50)
    
    config_off = StrategyConfig(
        name="cta_baseline",
        strategy_type=StrategyType.MACRO,
        params={
            "alignment_filter": False,
            "min_signal_strength": 0.0,
        },
        max_position_notional=1_000_000,
    )
    strategy_cls = get_strategy_class("macro_cta_trend")
    strategy_off = strategy_cls(config_off)
    
    # Run strategy on all bars (start after slow_window + 1 periods)
    signals_off = []
    start_idx = 201  # Need at least 202 periods (slow_window + 2)
    print(f"Running strategy from period {start_idx} to {len(close_prices)-1}")
    
    for i in range(start_idx, len(close_prices)):
        bar_data = {
            "close": close_prices[:i+1],  # Use all data up to current bar
            "symbols": symbols,
        }
        strategy_off.set_timestamp(df.index[i])
        sigs = strategy_off.on_data(bar_data)
        signals_off.extend(sigs)
    
    print(f"Signals generated (filter OFF): {len(signals_off)}")

    # Test 2: Alignment filter ON
    print("\n" + "-" * 50)
    print("Test 2: Alignment Filter ON")
    print("-" * 50)
    
    config_on = StrategyConfig(
        name="cta_alignment",
        strategy_type=StrategyType.MACRO,
        params={
            "alignment_filter": True,
            "min_signal_strength": 0.0,
        },
        max_position_notional=1_000_000,
    )
    strategy_on = strategy_cls(config_on)
    
    signals_on = []
    for i in range(start_idx, len(close_prices)):
        bar_data = {
            "close": close_prices[:i+1],
            "symbols": symbols,
        }
        strategy_on.set_timestamp(df.index[i])
        sigs = strategy_on.on_data(bar_data)
        signals_on.extend(sigs)
    
    print(f"Signals generated (filter ON): {len(signals_on)}")
    
    # Compare results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"Baseline signals: {len(signals_off)}")
    print(f"Alignment filter signals: {len(signals_on)}")
    print(f"Signal reduction: {len(signals_off) - len(signals_on)} ({(1 - len(signals_on)/max(len(signals_off),1))*100:.1f}%)")
    
    # Show diagnostic info from alignment filter run
    skipped = strategy_on.get_skipped_opportunities()
    print(f"\nSkipped trade opportunities (alignment mismatch): {skipped}")
    print(f"Total fast signals processed: {strategy_on._total_fast_signals}")

if __name__ == "__main__":
    test_alignment_filter()
