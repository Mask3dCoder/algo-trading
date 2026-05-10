"""Debug script to check fast signal values during COVID crash period."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

# Add parent directory to path for scripts package
PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from scripts.real_data_backtest import _df_to_data, _setup_strategy, STRATEGY_UNIVERSE, load_universe

# Load data
df = load_universe("cta", "2019-06-01", "2020-04-30", use_cache=False)

# Get SPY prices
spy = df[df["symbol"] == "SPY"].sort_values("timestamp")
print(f"SPY data points: {len(spy)}")
print(f"Date range: {spy['timestamp'].min()} to {spy['timestamp'].max()}")

# Compute fast signal (20-day SMA)
spy_close = spy["close"].values
dates = spy["timestamp"].values

print("\n--- Fast signal (price - SMA20) / SMA20 during crash period ---")
print(f"{'Date':<12} {'Price':>8} {'SMA20':>8} {'Fast':>8}")
for i in range(20, len(spy_close)):
    sma20 = np.mean(spy_close[max(0, i-20):i+1])
    fast = (spy_close[i] - sma20) / sma20 if sma20 > 0 else 0.0
    date_str = pd.Timestamp(dates[i]).strftime("%Y-%m-%d")
    if "2020-02" in date_str or "2020-03" in date_str or "2020-04" in date_str:
        print(f"{date_str:<12} {spy_close[i]:>8.2f} {sma20:>8.2f} {fast:>8.4f}")

# Now check what the strategy actually computes
print("\n--- Strategy on_data during crash period ---")
from quantcore.strategies.macro.cta_trend import CTATrendStrategy
from quantcore.strategies.base import StrategyConfig, StrategyType

cfg = StrategyConfig(
    name="macro_cta_trend",
    strategy_type=StrategyType.MACRO,
    params={
        "fast_window": 20,
        "medium_window": 50,
        "slow_window": 200,
        "vol_target": 0.15,
        "vol_lookback": 60,
        "rebalance_freq": 5,
    }
)
strat = CTATrendStrategy(cfg)
strat.initialize()

# Prepare data
df_all = load_universe("cta", "2019-06-01", "2020-06-30", use_cache=False)
data = _df_to_data(df_all)
close = data["close"]
symbols = data["symbols"]
spy_idx = symbols.index("SPY") if "SPY" in symbols else 0

print(f"\nSPY index: {spy_idx}")
print(f"Close shape: {close.shape}")

# Find crash period indices
timestamps = sorted(df_all["timestamp"].unique())
crash_start = pd.Timestamp("2020-02-19")
crash_end = pd.Timestamp("2020-03-23")

crash_indices = []
for i, ts in enumerate(timestamps):
    if crash_start <= pd.Timestamp(ts) <= crash_end:
        crash_indices.append(i)

print(f"\nCrash period has {len(crash_indices)} bars")
print(f"First few indices: {crash_indices[:5]}")

# Run strategy for a few bars
print("\n--- Strategy signals during crash period ---")
bar_count = 0
for idx in crash_indices[:10]:  # First 10 bars
    if idx >= close.shape[0]:
        break
    prices = close[:idx+1]  # Data up to current bar
    
    # Create data dict for on_data
    bar_data = {
        "close": prices,
        "symbols": symbols,
    }
    
    signals = strat.on_data(bar_data)
    
    ts = timestamps[idx] if idx < len(timestamps) else None
    spy_price = close[idx, spy_idx] if close.ndim == 2 else close[idx]
    
    if signals:
        for sig in signals:
            if sig.symbol == "SPY" or sig.symbol == "SPY":
                print(f"Bar {idx}: ts={ts}, SPY={spy_price:.2f}, signal={sig.direction}, strength={sig.strength:.4f}")
                bar_count += 1

if bar_count == 0:
    print("NO SIGNALS generated for SPY during first 10 crash bars!")
    print("Checking fast signal computation...")
    # Check fast signal manually
    if len(close) > 20:
        prices_20 = close[:21, spy_idx] if close.ndim == 2 else close[:21]
        sma = np.mean(prices_20[-20:])
        fast = (prices_20[-1] - sma) / sma if sma > 0 else 0.0
        print(f"Fast signal for SPY (bar 20): {fast:.4f}, sign={np.sign(fast)}")
