"""Simple test: run strategy on COVID data and print signals."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PARENT = REPO_ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

import numpy as np
import pandas as pd

from scripts.real_data_backtest import _df_to_data, _setup_strategy, load_universe
from quantcore.strategies.macro.cta_trend import CTATrendStrategy

# Load data
df = load_universe("cta", "2019-06-01", "2020-06-30", use_cache=False)
data = _df_to_data(df)
close = data["close"]
symbols = data["symbols"]
spy_idx = symbols.index("SPY") if "SPY" in symbols else 0

print(f"Symbols: {symbols}")
print(f"SPY index: {spy_idx}")
print(f"Close shape: {close.shape}")

# Create strategy
strat = _setup_strategy("macro_cta_trend")
strat.initialize()

# Run strategy for crash period (Feb 19 - Mar 23)
timestamps = sorted(df["timestamp"].unique())
# Strip timezones
timestamps = [pd.Timestamp(ts).replace(tzinfo=None) if hasattr(ts, 'tzinfo') and ts.tzinfo is not None else pd.Timestamp(ts) for ts in timestamps]

crash_start = pd.Timestamp("2020-02-19").replace(tzinfo=None)
crash_end = pd.Timestamp("2020-03-23").replace(tzinfo=None)

crash_indices = [i for i, ts in enumerate(timestamps) if crash_start <= pd.Timestamp(ts) <= crash_end]
print(f"\nCrash period: {len(crash_indices)} bars")

# Track signals for SPY
spy_signals = []
for idx in crash_indices:
    # Create data dict up to current bar
    bar_data = {
        "close": close[:idx+1],
        "symbols": symbols,
        "timestamp": timestamps[idx],
    }
    
    signals = strat.on_data(bar_data)
    
    for sig in signals:
        if sig.symbol == "SPY" or (hasattr(sig, 'symbol') and 'SPY' in str(sig.symbol)):
            spy_signals.append({
                "bar": idx,
                "ts": timestamps[idx],
                "direction": sig.direction,
                "strength": sig.strength,
                "metadata": sig.metadata,
            })

print(f"\nSPY signals during crash: {len(spy_signals)}")
for s in spy_signals[:10]:
    print(f"  Bar {s['bar']}: {s['ts']}, dir={s['direction']}, strength={s['strength']:.4f}, fast={s['metadata'].get('fast', 0):.4f}")

if len(spy_signals) == 0:
    print("\nNO SIGNALS FOR SPY DURING CRASH!")
    print("Let me check what the strategy computes for a specific bar...")
    
    # Check bar 170 (around Feb 19)
    test_idx = 170
    if test_idx < len(timestamps):
        bar_data = {
            "close": close[:test_idx+1],
            "symbols": symbols,
            "timestamp": timestamps[test_idx],
        }
        
        # Manually compute what the strategy should compute
        prices = close[:test_idx+1, spy_idx]
        fast = (prices[-1] - np.mean(prices[-20:])) / np.mean(prices[-20:]) if len(prices) >= 20 else 0.0
        print(f"\nManual check for bar {test_idx} (SPY):")
        print(f"  Price: {prices[-1]:.2f}")
        print(f"  SMA20: {np.mean(prices[-20:]):.2f}")
        print(f"  Fast signal: {fast:.4f}")
        print(f"  Fast dir: {np.sign(fast)}")
        
        # Call on_data
        signals = strat.on_data(bar_data)
        print(f"  Signals from strategy: {len(signals)}")
        for sig in signals:
            print(f"    {sig.symbol}: dir={sig.direction}, strength={sig.strength:.4f}")
