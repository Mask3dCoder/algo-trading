"""Quick diagnostic for factor_residual strategy."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from quantcore.strategies.macro.factor_residual import FactorResidualStrategy
from quantcore.strategies.base import StrategyConfig

# Create strategy
config = StrategyConfig(
    name="macro_factor_residual",
    strategy_type="macro",
    params={
        "rebalance_freq": 5,
        "reg_window": 252,
        "ridge_alpha": 1e-6,
        "resid_mom_window": 63,
        "resid_rev_weight": 0.3,
        "vol_lookback": 20,
        "min_weight": 0.01,
    }
)
strat = FactorResidualStrategy(config)
strat.initialize()

# Load some test data
from scripts.real_data_loader import load_universe
df = load_universe("original_5", "2023-01-01", "2024-12-31")

# Convert to data dict
symbols = sorted(df["symbol"].unique())
pivots = {}
for field in ["open", "high", "low", "close", "volume"]:
    p = df.pivot_table(index="timestamp", columns="symbol", values=field)
    pivots[field] = p[symbols].values

close = pivots["close"]
print(f"Data shape: {close.shape}")
print(f"Symbols: {symbols}")

# Run strategy for a few bars
rets = []
for t in range(300, 400):  # Just test 100 bars
    data = {
        "close": close[:t+1],
        "symbols": symbols,
    }
    signals = strat.on_data(data)
    
    if signals:
        print(f"\nBar {t}: Got {len(signals)} signals")
        for s in signals[:3]:  # Print first 3
            print(f"  {s.symbol}: direction={s.direction}, strength={s.strength:.4f}, "
                  f"weight={s.metadata.get('weight', 0):.4f}")
        
        # Simulate return
        if t < close.shape[0] - 1:
            bar_ret = 0.0
            total_w = 0.0
            for s in signals:
                idx = symbols.index(s.symbol)
                ret = (close[t+1, idx] - close[t, idx]) / close[t, idx]
                bar_ret += s.direction * s.strength * ret
                total_w += abs(s.strength)
            if total_w > 0:
                bar_ret /= total_w
                rets.append(bar_ret)

if rets:
    rets = np.array(rets)
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0.0
    print(f"\nDiagnostics over {len(rets)} bars:")
    print(f"  Mean return: {np.mean(rets):.6f}")
    print(f"  Std: {np.std(rets):.6f}")
    print(f"  Sharpe: {sharpe:.4f}")
else:
    print("\nNo returns generated - strategy not emitting signals or all zero")
