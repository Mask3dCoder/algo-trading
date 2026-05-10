"""Test residual signal generation step by step."""

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

# Load test data
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

# Run strategy and collect diagnostics
all_signals = []
all_weights = []
all_residuals = []

for t in range(300, 500):  # Test 200 bars
    data = {
        "close": close[:t+1],
        "symbols": symbols,
    }
    
    # Capture output
    import io
    import contextlib
    
    signals = strat.on_data(data)
    
    if signals:
        print(f"\nBar {t}: Got {len(signals)} signals")
        for s in signals[:5]:  # Print first 5
            print(f"  {s.symbol}: dir={s.direction}, strength={s.strength:.4f}, "
                  f"weight={s.metadata.get('weight', 0):.4f}, "
                  f"signal={s.metadata.get('signal', 0):.4f}, "
                  f"residual={s.metadata.get('residual', 0):.6f}")
        all_signals.append(signals)
    
    # Check internal state
    if hasattr(strat, '_last_betas') and strat._last_betas is not None:
        if t == 350:  # Print once
            print(f"\n[DIAG] Last betas shape: {strat._last_betas.shape}")
            print(f"[DIAG] Sample betas (first 2 assets):")
            for i in range(min(2, len(symbols))):
                print(f"  {symbols[i]}: {strat._last_betas[i, :]}")
            
            print(f"\n[DIAG] Checking residual values...")
            # Need to check residuals - but they're not stored
            # Let me add a method to check

print(f"\nTotal signal groups: {len(all_signals)}")
if all_signals:
    # Flatten and compute returns
    print("Strategy is generating signals")
else:
    print("WARNING: Strategy not generating any signals!")
    print("Possible causes:")
    print("1. Rebalance freq not met")
    print("2. Weights below min_weight")
    print("3. Factor neutrality constraints too tight")
    print("\nLet me check the weights directly...")
    
    # Manually call internal methods
    strat._days_since_rebalance = 5  # Force rebalance
    
    daily_rets = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
    factors = strat._build_factors(daily_rets)
    residuals = strat._compute_residuals(daily_rets, factors)
    signals = strat._compute_signals(residuals)
    weights = strat._compute_weights(signals, daily_rets)
    
    print(f"\n[DIAG] Signals: min={signals.min():.4f}, max={signals.max():.4f}, "
          f"mean={signals.mean():.4f}")
    print(f"[DIAG] Weights: min={weights.min():.4f}, max={weights.max():.4f}, "
          f"sum={weights.sum():.4f}")
    
    if strat._last_betas is not None:
        factor_exp = strat._last_betas.T @ weights
        print(f"[DIAG] Factor exposures: {factor_exp}")
        print(f"[DIAG] Market exposure: {weights.sum():.4f}")
