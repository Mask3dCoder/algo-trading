"""Simple test of fast/slow alignment filter."""

from __future__ import annotations
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
import numpy as np
import pandas as pd
from quantcore.backtest.walk_forward import anchored_walk_forward
from quantcore.risk.metrics import sharpe_ratio
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, load_universe

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

STRATEGY_ID = "macro_cta_trend"
UNIVERSE_NAME = "cta"
FULL_PERIOD = {"start": "2018-01-01", "end": "2024-12-31"}

def run_test(use_alignment=False):
    """Run backtest with optional alignment filter."""
    df = load_universe(UNIVERSE_NAME, FULL_PERIOD["start"], FULL_PERIOD["end"], use_cache=True)
    if df.empty:
        return None
    
    data = {}
    symbols = sorted(df["symbol"].unique())
    pivots = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field)
        pivots[field] = pivoted[symbols] if all(s in pivoted.columns for s in symbols) else pivoted
    if pivots["close"].empty:
        return None
    data = {
        "close": pivots["close"].values, "high": pivots["high"].values,
        "low": pivots["low"].values, "open": pivots["open"].values,
        "volume": pivots["volume"].values, "symbols": symbols,
    }
    for k in ["close", "high", "low", "open", "volume"]:
        arr = data[k]
        if np.any(np.isnan(arr)):
            data[k] = np.nan_to_num(pd.DataFrame(arr).ffill().values, nan=0.0)
    
    close = data["close"]
    n_periods = close.shape[0]
    timestamps = sorted(df["timestamp"].unique()) if "timestamp" in df.columns else None
    
    initial_train = min(200, n_periods // 3)
    test_size = min(50, max(20, n_periods // 10))
    embargo = max(1, int(0.02 * test_size))
    if initial_train + test_size + embargo >= n_periods:
        return None
    
    try:
        folds = anchored_walk_forward(n_periods, initial_train=initial_train,
            test_size=test_size, embargo_pct=embargo/test_size if test_size > 0 else 0.0)
    except Exception as e:
        log.debug("Fold failed: %s", e)
        return None
    
    all_returns = []
    positions = {}
    
    for fold in folds:
        if fold.train_end > n_periods or fold.test_end > n_periods:
            continue
        try:
            params = {
                "fast_window": 20, "medium_window": 50, "slow_window": 200,
                "fast_weight": 0.2, "medium_weight": 0.3, "slow_weight": 0.5,
                "vol_target": 0.15, "vol_lookback": 60, "rebalance_freq": 5,
                "vol_regime_threshold": 0.30, "vol_regime_lookback": 20, "min_vol_scale": 0.20,
                "alignment_filter": use_alignment,  # This is the key parameter
            }
            cfg = StrategyConfig(name=STRATEGY_ID, strategy_type=StrategyType.MACRO, params=params)
            cls = get_strategy_class(STRATEGY_ID)
            strat = cls(cfg)
            strat.initialize()
            
            # Warmup
            for bar_i in range(fold.train_start, fold.train_end):
                bar_data = {k: (v[:bar_i+1] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in data.items()}
                if timestamps and bar_i < len(timestamps):
                    ts = timestamps[bar_i]
                    if hasattr(strat, "set_timestamp"):
                        strat.set_timestamp(ts)
                    bar_data["timestamp"] = ts
                strat.on_data(bar_data)
            
            # OOS
            for bar_i in range(fold.test_start, fold.test_end - 1):
                bar_data = {k: (v[:bar_i+1] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in data.items()}
                if timestamps and bar_i < len(timestamps):
                    ts = timestamps[bar_i]
                    if hasattr(strat, "set_timestamp"):
                        strat.set_timestamp(ts)
                    bar_data["timestamp"] = ts
                
                signals = strat.on_data(bar_data) or []
                
                # Compute bar return
                if bar_i > fold.test_start:
                    bar_ret = 0.0
                    for s in signals:
                        if s.direction == 0:
                            continue
                        try:
                            asset_idx = symbols.index(s.symbol)
                            ret = (close[bar_i, asset_idx] - close[bar_i-1, asset_idx]) / close[bar_i-1, asset_idx]
                            bar_ret += float(ret) * float(s.strength) * float(s.direction)
                        except (ValueError, IndexError):
                            continue
                    all_returns.append(bar_ret)
                
                # Track trades
                for s in signals:
                    if s.direction == 0:
                        continue
                    if s.symbol not in positions:
                        positions[s.symbol] = {
                            "direction": s.direction,
                            "entry_price": float(data["close"][bar_i, symbols.index(s.symbol)]) if bar_i < data["close"].shape[0] else None,
                        }
                
                current_syms = {s.symbol for s in signals if s.direction != 0}
                for sym in list(positions.keys()):
                    if sym not in current_syms:
                        positions.pop(sym)
                        
        except Exception as e:
            log.debug("Fold failed: %s", e)
            continue
    
    if not all_returns:
        return None
    
    returns = np.array(all_returns)
    return {
        "sharpe": float(sharpe_ratio(returns)),
        "win_rate": float(np.mean(returns > 0)),
        "n_bars": len(returns),
    }

def main():
    print("=" * 70)
    print(" Phase 2.2: Simple Alignment Filter Test")
    print("=" * 70)
    
    # Baseline
    print("\nRunning: BASELINE (no filter)")
    baseline = run_test(use_alignment=False)
    if baseline:
        print(f"  Sharpe: {baseline['sharpe']:+.4f}  WinRate: {baseline['win_rate']:.1%}")
        print(f"  Bars: {baseline['n_bars']}")
    
    # Alignment filter
    print("\nRunning: Alignment Filter (fast/slow must agree)")
    aligned = run_test(use_alignment=True)
    if aligned:
        print(f"  Sharpe: {aligned['sharpe']:+.4f}  WinRate: {aligned['win_rate']:.1%}")
        print(f"  Bars: {aligned['n_bars']}")
    
    # Comparison
    if baseline and aligned:
        print("\n" + "=" * 80)
        print(" COMPARISON")
        print("=" * 80)
        diff = aligned['sharpe'] - baseline['sharpe']
        if diff > 0.05:
            print(f"Alignment filter IMPROVES Sharpe by {diff:+.4f} -> VALID")
        else:
            print(f"Alignment filter changes Sharpe by {diff:+.4f} -> NOT useful")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
