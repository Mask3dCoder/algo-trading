"""Walk-forward evaluation of factor_residual_v3."""
from __future__ import annotations

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from datetime import date

from quantcore.backtest.walk_forward import Fold, anchored_walk_forward
from quantcore.risk.metrics import sharpe_ratio, drawdown_stats
import quantcore.strategies.macro.factor_residual_v3  # noqa
from quantcore.strategies.base import StrategyConfig
from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, fetch_and_prepare, load_universe

# ── Config ──────────────────────────────────────────────────────────
STRATEGY_NAME = "macro_factor_residual_v3"
UNIVERSE_NAME = "expanded_15"
TC_SLIPPAGE = 0.0005   # 5 bps
TC_COMMISSION = 0.005   # $0.005/share
SYMBOLS = ["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC", "IWM", "EFA", "EEM", "HYG", "LQD", "USO", "SLV", "VNQ", "DIA"]

# Walk-forward params
INITIAL_TRAIN = 504   # 2 years
TEST_SIZE = 126       # 6 months
EMBARGO = 0.01

# ── Load data ───────────────────────────────────────────────────────
print("Loading data...")
data = load_universe(UNIVERSE_NAME, "2018-01-01", "2024-12-31", use_cache=True)
close_df, symbols = data["close"], data["symbols"]
print(f"  Universe: {len(symbols)} assets, {close_df.shape[0]} bars")
print(f"  Symbols: {symbols}")
print(f"  Date range: {close_df.index[0]} → {close_df.index[-1]}")

close_matrix = close_df.values
n_bars, n_assets = close_matrix.shape

# ── Walk-forward folds ──────────────────────────────────────────────
folds = anchored_walk_forward(n_bars, initial_train=INITIAL_TRAIN, test_size=TEST_SIZE, embargo_pct=EMBARGO)
print(f"\nWalk-forward: {len(folds)} folds")
for f in folds:
    train_dates = f"{close_df.index[f.train_start].date()} → {close_df.index[f.train_end-1].date()}"
    test_dates = f"{close_df.index[f.test_start].date()} → {close_df.index[f.test_end-1].date()}"
    print(f"  Train: [{f.train_start}:{f.train_end}] {train_dates}  Test: [{f.test_start}:{f.test_end}] {test_dates}")

# ── Run walk-forward ────────────────────────────────────────────────
StrategyClass = get_strategy_class(STRATEGY_NAME)
all_oos_rets = []
all_is_rets = []
fold_results = []

for fi, fold in enumerate(folds):
    # IS training
    is_close = close_matrix[fold.train_start:fold.train_end]
    is_dates = close_df.index[fold.train_start:fold.train_end]

    config = StrategyConfig(
        name=STRATEGY_NAME,
        params={
            "rebalance_freq": 5,
            "reg_window": 252,
            "ridge_alpha": 1e-6,
            "residual_window": 63,
            "vol_lookback": 20,
            "min_weight": 0.01,
        },
        max_position_notional=100_000.0,
    )

    strategy = StrategyClass(config)
    is_daily_rets = []

    # Walk through IS period to accumulate state (factor map, etc.)
    for t in range(252, is_close.shape[0]):  # need warmup
        window = is_close[:t+1]
        data_in = {"close": window, "symbols": symbols}
        signals = strategy.on_data(data_in)
        # Compute next-day returns for signals
        if t < is_close.shape[0] - 1 and signals:
            next_day_ret = (is_close[t+1] - is_close[t]) / np.maximum(is_close[t], 1e-10)
            port_ret = 0.0
            for sig in signals:
                idx = symbols.index(sig.symbol) if sig.symbol in symbols else -1
                if idx >= 0:
                    port_ret += sig.strength * sig.direction * next_day_ret[idx]
            is_daily_rets.append(port_ret)

    is_sr = sharpe_ratio(np.array(is_daily_rets)) if is_daily_rets else 0.0
    all_is_rets.extend(is_daily_rets)

    # OOS testing
    oos_close = close_matrix[fold.test_start:fold.test_end]
    oos_dates = close_df.index[fold.test_start:fold.test_end]
    oos_daily_rets = []

    # Reset strategy for OOS
    strategy2 = StrategyClass(config)
    # Walk through combined IS+OOS with rebalance
    combined = close_matrix[fold.train_start:fold.test_end]
    for t in range(252, combined.shape[0]):
        window = combined[:t+1]
        data_in = {"close": window, "symbols": symbols}
        signals = strategy2.on_data(data_in)
        if t < combined.shape[0] - 1 and signals:
            next_day_ret = (combined[t+1] - combined[t]) / np.maximum(combined[t], 1e-10)
            port_ret = 0.0
            for sig in signals:
                idx = symbols.index(sig.symbol) if sig.symbol in symbols else -1
                if idx >= 0:
                    port_ret += sig.strength * sig.direction * next_day_ret[idx]
            # Only count OOS period returns
            if t >= (oos_close.shape[0] - (combined.shape[0] - (fold.test_end - fold.train_start))):
                pass
            bar_in_oos = t >= (combined.shape[0] - oos_close.shape[0])
            if bar_in_oos and t < combined.shape[0] - 1:
                oos_daily_rets.append(port_ret)

    oos_sr = sharpe_ratio(np.array(oos_daily_rets)) if oos_daily_rets else 0.0
    dd = drawdown_stats(np.array(oos_daily_rets)) if oos_daily_rets else {}
    all_oos_rets.extend(oos_daily_rets)

    fold_results.append({
        "fold": fi,
        "oos_sr": oos_sr,
        "is_sr": is_sr,
        "oos_ret": np.array(oos_daily_rets) if oos_daily_rets else np.array([]),
    })

    print(f"  Fold {fi}: IS SR={is_sr:.4f}, OOS SR={oos_sr:.4f}, MaxDD={dd.get('max_drawdown', 0)*100:.2f}%")

# ── Aggregate ───────────────────────────────────────────────────────
all_oos = np.array(all_oos_rets)
all_is = np.array(all_is_rets)
agg_sr = sharpe_ratio(all_oos) if len(all_oos) > 0 else 0.0
agg_is_sr = sharpe_ratio(all_is) if len(all_is) > 0 else 0.0
agg_dd = drawdown_stats(all_oos) if len(all_oos) > 0 else {}

print(f"\n{'='*70}")
print(f"FACTOR RESIDUAL V3 — WALK-FORWARD RESULTS")
print(f"{'='*70}")
print(f"OOS Sharpe Ratio:       {agg_sr:.4f}")
print(f"IS Sharpe Ratio:        {agg_is_sr:.4f}")
print(f"OOS/IS Degradation:     {agg_sr/agg_is_sr:.2f}" if agg_is_sr != 0 else "OOS/IS Degradation:     N/A")
print(f"OOS Max Drawdown:       {agg_dd.get('max_drawdown', 0)*100:.2f}%")
print(f"OOS Annualized Return:  {np.mean(all_oos)*252*100:.2f}%")
print(f"OOS Annualized Vol:     {np.std(all_oos)*np.sqrt(252)*100:.2f}%")
if len(all_oos) > 0:
    print(f"OOS Win Rate:           {np.mean(all_oos > 0)*100:.1f}%")
    print(f"OOS Profit Factor:      {np.sum(all_oos[all_oos>0]) / abs(np.sum(all_oos[all_oos<0])):.2f}" if np.sum(all_oos[all_oos<0]) != 0 else "OOS Profit Factor:      N/A")
    print(f"OOS Daily Mean:         {np.mean(all_oos)*10000:.2f} bps")
    print(f"OOS Daily Std:          {np.std(all_oos)*10000:.2f} bps")

# ── Verdict ─────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"ACCEPTANCE CHECK")
print(f"{'='*70}")
oos_pass = agg_sr > 0.5
dd_pass = agg_dd.get('max_drawdown', 1.0) < 0.20
print(f"OOS Sharpe > 0.5:  {'PASS' if oos_pass else 'FAIL'} ({agg_sr:.4f})")
print(f"Max DD < 20%:      {'PASS' if dd_pass else 'FAIL'} ({agg_dd.get('max_drawdown', 0)*100:.2f}%)")
print(f"VERDICT: {'ACCEPT' if oos_pass and dd_pass else 'REJECT'}")