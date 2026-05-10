"""Phase 2.0: Trade Quality Attribution for macro_cta_trend."""

from __future__ import annotations
import sys
from pathlib import Path

# Add repo root to path BEFORE importing quantcore
repo_root = Path.cwd()
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import random
import logging
from typing import Any
import numpy as np
import pandas as pd
from quantcore.backtest.walk_forward import anchored_walk_forward
from quantcore.risk.metrics import sharpe_ratio
import quantcore.strategies.macro.cta_trend
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, load_universe, validate_data

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

THRESHOLD = 0.2
STRATEGY_ID = "macro_cta_trend"
UNIVERSE_NAME = "cta"
FULL_PERIOD = {"start": "2018-01-01", "end": "2024-12-31"}
STRATEGY_PARAMS = {
    "fast_window": 20, "medium_window": 50, "slow_window": 200,
    "fast_weight": 0.2, "medium_weight": 0.3, "slow_weight": 0.5,
    "vol_target": 0.15, "vol_lookback": 60, "rebalance_freq": 5,
    "vol_regime_threshold": 0.30, "vol_regime_lookback": 20, "min_vol_scale": 0.20,
}

def _df_to_data(df):
    symbols = sorted(df["symbol"].unique())
    n_assets = len(symbols)
    if n_assets == 0:
        return {"close": np.array([[]], dtype=float), "symbols": []}
    pivots = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field)
        pivots[field] = pivoted[symbols] if all(s in pivoted.columns for s in symbols) else pivoted
    if pivots["close"].empty:
        return {"close": np.array([[]], dtype=float), "symbols": symbols}
    data = {
        "close": pivots["close"].values, "high": pivots["high"].values,
        "low": pivots["low"].values, "open": pivots["open"].values,
        "volume": pivots["volume"].values, "symbols": symbols,
    }
    for k in ["close", "high", "low", "open", "volume"]:
        arr = data[k]
        if np.any(np.isnan(arr)):
            data[k] = np.nan_to_num(pd.DataFrame(arr).ffill().values, nan=0.0)
    return data

def simulate_skipped_pnl(opp, close, symbols, hold_bars=10):
    asset = opp.get("asset")
    entry_price = opp.get("entry_price")
    direction = opp.get("direction", 0)
    if asset not in symbols or entry_price is None or direction == 0:
        return None
    try:
        asset_idx = symbols.index(asset)
    except ValueError:
        return None
    entry_bar = opp.get("bar_idx", 0)
    exit_bar = min(entry_bar + hold_bars, close.shape[0] - 1)
    if exit_bar <= entry_bar:
        return None
    exit_price = float(close[exit_bar, asset_idx])
    pnl_abs = (exit_price - entry_price) * direction
    pnl_pct = pnl_abs / entry_price if entry_price > 0 else 0.0
    return {"pnl_pct": float(pnl_pct)}

def run_strategy_with_attribution(strategy_id, df, threshold, run_random=False, random_rate=0.0):
    data = _df_to_data(df)
    close = data["close"]
    n_periods = close.shape[0]
    symbols = data["symbols"]
    timestamps = sorted(df["timestamp"].unique()) if "timestamp" in df.columns else None
    if n_periods < 100 or len(symbols) == 0:
        return None
    initial_train = min(200, n_periods // 3)
    test_size = min(50, max(20, n_periods // 10))
    embargo = max(1, int(0.02 * test_size))
    if initial_train + test_size + embargo >= n_periods:
        initial_train = max(50, n_periods // 4)
        test_size = max(10, n_periods // 8)
        embargo = max(0, int(0.02 * test_size))
    if initial_train + test_size + embargo >= n_periods:
        return None
    try:
        folds = anchored_walk_forward(n_periods, initial_train=initial_train,
            test_size=test_size, embargo_pct=embargo/test_size if test_size > 0 else 0.0)
    except Exception:
        return None
    all_opportunities = []
    all_trades = []
    rng = random.Random(42)
    for fold in folds:
        if fold.train_end > n_periods or fold.test_end > n_periods:
            continue
        try:
            params = dict(STRATEGY_PARAMS)
            params["min_signal_strength"] = threshold if not run_random else 0.0
            cfg = StrategyConfig(name=strategy_id, strategy_type=StrategyType.MACRO, params=params)
            cls = get_strategy_class(strategy_id)
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
            strat.reset_diagnostics()
            positions = {}
            for bar_i in range(fold.test_start, fold.test_end - 1):
                bar_data = {k: (v[:bar_i+1] if isinstance(v, np.ndarray) and v.ndim >= 1 else v) for k, v in data.items()}
                if timestamps and bar_i < len(timestamps):
                    ts = timestamps[bar_i]
                    if hasattr(strat, "set_timestamp"):
                        strat.set_timestamp(ts)
                    bar_data["timestamp"] = ts
                signals = strat.on_data(bar_data) or []
                opps = strat.get_all_opportunities() if hasattr(strat, "get_all_opportunities") else []
                for opp in opps:
                    opp["bar_idx"] = bar_i
                    if run_random and opp.get("was_executed", False) and rng.random() < random_rate:
                        opp["was_executed"] = False
                        opp["randomly_skipped"] = True
                    all_opportunities.append(opp)
                for s in signals:
                    if s.direction == 0:
                        continue
                    if s.symbol not in positions:
                        positions[s.symbol] = {
                            "direction": s.direction,
                            "entry_ts": bar_data.get("timestamp"),
                            "entry_price": float(data["close"][bar_i, data["symbols"].index(s.symbol)]) if bar_i < data["close"].shape[0] else None,
                        }
                current_syms = {s.symbol for s in signals if s.direction != 0}
                for sym in list(positions.keys()):
                    if sym not in current_syms:
                        pos = positions.pop(sym)
                        exit_price = float(data["close"][bar_i, data["symbols"].index(sym)]) if bar_i < data["close"].shape[0] else None
                        pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
                        pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
                        all_trades.append({"pnl_pct": float(pnl_pct), "symbol": sym})
            strat.reset_diagnostics()
        except Exception as e:
            log.debug("Fold failed: %s", e)
            continue
    return {"opportunities": all_opportunities, "trades": all_trades}

def compute_metrics(pnls):
    if not pnls or len(pnls) == 0:
        return {"n": 0, "avg": 0.0, "sharpe": 0.0, "win_rate": 0.0}
    arr = np.array(pnls)
    return {
        "n": len(arr),
        "avg": float(np.mean(arr)),
        "sharpe": float(sharpe_ratio(arr)),
        "win_rate": float(np.mean(arr > 0)),
    }

def main():
    print("=" * 70)
    print(" Phase 2.0: Trade Quality Attribution")
    print("=" * 70)
    print("\nLoading data...")
    df = load_universe(UNIVERSE_NAME, FULL_PERIOD["start"], FULL_PERIOD["end"], use_cache=True)
    if df.empty:
        print("ERROR: No data")
        return
    result = run_strategy_with_attribution(STRATEGY_ID, df, THRESHOLD)
    if not result:
        print("ERROR: No results")
        return
    opps = result["opportunities"]
    executed = [o for o in opps if o.get("was_executed", False)]
    skipped = [o for o in opps if not o.get("was_executed", False)]
    print("\nTrade Attribution Results (threshold={}):".format(THRESHOLD))
    print("  Total opportunities: {}".format(len(opps)))
    print("  Executed: {} ({:.1f}%)".format(len(executed), len(executed)/max(len(opps),1)*100))
    print("  Skipped: {} ({:.1f}%)".format(len(skipped), len(skipped)/max(len(opps),1)*100))
    data = _df_to_data(df)
    close = data["close"]
    symbols = data["symbols"]
    skipped_pnls = []
    for opp in skipped:
        pnl_info = simulate_skipped_pnl(opp, close, symbols)
        if pnl_info:
            skipped_pnls.append(pnl_info["pnl_pct"])
    executed_pnls = [t["pnl_pct"] for t in result["trades"]]
    exec_metrics = compute_metrics(executed_pnls)
    skip_metrics = compute_metrics(skipped_pnls)
    print("\n" + "-" * 60)
    print("COMPARISON: Executed vs Skipped")
    print("-" * 60)
    print("{:<15s} {:>12s} {:>12s}".format('Metric', 'Executed', 'Skipped'))
    print("-" * 15 + " " + "-" * 12 + " " + "-" * 12)
    print("{:<15s} {:>12d} {:>12d}".format('N', exec_metrics['n'], skip_metrics['n']))
    print("{:<15s} {:>+12.6f} {:>+12.6f}".format('Avg PnL', exec_metrics['avg'], skip_metrics['avg']))
    print("{:<15s} {:>+12.4f} {:>+12.4f}".format('Sharpe', exec_metrics['sharpe'], skip_metrics['sharpe']))
    print("{:<15s} {:>11.1%} {:>11.1%}".format('Win Rate', exec_metrics['win_rate'], skip_metrics['win_rate']))
    avg_diff = skip_metrics['avg'] - exec_metrics['avg']
    print("\n" + "-" * 60)
    print("INTERPRETATION")
    print("-" * 60)
    if avg_diff < -0.001:
        print("CASE A: Skipped trades are WORSE -> Fast signal has real edge")
        print("  -> Keep and refine the filter")
    elif abs(avg_diff) <= 0.001:
        print("CASE B: Skipped approx Executed -> Filter is just reducing variance")
        print("  -> Equivalent to lowering position size")
    else:
        print("CASE C: Skipped trades are BETTER -> You are filtering out alpha!")
        print("  -> Strategy is mis-specified")
    skip_rate = len(skipped) / max(len(opps), 1)
    print("\n" + "-" * 60)
    print("RANDOM SKIPPING CONTROL")
    print("-" * 60)
    print("Running random skip at rate: {:.1%}".format(skip_rate))
    result_random = run_strategy_with_attribution(STRATEGY_ID, df, THRESHOLD, run_random=True, random_rate=skip_rate)
    if result_random:
        random_pnls = [t["pnl_pct"] for t in result_random["trades"]]
        random_metrics = compute_metrics(random_pnls)
        print("Random skip Sharpe: {:+.4f}".format(random_metrics['sharpe']))
        print("Signal-gated Sharpe: {:+.4f}".format(exec_metrics['sharpe']))
        if abs(random_metrics['sharpe'] - exec_metrics['sharpe']) < 0.02:
            print("-> Filter is NOT intelligent - just reducing activity")
        else:
            print("-> Filter has intelligence - different from random")
    output = {
        "threshold": THRESHOLD,
        "executed_metrics": exec_metrics,
        "skipped_metrics": skip_metrics,
        "skip_rate": skip_rate,
        "interpretation": "CASE A" if avg_diff < -0.001 else ("CASE C" if avg_diff > 0.001 else "CASE B"),
    }
    output_path = repo_root / "results" / "phase2_0_trade_attribution.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print("\nResults saved to: {}".format(output_path))

if __name__ == "__main__":
    main()
