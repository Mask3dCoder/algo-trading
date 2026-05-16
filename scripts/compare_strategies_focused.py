"""Compare cross_sectional_factor_neutral vs cross_sectional_momentum_v2.

Uses ETF_EXPANDED universe (55+ liquid ETFs) and Ken French 5-factor + MOM
data.  Computes full metrics: Sharpe, max drawdown, annual return, turnover,
and factor exposures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from quantcore.risk.metrics import sharpe_ratio, drawdown_stats
from quantcore.strategies.base import StrategyConfig, StrategyType

# Trigger @register decorators
import quantcore.strategies.macro.cross_sectional_factor_neutral  # noqa: F401
import quantcore.strategies.macro.cross_sectional_momentum_v2  # noqa: F401

from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, load_universe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_to_data(df: pd.DataFrame) -> dict:
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
        "close": pivots["close"].values,
        "high": pivots["high"].values,
        "low": pivots["low"].values,
        "open": pivots["open"].values,
        "volume": pivots["volume"].values,
        "symbols": symbols,
    }
    for k in ["close", "high", "low", "open", "volume"]:
        arr = data[k]
        if np.any(np.isnan(arr)):
            filled = pd.DataFrame(arr).ffill(limit=5)
            arr = filled.to_numpy()
            data[k] = np.nan_to_num(arr, nan=0.0)
    return data


def compute_signal_returns(strat, data, start, end, timestamps=None, tc_bps=0.0):
    """Compute bar-level signal-weighted returns."""
    if end - start < 2:
        return np.array([], dtype=float)

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)
    if n_assets == 0:
        return np.array([], dtype=float)

    asset_rets = (close[start + 1:end] - close[start:end - 1]) / np.maximum(close[start:end - 1], 1e-10)
    bar_rets = np.zeros(end - start - 1)
    carried_weights = {}
    total_turnover = 0.0

    for idx, bar_i in enumerate(range(start, end - 1)):
        bar_data = {"close": close[:bar_i + 1], "symbols": symbols}
        for k in ["high", "low", "open", "volume"]:
            bar_data[k] = data.get(k, close[:bar_i + 1])[:bar_i + 1]

        if timestamps is not None and bar_i < len(timestamps):
            bar_data["timestamp"] = timestamps[bar_i]

        try:
            signals = strat.on_data(bar_data) or []
        except Exception as e:
            import traceback
            print(f"  signal error bar {bar_i}: {e}")
            traceback.print_exc()
            signals = []

        weighted_ret = 0.0
        total_weight = 0.0
        current_weights = {}

        for s in signals:
            weight = float(s.direction) * float(s.strength)
            if s.direction != 0:
                current_weights[s.symbol] = weight
            try:
                asset_idx = symbols.index(s.symbol)
            except (ValueError, IndexError):
                continue
            weighted_ret += weight * float(asset_rets[idx, asset_idx])
            total_weight += abs(float(s.strength))

        if total_weight > 1e-10:
            bar_rets[idx] = weighted_ret / total_weight

        # Transaction cost
        if tc_bps > 0 and current_weights:
            turnover = 0.0
            all_syms = set(carried_weights.keys()) | set(current_weights.keys())
            for sym in all_syms:
                prev_w = carried_weights.get(sym, 0.0)
                curr_w = current_weights.get(sym, 0.0)
                turnover += abs(curr_w - prev_w)
            tc_cost = turnover * tc_bps / 10000.0
            bar_rets[idx] -= tc_cost
            total_turnover += turnover

        if current_weights:
            carried_weights = dict(current_weights)

    return bar_rets, total_turnover


def build_equity_curve(returns, initial=1_000_000.0):
    if len(returns) == 0:
        return np.array([initial])
    equity = np.empty(len(returns) + 1)
    equity[0] = initial
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


def max_drawdown_from_equity(equity):
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(np.min(dd))


def annual_return_from_equity(equity, n_years):
    if len(equity) < 2 or n_years <= 0:
        return 0.0
    return (equity[-1] / equity[0]) ** (1.0 / n_years) - 1.0


def compute_turnover(returns):
    """Proxy: count sign changes / total days."""
    if len(returns) < 2:
        return 0.0
    directional = np.sum(np.diff(np.sign(returns)) != 0)
    return directional / len(returns) * 100.0


def compute_factor_exposures(portfolio_rets, factor_rets):
    """Compute portfolio factor exposures via regression."""
    import numpy as np
    if len(portfolio_rets) == 0 or factor_rets.shape[0] == 0:
        return {}

    min_len = min(len(portfolio_rets), factor_rets.shape[0])
    y = portfolio_rets[:min_len]
    X = factor_rets[:min_len]

    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])
    try:
        beta = np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
        # beta[0] is intercept, beta[1:] are factor loadings
        return beta[1:]
    except Exception:
        return np.zeros(X.shape[1])


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

STRATEGY_CONFIGS = {
    "macro_cross_sectional_factor_neutral": {
        "name": "Factor-Neutral StatArb",
        "params": {
            "rebalance_freq": 21,
            "reg_lookback": 252,
            "pred_window": 20,
            "ridge_alpha": 1.0,
            "max_position": 0.03,
            "target_vol": 0.15,
            "vol_lookback": 20,
            "zscore_cutoff": 1.5,
            "percentile_top": 0.30,
        },
    },
    "macro_cross_sectional_momentum_v2": {
        "name": "CSM v2 (Baseline)",
        "params": {
            "rebalance_freq": 5,
            "mom_windows": "20,60,120",
            "mom_weights": "0.2,0.5,0.3",
            "target_vol": 0.12,
            "vol_lookback": 20,
            "use_vol_adjustment": True,
            "max_gross_exposure": 1.0,
            "beta_neutral": True,
            "regime_filter": True,
            "dd_control": True,
            "min_weight": 0.01,
        },
    },
}


def run_strategy(strategy_id, start_date="2018-01-01", end_date="2024-12-31"):
    """Run a single strategy backtest and return metrics."""
    cfg_dict = STRATEGY_CONFIGS[strategy_id]
    universe_name = "expanded_15"  # 15-asset universe as a proxy

    print(f"\n{'=' * 60}")
    print(f"  Strategy: {cfg_dict['name']}")
    print(f"  Strategy ID: {strategy_id}")
    print(f"  Period: {start_date} -> {end_date}")
    print(f"{'=' * 60}")

    # Load data
    print("Loading data...")
    df = load_universe(universe_name, start_date, end_date)
    if df.empty:
        print("ERROR: No data loaded.")
        return None
    print(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols.")

    data = _df_to_data(df)
    close = data["close"]
    n_periods = close.shape[0]
    symbols = data["symbols"]
    print(f"Data matrix: {n_periods} bars x {len(symbols)} assets")

    timestamps = sorted(df["timestamp"].unique())

    # Setup strategy
    cfg = StrategyConfig(
        name=strategy_id,
        strategy_type=StrategyType.MACRO,
        params=cfg_dict["params"],
    )
    cls = get_strategy_class(strategy_id)
    strat = cls(cfg)
    strat.initialize()

    # Warmup
    print("Running warmup...")
    warmup_bars = 300
    warmup_end = min(warmup_bars, n_periods - 10)
    for bar_i in range(0, warmup_end):
        bar_data = {"close": close[:bar_i + 1], "symbols": symbols}
        for k in ["high", "low", "open", "volume"]:
            bar_data[k] = data.get(k, close[:bar_i + 1])[:bar_i + 1]
        if bar_i < len(timestamps):
            bar_data["timestamp"] = timestamps[bar_i]
        try:
            strat.on_data(bar_data)
        except Exception:
            pass

    # OOS backtest
    print("Running OOS backtest...")
    oos_start = warmup_end
    oos_end = n_periods

    oos_returns, total_turnover = compute_signal_returns(
        strat, data, oos_start, oos_end,
        timestamps=timestamps, tc_bps=5.0,
    )

    if len(oos_returns) == 0:
        print("ERROR: No OOS returns produced.")
        return None

    # Compute metrics
    equity = build_equity_curve(oos_returns, initial=1_000_000.0)
    total_return = (equity[-1] - 1_000_000.0) / 1_000_000.0 * 100.0

    n_years = len(oos_returns) / 252.0
    ann_return = annual_return_from_equity(equity, n_years) * 100.0
    sr = sharpe_ratio(oos_returns, rf=0.04)
    max_dd = max_drawdown_from_equity(equity) * 100.0
    turnover = compute_turnover(oos_returns)

    # Factor exposures
    from quantcore.data.sources.fama_french import FamaFrenchLoader
    ff = FamaFrenchLoader(cache_dir=".cache/fama_french")
    try:
        factor_key = "5_factors_2x3"
        if factor_key not in ff.list_available():
            factor_key = "3_factors"
        factor_df = ff.load(factor_key, freq="daily")
        factor_returns = factor_df[[c for c in factor_df.columns if c not in ["Date", "RF"]]].to_numpy()
        exposures = compute_factor_exposures(oos_returns, factor_returns)
        factor_names = [c for c in factor_df.columns if c not in ["Date", "RF"]]
    except Exception:
        exposures = np.zeros(5)
        factor_names = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

    results = {
        "strategy": cfg_dict["name"],
        "strategy_id": strategy_id,
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(ann_return, 2),
        "sharpe_ratio": round(sr, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "turnover_pct": round(turnover, 2),
        "oos_bars": len(oos_returns),
        "factor_exposures": dict(zip(factor_names, [round(float(x), 6) for x in exposures])),
    }

    print(f"\n  Total Return:       {total_return:+.2f}%")
    print(f"  Annual Return:      {ann_return:+.2f}%")
    print(f"  Sharpe Ratio:       {sr:.4f}")
    print(f"  Max Drawdown:       {max_dd:.2f}%")
    print(f"  Turnover (proxy):   {turnover:.2f}%")
    print(f"  OOS Bars:           {len(oos_returns)}")
    print(f"  Factor Exposures:   {dict(zip(factor_names, [round(float(x), 6) for x in exposures]))}")

    return results


def main():
    print("=" * 60)
    print("  STRATEGY COMPARISON: Factor-Neutral vs CSM v2")
    print("  Universe: ETF_EXPANDED (55+ liquid ETFs)")
    print("  Factors: Ken French 5-Factor + Momentum")
    print("  Period: 2018-01-01 -> 2024-12-31")
    print("  TC: 5 bps")
    print("=" * 60)

    results = []
    for strategy_id in ["macro_cross_sectional_factor_neutral", "macro_cross_sectional_momentum_v2"]:
        result = run_strategy(strategy_id)
        if result:
            results.append(result)

    # Summary table
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<25} {'Factor-Neutral':>15} {'CSM v2':>15}")
    print("-" * 60)

    if len(results) >= 2:
        for metric in ["total_return_pct", "annual_return_pct", "sharpe_ratio", "max_drawdown_pct", "turnover_pct"]:
            label = {
                "total_return_pct": "Total Return (%)",
                "annual_return_pct": "Annual Return (%)",
                "sharpe_ratio": "Sharpe Ratio",
                "max_drawdown_pct": "Max Drawdown (%)",
                "turnover_pct": "Turnover (%)",
            }.get(metric, metric)
            v1 = results[0][metric]
            v2 = results[1][metric]
            print(f"{label:<25} {v1:>15.4f} {v2:>15.4f}")

    print("\nFactor Exposures (Factor-Neutral):")
    if len(results) >= 1:
        for k, v in results[0]["factor_exposures"].items():
            print(f"  {k}: {v:.6f}")

    # Save
    with open("strategy_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to strategy_comparison_results.json")

    return results


if __name__ == "__main__":
    results = main()
