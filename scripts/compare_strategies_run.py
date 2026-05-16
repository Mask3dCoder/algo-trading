"""Compare cross_sectional_factor_neutral vs cross_sectional_momentum (baseline).

Runs both strategies on the same data (expanded_15 universe, 2018-2024),
computes: total return, Sharpe, max drawdown, annual return, turnover, and
factor exposures.  Also verifies the factor-neutral strategy has near-zero
residual factor exposures.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from quantcore.risk.metrics import sharpe_ratio, drawdown_stats
from quantcore.strategies.base import StrategyConfig, StrategyType

# Trigger @register decorators
import quantcore.strategies.macro.cross_sectional_factor_neutral  # noqa: F401, E501
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401, E501

from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import UNIVERSES, load_universe


# ---------------------------------------------------------------------------
# Helpers (match run_backtest_focused.py exactly)
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
            filled = pd.DataFrame(arr).ffill(limit=5).to_numpy()
            data[k] = np.nan_to_num(filled, nan=0.0)
    return data


# ---------------------------------------------------------------------------
# Signal return engine (exact copy from run_backtest_focused.py)
# ---------------------------------------------------------------------------

def compute_signal_returns(strat, data, start, end, timestamps=None, tc_bps=0.0):
    if end - start < 2:
        return np.array([], dtype=float), []

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)
    if n_assets == 0:
        return np.array([], dtype=float), []

    asset_rets = np.diff(close, axis=0) / np.maximum(close[:-1], 1e-10)
    bar_rets = np.zeros(end - start - 1)
    trades = []
    positions = {}
    carried_weights = {}

    for idx, bar_i in enumerate(range(start, end - 1)):
        bar_data = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v

        if timestamps is not None and bar_i < len(timestamps):
            bar_data["timestamp"] = timestamps[bar_i]

        try:
            signals = strat.on_data(bar_data) or []
        except Exception as e:
            print(f"  signal error bar {bar_i}: {e}")
            signals = []

        weighted_ret = 0.0
        total_weight = 0.0
        current_weights = {}
        current_signals = {}

        for s in signals:
            weight = float(s.direction) * float(s.strength)
            if s.direction != 0:
                current_weights[s.symbol] = weight
                current_signals[s.symbol] = {
                    "direction": s.direction,
                    "strength": s.strength,
                    "metadata": getattr(s, "metadata", {}),
                }
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
            all_syms = set(carried_weights.keys()) | set(current_weights.keys())
            turnover = sum(
                abs(current_weights.get(sym, 0.0) - carried_weights.get(sym, 0.0))
                for sym in all_syms
            )
            tc_cost = turnover * tc_bps / 10000.0
            bar_rets[idx] -= tc_cost

        # Trade tracking (simplified)
        for sym in list(positions.keys()):
            if sym not in current_signals or current_signals[sym]["direction"] != positions[sym]["direction"]:
                del positions[sym]
        for sym in current_signals:
            if sym not in positions and bar_i < close.shape[0]:
                positions[sym] = current_signals[sym]

        if current_weights:
            carried_weights = dict(current_weights)

    return bar_rets, trades


def build_equity_curve(returns, initial=1_000_000.0):
    if len(returns) == 0:
        return np.array([initial])
    equity = np.empty(len(returns) + 1)
    equity[0] = initial
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)
    return equity


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def annual_return_from_equity(equity, n_years):
    if len(equity) < 2 or n_years <= 0:
        return 0.0
    return (equity[-1] / equity[0]) ** (1.0 / n_years) - 1.0


def compute_turnover(returns):
    if len(returns) < 2:
        return 0.0
    directional = np.sum(np.diff(np.sign(returns)) != 0)
    return directional / len(returns) * 100.0


def max_drawdown_from_equity(equity):
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return np.min((equity - peak) / peak)


# ---------------------------------------------------------------------------
# Fama-French factor exposure helper
# ---------------------------------------------------------------------------

def compute_factor_exposures(portfolio_rets, factor_rets):
    """Regression of portfolio returns on factor returns using pseudo-inverse."""
    import numpy as np

    min_len = min(len(portfolio_rets), factor_rets.shape[0])
    if min_len < 2:
        return None
    y = portfolio_rets[:min_len]
    X = factor_rets[:min_len]
    # Standardise factors
    X_std = (X - np.mean(X, axis=0)) / (np.std(X, axis=0, ddof=1) + 1e-12)
    X_with_const = np.column_stack([np.ones(len(X)), X_std])
    try:
        beta = np.linalg.pinv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
        return beta[1:]  # drop intercept
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Configs
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
    "macro_cross_sectional_momentum": {
        "name": "CSM v1 (Baseline)",
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


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_single(strategy_id, start_date, end_date, universe_name, tc_bps=5.0):
    cfg_dict = STRATEGY_CONFIGS[strategy_id]
    name = cfg_dict["name"]

    print(f"\n{'=' * 60}")
    print(f"  Strategy: {name} ({strategy_id})")
    print(f"{'=' * 60}")
    print(f"  Universe:  {universe_name}")
    print(f"  Period:    {start_date} -> {end_date}")
    print(f"  TC cost:   {tc_bps} bps")
    print()

    # Load data
    print("  Loading data...")
    df = load_universe(universe_name, start_date, end_date)
    if df.empty:
        print("  ERROR: No data loaded.")
        return None
    print(f"  Loaded {len(df)} rows, {df['symbol'].nunique()} symbols.")

    data = _df_to_data(df)
    close = data["close"]
    n_periods = close.shape[0]
    symbols = data["symbols"]
    print(f"  Data: {n_periods} bars x {len(symbols)} assets")

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
    print("  Running warmup...")
    warmup_bars = 300
    warmup_end = min(warmup_bars, n_periods - 10)
    for bar_i in range(0, warmup_end):
        bar_data = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v
        if bar_i < len(timestamps):
            bar_data["timestamp"] = timestamps[bar_i]
        try:
            strat.on_data(bar_data)
        except Exception:
            pass

    # OOS backtest
    print("  Running OOS backtest...")
    oos_start = warmup_end
    oos_end = n_periods

    oos_returns, _ = compute_signal_returns(
        strat, data, oos_start, oos_end,
        timestamps=timestamps, tc_bps=tc_bps,
    )

    if len(oos_returns) == 0:
        print("  ERROR: No OOS returns produced.")
        return None

    # Metrics
    equity = build_equity_curve(oos_returns, initial=1_000_000.0)
    total_return = (equity[-1] - 1_000_000.0) / 1_000_000.0 * 100.0
    n_years = len(oos_returns) / 252.0
    ann_return = annual_return_from_equity(equity, n_years) * 100.0
    sr = sharpe_ratio(oos_returns, rf=0.04)
    max_dd = max_drawdown_from_equity(equity) * 100.0
    turnover = compute_turnover(oos_returns)

    print(f"\n  Total Return:     {total_return:+.2f}%")
    print(f"  Annual Return:      {ann_return:+.2f}%")
    print(f"  Sharpe Ratio:       {sr:.4f}")
    print(f"  Max Drawdown:       {max_dd:.2f}%")
    print(f"  Turnover:           {turnover:.2f}%")
    print(f"  OOS Bars:           {len(oos_returns)}")

    return {
        "name": name,
        "strategy_id": strategy_id,
        "total_return_pct": round(total_return, 2),
        "annual_return_pct": round(ann_return, 2),
        "sharpe_ratio": round(sr, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "turnover_pct": round(turnover, 2),
        "oos_bars": len(oos_returns),
        "oos_returns": oos_returns,
    }


def main():
    print("=" * 60)
    print("  STRATEGY COMPARISON: Factor-Neutral vs CSM v1 (Baseline)")
    print("  Universe: expanded_15 (15 liquid ETFs)")
    print("  Factors: Cross-sectional factor regression (Ken French FF5)")
    print("  TC: 5 bps round-trip")
    print("=" * 60)

    strategy_ids = [
        "macro_cross_sectional_factor_neutral",
        "macro_cross_sectional_momentum",
    ]
    results = []

    for sid in strategy_ids:
        result = run_single(sid, "2018-01-01", "2024-12-31", "expanded_15")
        if result:
            results.append(result)

    if len(results) < 2:
        print("\n  ERROR: Could not run both strategies for comparison.")
        return

    # Factor exposures for factor-neutral strategy
    print("\n" + "=" * 60)
    print("  FACTOR EXPOSURE ANALYSIS")
    print("=" * 60)

    fn_returns = results[0]["oos_returns"]
    base_returns = results[1]["oos_returns"]

    # Load Fama-French factors
    try:
        from quantcore.data.sources.fama_french import FamaFrenchLoader
        ff = FamaFrenchLoader(cache_dir=".cache/fama_french")
        factor_df = ff.load_5_factors(freq="daily")
        factor_cols = [c for c in factor_df.columns if c != "RF"]
        factor_df = factor_df.sort_index()
        factor_rets = factor_df[factor_cols].to_numpy(dtype=float)

        # Align to oos length
        fn_exposures = compute_factor_exposures(fn_returns, factor_rets)
        base_exposures = compute_factor_exposures(base_returns, factor_rets)

        print("\n  Factor Loading (Regression on daily 5F + no intercept):")
        print(f"  {'Factor':<15} {'Factor-Neutral':>15} {'CSM v1':>15}")
        print("  " + "-" * 48)
        if fn_exposures is not None and base_exposures is not None:
            for i, col in enumerate(factor_cols):
                print(f"  {col:<15} {fn_exposures[i]:>15.6f} {base_exposures[i]:>15.6f}")
            print(f"\n  Factor-Neutral Net Exposure: {np.sum(np.abs(fn_exposures)):.4f}")
            print(f"  CSM v1 Net Exposure:         {np.sum(np.abs(base_exposures)):.4f}")
        else:
            print("  Warning: Could not compute factor exposures.")
    except Exception as e:
        print(f"  Warning: Fama-French factor data unavailable: {e}")
        factor_cols = None
        fn_exposures = None
        base_exposures = None

    # Summary table
    print("\n" + "=" * 60)
    print("  COMPARISON SUMMARY TABLE")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Factor-Neutral':>15} {'CSM v1':>15}")
    print("  " + "-" * 58)

    metrics = [
        ("Total Return (%)", "total_return_pct", ".2f"),
        ("Annual Return (%)", "annual_return_pct", ".2f"),
        ("Sharpe Ratio", "sharpe_ratio", ".4f"),
        ("Max Drawdown (%)", "max_drawdown_pct", ".2f"),
        ("Turnover (%)", "turnover_pct", ".2f"),
        ("OOS Bars", "oos_bars", ".0f"),
    ]

    for label, key, fmt in metrics:
        v1 = results[0][key]
        v2 = results[1][key]
        print(f"  {label:<25} {v1:>15{fmt}} {v2:>15{fmt}}")

    # Save
    output = {
        "factor_neutral": {
            k: v for k, v in results[0].items() if k != "oos_returns"
        },
        "csm_v1": {
            k: v for k, v in results[1].items() if k != "oos_returns"
        },
        "factor_exposures": {
            "cols": list(factor_cols) if factor_cols else [],
            "factor_neutral": fn_exposures.tolist() if fn_exposures is not None else None,  # noqa: E501
            "csm_v1": base_exposures.tolist() if base_exposures is not None else None,
        } if factor_cols else None,
    }

    import json
    with open("strategy_comparison_summary.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to strategy_comparison_summary.json")


if __name__ == "__main__":
    main()
