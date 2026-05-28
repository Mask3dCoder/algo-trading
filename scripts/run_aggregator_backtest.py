"""Run multi-strategy aggregator backtest with walk-forward validation.

Loads macro strategies, wraps them in MultiStrategyAggregator with risk-parity
weighting, runs OOS backtest on the expanded_15 universe, and compares combined
performance to each standalone strategy.

Usage:
    python scripts/run_aggregator_backtest.py
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

from quantcore.portfolio.aggregator import (
    AggregatorConfig,
    MultiStrategyAggregator,
    SubStrategy,
)
from quantcore.risk.metrics import sharpe_ratio
from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class
from scripts.real_data_loader import load_universe

# Trigger @register decorators
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401
import quantcore.strategies.macro.state_selective_momentum  # noqa: F401
import quantcore.strategies.macro.factor_residual_v3  # noqa: F401
import quantcore.strategies.macro.cross_sectional_factor_neutral  # noqa: F401


# ---------------------------------------------------------------------------
# Strategy configs
# ---------------------------------------------------------------------------

STRATEGY_CONFIGS: dict[str, dict] = {
    "macro_cross_sectional_momentum": {
        "universe": "expanded_15",
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
    "macro_state_selective_momentum": {
        "universe": "expanded_15",
        "params": {
            "rebalance_freq": 21,
            "mom_windows": "20,60,120",
            "mom_weights": "0.2,0.5,0.3",
            "target_vol": 0.12,
            "vol_lookback": 20,
            "regime_lookback": 60,
            "high_vol_threshold": 0.20,
            "low_vol_threshold": 0.10,
            "max_position": 0.20,
        },
    },
    "macro_factor_residual_v3": {
        "universe": "expanded_15",
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
    "macro_cross_sectional_factor_neutral": {
        "universe": "expanded_15",
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
}


# ---------------------------------------------------------------------------
# Data helpers (from run_backtest_focused.py)
# ---------------------------------------------------------------------------

def _causal_ffill_2d(arr: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame(arr)
    filled = df.ffill(limit=max_gap)
    fill_mask = df.isna().values & (~filled.isna().values)
    return filled.values, fill_mask


def _df_to_data(df: pd.DataFrame) -> dict:
    symbols = sorted(df["symbol"].unique())
    n_assets = len(symbols)
    if n_assets == 0:
        return {"close": np.array([[]], dtype=float), "symbols": []}

    pivots: dict[str, pd.DataFrame] = {}
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
            df_filled, _ = _causal_ffill_2d(arr, max_gap=5)
            data[k] = np.nan_to_num(df_filled, nan=0.0)
    return data


# ---------------------------------------------------------------------------
# Signal-based return computation (adapted from run_backtest_focused.py)
# ---------------------------------------------------------------------------

def compute_signal_returns(
    strat,
    data: dict,
    start: int,
    end: int,
    timestamps: list | None = None,
    tc_bps: float = 5.0,
) -> tuple[np.ndarray, list[dict]]:
    """Compute bar-level signal-weighted returns with trade tracking."""
    if end - start < 2:
        return np.array([], dtype=float), []

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)
    if n_assets == 0:
        return np.array([], dtype=float), []

    asset_rets = (close[start + 1:end] - close[start:end - 1]) / np.maximum(close[start:end - 1], 1e-10)
    bar_rets = np.zeros(end - start - 1)
    carried_weights: dict[str, float] = {}
    positions: dict[str, dict] = {}
    trades: list[dict] = []

    for idx, bar_i in enumerate(range(start, end - 1)):
        bar_data: dict = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v

        if timestamps is not None and bar_i < len(timestamps):
            ts = timestamps[bar_i]
            if hasattr(strat, 'set_timestamp'):
                strat.set_timestamp(ts)
            bar_data["timestamp"] = ts

        try:
            signals = strat.on_data(bar_data) or []
        except Exception:
            signals = []

        weighted_ret = 0.0
        total_weight = 0.0
        current_weights: dict[str, float] = {}
        current_signals: dict[str, dict] = {}

        for s in signals:
            weight = float(s.direction) * float(s.strength)
            if s.direction != 0:
                current_weights[s.symbol] = weight
                current_signals[s.symbol] = {
                    "direction": s.direction,
                    "strength": s.strength,
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
            turnover = 0.0
            all_syms = set(carried_weights.keys()) | set(current_weights.keys())
            for sym in all_syms:
                prev_w = carried_weights.get(sym, 0.0)
                curr_w = current_weights.get(sym, 0.0)
                turnover += abs(curr_w - prev_w)
            bar_rets[idx] -= turnover * tc_bps / 10000.0

        # Trade tracking
        if current_signals:
            for sym in list(positions.keys()):
                pos = positions[sym]
                exit_ts = timestamps[bar_i] if timestamps and bar_i < len(timestamps) else None
                exit_price = float(close[bar_i, symbols.index(sym)]) if bar_i < close.shape[0] else None
                if sym not in current_signals or current_signals[sym]["direction"] != pos["direction"]:
                    pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
                    pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
                    trades.append({
                        "symbol": sym, "direction": pos["direction"],
                        "entry_ts": pos["entry_ts"], "exit_ts": exit_ts,
                        "entry_price": pos["entry_price"], "exit_price": exit_price,
                        "pnl_abs": pnl_abs, "pnl_pct": pnl_pct,
                    })
                    positions.pop(sym)

            for sym in current_signals:
                if sym not in positions:
                    sig = current_signals[sym]
                    entry_price = float(close[bar_i, symbols.index(sym)]) if bar_i < close.shape[0] else None
                    entry_ts = timestamps[bar_i] if timestamps and bar_i < len(timestamps) else None
                    positions[sym] = {
                        "direction": sig["direction"],
                        "entry_ts": entry_ts,
                        "entry_price": entry_price,
                    }

        if current_weights:
            carried_weights = dict(current_weights)

    # Close remaining positions
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        exit_ts = timestamps[end - 1] if timestamps and end - 1 < len(timestamps) else None
        exit_price = float(close[end - 1, symbols.index(sym)]) if end - 1 < close.shape[0] else None
        pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
        pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
        trades.append({
            "symbol": sym, "direction": pos["direction"],
            "entry_ts": pos["entry_ts"], "exit_ts": exit_ts,
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "pnl_abs": pnl_abs, "pnl_pct": pnl_pct,
        })

    return bar_rets, trades


# ---------------------------------------------------------------------------
# Walk-forward with anchored expanding window
# ---------------------------------------------------------------------------

def anchored_walk_forward_returns(
    strat_factory,
    data: dict,
    timestamps: list,
    train_bars: int = 2520,   # ~10 years
    test_bars: int = 252,     # ~1 year
    step_bars: int = 252,     # step by 1 year
    tc_bps: float = 5.0,
    warmup_bars: int = 300,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Anchored expanding-window walk-forward backtest.

    Returns (oos_returns, all_trades, fold_results).
    """
    n = len(timestamps)
    all_returns: list[np.ndarray] = []
    all_trades: list[dict] = []
    fold_results: list[dict] = []

    fold_start = max(train_bars, warmup_bars)
    fold_idx = 0

    while fold_start + test_bars <= n:
        train_end = fold_start
        test_end = min(fold_start + test_bars, n)

        # Fresh strategy instance for this fold
        strat = strat_factory()

        # Warmup: feed training data
        warmup_end = min(warmup_bars, train_end - 10)
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
                ts = timestamps[bar_i]
                if hasattr(strat, 'set_timestamp'):
                    strat.set_timestamp(ts)
                bar_data["timestamp"] = ts
            try:
                strat.on_data(bar_data)
            except Exception:
                pass

        # Train: continue feeding from warmup_end to train_end
        for bar_i in range(warmup_end, train_end):
            bar_data = {}
            for k, v in data.items():
                if k == "symbols":
                    bar_data[k] = v
                elif isinstance(v, np.ndarray) and v.ndim >= 1:
                    bar_data[k] = v[:bar_i + 1]
                else:
                    bar_data[k] = v
            if bar_i < len(timestamps):
                ts = timestamps[bar_i]
                if hasattr(strat, 'set_timestamp'):
                    strat.set_timestamp(ts)
                bar_data["timestamp"] = ts
            try:
                strat.on_data(bar_data)
            except Exception:
                pass

        # Test: collect OOS returns
        fold_rets, fold_trades = compute_signal_returns(
            strat, data, train_end, test_end,
            timestamps=timestamps, tc_bps=tc_bps,
        )

        fold_sharpe = float(np.nan)
        if len(fold_rets) > 5:
            std = float(np.std(fold_rets, ddof=1))
            if std > 1e-10:
                fold_sharpe = (float(np.mean(fold_rets)) / std) * np.sqrt(252.0)

        fold_results.append({
            "fold": fold_idx,
            "train_range": [0, train_end],
            "test_range": [train_end, test_end],
            "oos_sharpe": fold_sharpe,
            "n_bars": len(fold_rets),
        })

        all_returns.append(fold_rets)
        all_trades.extend(fold_trades)

        fold_start += step_bars
        fold_idx += 1

    if not all_returns:
        return np.array([], dtype=float), [], fold_results

    return np.concatenate(all_returns), all_trades, fold_results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    returns: np.ndarray,
    trades: list[dict],
    rf_rate: float = 0.04,
) -> dict:
    """Compute standard backtest metrics."""
    if len(returns) == 0:
        return {}

    equity = np.empty(len(returns) + 1)
    equity[0] = 1_000_000.0
    for i, r in enumerate(returns):
        equity[i + 1] = equity[i] * (1.0 + r)

    total_ret = (equity[-1] - equity[0]) / equity[0] * 100.0
    sr = sharpe_ratio(returns, rf=rf_rate)

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd)) * 100.0

    pnls = [t["pnl_abs"] for t in trades if t.get("pnl_abs") is not None]
    win_rate = float(np.mean(np.array(pnls) > 0)) * 100.0 if pnls else 0.0

    # Bootstrap 90% CI for Sharpe
    ci_lower, ci_upper = _bootstrap_sharpe_ci(returns, rf_rate, alpha=0.10)

    return {
        "total_return_pct": round(total_ret, 2),
        "sharpe_ratio": round(sr, 4),
        "sharpe_ci_lower": round(ci_lower, 4),
        "sharpe_ci_upper": round(ci_upper, 4),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate_pct": round(win_rate, 1),
        "n_trades": len(trades),
        "final_equity": round(float(equity[-1]), 2),
        "n_bars": len(returns),
    }


def _bootstrap_sharpe_ci(
    returns: np.ndarray,
    rf_rate: float = 0.04,
    alpha: float = 0.10,
    n_boot: int = 2000,
) -> tuple[float, float]:
    """Bootstrap confidence interval for annualized Sharpe ratio."""
    if len(returns) < 20:
        return float("nan"), float("nan")

    rng = np.random.default_rng(42)
    sharpes = np.empty(n_boot)
    n = len(returns)

    for i in range(n_boot):
        sample = rng.choice(returns, size=n, replace=True)
        std = float(np.std(sample, ddof=1))
        if std < 1e-10:
            sharpes[i] = 0.0
        else:
            rf_daily = (1.0 + rf_rate) ** (1.0 / 252.0) - 1.0
            sharpes[i] = (float(np.mean(sample)) - rf_daily) / std * np.sqrt(252.0)

    lo = float(np.percentile(sharpes, 100 * alpha / 2))
    hi = float(np.percentile(sharpes, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    # ── Configuration ──
    universe_name = "expanded_15"
    start_date = "2018-01-01"
    end_date = "2024-12-31"
    warmup_bars = 300
    tc_bps = 5.0
    rf_rate = 0.04

    # Strategies to include in the aggregator
    strategy_ids = [
        "macro_cross_sectional_momentum",
        "macro_state_selective_momentum",
        "macro_factor_residual_v3",
        "macro_cross_sectional_factor_neutral",
    ]

    print("=" * 70)
    print("  MULTI-STRATEGY AGGREGATOR BACKTEST")
    print("=" * 70)
    print(f"  Universe:    {universe_name}")
    print(f"  Period:      {start_date} -> {end_date}")
    print(f"  Strategies:  {len(strategy_ids)}")
    print(f"  Method:      risk_parity")
    print(f"  TC:          {tc_bps} bps")
    print(f"  Risk-free:   {rf_rate:.0%} annual")
    print()

    # ── Load data ──
    print("Loading data...")
    df = load_universe(universe_name, start_date, end_date)
    if df.empty:
        print("ERROR: No data loaded.")
        return {}
    data = _df_to_data(df)
    timestamps = sorted(df["timestamp"].unique())
    n_periods = data["close"].shape[0]
    print(f"  {n_periods} bars x {len(data['symbols'])} assets\n")

    # ── Run standalone strategies ──
    standalone_results: dict[str, dict] = {}

    for sid in strategy_ids:
        if sid not in STRATEGY_CONFIGS:
            print(f"  SKIP {sid}: no config defined")
            continue

        print(f"Running standalone: {sid}...")
        cfg_dict = STRATEGY_CONFIGS[sid]
        cfg = StrategyConfig(
            name=sid,
            strategy_type=StrategyType.MACRO,
            params=cfg_dict["params"],
        )

        try:
            cls = get_strategy_class(sid)
        except KeyError:
            print(f"  SKIP {sid}: not registered")
            continue

        def factory(_cls=cls, _cfg=cfg):
            s = _cls(_cfg)
            s.initialize()
            return s

        oos_returns, trades, fold_results = anchored_walk_forward_returns(
            factory, data, timestamps,
            train_bars=756, test_bars=126, step_bars=126,
            tc_bps=tc_bps, warmup_bars=warmup_bars,
        )

        if len(oos_returns) == 0:
            print(f"  SKIP {sid}: no OOS returns")
            continue

        metrics = compute_metrics(oos_returns, trades, rf_rate)
        standalone_results[sid] = metrics
        print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}  "
              f"CI: [{metrics['sharpe_ci_lower']:.4f}, {metrics['sharpe_ci_upper']:.4f}]  "
              f"MaxDD: {metrics['max_drawdown_pct']:.2f}%  "
              f"Trades: {metrics['n_trades']}")

    # ── Run aggregator ──
    print("\nRunning aggregator (risk_parity)...")
    sub_strategies = []
    for sid in strategy_ids:
        if sid not in STRATEGY_CONFIGS:
            continue
        cfg_dict = STRATEGY_CONFIGS[sid]
        cfg = StrategyConfig(
            name=sid,
            strategy_type=StrategyType.MACRO,
            params=cfg_dict["params"],
        )
        try:
            cls = get_strategy_class(sid)
            sub_strategies.append(SubStrategy(
                strategy_id=sid,
                strategy_cls=cls,
                config=cfg,
            ))
        except KeyError:
            pass

    if not sub_strategies:
        print("ERROR: No valid sub-strategies for aggregator.")
        return {}

    agg_config = AggregatorConfig(
        name="multi_strategy_aggregator",
        combination_method="risk_parity",
        weight_update_freq=5,
        vol_lookback=60,
        max_position_pct=0.05,
        max_gross_leverage=2.0,
        beta_neutral=True,
        beta_lookback=60,
        market_proxy="SPY",
        negative_sharpe_cutoff=0.0,
        initial_cash=1_000_000.0,
        rf_rate=rf_rate,
    )

    def agg_factory():
        fresh_subs = []
        for sid in strategy_ids:
            if sid not in STRATEGY_CONFIGS:
                continue
            cfg_dict = STRATEGY_CONFIGS[sid]
            cfg = StrategyConfig(
                name=sid,
                strategy_type=StrategyType.MACRO,
                params=cfg_dict["params"],
            )
            try:
                cls = get_strategy_class(sid)
                fresh_subs.append(SubStrategy(
                    strategy_id=sid,
                    strategy_cls=cls,
                    config=cfg,
                ))
            except KeyError:
                pass
        agg = MultiStrategyAggregator(fresh_subs, agg_config)
        return agg

    oos_returns, trades, fold_results = anchored_walk_forward_returns(
        agg_factory, data, timestamps,
        train_bars=756, test_bars=126, step_bars=126,
        tc_bps=tc_bps, warmup_bars=warmup_bars,
    )

    if len(oos_returns) == 0:
        print("ERROR: Aggregator produced no OOS returns.")
        return {}

    agg_metrics = compute_metrics(oos_returns, trades, rf_rate)

    # ── Diagnostic runs: equal_weight, mean_variance, single-strategy ──
    diagnostic_results: dict[str, dict] = {}

    for method in ["equal_weight", "mean_variance"]:
        print(f"\nRunning aggregator ({method})...")
        def diag_factory(_method=method, _sids=strategy_ids):
            fresh_subs = []
            for sid in _sids:
                if sid not in STRATEGY_CONFIGS:
                    continue
                cfg_dict = STRATEGY_CONFIGS[sid]
                cfg = StrategyConfig(
                    name=sid, strategy_type=StrategyType.MACRO,
                    params=cfg_dict["params"],
                )
                try:
                    cls = get_strategy_class(sid)
                    fresh_subs.append(SubStrategy(strategy_id=sid, strategy_cls=cls, config=cfg))
                except KeyError:
                    pass
            diag_cfg = AggregatorConfig(
                name=f"agg_{_method}", combination_method=_method,
                weight_update_freq=5, vol_lookback=60, max_position_pct=0.05,
                max_gross_leverage=2.0, beta_neutral=True, beta_lookback=60,
                market_proxy="SPY", negative_sharpe_cutoff=0.0,
                initial_cash=1_000_000.0, rf_rate=rf_rate,
            )
            return MultiStrategyAggregator(fresh_subs, diag_cfg)

        diag_oos, diag_trades, diag_folds = anchored_walk_forward_returns(
            diag_factory, data, timestamps,
            train_bars=756, test_bars=126, step_bars=126,
            tc_bps=tc_bps, warmup_bars=warmup_bars,
        )
        if len(diag_oos) > 0:
            diagnostic_results[method] = compute_metrics(diag_oos, diag_trades, rf_rate)
            m = diagnostic_results[method]
            print(f"  Sharpe: {m['sharpe_ratio']:.4f}  MaxDD: {m['max_drawdown_pct']:.2f}%  Trades: {m['n_trades']}")

    # Single-strategy aggregator (best standalone only)
    best_sid = max(standalone_results, key=lambda k: standalone_results[k]["sharpe_ratio"]) if standalone_results else None
    if best_sid:
        print(f"\nRunning single-strategy aggregator ({best_sid})...")
        def single_factory(_sid=best_sid):
            cfg_dict = STRATEGY_CONFIGS[_sid]
            cfg = StrategyConfig(name=_sid, strategy_type=StrategyType.MACRO, params=cfg_dict["params"])
            cls = get_strategy_class(_sid)
            subs = [SubStrategy(strategy_id=_sid, strategy_cls=cls, config=cfg)]
            single_cfg = AggregatorConfig(
                name=f"agg_single_{_sid}", combination_method="equal_weight",
                weight_update_freq=5, vol_lookback=60, max_position_pct=0.05,
                max_gross_leverage=2.0, beta_neutral=True, beta_lookback=60,
                market_proxy="SPY", negative_sharpe_cutoff=0.0,
                initial_cash=1_000_000.0, rf_rate=rf_rate,
            )
            return MultiStrategyAggregator(subs, single_cfg)

        single_oos, single_trades, single_folds = anchored_walk_forward_returns(
            single_factory, data, timestamps,
            train_bars=756, test_bars=126, step_bars=126,
            tc_bps=tc_bps, warmup_bars=warmup_bars,
        )
        if len(single_oos) > 0:
            diagnostic_results[f"single_{best_sid}"] = compute_metrics(single_oos, single_trades, rf_rate)
            m = diagnostic_results[f"single_{best_sid}"]
            print(f"  Sharpe: {m['sharpe_ratio']:.4f}  MaxDD: {m['max_drawdown_pct']:.2f}%  Trades: {m['n_trades']}")

    # ── Print comparison ──
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)
    header = f"  {'Strategy':<42s} {'Sharpe':>8s} {'CI Lower':>10s} {'MaxDD':>8s} {'Trades':>7s}"
    print(header)
    print("  " + "-" * 66)

    for sid, m in standalone_results.items():
        print(f"  {sid:<42s} {m['sharpe_ratio']:>8.4f} {m['sharpe_ci_lower']:>10.4f} "
              f"{m['max_drawdown_pct']:>7.2f}% {m['n_trades']:>6d}")

    print("  " + "-" * 66)
    print(f"  {'AGGREGATOR (risk_parity)':<42s} {agg_metrics['sharpe_ratio']:>8.4f} "
          f"{agg_metrics['sharpe_ci_lower']:>10.4f} "
          f"{agg_metrics['max_drawdown_pct']:>7.2f}% {agg_metrics['n_trades']:>6d}")

    for method_name, m in diagnostic_results.items():
        ci_lower = m.get('sharpe_ci_lower', float('nan'))
        print(f"  {f'AGGREGATOR ({method_name})':<42s} {m['sharpe_ratio']:>8.4f} "
              f"{ci_lower:>10.4f} "
              f"{m['max_drawdown_pct']:>7.2f}% {m['n_trades']:>6d}")
    print("=" * 70)

    # ── Success criteria ──
    best_standalone_sr = max(m["sharpe_ratio"] for m in standalone_results.values()) if standalone_results else 0.0
    worst_standalone_dd = min(m["max_drawdown_pct"] for m in standalone_results.values()) if standalone_results else 0.0

    agg_sr = agg_metrics["sharpe_ratio"]
    agg_dd = agg_metrics["max_drawdown_pct"]

    sr_pass = agg_sr >= best_standalone_sr
    dd_pass = agg_dd > worst_standalone_dd  # less negative = better

    print(f"\n  Best standalone Sharpe:    {best_standalone_sr:.4f}")
    print(f"  Aggregator Sharpe:         {agg_sr:.4f}  {'PASS' if sr_pass else 'FAIL'}")
    print(f"  Worst standalone MaxDD:    {worst_standalone_dd:.2f}%")
    print(f"  Aggregator MaxDD:          {agg_dd:.2f}%  {'PASS' if dd_pass else 'FAIL'}")

    # ── Save results ──
    results = {
        "config": {
            "universe": universe_name,
            "period": f"{start_date} -> {end_date}",
            "combination_method": "risk_parity",
            "strategies": strategy_ids,
            "tc_bps": tc_bps,
            "rf_rate": rf_rate,
            "train_bars": 756,
            "test_bars": 126,
        },
        "standalone": standalone_results,
        "aggregator": agg_metrics,
        "diagnostics": diagnostic_results,
        "criteria": {
            "combined_sharpe_gte_best": sr_pass,
            "combined_dd_better_than_worst": dd_pass,
            "best_standalone_sharpe": best_standalone_sr,
            "worst_standalone_dd": worst_standalone_dd,
        },
    }

    out_path = REPO_ROOT / "results" / "aggregator_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {out_path}")

    # ── Fold details ──
    print("\n  Walk-Forward Fold Details (Aggregator):")
    for fold in fold_results:
        sharpe_str = f"{fold['oos_sharpe']:.4f}" if not np.isnan(fold['oos_sharpe']) else "NaN"
        print(f"    Fold {fold['fold']}: train [0,{fold['train_range'][1]}] "
              f"test [{fold['test_range'][0]},{fold['test_range'][1]}] "
              f"OOS Sharpe={sharpe_str} ({fold['n_bars']} bars)")

    return results


if __name__ == "__main__":
    results = main()
