"""Phase 1.8: Signal Strength Filter Hypothesis Test for macro_cta_trend.

Tests whether filtering weak fast signals improves trade quality.
Runs exactly thresholds [0.1, 0.2, 0.3] against a clean baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from quantcore.backtest.walk_forward import Fold, anchored_walk_forward
from quantcore.risk.metrics import sharpe_ratio, sortino_ratio, drawdown_stats

# Import strategy modules to trigger @register decorators
import quantcore.strategies.macro.cta_trend  # noqa: F401

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class

from scripts.real_data_loader import UNIVERSES, load_universe, validate_data

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

THRESHOLDS = [0.0, 0.1, 0.2, 0.3]  # 0.0 = baseline (no filter)
STRATEGY_ID = "macro_cta_trend"
UNIVERSE_NAME = "cta"

FULL_PERIOD = {"start": "2018-01-01", "end": "2024-12-31"}

STRATEGY_PARAMS: dict[str, Any] = {
    "fast_window": 20,
    "medium_window": 50,
    "slow_window": 200,
    "fast_weight": 0.2,
    "medium_weight": 0.3,
    "slow_weight": 0.5,
    "vol_target": 0.15,
    "vol_lookback": 60,
    "rebalance_freq": 5,
    "vol_regime_enabled": True,
    "vol_regime_threshold": 0.30,
    "vol_regime_lookback": 20,
    "min_vol_scale": 0.20,
}

# ---------------------------------------------------------------------------
# Data conversion (same as real_data_backtest.py)
# ---------------------------------------------------------------------------

def _df_to_data(df: pd.DataFrame) -> dict:
    """Convert combined OHLCV DataFrame to strategy data dict with numpy arrays."""
    symbols = sorted(df["symbol"].unique())
    n_assets = len(symbols)

    if n_assets == 0:
        return {"close": np.array([[]], dtype=float), "symbols": []}

    pivots: dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field)
        pivots[field] = (
            pivoted[symbols] if all(s in pivoted.columns for s in symbols) else pivoted
        )

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
            df_filled = pd.DataFrame(arr).ffill().values
            data[k] = np.nan_to_num(df_filled, nan=0.0)

    return data


# ---------------------------------------------------------------------------
# Holding days calc
# ---------------------------------------------------------------------------

def _holding_days(entry_ts, exit_ts) -> float:
    if entry_ts is None or exit_ts is None:
        return 0.0
    try:
        e = pd.Timestamp(entry_ts)
        x = pd.Timestamp(exit_ts)
        return (x - e).total_seconds() / 86400.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Signal returns with trade tracking (CLEAN -- no loss-based exits)
# ---------------------------------------------------------------------------

def _compute_signal_returns_and_trades(
    strat: Any,
    data: dict,
    start: int,
    end: int,
    timestamps: list | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Compute bar-level returns and track trades.

    NO loss-based exits. NO min hold. Pure signal-driven entry/exit.
    """
    trades: list[dict] = []
    if end - start < 2:
        return np.array([], dtype=float), trades

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)

    if n_assets == 0:
        return np.array([], dtype=float), trades

    asset_rets = (
        close[start + 1:end] - close[start:end - 1]
    ) / np.maximum(close[start:end - 1], 1e-10)

    bar_rets = np.zeros(end - start - 1)
    carried_weights: dict[str, float] = {}
    positions: dict[str, dict] = {}  # symbol -> position state

    # Track signal filtering stats
    total_fast_signals = 0
    filtered_fast_signals = 0

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
            if hasattr(strat, "set_timestamp"):
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
                    "metadata": s.metadata,
                }
            try:
                asset_idx = symbols.index(s.symbol)
            except (ValueError, IndexError):
                continue
            weighted_ret += weight * float(asset_rets[idx, asset_idx])
            total_weight += abs(float(s.strength))

        if total_weight > 1e-10:
            bar_rets[idx] = weighted_ret / total_weight

        # Track fast signal filtering stats from metadata (once per bar per asset)
        # Signals with fast_filtered=True had their fast contribution zeroed out
        # We estimate filter rate from strategy diagnostics collected at fold end

        # --- Trade tracking (signal-driven only, no loss exits) ---
        for sym in list(positions.keys()):
            pos = positions[sym]
            exit_ts = timestamps[bar_i] if timestamps else None
            exit_price = (
                close[bar_i, symbols.index(sym)]
                if bar_i < close.shape[0] else None
            )

            # Signal-driven exit only (direction change or no signal)
            if sym not in current_signals or current_signals[sym]["direction"] != pos["direction"]:
                holding_days = _holding_days(pos["entry_ts"], exit_ts)
                pnl_abs = (
                    (exit_price - pos["entry_price"]) * pos["direction"]
                    if exit_price and pos["entry_price"] else 0.0
                )
                pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
                trades.append({
                    "timestamp_entry": pos["entry_ts"],
                    "timestamp_exit": exit_ts,
                    "position_direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": exit_price,
                    "pnl_abs": pnl_abs,
                    "pnl_pct": pnl_pct,
                    "holding_days": holding_days,
                    "symbol": sym,
                    "exit_reason": "signal",
                })
                positions.pop(sym)

        # Open new positions
        for sym in current_signals:
            if sym not in positions:
                sig = current_signals[sym]
                entry_price = (
                    close[bar_i, symbols.index(sym)]
                    if bar_i < close.shape[0] else None
                )
                entry_ts = timestamps[bar_i] if timestamps else None
                positions[sym] = {
                    "direction": sig["direction"],
                    "entry_ts": entry_ts,
                    "entry_price": entry_price,
                }

        if current_weights:
            carried_weights = dict(current_weights)

    # Close remaining positions at end
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        exit_ts = timestamps[end - 1] if timestamps else None
        exit_price = (
            close[end - 1, symbols.index(sym)]
            if end - 1 < close.shape[0] else None
        )
        holding_days = _holding_days(pos["entry_ts"], exit_ts)
        pnl_abs = (
            (exit_price - pos["entry_price"]) * pos["direction"]
            if exit_price and pos["entry_price"] else 0.0
        )
        pnl_pct = pnl_abs / pos["entry_price"] if pos["entry_price"] else 0.0
        trades.append({
            "timestamp_entry": pos["entry_ts"],
            "timestamp_exit": exit_ts,
            "position_direction": pos["direction"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "symbol": sym,
        })

    return bar_rets, trades


# ---------------------------------------------------------------------------
# Walk-forward backtest with trade tracking
# ---------------------------------------------------------------------------

def run_full_period_with_trades(
    strategy_id: str,
    df: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Run anchored walk-forward with full trade tracking."""
    result: dict[str, Any] = {
        "oos_sharpe": None,
        "oos_sortino": None,
        "max_dd": 0.0,
        "win_rate": 0.0,
        "n_folds": 0,
        "n_oos_bars": 0,
        "trades": [],
        "error": None,
    }

    try:
        data = _df_to_data(df)
    except Exception as e:
        result["error"] = f"Data conversion failed: {e}"
        return result

    close = data["close"]
    n_periods = close.shape[0]
    symbols = data["symbols"]

    if n_periods < 100 or len(symbols) == 0:
        result["error"] = f"Insufficient data: {n_periods} bars, {len(symbols)} symbols"
        return result

    timestamps = sorted(df["timestamp"].unique()) if "timestamp" in df.columns else None

    initial_train = min(200, n_periods // 3)
    test_size = min(50, max(20, n_periods // 10))
    embargo = max(1, int(0.02 * test_size))

    if initial_train + test_size + embargo >= n_periods:
        initial_train = max(50, n_periods // 4)
        test_size = max(10, n_periods // 8)
        embargo = max(0, int(0.02 * test_size))

    if initial_train + test_size + embargo >= n_periods:
        result["error"] = f"Insufficient bars ({n_periods}) for walk-forward"
        return result

    try:
        folds = anchored_walk_forward(
            n_periods,
            initial_train=initial_train,
            test_size=test_size,
            embargo_pct=embargo / test_size if test_size > 0 else 0.0,
        )
    except Exception as e:
        result["error"] = f"Fold generation failed: {e}"
        return result

    all_oos_rets: list[np.ndarray] = []
    all_trades: list[dict] = []
    total_fast_signals = 0
    filtered_fast_signals = 0

    for fold in folds:
        if fold.train_end > n_periods or fold.test_end > n_periods:
            continue

        try:
            params = dict(STRATEGY_PARAMS)
            params["min_signal_strength"] = threshold

            cfg = StrategyConfig(
                name=strategy_id,
                strategy_type=StrategyType.MACRO,
                params=params,
            )
            cls = get_strategy_class(strategy_id)
            strat = cls(cfg)
            strat.initialize()

            # Warmup
            _compute_signal_returns_and_trades(strat, data, fold.train_start, fold.train_end)

            # OOS with trade tracking
            oos_rets, trades = _compute_signal_returns_and_trades(
                strat, data, fold.test_start, fold.test_end, timestamps=timestamps
            )

            if len(oos_rets) > 2:
                all_oos_rets.append(oos_rets)
                all_trades.extend(trades)

                # Count signal filtering from strategy diagnostics
                if hasattr(strat, "_total_fast_signals"):
                    total_fast_signals += strat._total_fast_signals
                if hasattr(strat, "_filtered_fast_signals"):
                    filtered_fast_signals += strat._filtered_fast_signals

        except Exception as e:
            log.debug("Fold (%d, %d) failed: %s", fold.test_start, fold.test_end, e)
            continue

    if not all_oos_rets:
        result["error"] = "No folds completed successfully"
        return result

    oos_returns = np.concatenate(all_oos_rets)

    if len(oos_returns) < 3:
        result["error"] = f"Insufficient OOS data: {len(oos_returns)} bars"
        return result

    result["n_folds"] = len(all_oos_rets)
    result["oos_sharpe"] = sharpe_ratio(oos_returns)
    result["oos_sortino"] = sortino_ratio(oos_returns)
    result["max_dd"] = drawdown_stats(oos_returns).max_drawdown
    result["win_rate"] = float(np.mean(oos_returns > 0))
    result["n_oos_bars"] = len(oos_returns)
    result["trades"] = all_trades
    result["total_fast_signals"] = total_fast_signals
    result["filtered_fast_signals"] = filtered_fast_signals

    return result


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_trade_metrics(trades: list[dict]) -> dict[str, Any]:
    """Compute all required trade quality metrics."""
    if not trades:
        return {
            "n_trades": 0,
            "avg_pnl": 0.0,
            "median_pnl": 0.0,
            "win_rate": 0.0,
            "whipsaw_pct": 0.0,
            "whipsaw_pnl_contrib": 0.0,
            "pnl_p10": 0.0,
            "pnl_p25": 0.0,
            "pnl_p50": 0.0,
            "pnl_p75": 0.0,
            "pnl_p90": 0.0,
        }

    pnls = np.array([t["pnl_pct"] for t in trades])
    wins = pnls > 0
    whipsaws = [(t["holding_days"] <= 5 and t["pnl_pct"] < 0) for t in trades]

    total_pnl = float(np.sum(pnls))
    whipsaw_pnl = float(np.sum([t["pnl_pct"] for t in trades if t["holding_days"] <= 5 and t["pnl_pct"] < 0]))

    return {
        "n_trades": len(trades),
        "avg_pnl": float(np.mean(pnls)),
        "median_pnl": float(np.median(pnls)),
        "win_rate": float(np.mean(wins)),
        "whipsaw_pct": float(np.mean(whipsaws)),
        "whipsaw_pnl_contrib": whipsaw_pnl / total_pnl if total_pnl != 0 else 0.0,
        "pnl_p10": float(np.percentile(pnls, 10)),
        "pnl_p25": float(np.percentile(pnls, 25)),
        "pnl_p50": float(np.percentile(pnls, 50)),
        "pnl_p75": float(np.percentile(pnls, 75)),
        "pnl_p90": float(np.percentile(pnls, 90)),
    }


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment() -> dict[str, Any]:
    """Run the full Phase 1.8 experiment grid."""
    print("=" * 70)
    print(" Phase 1.8: Signal Strength Filter Hypothesis Test")
    print(f" Strategy: {STRATEGY_ID}")
    print(f" Threshold grid: {THRESHOLDS}")
    print("=" * 70)

    # Load data once
    print("\n Loading data...")
    df = load_universe(UNIVERSE_NAME, FULL_PERIOD["start"], FULL_PERIOD["end"], use_cache=True)
    if df.empty:
        print(" ERROR: No data returned")
        return {}

    expected_symbols = UNIVERSES[UNIVERSE_NAME]["symbols"]
    errs = validate_data(df, expected_symbols=expected_symbols, name=f"{STRATEGY_ID}/full")
    if errs:
        print(f" Data quality issues: {errs[:3]}")

    results: list[dict] = []
    baseline_trades = 0

    for threshold in THRESHOLDS:
        print(f"\n{'-' * 60}")
        label = "BASELINE" if threshold == 0.0 else f"threshold={threshold:.1f}"
        print(f" Running: {label}")

        result = run_full_period_with_trades(STRATEGY_ID, df, threshold)

        if result.get("error"):
            print(f" ERROR: {result['error']}")
            continue

        metrics = compute_trade_metrics(result["trades"])

        if threshold == 0.0:
            baseline_trades = metrics["n_trades"]

        trade_reduction = (
            (baseline_trades - metrics["n_trades"]) / baseline_trades * 100
            if baseline_trades > 0 and threshold > 0 else 0.0
        )

        total_fast = result.get("total_fast_signals", 0)
        filtered_fast = result.get("filtered_fast_signals", 0)
        filtered_pct = (
            filtered_fast / total_fast * 100 if total_fast > 0 else 0.0
        )

        record = {
            "threshold": threshold,
            "sharpe": result["oos_sharpe"],
            "sortino": result["oos_sortino"],
            "max_dd": result["max_dd"],
            "win_rate_bar": result["win_rate"],
            "n_folds": result["n_folds"],
            "n_oos_bars": result["n_oos_bars"],
            **metrics,
            "trade_reduction_pct": trade_reduction,
            "filtered_signals_pct": filtered_pct,
        }
        results.append(record)

        # Print summary
        print(f"  Sharpe: {result['oos_sharpe']:+.4f}  "
              f"MaxDD: {result['max_dd']:.2%}  "
              f"BarWR: {result['win_rate']:.1%}")
        print(f"  Trades: {metrics['n_trades']}  "
              f"AvgPnL: {metrics['avg_pnl']:.4f}  "
              f"MedPnL: {metrics['median_pnl']:.4f}")
        print(f"  WinRate: {metrics['win_rate']:.1%}  "
              f"Whipsaw%: {metrics['whipsaw_pct']:.1%}  "
              f"TradeReduction: {trade_reduction:.1f}%")

    return {"results": results}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_comparison_table(results: list[dict]) -> None:
    """Print the required comparison table."""
    print("\n" + "=" * 80)
    print(" COMPARISON TABLE")
    print("=" * 80)

    header = (
        f" {'Threshold':<10s} {'Sharpe':>8s} {'Trades':>7s} "
        f"{'Avg PnL':>9s} {'Whipsaw%':>9s} {'Filtered%':>10s}"
    )
    print(header)
    print(f" {'-' * 10} {'-' * 8} {'-' * 7} {'-' * 9} {'-' * 9} {'-' * 10}")

    baseline_sharpe = None
    baseline_avg_pnl = None
    baseline_whipsaw = None
    baseline_trades = None

    for r in results:
        if r["threshold"] == 0.0:
            baseline_sharpe = r["sharpe"]
            baseline_avg_pnl = r["avg_pnl"]
            baseline_whipsaw = r["whipsaw_pct"]
            baseline_trades = r["n_trades"]

    for r in results:
        thresh_str = "BASELINE" if r["threshold"] == 0.0 else f"{r['threshold']:.1f}"
        print(
            f" {thresh_str:<10s} {r['sharpe']:>+8.4f} {r['n_trades']:>7d} "
            f"{r['avg_pnl']:>+9.4f} {r['whipsaw_pct'] * 100:>8.1f}% "
            f"{r['filtered_signals_pct']:>9.1f}%"
        )

    print("\n" + "=" * 80)
    print(" INTERPRETATION")
    print("=" * 80)

    if baseline_sharpe is None or len(results) < 2:
        print(" INSUFFICIENT DATA for interpretation")
        return

    # Answer the 4 required questions
    print("\n#1 Does filtering improve Sharpe?")
    best_sharpe = max(r["sharpe"] for r in results if r["threshold"] > 0)
    best_thresh = next(r["threshold"] for r in results if r["sharpe"] == best_sharpe and r["threshold"] > 0)
    sharpe_improved = best_sharpe > baseline_sharpe
    if sharpe_improved:
        print(f"   YES -- best Sharpe {best_sharpe:+.4f} at threshold={best_thresh} (baseline {baseline_sharpe:+.4f})")
    else:
        print(f"   NO -- best filtered Sharpe {best_sharpe:+.4f} <= baseline {baseline_sharpe:+.4f}")

    print("\n#2 Does avg PnL per trade increase?")
    best_avg = max(r["avg_pnl"] for r in results if r["threshold"] > 0)
    best_avg_thresh = next(r["threshold"] for r in results if r["avg_pnl"] == best_avg and r["threshold"] > 0)
    avg_improved = best_avg > baseline_avg_pnl
    if avg_improved:
        print(f"   YES -- best avg PnL {best_avg:+.4f} at threshold={best_avg_thresh} (baseline {baseline_avg_pnl:+.4f})")
    else:
        print(f"   NO -- best filtered avg {best_avg:+.4f} <= baseline {baseline_avg_pnl:+.4f}")

    print("\n#3 Are whipsaws reduced?")
    best_whip = min(r["whipsaw_pct"] for r in results if r["threshold"] > 0)
    best_whip_thresh = next(r["threshold"] for r in results if r["whipsaw_pct"] == best_whip and r["threshold"] > 0)
    whip_reduced = best_whip < baseline_whipsaw
    if whip_reduced:
        print(f"   YES -- lowest whipsaw {best_whip * 100:.1f}% at threshold={best_whip_thresh} (baseline {baseline_whipsaw * 100:.1f}%)")
    else:
        print(f"   NO -- best filtered whipsaw {best_whip * 100:.1f}% >= baseline {baseline_whipsaw * 100:.1f}%")

    print("\n#4 Is trade reduction proportional or excessive?")
    for r in results:
        if r["threshold"] > 0:
            reduction = r["trade_reduction_pct"]
            label = "PROPORTIONAL" if reduction < 50 else "EXCESSIVE"
            print(f"   threshold={r['threshold']:.1f}: {reduction:.1f}% reduction -> {label}")

    # Success/failure criteria check
    print("\n" + "=" * 80)
    print(" VERDICT")
    print("=" * 80)

    # Check all success criteria against best threshold
    best = max(
        (r for r in results if r["threshold"] > 0),
        key=lambda r: r["sharpe"],
    )

    success_checks = {
        "Sharpe improves meaningfully": best["sharpe"] > baseline_sharpe + 0.05,
        "Avg PnL increases": best["avg_pnl"] > baseline_avg_pnl + 0.0001,
        "Whipsaw decreases": best["whipsaw_pct"] < baseline_whipsaw - 0.001,
        "Trade reduction < 50%": best["trade_reduction_pct"] < 50,
    }

    n_success = sum(success_checks.values())
    print(f"\n Best threshold: {best['threshold']:.1f}")
    print(f" Success criteria: {n_success}/{len(success_checks)} passed")
    for criterion, passed in success_checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"   [{status}] {criterion}")

    if n_success >= 3:
        print("\n>>> Signal strength filter is VALID")
    else:
        print("\n>>> Signal strength filter is NOT useful")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    output = run_experiment()
    if not output.get("results"):
        print("\n Experiment produced no results")
        return

    print_comparison_table(output["results"])

    # Save results
    output_path = REPO_ROOT / "results" / "phase1_8_signal_strength.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\n Results saved to: {output_path}")


if __name__ == "__main__":
    main()
