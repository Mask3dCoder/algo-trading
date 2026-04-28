#!/usr/bin/env python
"""Phase A real-data backtest: walk-forward validation on real market data.

Loads yfinance data for asset universes, runs anchored walk-forward
validation for both deploy strategies (macro_cta_trend, macro_risk_parity),
and compares OOS Sharpe ratios against pass criteria.

Usage:
    python scripts/real_data_backtest.py                         # All periods
    python scripts/real_data_backtest.py --period full           # Single period
    python scripts/real_data_backtest.py --list-periods          # Show periods
    python scripts/real_data_backtest.py --verbose               # Debug logging
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from typing import Any

import numpy as np
import pandas as pd

from quantcore.backtest.walk_forward import Fold, anchored_walk_forward
from quantcore.risk.metrics import sharpe_ratio
# Import strategy modules to trigger @register decorators
import quantcore.strategies.macro.cta_trend  # noqa: F401
import quantcore.strategies.macro.risk_parity  # noqa: F401

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class

from scripts.real_data_loader import DataQualityError, load_universe, validate_data

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Time periods for backtesting
# ---------------------------------------------------------------------------

ALL_PERIODS: dict[str, dict[str, str]] = {
    "full": {
        "start": "2008-01-01",
        "end": "2024-12-31",
        "label": "Full Sample (2008–2024)",
    },
    "pre_covid": {
        "start": "2008-01-01",
        "end": "2019-12-31",
        "label": "Pre-COVID (2008–2019)",
    },
    "covid": {
        "start": "2020-01-01",
        "end": "2021-12-31",
        "label": "COVID (2020–2021)",
    },
    "rate_hikes": {
        "start": "2022-01-01",
        "end": "2023-12-31",
        "label": "Rate Hikes (2022–2023)",
    },
    "gfc": {
        "start": "2008-01-01",
        "end": "2009-12-31",
        "label": "GFC (2008–2009)",
    },
}

# ---------------------------------------------------------------------------
# Pass criteria: minimum OOS Sharpe per (strategy, period)
# ---------------------------------------------------------------------------
# Keys are (strategy_id, period_id). Missing entries are informational only.

PASS_CRITERIA: dict[tuple[str, str], float] = {
    ("macro_cta_trend", "full"): 0.80,
    ("macro_cta_trend", "pre_covid"): 0.60,
    ("macro_cta_trend", "covid"): 0.30,
    ("macro_cta_trend", "rate_hikes"): 0.30,
    ("macro_cta_trend", "gfc"): 0.00,
    ("macro_risk_parity", "full"): 0.50,
    ("macro_risk_parity", "pre_covid"): 0.40,
    ("macro_risk_parity", "covid"): 0.00,
    ("macro_risk_parity", "rate_hikes"): -0.30,
    ("macro_risk_parity", "gfc"): -0.50,
}

# ---------------------------------------------------------------------------
# Strategy parameter defaults (from deployment_manifest.yaml recommended_params)
# ---------------------------------------------------------------------------

STRATEGY_PARAMS: dict[str, dict[str, Any]] = {
    "macro_cta_trend": {
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
    },
    "macro_risk_parity": {
        "vol_target": 0.10,
        "lookback": 100,
        "rebalance_freq": 21,
        "regime_overlay": True,
        "max_leverage": 1.0,
        "vol_regime_enabled": True,
        "vol_regime_threshold": 0.40,
        "vol_regime_lookback": 20,
        "min_vol_scale": 0.30,
    },
}

# Universe per strategy
STRATEGY_UNIVERSE: dict[str, str] = {
    "macro_cta_trend": "cta",
    "macro_risk_parity": "equity",
}

# ---------------------------------------------------------------------------
# DataFrame -> numpy data dict conversion
# ---------------------------------------------------------------------------


def _df_to_data(df: pd.DataFrame) -> dict:
    """Convert combined OHLCV DataFrame to strategy data dict with numpy arrays.

    Pivots the long-form DataFrame into per-field 2D arrays of shape
    (n_bars, n_assets) indexed by timestamp and symbol.
    """
    symbols = sorted(df["symbol"].unique())
    n_assets = len(symbols)

    if n_assets == 0:
        return {"close": np.array([[]], dtype=float), "symbols": []}

    # Pivot each field
    pivots: dict[str, pd.DataFrame] = {}
    for field in ["open", "high", "low", "close", "volume"]:
        pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field)
        # Reindex to ensure symbol ordering
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

    # Fill any remaining NaN (some symbols may start later)
    for k in ["close", "high", "low", "open", "volume"]:
        arr = data[k]
        if np.any(np.isnan(arr)):
            arr = np.nan_to_num(arr, nan=0.0)
            data[k] = arr

    return data


# ---------------------------------------------------------------------------
# Signal-based return computation (mirrors strategy_audit.py pattern)
# ---------------------------------------------------------------------------


def _compute_signal_returns(
    strat: Any,
    data: dict,
    start: int,
    end: int,
) -> np.ndarray:
    """Compute bar-level signal-weighted returns over data[start:end).

    For each bar i in [start, end-1), calls strat.on_data() with cumulative
    data[:i+1] and computes the weighted forward return.

    Returns array of length (end - start - 1). Empty if < 2 bars.
    """
    if end - start < 2:
        return np.array([], dtype=float)

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)

    if n_assets == 0:
        return np.array([], dtype=float)

    # Forward asset returns over evaluation window
    asset_rets = (close[start + 1: end] - close[start: end - 1]) / np.maximum(close[start: end - 1], 1e-10)

    bar_rets = np.zeros(end - start - 1)

    for idx, bar_i in enumerate(range(start, end - 1)):
        bar_data: dict = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v

        try:
            signals = strat.on_data(bar_data) or []
        except Exception:
            signals = []

        weighted_ret = 0.0
        total_weight = 0.0
        for s in signals:
            if s.direction != 0:
                try:
                    asset_idx = symbols.index(s.symbol)
                except (ValueError, IndexError):
                    continue
                weighted_ret += float(s.direction) * float(s.strength) * float(asset_rets[idx, asset_idx])
                total_weight += abs(float(s.strength))

        if total_weight > 1e-10:
            bar_rets[idx] = weighted_ret / total_weight

    return bar_rets


# ---------------------------------------------------------------------------
# Walk-forward backtest for a single (strategy, period) pair
# ---------------------------------------------------------------------------


def _setup_strategy(strategy_id: str) -> Any:
    """Create and initialize a strategy instance from the registry."""
    params = STRATEGY_PARAMS.get(strategy_id, {})
    stype = StrategyType.MACRO
    cfg = StrategyConfig(
        name=strategy_id,
        strategy_type=stype,
        params=dict(params),
    )
    cls = get_strategy_class(strategy_id)
    strat = cls(cfg)
    strat.initialize()
    return strat


def run_period_backtest(
    strategy_id: str,
    df: pd.DataFrame,
    folds: list[Fold],
) -> dict[str, Any]:
    """Run walk-forward backtest for a single strategy on a data period.

    Args:
        strategy_id: Strategy registry name.
        df: Combined OHLCV DataFrame for the period.
        folds: Walk-forward Fold list.

    Returns:
        Dict with results:
            'strategy': strategy_id
            'n_folds': completed folds
            'oos_sharpes': list of per-fold OOS Sharpe ratios
            'oos_sharpe': aggregate OOS Sharpe (concatenated returns)
            'n_oos_bars': total OOS bars across folds
            'fold_results': list of per-fold dicts
            'error': error message if failed
    """
    result: dict[str, Any] = {
        "strategy": strategy_id,
        "n_folds": 0,
        "oos_sharpes": [],
        "oos_sharpe": 0.0,
        "n_oos_bars": 0,
        "fold_results": [],
        "error": None,
    }

    # Convert DataFrame to numpy dict
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

    fold_results: list[dict] = []
    all_oos_rets: list[np.ndarray] = []

    for fold in folds:
        # Ensure fold bounds are within data
        if fold.train_end > n_periods or fold.test_end > n_periods:
            continue

        try:
            strat = _setup_strategy(strategy_id)

            # In-sample: build strategy state
            _compute_signal_returns(strat, data, fold.train_start, fold.train_end)

            # Out-of-sample: collect OOS returns from trained state
            oos_rets = _compute_signal_returns(strat, data, fold.test_start, fold.test_end)

            if len(oos_rets) > 2:
                fold_sr = sharpe_ratio(oos_rets, annualize=False)
                fold_results.append({
                    "train": (fold.train_start, fold.train_end),
                    "test": (fold.test_start, fold.test_end),
                    "oos_sharpe": fold_sr,
                    "n_oos_bars": len(oos_rets),
                })
                all_oos_rets.append(oos_rets)

        except Exception as e:
            log.debug("Fold (%d, %d) failed: %s", fold.test_start, fold.test_end, e)
            continue

    if not all_oos_rets:
        result["error"] = "No folds completed successfully"
        return result

    # Aggregate OOS returns
    oos_returns = np.concatenate(all_oos_rets) if len(all_oos_rets) > 0 else np.array([])

    if len(oos_returns) < 3:
        result["error"] = f"Insufficient OOS data: {len(oos_returns)} bars"
        return result

    result["n_folds"] = len(fold_results)
    result["oos_sharpes"] = [f["oos_sharpe"] for f in fold_results]
    result["oos_sharpe"] = sharpe_ratio(oos_returns)
    result["n_oos_bars"] = len(oos_returns)
    result["fold_results"] = fold_results

    return result


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def _fmt(val: Any, decimals: int = 2) -> str:
    """Format a number with sign prefix."""
    if val is None:
        return "N/A".rjust(8)
    return f"{val:+.{decimals}f}"


def print_results_table(all_results: list[dict[str, Any]]) -> None:
    """Print a formatted results table with pass/fail indicators."""
    print()
    print("=" * 90)
    print("  PHASE A -- REAL-DATA WALK-FORWARD BACKTEST RESULTS")
    print("=" * 90)

    # Header
    print(f"  {'Strategy':<20s} {'Period':<20s} {'OOS Sharpe':>10s} "
          f"{'Target':>8s} {'Status':>8s} {'Folds':>6s} {'OOS Bars':>9s}")
    print(f"  {'-' * 20} {'-' * 20} {'-' * 10} {'-' * 8} "
          f"{'-' * 8} {'-' * 6} {'-' * 9}")

    passed = 0
    failed = 0
    total = 0

    for r in all_results:
        strategy_id = r["strategy"]
        period_id = r["period"]
        period_label = r.get("period_label", period_id)
        oos_sr = r.get("oos_sharpe")
        error = r.get("error")

        criterion = PASS_CRITERIA.get((strategy_id, period_id))
        if criterion is not None and oos_sr is not None:
            status = "PASS" if oos_sr >= criterion else "FAIL"
            if status == "PASS":
                passed += 1
            else:
                failed += 1
        else:
            status = "INFO"

        total += 1

        sr_str = _fmt(oos_sr, 3) if oos_sr is not None else "ERROR"
        crit_str = f">{criterion:.2f}" if criterion is not None else "--"
        folds_str = str(r.get("n_folds", 0)).rjust(4)
        bars_str = str(r.get("n_oos_bars", 0)).rjust(7)

        print(f"  {strategy_id:<20s} {period_label:<20s} {sr_str:>10s} "
              f"{crit_str:>8s} {status:>8s} {folds_str:>6s} {bars_str:>9s}")

        if error:
            print(f"  {'':>20s} {'':>20s} ERROR: {error}")

    print(f"  {'-' * 20} {'-' * 20} {'-' * 10} {'-' * 8} "
          f"{'-' * 8} {'-' * 6} {'-' * 9}")
    print(f"\n  Results: {passed} passed, {failed} failed, {total} total")

    # Detailed fold breakdown
    print()
    print("-" * 90)
    print("  PER-FOLD BREAKDOWN")
    print("-" * 90)

    for r in all_results:
        if not r.get("fold_results"):
            continue
        strategy_id = r["strategy"]
        period_label = r.get("period_label", r["period"])
        print(f"\n  {strategy_id} / {period_label}:")
        for fr in r["fold_results"]:
            train = f"{fr['train'][0]}-{fr['train'][1]}"
            test = f"{fr['test'][0]}-{fr['test'][1]}"
            sr = fr["oos_sharpe"]
            n_bars = fr["n_oos_bars"]
            print(f"    Fold train=({train}) -> test=({test})  "
                  f"OOS Sharpe={sr:+.4f}  bars={n_bars}")


def print_pass_fail_summary(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute and print pass/fail summary across all strategy-period combos.

    Returns a summary dict for programmatic use.
    """
    summary: dict[str, Any] = {
        "n_passed": 0,
        "n_failed": 0,
        "n_total": 0,
        "results_by_strategy": {},
    }

    for r in all_results:
        strategy_id = r["strategy"]
        period_id = r["period"]
        oos_sr = r.get("oos_sharpe")
        criterion = PASS_CRITERIA.get((strategy_id, period_id))
        error = r.get("error")

        if strategy_id not in summary["results_by_strategy"]:
            summary["results_by_strategy"][strategy_id] = []

        passed = False
        if criterion is not None and oos_sr is not None and not error:
            passed = oos_sr >= criterion

        entry = {
            "period": period_id,
            "oos_sharpe": oos_sr,
            "criterion": criterion,
            "passed": passed,
            "error": error,
        }
        summary["results_by_strategy"][strategy_id].append(entry)

        if criterion is not None and oos_sr is not None and not error:
            if passed:
                summary["n_passed"] += 1
            else:
                summary["n_failed"] += 1
            summary["n_total"] += 1

    # Overall verdict
    print()
    print("=" * 90)
    print("  VERDICT")
    print("=" * 90)
    if summary["n_failed"] == 0 and summary["n_total"] > 0:
        print(f"\n  ALL {summary['n_total']} CRITERIA PASSED -- Phase A validation successful.\n"
              f"  Both strategies pass real-data walk-forward validation.\n"
              f"  Proceed to Phase B (Paper Trading Setup).")
        summary["verdict"] = "PASS"
    elif summary["n_total"] == 0:
        print(f"\n  No criteria evaluated -- check backtest results above.\n")
        summary["verdict"] = "NO_DATA"
    else:
        print(f"\n  {summary['n_passed']}/{summary['n_total']} criteria passed -- "
              f"{summary['n_failed']} failure(s).\n")
        for sid, entries in summary["results_by_strategy"].items():
            for e in entries:
                if not e["passed"] and e["criterion"] is not None:
                    print(f"  FAIL: {sid}/{e['period']} -- "
                          f"OOS Sharpe {e['oos_sharpe']:+.3f} "
                          f"< {e['criterion']:+.2f}")
        print(f"\n  Review failures before proceeding to Phase B.")
        summary["verdict"] = "FAIL"

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run Phase A backtest for all strategy-period combinations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase A real-data walk-forward backtest"
    )
    parser.add_argument("--period", type=str, default=None,
                        help="Period to backtest (default: all)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Strategy to backtest (default: both)")
    parser.add_argument("--list-periods", action="store_true",
                        help="Print period definitions and exit")
    parser.add_argument("--use-cache", action="store_true", default=True,
                        help="Use parquet cache for yfinance data")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if args.list_periods:
        print(f"{'Period':<15s} {'Start':<12s} {'End':<12s} Label")
        print("-" * 60)
        for pid, pinfo in ALL_PERIODS.items():
            print(f"{pid:<15s} {pinfo['start']:<12s} {pinfo['end']:<12s} {pinfo['label']}")
        return

    # Determine which strategies and periods to run
    strategies_to_run: list[str] = (
        [args.strategy] if args.strategy else list(STRATEGY_PARAMS.keys())
    )
    periods_to_run: dict[str, dict[str, str]] = (
        {args.period: ALL_PERIODS[args.period]} if args.period else ALL_PERIODS
    )

    all_results: list[dict[str, Any]] = []

    for strategy_id in strategies_to_run:
        if strategy_id not in STRATEGY_UNIVERSE:
            print(f"\n  ERROR: Unknown strategy '{strategy_id}'")
            continue

        universe_name = STRATEGY_UNIVERSE[strategy_id]

        for period_id, period_info in periods_to_run.items():
            start = period_info["start"]
            end = period_info["end"]
            label = period_info["label"]

            print(f"\n  {'-' * 60}")
            print(f"  Backtest: {strategy_id} / {label}")
            print(f"  Universe: {universe_name}")
            print(f"  Period:   {start} -> {end}")
            print(f"  {'-' * 60}")

            # Load data
            try:
                df = load_universe(universe_name, start, end, use_cache=args.use_cache)
            except Exception as e:
                print(f"  ERROR loading data: {e}")
                all_results.append({
                    "strategy": strategy_id,
                    "period": period_id,
                    "period_label": label,
                    "error": f"Data load failed: {e}",
                    "oos_sharpe": None,
                })
                continue

            if df.empty:
                print(f"  WARNING: No data for {universe_name} ({start} -> {end})")
                all_results.append({
                    "strategy": strategy_id,
                    "period": period_id,
                    "period_label": label,
                    "error": "Empty dataset",
                    "oos_sharpe": None,
                })
                continue

            # Validate data quality
            vresult = validate_data(df, name=f"{strategy_id}/{period_id}")
            if not vresult["passed"]:
                print(f"  Data quality issues ({len(vresult['errors'])} errors):")
                for err in vresult["errors"][:3]:
                    print(f"    - {err}")

            # Generate walk-forward folds
            n_bars = df["timestamp"].nunique()
            initial_train = min(200, n_bars // 3)
            test_size = min(50, max(20, n_bars // 10))
            embargo = max(1, int(0.02 * test_size))

            if initial_train + test_size + embargo >= n_bars:
                # Reduce train size if data is limited
                initial_train = max(50, n_bars // 4)
                test_size = max(10, n_bars // 8)
                embargo = max(0, int(0.02 * test_size))

            if initial_train + test_size + embargo >= n_bars:
                print(f"  WARNING: Too few bars ({n_bars}) for walk-forward")
                all_results.append({
                    "strategy": strategy_id,
                    "period": period_id,
                    "period_label": label,
                    "error": f"Insufficient bars ({n_bars})",
                    "oos_sharpe": None,
                })
                continue

            try:
                folds = anchored_walk_forward(
                    n_bars,
                    initial_train=initial_train,
                    test_size=test_size,
                    embargo_pct=embargo / test_size if test_size > 0 else 0.0,
                )
            except Exception as e:
                print(f"  ERROR generating folds: {e}")
                all_results.append({
                    "strategy": strategy_id,
                    "period": period_id,
                    "period_label": label,
                    "error": f"Fold generation failed: {e}",
                    "oos_sharpe": None,
                })
                continue

            print(f"  Bars: {n_bars}, Folds: {len(folds)}, "
                  f"Train: {initial_train}, Test: {test_size}")

            # Run backtest
            result = run_period_backtest(strategy_id, df, folds)
            result["strategy"] = strategy_id
            result["period"] = period_id
            result["period_label"] = label

            oos_sr = result.get("oos_sharpe")
            error = result.get("error")
            if oos_sr is not None:
                print(f"  OOS Sharpe: {oos_sr:+.4f} ({result['n_folds']} folds, "
                      f"{result['n_oos_bars']} OOS bars)")
            else:
                print(f"  ERROR: {error}")

            all_results.append(result)

    # Print final results table
    if all_results:
        print_results_table(all_results)
        print_pass_fail_summary(all_results)


if __name__ == "__main__":
    main()
