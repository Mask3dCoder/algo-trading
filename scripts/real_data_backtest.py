"""Phase A real-data validation -- walk-forward backtest on real market data.

Loads yfinance data for CTA and RP universes, runs anchored walk-forward
validation for both deploy strategies (macro_cta_trend, macro_risk_parity),
evaluates regime-specific performance, and produces a pass/fail report.

Usage:
    python scripts/real_data_backtest.py                          # All strategies
    python scripts/real_data_backtest.py --strategy macro_cta_trend  # Single
    python scripts/real_data_backtest.py --regime-only            # Skip full
    python scripts/real_data_backtest.py --verbose                # Debug logging
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
import quantcore.strategies.macro.risk_parity  # noqa: F401

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class

from scripts.real_data_loader import UNIVERSES, fetch_and_prepare, load_universe, validate_data

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy -> universe mapping
# ---------------------------------------------------------------------------

STRATEGY_UNIVERSE: dict[str, str] = {
    "macro_cta_trend": "cta",
    "macro_risk_parity": "rp",
}

# ---------------------------------------------------------------------------
# Regime periods with warmup
# ---------------------------------------------------------------------------
# Each regime has:
#   warmup_start: data before this is used to build strategy state
#   start: start of OOS evaluation window
#   end:   end of OOS evaluation window
#   label: display name

REGIME_PERIODS: dict[str, dict[str, str]] = {
    "gfc": {
        "warmup_start": "2006-01-01",
        "start": "2008-01-01",
        "end": "2009-12-31",
        "label": "2008 GFC",
    },
    "q4_vol": {
        "warmup_start": "2018-01-01",
        "start": "2018-10-01",
        "end": "2018-12-31",
        "label": "2018 Q4 Vol",
    },
    "covid": {
        "warmup_start": "2019-06-01",
        "start": "2020-01-01",
        "end": "2021-12-31",
        "label": "2020 COVID",
    },
    "rate_hikes": {
        "warmup_start": "2021-06-01",
        "start": "2022-01-01",
        "end": "2023-12-31",
        "label": "2022 Rate Hikes",
    },
    "bull": {
        "warmup_start": "2022-06-01",
        "start": "2023-01-01",
        "end": "2024-12-31",
        "label": "2023-2024 Bull",
    },
}

# Full period (primary validation)
FULL_PERIOD: dict[str, str] = {
    "start": "2018-01-01",
    "end": "2024-12-31",
}

# ---------------------------------------------------------------------------
# Strategy params (from deployment_manifest.yaml recommended_params)
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

# ---------------------------------------------------------------------------
# Pass criteria (min OOS Sharpe per regime)
# ---------------------------------------------------------------------------

PASS_CRITERIA: dict[tuple[str, str], float] = {
    # Full period (walk-forward)
    ("macro_cta_trend", "full"): 0.50,
    ("macro_risk_parity", "full"): 0.30,
    # Regime periods
    ("macro_cta_trend", "gfc"): 0.00,
    ("macro_cta_trend", "covid"): 0.20,
    ("macro_cta_trend", "rate_hikes"): 0.20,
    ("macro_cta_trend", "q4_vol"): -0.10,
    ("macro_cta_trend", "bull"): 0.30,
    ("macro_risk_parity", "gfc"): -0.30,
    ("macro_risk_parity", "covid"): 0.00,
    ("macro_risk_parity", "rate_hikes"): -0.20,
    ("macro_risk_parity", "q4_vol"): -0.30,
    ("macro_risk_parity", "bull"): 0.20,
}

SHOWSTOPPER_THRESHOLDS: dict[tuple[str, str], float] = {
    # If OOS Sharpe falls below this, it is a showstopper
    ("macro_cta_trend", "full"): 0.00,
    ("macro_cta_trend", "gfc"): -0.50,
    ("macro_cta_trend", "covid"): -0.20,
    ("macro_cta_trend", "rate_hikes"): -0.20,
    ("macro_cta_trend", "bull"): 0.00,
    ("macro_risk_parity", "full"): -0.10,
    ("macro_risk_parity", "gfc"): -0.80,
    ("macro_risk_parity", "covid"): -0.30,
    ("macro_risk_parity", "rate_hikes"): -0.50,
    ("macro_risk_parity", "bull"): -0.10,
}

# ---------------------------------------------------------------------------
# DataFrame -> numpy data dict conversion
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
            # Forward-fill along time axis, then zero-fill leading NaN
            df_filled = pd.DataFrame(arr).ffill().values
            data[k] = np.nan_to_num(df_filled, nan=0.0)

    return data


# ---------------------------------------------------------------------------
# Signal-based return computation
# ---------------------------------------------------------------------------


def _compute_signal_returns(
    strat: Any,
    data: dict,
    start: int,
    end: int,
    tc_bps: float = 0.0,
) -> np.ndarray:
    """Compute bar-level signal-weighted returns over data[start:end).

    Supports transaction costs via tc_bps (turnover cost in basis points).
    When tc_bps > 0, tracks per-symbol position weights and applies a
    cost proportional to absolute weight changes between consecutive bars.
    """
    if end - start < 2:
        return np.array([], dtype=float)

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)

    if n_assets == 0:
        return np.array([], dtype=float)

    asset_rets = (
        close[start + 1: end] - close[start: end - 1]
    ) / np.maximum(close[start: end - 1], 1e-10)

    bar_rets = np.zeros(end - start - 1)
    carried_weights: dict[str, float] = {}  # persistent positions between signals

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
        current_weights: dict[str, float] = {}

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

        # Transaction cost: only on bars where signals actually fire
        if tc_bps > 0 and current_weights:
            turnover = 0.0
            all_syms = set(carried_weights.keys()) | set(current_weights.keys())
            for sym in all_syms:
                prev_w = carried_weights.get(sym, 0.0)
                curr_w = current_weights.get(sym, 0.0)
                turnover += abs(curr_w - prev_w)
            tc_cost = turnover * tc_bps / 10000.0
            bar_rets[idx] -= tc_cost

        # Update carried weights only when new signals arrive
        if current_weights:
            carried_weights = dict(current_weights)

    return bar_rets


# ---------------------------------------------------------------------------
# Strategy setup
# ---------------------------------------------------------------------------


def _setup_strategy(strategy_id: str) -> Any:
    """Create and initialize a strategy instance from the registry."""
    params = STRATEGY_PARAMS.get(strategy_id, {})
    cfg = StrategyConfig(
        name=strategy_id,
        strategy_type=StrategyType.MACRO,
        params=dict(params),
    )
    cls = get_strategy_class(strategy_id)
    strat = cls(cfg)
    strat.initialize()
    return strat


# ---------------------------------------------------------------------------
# Full period walk-forward backtest
# ---------------------------------------------------------------------------


def run_full_period_backtest(
    strategy_id: str,
    df: pd.DataFrame,
    tc_bps: float = 0.0,
) -> dict[str, Any]:
    """Run anchored walk-forward on the full period.

    Returns dict with keys: oos_sharpe, oos_sortino, max_dd, win_rate,
    n_folds, n_oos_bars, oos_sharpes (per fold), fold_results, error.
    """
    result: dict[str, Any] = {
        "oos_sharpe": None,
        "oos_sortino": None,
        "max_dd": 0.0,
        "win_rate": 0.0,
        "n_folds": 0,
        "n_oos_bars": 0,
        "oos_sharpes": [],
        "fold_results": [],
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

    fold_results: list[dict] = []
    all_oos_rets: list[np.ndarray] = []

    for fold in folds:
        if fold.train_end > n_periods or fold.test_end > n_periods:
            continue

        try:
            strat = _setup_strategy(strategy_id)

            # In-sample: build strategy state
            _compute_signal_returns(strat, data, fold.train_start, fold.train_end)

            # Out-of-sample: collect returns
            oos_rets = _compute_signal_returns(strat, data, fold.test_start, fold.test_end, tc_bps=tc_bps)

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

    oos_returns = np.concatenate(all_oos_rets) if len(all_oos_rets) > 0 else np.array([])

    if len(oos_returns) < 3:
        result["error"] = f"Insufficient OOS data: {len(oos_returns)} bars"
        return result

    result["n_folds"] = len(fold_results)
    result["oos_sharpes"] = [f["oos_sharpe"] for f in fold_results]
    result["oos_sharpe"] = sharpe_ratio(oos_returns)
    result["oos_sortino"] = sortino_ratio(oos_returns)
    result["max_dd"] = drawdown_stats(oos_returns).max_drawdown
    result["win_rate"] = float(np.mean(oos_returns > 0))
    result["n_oos_bars"] = len(oos_returns)
    result["fold_results"] = fold_results
    result["oos_returns"] = oos_returns.tolist()

    return result


# ---------------------------------------------------------------------------
# Regime backtest (single train/test split with warmup)
# ---------------------------------------------------------------------------


def run_regime_backtest(
    strategy_id: str,
    df: pd.DataFrame,
    warmup_end_idx: int,
    tc_bps: float = 0.0,
) -> dict[str, Any]:
    """Run a single train/test split backtest.

    warmup_end_idx is the bar index where warmup ends and OOS evaluation begins.
    Bars [0, warmup_end_idx) are used to build strategy state (no returns collected).
    Bars [warmup_end_idx, end) generate OOS returns.

    Returns dict with keys: oos_sharpe, oos_sortino, max_dd, win_rate, n_oos_bars, error.
    """
    result: dict[str, Any] = {
        "oos_sharpe": None,
        "oos_sortino": None,
        "max_dd": 0.0,
        "win_rate": 0.0,
        "n_oos_bars": 0,
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

    if n_periods < warmup_end_idx + 3 or len(symbols) == 0:
        result["error"] = f"Insufficient data: {n_periods} bars, warmup end {warmup_end_idx}"
        return result

    try:
        strat = _setup_strategy(strategy_id)

        # Warmup: build strategy state (no returns collected)
        _compute_signal_returns(strat, data, 0, warmup_end_idx)

        # OOS: collect returns from trained state
        oos_rets = _compute_signal_returns(strat, data, warmup_end_idx, n_periods, tc_bps=tc_bps)

    except Exception as e:
        result["error"] = f"Backtest failed: {e}"
        return result

    if len(oos_rets) < 3:
        result["error"] = f"Insufficient OOS data: {len(oos_rets)} bars"
        return result

    result["oos_sharpe"] = sharpe_ratio(oos_rets)
    result["oos_sortino"] = sortino_ratio(oos_rets)
    result["max_dd"] = drawdown_stats(oos_rets).max_drawdown
    result["win_rate"] = float(np.mean(oos_rets > 0))
    result["n_oos_bars"] = len(oos_rets)
    result["oos_returns"] = oos_rets.tolist()

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _fmt(val: Any, decimals: int = 2) -> str:
    """Format a number with sign prefix."""
    if val is None:
        return "N/A".rjust(8)
    return f"{val:+.{decimals}f}"


def _pct(val: float | None) -> str:
    """Format as percentage."""
    if val is None:
        return "N/A".rjust(8)
    return f"{val * 100:.1f}%".rjust(8)


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------


def run_validation(
    strategies: list[str] | None = None,
    regime_only: bool = False,
    use_cache: bool = True,
) -> list[dict[str, Any]]:
    """Run Phase A validation across all strategy/period combinations.

    Returns list of result dicts (one per strategy-period pair).
    """
    if strategies is None:
        strategies = list(STRATEGY_PARAMS.keys())

    all_results: list[dict[str, Any]] = []

    for strategy_id in strategies:
        if strategy_id not in STRATEGY_UNIVERSE:
            log.error("Unknown strategy '%s'", strategy_id)
            continue

        universe_name = STRATEGY_UNIVERSE[strategy_id]
        expected_symbols = UNIVERSES[universe_name]["symbols"]

        # ── Full period (walk-forward) ──
        if not regime_only:
            print(f"\n{'=' * 60}")
            print(f"  Full period backtest: {strategy_id}")
            print(f"  Universe: {universe_name} ({', '.join(expected_symbols)})")
            print(f"  Period: {FULL_PERIOD['start']} -> {FULL_PERIOD['end']}")
            print(f"{'=' * 60}")

            try:
                df = load_universe(universe_name, FULL_PERIOD["start"], FULL_PERIOD["end"],
                                   use_cache=use_cache)
            except Exception as e:
                print(f"  ERROR loading data: {e}")
                all_results.append({
                    "strategy": strategy_id,
                    "period": "full",
                    "period_label": "Full Period (2018-2024)",
                    "error": f"Data load failed: {e}",
                    "oos_sharpe": None,
                })
                continue

            if df.empty:
                print(f"  WARNING: No data returned")
                all_results.append({
                    "strategy": strategy_id,
                    "period": "full",
                    "period_label": "Full Period (2018-2024)",
                    "error": "Empty dataset",
                    "oos_sharpe": None,
                })
                continue

            # Validate data quality
            errs = validate_data(df, expected_symbols=expected_symbols,
                                 name=f"{strategy_id}/full")
            if errs:
                print(f"  Data quality issues ({len(errs)}):")
                for e in errs[:5]:
                    print(f"    - {e}")

            # Validate explicitly (raise on failure)
            try:
                errs = validate_data(df, expected_symbols=expected_symbols,
                                     name=f"{strategy_id}/full")
                if errs:
                    raise ValueError(f"Data quality: {'; '.join(errs[:3])}")
            except ValueError as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "strategy": strategy_id,
                    "period": "full",
                    "period_label": "Full Period (2018-2024)",
                    "error": str(e),
                    "oos_sharpe": None,
                })
                continue

            result = run_full_period_backtest(strategy_id, df)
            result["strategy"] = strategy_id
            result["period"] = "full"
            result["period_label"] = "Full Period (2018-2024)"

            if result.get("oos_sharpe") is not None:
                print(f"  OOS Sharpe: {result['oos_sharpe']:+.4f}  "
                      f"Sortino: {result['oos_sortino']:+.4f}  "
                      f"Max DD: {result['max_dd']:.2%}  "
                      f"Win Rate: {result['win_rate']:.1%}")
                print(f"  Folds: {result['n_folds']}  OOS bars: {result['n_oos_bars']}")
            else:
                print(f"  ERROR: {result.get('error')}")

            all_results.append(result)

        # ── Regime periods ──
        for regime_id, regime_info in REGIME_PERIODS.items():
            warmup_start = regime_info["warmup_start"]
            regime_start = regime_info["start"]
            regime_end = regime_info["end"]

            print(f"\n  {'-' * 50}")
            print(f"  Regime backtest: {strategy_id} / {regime_info['label']}")
            print(f"  Data: {warmup_start} (warmup) -> {regime_start} (OOS) -> {regime_end}")

            try:
                df = load_universe(universe_name, warmup_start, regime_end,
                                   use_cache=use_cache)
            except Exception as e:
                print(f"  ERROR loading data: {e}")
                all_results.append({
                    "strategy": strategy_id,
                    "period": regime_id,
                    "period_label": regime_info["label"],
                    "error": f"Data load failed: {e}",
                    "oos_sharpe": None,
                })
                continue

            if df.empty:
                print(f"  WARNING: No data returned")
                all_results.append({
                    "strategy": strategy_id,
                    "period": regime_id,
                    "period_label": regime_info["label"],
                    "error": "Empty dataset",
                    "oos_sharpe": None,
                })
                continue

            # Validate data
            try:
                errs = validate_data(df, expected_symbols=expected_symbols,
                                     name=f"{strategy_id}/{regime_id}")
                if errs:
                    raise ValueError(f"Data quality: {'; '.join(errs[:3])}")
            except ValueError as e:
                print(f"  ERROR: {e}")
                all_results.append({
                    "strategy": strategy_id,
                    "period": regime_id,
                    "period_label": regime_info["label"],
                    "error": str(e),
                    "oos_sharpe": None,
                })
                continue

            # Find the bar index where regime OOS begins
            all_dates = sorted(df["timestamp"].unique())
            regime_start_date = pd.Timestamp(regime_start)

            # Strip timezone for comparison if data has tz-aware timestamps
            if hasattr(all_dates[0], "tz") and all_dates[0].tz is not None:
                regime_start_date = regime_start_date.tz_localize(all_dates[0].tz)
            elif hasattr(regime_start_date, "tz") and regime_start_date.tz is not None:
                all_dates = [d.tz_localize(None) for d in all_dates]

            warmup_end_idx = None
            for i, d in enumerate(all_dates):
                if d >= regime_start_date:
                    warmup_end_idx = i
                    break

            if warmup_end_idx is None or warmup_end_idx < 10:
                print(f"  ERROR: Regime start {regime_start} not found in data")
                all_results.append({
                    "strategy": strategy_id,
                    "period": regime_id,
                    "period_label": regime_info["label"],
                    "error": "Regime start not found in data",
                    "oos_sharpe": None,
                })
                continue

            result = run_regime_backtest(strategy_id, df, warmup_end_idx)
            result["strategy"] = strategy_id
            result["period"] = regime_id
            result["period_label"] = regime_info["label"]

            if result.get("oos_sharpe") is not None:
                print(f"  OOS Sharpe: {result['oos_sharpe']:+.4f}  "
                      f"Sortino: {result['oos_sortino']:+.4f}  "
                      f"Max DD: {result['max_dd']:.2%}  "
                      f"Win Rate: {result['win_rate']:.1%}  "
                      f"OOS bars: {result['n_oos_bars']}")
            else:
                print(f"  ERROR: {result.get('error')}")

            all_results.append(result)

    return all_results


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------


def print_results_table(all_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Print formatted results table with PASS/FAIL per criterion.

    Returns summary dict with verdict.
    """
    print()
    print("=" * 110)
    print("  PHASE A -- REAL-DATA VALIDATION RESULTS")
    print("=" * 110)

    header = (
        f"  {'Strategy':<20s} {'Period':<18s} {'OOS SR':>8s} "
        f"{'Sortino':>8s} {'Max DD':>8s} {'Win Rate':>8s} "
        f"{'Bars':>6s} {'Status':>8s}"
    )
    print(header)
    print(f"  {'-' * 20} {'-' * 18} {'-' * 8} {'-' * 8} "
          f"{'-' * 8} {'-' * 8} {'-' * 6} {'-' * 8}")

    n_passed = 0
    n_failed = 0
    n_showstopper = 0
    n_total = 0

    for r in all_results:
        strategy_id = r["strategy"]
        period_id = r["period"]
        period_label = r.get("period_label", period_id)
        oos_sr = r.get("oos_sharpe")
        oos_sort = r.get("oos_sortino")
        dd = r.get("max_dd")
        wr = r.get("win_rate")
        n_bars = r.get("n_oos_bars", 0)
        error = r.get("error")

        # Check pass criterion
        criterion = PASS_CRITERIA.get((strategy_id, period_id))
        showstopper = SHOWSTOPPER_THRESHOLDS.get((strategy_id, period_id))

        if error:
            status = "ERROR"
        elif oos_sr is not None and criterion is not None:
            if showstopper is not None and oos_sr < showstopper:
                status = "SHOWSTOP"
                n_showstopper += 1
                n_failed += 1
            elif oos_sr >= criterion:
                status = "PASS"
                n_passed += 1
            else:
                status = "FAIL"
                n_failed += 1
        else:
            status = "INFO"

        n_total += 1

        sr_s = _fmt(oos_sr, 3) if oos_sr is not None else "ERROR".rjust(8)
        sort_s = _fmt(oos_sort, 3) if oos_sort is not None else "N/A".rjust(8)
        dd_s = _pct(dd) if dd is not None else "N/A".rjust(8)
        wr_s = _pct(wr) if wr is not None else "N/A".rjust(8)
        bars_s = str(n_bars).rjust(6) if n_bars else "N/A".rjust(6)

        print(f"  {strategy_id:<20s} {period_label:<18s} {sr_s:>8s} "
              f"{sort_s:>8s} {dd_s:>8s} {wr_s:>8s} {bars_s:>6s} {status:>8s}")

        if error:
            print(f"  {'':>20s} {'':>18s} ERROR: {error}")

    print(f"  {'-' * 20} {'-' * 18} {'-' * 8} {'-' * 8} "
          f"{'-' * 8} {'-' * 8} {'-' * 6} {'-' * 8}")

    print(f"\n  PASS: {n_passed}  FAIL: {n_failed}  "
          f"SHOWSTOPPER: {n_showstopper}  TOTAL: {n_total}")

    # Print showstopper details
    if n_showstopper > 0:
        print(f"\n  {'!' * 50}")
        print(f"  SHOWSTOPPER FAILURES -- review before deployment")
        print(f"  {'!' * 50}")
        for r in all_results:
            sid = r["strategy"]
            pid = r["period"]
            oos_sr = r.get("oos_sharpe")
            st = SHOWSTOPPER_THRESHOLDS.get((sid, pid))
            if oos_sr is not None and st is not None and oos_sr < st:
                print(f"    {sid}/{pid}: OOS SR {oos_sr:+.3f} < {st:+.2f} (showstopper)")

    # Print per-strategy pass/fail details
    print(f"\n  {'=' * 50}")
    print(f"  PER-STRATEGY SUMMARY")
    print(f"  {'=' * 50}")

    strategies_done = sorted(set(r["strategy"] for r in all_results))
    for sid in strategies_done:
        strat_results = [r for r in all_results if r["strategy"] == sid]
        s_passed = 0
        s_failed = 0
        for r in strat_results:
            criterion = PASS_CRITERIA.get((sid, r["period"]))
            oos_sr = r.get("oos_sharpe")
            if criterion is not None and oos_sr is not None and not r.get("error"):
                if oos_sr >= criterion:
                    s_passed += 1
                else:
                    s_failed += 1

        print(f"\n  {sid}:")
        for r in strat_results:
            pid = r["period"]
            oos_sr = r.get("oos_sharpe")
            criterion = PASS_CRITERIA.get((sid, pid))
            dd = r.get("max_dd")
            wr = r.get("win_rate")
            label = r.get("period_label", pid)

            sr_s = _fmt(oos_sr, 3) if oos_sr is not None else "ERROR"
            crit_s = f">{criterion:.2f}" if criterion is not None else "--"
            dd_s = _pct(dd) if dd is not None else "N/A"
            wr_s = _pct(wr) if wr is not None else "N/A"

            if oos_sr is not None and criterion is not None:
                if oos_sr >= criterion:
                    flag = "PASS"
                else:
                    flag = "FAIL"
            else:
                flag = "INFO"

            print(f"    {label:<18s} SR={sr_s}  (req {crit_s})  "
                  f"DD={dd_s}  WR={wr_s}  [{flag}]")

            if r.get("error"):
                print(f"    {'':<18s} ERROR: {r['error']}")

    # Overall verdict
    print(f"\n  {'=' * 50}")
    print(f"  OVERALL VERDICT")
    print(f"  {'=' * 50}")

    summary: dict[str, Any] = {
        "n_passed": n_passed,
        "n_failed": n_failed,
        "n_showstopper": n_showstopper,
        "n_total": n_total,
    }

    if n_showstopper > 0:
        print(f"\n  FAIL -- {n_showstopper} SHOWSTOPPER(S) detected.")
        print(f"  Review showstoppers before proceeding to Phase B.")
        summary["verdict"] = "SHOWSTOPPER"
    elif n_failed == 0 and n_total > 0:
        print(f"\n  ALL {n_total} CRITERIA PASSED -- Phase A validation successful.")
        print(f"  Both strategies pass real-data validation.")
        summary["verdict"] = "PASS"
    elif n_failed > 0:
        print(f"\n  {n_passed}/{n_total} criteria passed ({n_failed} failure(s)).")
        summary["verdict"] = "FAIL"
    else:
        print(f"\n  No criteria evaluated.")
        summary["verdict"] = "NO_DATA"

    return summary


def save_results_json(all_results: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Save results and summary to results/real_data_validation.json."""
    output_path = REPO_ROOT / "results" / "real_data_validation.json"

    # Convert numpy types and non-serializable objects
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    output = {
        "phase": "A",
        "description": "Real-data validation results for Phase A",
        "strategies_evaluated": list(STRATEGY_PARAMS.keys()),
        "results": all_results,
        "summary": summary,
    }

    # Deep-convert all values
    output = json.loads(json.dumps(output, default=_serialize))

    output_path.write_text(
        json.dumps(output, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  Results saved to: {output_path}")


# ---------------------------------------------------------------------------
# Pass criteria report (showstopper check)
# ---------------------------------------------------------------------------


def check_showstoppers(all_results: list[dict[str, Any]]) -> list[str]:
    """Check all results against showstopper thresholds.

    Returns list of showstopper violation messages (empty = no violations).
    Showstoppers halt Phase A with a FAIL verdict.
    """
    violations: list[str] = []
    for r in all_results:
        sid = r["strategy"]
        pid = r["period"]
        oos_sr = r.get("oos_sharpe")
        threshold = SHOWSTOPPER_THRESHOLDS.get((sid, pid))

        if oos_sr is not None and threshold is not None and oos_sr < threshold:
            violations.append(
                f"SHOWSTOPPER: {sid}/{pid} OOS SR {oos_sr:+.3f} < {threshold:+.2f}"
            )

    return violations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase A real-data validation"
    )
    parser.add_argument("--strategy", type=str, default=None,
                        help="Strategy to validate (default: both)")
    parser.add_argument("--regime-only", action="store_true",
                        help="Skip full period walk-forward, only regimes")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable parquet cache for yfinance")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    strategies: list[str] | None = [args.strategy] if args.strategy else None
    use_cache = not args.no_cache

    print("=" * 60)
    print("  Phase A: Real-Data Validation")
    print("  Strategies: ", strategies or list(STRATEGY_PARAMS.keys()))
    print("  Full period: 2018-2024 ", "(skipped)" if args.regime_only else "")
    print(f"  Regimes: {list(REGIME_PERIODS.keys())}")
    print(f"  Cache: {'disabled' if args.no_cache else 'enabled'}")
    print("=" * 60)

    all_results = run_validation(
        strategies=strategies,
        regime_only=args.regime_only,
        use_cache=use_cache,
    )

    if all_results:
        summary = print_results_table(all_results)

        # Check showstoppers
        violations = check_showstoppers(all_results)
        if violations:
            print(f"\n  {'!' * 50}")
            print(f"  SHOWSTOPPER VIOLATIONS FOUND")
            print(f"  {'!' * 50}")
            for v in violations:
                print(f"  {v}")

        # Save JSON
        save_results_json(all_results, summary)

    print(f"\n  {'=' * 60}")
    print(f"  Phase A complete.")
    print(f"  {'=' * 60}")


if __name__ == "__main__":
    main()
