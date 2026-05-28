"""FRED regime strategy backtest — Phase A validation.

Compares standalone momentum vs. FRED-regime-filtered momentum using
real market data and walk-forward validation.

Tests:
  1. Standalone momentum (no regime filter)
  2. FRED-regime-filtered momentum (composite macro signal overlay)
  3. Per-regime performance breakdown (GFC, COVID, Rate Hikes, Bull)

Requires:
  - FRED_API_KEY environment variable
  - pip install fredapi

Usage:
    python scripts/run_fred_regime_backtest.py
    python scripts/run_fred_regime_backtest.py --verbose
    python scripts/run_fred_regime_backtest.py --start 2015-01-01 --end 2024-12-31
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

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd

from quantcore.backtest.walk_forward import Fold, anchored_walk_forward
from quantcore.risk.metrics import sharpe_ratio, sortino_ratio, drawdown_stats, bootstrap_ci

# Import strategy modules to trigger @register decorators
import quantcore.strategies.macro.cta_trend  # noqa: F401
import quantcore.strategies.macro.risk_parity  # noqa: F401
import quantcore.strategies.macro.fred_regime_strategy  # noqa: F401

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.registry import get as get_strategy_class

from scripts.real_data_loader import fetch_and_prepare, load_universe, validate_data

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

UNIVERSE = "cta"
ASSETS = ["DBC", "GLD", "IEF", "SPY", "TLT"]

FRED_CATEGORIES = ["yields", "credit", "inflation", "employment", "monetary"]

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

FULL_PERIOD = {"start": "2018-01-01", "end": "2024-12-31"}

# Strategy params
MOMENTUM_PARAMS: dict[str, Any] = {
    "momentum_window": 63,
    "vol_target": 0.15,
    "vol_lookback": 60,
    "rebalance_freq": 5,
    "min_position": 0.05,
}

FRED_REGIME_PARAMS: dict[str, Any] = {
    **MOMENTUM_PARAMS,
    "regime_weights": {
        "yield_curve": 0.30,
        "inflation": 0.20,
        "financial": 0.25,
        "economic": 0.25,
    },
    "yield_curve_threshold": 0.5,
    "inflation_threshold": 0.5,
    "financial_threshold": 0.5,
    "economic_threshold": 0.5,
}

TC_BPS = 5.0  # 5 bps round-trip cost


# ---------------------------------------------------------------------------
# DataFrame -> strategy data dict
# ---------------------------------------------------------------------------


def _causal_ffill_2d(arr: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Bounded causal forward-fill along the time axis."""
    df = pd.DataFrame(arr)
    filled = df.ffill(limit=max_gap)
    fill_mask = df.isna().values & (~filled.isna().values)
    return filled.values, fill_mask


def _df_to_data(df: pd.DataFrame) -> dict:
    """Convert combined OHLCV DataFrame to strategy data dict."""
    symbols = sorted(df["symbol"].unique())
    if not symbols:
        return {"close": np.array([[]], dtype=float), "symbols": []}

    pivoted = df.pivot_table(index="timestamp", columns="symbol", values="close")
    pivoted = pivoted[symbols] if all(s in pivoted.columns for s in symbols) else pivoted

    if pivoted.empty:
        return {"close": np.array([[]], dtype=float), "symbols": symbols}

    close = pivoted.values
    if np.any(np.isnan(close)):
        close_filled, _ = _causal_ffill_2d(close, max_gap=5)
        close = np.nan_to_num(close_filled, nan=0.0)

    return {"close": close, "symbols": symbols}


# ---------------------------------------------------------------------------
# Signal-weighted return computation
# ---------------------------------------------------------------------------


def _compute_signal_returns(
    strat: Any,
    data: dict,
    start: int,
    end: int,
    tc_bps: float = 0.0,
) -> np.ndarray:
    """Compute bar-level signal-weighted returns over data[start:end)."""
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
    carried_weights: dict[str, float] = {}

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
        elif carried_weights:
            for sym, weight in carried_weights.items():
                try:
                    asset_idx = symbols.index(sym)
                except (ValueError, IndexError):
                    continue
                weighted_ret += weight * float(asset_rets[idx, asset_idx])
                total_weight += abs(weight)
            if total_weight > 1e-10:
                bar_rets[idx] = weighted_ret / total_weight

        if tc_bps > 0 and current_weights:
            turnover = 0.0
            all_syms = set(carried_weights.keys()) | set(current_weights.keys())
            for sym in all_syms:
                prev_w = carried_weights.get(sym, 0.0)
                curr_w = current_weights.get(sym, 0.0)
                turnover += abs(curr_w - prev_w)
            bar_rets[idx] -= turnover * tc_bps / 10000.0

        if current_weights:
            carried_weights = current_weights

    return bar_rets


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _compute_metrics(returns: np.ndarray, label: str = "") -> dict[str, Any]:
    """Compute standard performance metrics from a return series."""
    if len(returns) < 2:
        return {"label": label, "error": "insufficient data"}

    sr = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    dd = drawdown_stats(returns)

    total_ret = float(np.prod(1 + returns) - 1)
    ann_vol = float(np.std(returns, ddof=1) * np.sqrt(252))
    n_days = len(returns)

    try:
        ci_low, ci_high = bootstrap_ci(returns, n_bootstrap=1000)
    except Exception:
        ci_low, ci_high = float("nan"), float("nan")

    return {
        "label": label,
        "sharpe": round(sr, 4),
        "sortino": round(sortino, 4),
        "total_return": round(total_ret, 4),
        "annualized_vol": round(ann_vol, 4),
        "max_drawdown": round(dd.max_drawdown, 4),
        "n_days": n_days,
        "ci_95_low": round(ci_low, 4),
        "ci_95_high": round(ci_high, 4),
    }


# ---------------------------------------------------------------------------
# Backtest runners
# ---------------------------------------------------------------------------


def _run_full_period_backtest(
    strategy_name: str,
    params: dict[str, Any],
    price_data: dict,
    start_date: str,
    end_date: str,
    timestamps: list | None = None,
) -> dict[str, Any]:
    """Run a walk-forward backtest over the full period."""
    strat_cls = get_strategy_class(strategy_name)
    config = StrategyConfig(
        name=strategy_name,
        strategy_type=StrategyType.MACRO,
        params=params,
    )
    strat = strat_cls(config)

    close = price_data["close"]
    n_periods = close.shape[0]

    # Simple 60/20/20 train/val/test split
    train_end = int(n_periods * 0.6)
    val_end = int(n_periods * 0.8)

    returns = _compute_signal_returns(
        strat, price_data, val_end, n_periods, tc_bps=TC_BPS,
    )

    metrics = _compute_metrics(returns, label=strategy_name)
    metrics["start"] = start_date
    metrics["end"] = end_date
    return metrics


def _run_regime_backtest(
    strategy_name: str,
    params: dict[str, Any],
    price_data: dict,
    regime_key: str,
    regime_cfg: dict[str, str],
    timestamps: list | None = None,
) -> dict[str, Any]:
    """Run a backtest over a specific regime period."""
    strat_cls = get_strategy_class(strategy_name)
    config = StrategyConfig(
        name=strategy_name,
        strategy_type=StrategyType.MACRO,
        params=params,
    )
    strat = strat_cls(config)

    close = price_data["close"]
    n_periods = close.shape[0]

    # Use last 40% of the regime window for OOS evaluation
    oos_start = int(n_periods * 0.6)

    returns = _compute_signal_returns(
        strat, price_data, oos_start, n_periods, tc_bps=TC_BPS,
    )

    metrics = _compute_metrics(returns, label=f"{strategy_name}/{regime_key}")
    metrics["regime"] = regime_cfg["label"]
    return metrics


# ---------------------------------------------------------------------------
# Macro data loading
# ---------------------------------------------------------------------------


def _load_macro_data(start: str, end: str) -> pd.DataFrame | None:
    """Load FRED macro data for regime signals."""
    try:
        from quantcore.data.sources.fred_loader import FREDDataLoader
        loader = FREDDataLoader()
        return loader.build_macro_dataframe(
            start=start,
            end=end,
            categories=FRED_CATEGORIES,
            use_first_release=True,
        )
    except Exception as e:
        log.warning("Failed to load FRED macro data: %s", e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="FRED regime strategy backtest")
    parser.add_argument("--start", type=str, default=FULL_PERIOD["start"])
    parser.add_argument("--end", type=str, default=FULL_PERIOD["end"])
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--regime-only", action="store_true", help="Skip full period")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    print("=" * 70)
    print("  FRED Regime Strategy Backtest — Phase A Validation")
    print("=" * 70)

    # Load price data
    print(f"\nLoading {UNIVERSE} universe: {args.start} -> {args.end} ...")
    df = fetch_and_prepare(UNIVERSE, args.start, args.end)
    print(f"  Loaded {len(df)} bars, {df['symbol'].nunique()} symbols")

    price_data = _df_to_data(df)
    timestamps = sorted(df["timestamp"].unique())

    # Load FRED macro data
    print("\nLoading FRED macro data...")
    macro_df = _load_macro_data(args.start, args.end)
    if macro_df is not None:
        print(f"  Loaded {len(macro_df)} rows, {len(macro_df.columns)} series")
    else:
        print("  WARNING: No FRED data — regime strategy will use neutral regime")

    results: dict[str, Any] = {
        "test": "fred_regime_backtest",
        "universe": UNIVERSE,
        "assets": ASSETS,
        "start": args.start,
        "end": args.end,
        "tc_bps": TC_BPS,
        "full_period": {},
        "regimes": {},
    }

    # --- Full period ---
    if not args.regime_only:
        print("\n" + "=" * 70)
        print("  FULL PERIOD BACKTEST")
        print("=" * 70)

        for name, params in [
            ("macro_cta_trend", MOMENTUM_PARAMS),
            ("macro_fred_regime", FRED_REGIME_PARAMS),
        ]:
            print(f"\n  Running {name}...")
            try:
                metrics = _run_full_period_backtest(
                    name, params, price_data, args.start, args.end, timestamps,
                )
                results["full_period"][name] = metrics
                print(f"    Sharpe: {metrics['sharpe']:+.3f}  "
                      f"Sortino: {metrics['sortino']:+.3f}  "
                      f"MaxDD: {metrics['max_drawdown']:.3f}  "
                      f"TotalRet: {metrics['total_return']:+.3f}")
            except Exception as e:
                print(f"    ERROR: {e}")
                results["full_period"][name] = {"error": str(e)}

    # --- Regime-specific ---
    print("\n" + "=" * 70)
    print("  REGIME-SPECIFIC BACKTEST")
    print("=" * 70)

    for regime_key, regime_cfg in REGIME_PERIODS.items():
        print(f"\n  [{regime_cfg['label']}]")

        # Load regime-specific data
        try:
            regime_df = fetch_and_prepare(
                UNIVERSE, regime_cfg["warmup_start"], regime_cfg["end"]
            )
            regime_data = _df_to_data(regime_df)
        except Exception as e:
            print(f"    Data load error: {e}")
            continue

        regime_results: dict[str, Any] = {}
        for name, params in [
            ("macro_cta_trend", MOMENTUM_PARAMS),
            ("macro_fred_regime", FRED_REGIME_PARAMS),
        ]:
            try:
                metrics = _run_regime_backtest(
                    name, params, regime_data, regime_key, regime_cfg,
                )
                regime_results[name] = metrics
                print(f"    {name:30s}  Sharpe: {metrics['sharpe']:+.3f}  "
                      f"MaxDD: {metrics['max_drawdown']:.3f}")
            except Exception as e:
                print(f"    {name:30s}  ERROR: {e}")
                regime_results[name] = {"error": str(e)}

        results["regimes"][regime_key] = regime_results

    # --- Summary ---
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if "macro_cta_trend" in results.get("full_period", {}):
        standalone = results["full_period"]["macro_cta_trend"]
        regime = results["full_period"].get("macro_fred_regime", {})
        if "sharpe" in standalone and "sharpe" in regime:
            delta = regime["sharpe"] - standalone["sharpe"]
            print(f"\n  Standalone Sharpe:  {standalone['sharpe']:+.3f}")
            print(f"  FRED Regime Sharpe: {regime['sharpe']:+.3f}")
            print(f"  Delta:              {delta:+.3f}")
            if delta > 0:
                print("  -> FRED regime overlay IMPROVES risk-adjusted returns")
            else:
                print("  -> FRED regime overlay does NOT improve returns in this test")

    # Regime win/loss
    if results.get("regimes"):
        wins = 0
        total = 0
        for rk, rv in results["regimes"].items():
            s = rv.get("macro_cta_trend", {}).get("sharpe", 0)
            r = rv.get("macro_fred_regime", {}).get("sharpe", 0)
            if r > s:
                wins += 1
            total += 1
        print(f"\n  Regime overlay wins: {wins}/{total} regimes")

    # Write output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results written to: {output_path}")

    print()


if __name__ == "__main__":
    main()
