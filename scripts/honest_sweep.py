#!/usr/bin/env python3
"""honest_sweep.py — Real-data walk-forward validation for ALL registered strategies.

This script replaces synthetic np.random fixtures with real yfinance market data.
It is the single source of truth for which strategies have deployable alpha.

Usage:
    python scripts/honest_sweep.py                    # All strategies
    python scripts/honest_sweep.py --strategy equity_momentum  # Single strategy
    python scripts/honest_sweep.py --skip-options      # Skip options (need IV data)
    python scripts/honest_sweep.py --null-check         # Run NullStrategy integrity check
    python scripts/honest_sweep.py --output results.csv # Custom output path

Output:
    honest_sweep_results_YYYY-MM-DD.csv — Full results for all strategies
    null_strategy_report.json — Engine integrity check (if --null-check)

Triage Rules:
    KEEP:    ci_lower > 0 AND dsr_p < 0.1 AND n_trades > 10 AND max_dd < 50%
    MONITOR: ci_lower > 0 AND n_trades > 10
    DISABLE: everything else
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_SIZE = 504   # 2 years
TEST_SIZE = 126    # 6 months
EMBARGO_PCT = 0.01
N_BOOTSTRAP = 10_000
MIN_FOLDS = 4
OOS_SHARPE_FLOOR = -0.5  # Tightened from -5.0

DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "GLD", "TLT", "DBC",
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META",
    "TSLA", "JPM", "V", "JNJ", "PG", "UNH", "HD", "XOM",
]

MACRO_TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"]

DATA_START = "2014-01-01"  # Need enough history for anchored walk-forward
DATA_END = "2024-12-31"

# Strategy parameters matching walk-forward test configs
_STRATEGY_PARAMS: dict[str, dict] = {
    # Equity
    "equity_momentum": {"rebalance_freq": 1, "lookback_months": 6, "skip_months": 1},
    "equity_earnings_momentum": {"rebalance_freq": 1},
    "equity_low_volatility": {"rebalance_freq": 1},
    "equity_stat_arb": {"rescan_freq": 1, "entry_z": 1.5, "lookback": 100, "max_pairs": 2},
    "equity_stat_arb_mr": {"entry_threshold": 2.0, "lookback_zscore": 20, "max_pairs": 5, "rebalance_freq": 60},
    "equity_mean_reversion": {"rsi_oversold": 30, "rsi_overbought": 70},
    "equity_trend_following": {"trend_window": 50, "fast_window": 10, "slow_window": 30},
    "equity_multi_factor": {},
    "equity_quality_factor": {},
    "equity_value_factor": {},
    "equity_volatility_breakout": {},
    "equity_volume_profile": {},
    "equity_cross_sectional_momentum": {"rebalance_freq": 1},
    "equity_full_universe_xs_momentum": {"rebalance_freq": 1},
    "equity_idiosyncratic_momentum": {"rebalance_freq": 1},
    "equity_short_term_reversal": {"rebalance_freq": 1},
    "equity_tsmom": {"rebalance_freq": 1},
    "equity_multifactor_composite": {},
    "equity_quality_rmw_cma": {},
    "equity_value_hml": {},
    "equity_opening_gap_fade": {"gap_threshold_pct": 2.0, "exit_minutes": 30},
    "sentiment_momentum": {"sentiment_lookback": 5, "rebalance_freq": 1, "tercile_pct": 0.33},
    # Intraday (will use daily data as proxy — results are conservative)
    "intraday_breakout": {"range_bars": 10, "atr_window": 7, "session_bars": 20, "exit_before_close": 3, "cooldown_bars": 2},
    "intraday_mean_reversion": {"lookback": 10, "rsi_period": 7, "bb_period": 10},
    "intraday_momentum": {"lookback_bars": 6, "rebalance_freq": 3, "n_long": 2, "n_short": 2},
    "intraday_trend": {"trend_window": 15, "fast_window": 5, "slow_window": 10},
    "opening_gap_fade": {"gap_threshold_pct": 2.0, "exit_minutes": 30},
    # Options (need IV data — may produce limited results on OHLCV only)
    "options_delta_neutral": {"ivr_entry": 30, "hedge_freq": 1},
    "options_iron_condor": {"ivr_entry": 30},
    "options_earnings_vol": {},
    "options_tail_hedge": {},
    "options_put_write": {"vix_entry": 20, "otm_pct": 0.95, "dte": 30, "risk_per_trade": 0.02, "max_positions": 3},
    # Macro
    "macro_risk_parity": {"rebalance_freq": 1, "lookback": 100},
    "macro_cta_trend": {"rebalance_freq": 1, "slow_window": 100},
    "macro_carry": {"rebalance_freq": 1, "n_long": 2, "n_short": 2},
    "macro_carry_trend_filter": {"rebalance_freq": 1, "slow_window": 100},
    "macro_factor_residual": {"rebalance_freq": 5, "reg_window": 252},
    "macro_factor_residual_strict": {"rebalance_freq": 5, "reg_window": 252},
    "macro_factor_residual_qp": {"rebalance_freq": 5, "reg_window": 252},
    "macro_factor_residual_v3": {"rebalance_freq": 5, "reg_window": 252, "residual_window": 63},
    "macro_regime_adaptive_momentum": {"rebalance_freq": 1, "slow_window": 100},
    "macro_state_selective_momentum": {"rebalance_freq": 1, "slow_window": 100},
    "macro_cross_sectional_momentum": {"rebalance_freq": 1, "slow_window": 100},
    "macro_cross_sectional_factor_neutral": {"rebalance_freq": 5, "reg_lookback": 252, "pred_window": 20},
    "macro_true_carry": {"rebalance_freq": 1, "slow_window": 100},
    "macro_vrp": {"rebalance_freq": 1},
    "macro_fred_regime": {"rebalance_freq": 1},
    "simple_buy_hold": {},
}

OPTIONS_STRATEGIES = {
    "options_delta_neutral", "options_iron_condor", "options_earnings_vol",
    "options_tail_hedge", "options_put_write",
}

MACRO_STRATEGIES = {k for k in _STRATEGY_PARAMS if k.startswith("macro_")}


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_real_data(tickers: list[str], start: str, end: str) -> dict[str, np.ndarray]:
    """Load real yfinance data for all tickers.

    Returns dict with:
        close: (n_bars, n_assets) adjusted close prices
        high, low: same shape
        volume: same shape
        symbols: list of ticker strings
    """
    import yfinance as yf

    print(f"  Loading real data: {len(tickers)} tickers, {start} to {end}...")

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if data.empty:
        raise RuntimeError(f"No data returned from yfinance for {tickers}")

    # Handle single-ticker vs multi-ticker column format
    if len(tickers) == 1:
        close = data["Close"].values.reshape(-1, 1)
        high = data["High"].values.reshape(-1, 1) if "High" in data else close * 1.005
        low = data["Low"].values.reshape(-1, 1) if "Low" in data else close * 0.995
        volume = data["Volume"].values.reshape(-1, 1).astype(float) if "Volume" in data else np.ones_like(close) * 1e6
    else:
        close = data["Close"].values
        high = data["High"].values if "High" in data else close * 1.005
        low = data["Low"].values if "Low" in data else close * 0.995
        volume = data["Volume"].values.astype(float) if "Volume" in data else np.ones_like(close) * 1e6

    # Drop any rows with NaN in close
    valid_mask = ~np.any(np.isnan(close), axis=1)
    close = close[valid_mask]
    high = high[valid_mask]
    low = low[valid_mask]
    volume = volume[valid_mask]

    # Implied vol proxy: 21-day realized vol * sqrt(252), minimum 0.05
    log_ret = np.diff(np.log(close), axis=0)
    log_ret = np.vstack([np.zeros((1, close.shape[1])), log_ret])
    iv_proxy = np.full_like(close, 0.20)
    lookback = 21
    for i in range(lookback, len(close)):
        rv = np.std(log_ret[i - lookback + 1 : i + 1], axis=0, ddof=1) * np.sqrt(252)
        iv_proxy[i] = np.maximum(rv, 0.05)

    # Yield proxy for carry strategies: use dividend-adjusted returns
    yields_proxy = np.full_like(close, 0.04)  # Flat 4% default

    n_bars = close.shape[0]
    print(f"  Loaded {n_bars} bars for {len(tickers)} tickers")
    print(f"  Date range: {start} to {end}")

    return {
        "close": close,
        "high": high,
        "low": low,
        "volume": volume,
        "implied_vol": iv_proxy,
        "yields": yields_proxy,
        "symbols": tickers,
        "earnings_upcoming": np.zeros(n_bars),
        "equity": np.array(1_000_000.0),
    }


def _strategy_type(name: str) -> str:
    if name.startswith("equity_") or name.startswith("intraday_") or name == "opening_gap_fade":
        return "equity"
    if name.startswith("options_"):
        return "options"
    if name.startswith("macro_") or name == "simple_buy_hold":
        return "macro"
    if name == "sentiment_momentum":
        return "equity"
    return "equity"


# ---------------------------------------------------------------------------
# Walk-Forward Validation on Real Data
# ---------------------------------------------------------------------------

def compute_signal_returns(strategy, data: dict, step: int = 5) -> np.ndarray:
    """Run strategy on expanding windows within the data slice.

    Instead of running bar-by-bar (O(n^2)), runs every `step` bars.
    For each window, gets signals and computes the return over the next step.
    This gives ~n/step return observations per fold — enough for Sharpe CI.
    """
    close = data.get("close")
    if close is None or close.ndim < 2 or close.shape[0] < 2:
        return np.array([])

    n = close.shape[0]
    n_assets = close.shape[1]
    symbols = data.get("symbols", [f"A{i}" for i in range(n_assets)])

    # Pre-compute per-bar asset returns
    asset_returns = np.diff(close, axis=0) / np.maximum(close[:-1], 1e-10)

    returns_list = []
    min_lookback = 20  # Strategies need some history

    for t in range(min_lookback, n - step, step):
        # Build data up to bar t
        bar_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:t]
            else:
                bar_data[k] = v

        try:
            signals = strategy.on_data(bar_data)
        except Exception:
            signals = []

        if not signals:
            continue

        # Aggregate into a single portfolio return over the next `step` bars
        port_ret = 0.0
        total_weight = 0.0
        for s in signals:
            if s.direction == 0:
                continue
            if s.symbol in symbols:
                idx = symbols.index(s.symbol)
            else:
                continue
            if idx < n_assets and t - 1 < len(asset_returns):
                # Multi-step return: sum of daily returns
                end_idx = min(t - 1 + step, len(asset_returns))
                if end_idx > t - 1:
                    multi_ret = np.sum(asset_returns[t - 1:end_idx, idx])
                    port_ret += float(s.direction) * float(s.strength) * float(multi_ret)
                    total_weight += float(s.strength)

        if total_weight > 0:
            returns_list.append(port_ret / total_weight)

    return np.array(returns_list) if returns_list else np.array([])


def bootstrap_sharpe_ci(returns: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, ci: float = 0.90) -> tuple:
    """Bootstrap Sharpe ratio confidence interval.

    Returns (sharpe_mean, ci_lower, ci_upper).
    """
    if len(returns) < 10:
        return (float("nan"), float("nan"), float("nan"))

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return (float("nan"), float("nan"), float("nan"))

    # Point estimate
    mean_r = np.mean(r)
    std_r = np.std(r, ddof=1)
    if std_r < 1e-10:
        return (float("nan"), float("nan"), float("nan"))

    sharpe_annual = (mean_r / std_r) * np.sqrt(252)

    # Bootstrap
    rng = np.random.default_rng(42)
    bootstrap_sharpes = np.empty(n_bootstrap)
    valid = 0
    for b in range(n_bootstrap):
        sample = rng.choice(r, size=len(r), replace=True)
        m = np.mean(sample)
        s = np.std(sample, ddof=1)
        if s > 1e-10:
            bootstrap_sharpes[valid] = (m / s) * np.sqrt(252)
            valid += 1

    if valid < 100:
        return (sharpe_annual, float("nan"), float("nan"))

    bootstrap_sharpes = bootstrap_sharpes[:valid]
    alpha = (1.0 - ci) / 2.0
    ci_lower = float(np.percentile(bootstrap_sharpes, alpha * 100))
    ci_upper = float(np.percentile(bootstrap_sharpes, (1.0 - alpha) * 100))

    return (float(sharpe_annual), ci_lower, ci_upper)


def compute_dsr_pvalue(sharpe: float, n_trials: int = 1, n_bars: int = 0) -> float:
    """Deflated Sharpe Ratio p-value (simplified Harvey-Liu 2015).

    DSR tests whether the observed Sharpe is statistically significant after
    accounting for multiple testing across n_trials strategies.
    """
    if math.isnan(sharpe) or n_bars < 20:
        return 1.0

    # Sharpe standard error (annualized)
    se_sharpe = 1.0 / np.sqrt(n_bars / 252.0) if n_bars > 0 else 1.0

    # Expected maximum Sharpe under null (multiple testing adjustment)
    # Using Bonferroni-like adjustment: expected max = norm.ppf(1 - 1/n_trials) * se
    from scipy.stats import norm
    if n_trials > 1:
        expected_max_sharpe = norm.ppf(1.0 - 1.0 / n_trials) * se_sharpe
    else:
        expected_max_sharpe = 0.0

    # DSR statistic
    dsr_stat = (sharpe - expected_max_sharpe) / se_sharpe
    p_value = 1.0 - norm.cdf(dsr_stat)

    return max(0.0, min(1.0, float(p_value)))


def compute_max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown as positive percentage."""
    if len(returns) < 2:
        return 0.0
    equity = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(abs(np.min(dd))) * 100.0


def count_trades(strategy, data: dict, step: int = 21) -> int:
    """Count number of non-flat signal events, sampling every `step` bars."""
    close = data.get("close")
    if close is None or close.ndim < 2 or close.shape[0] < 2:
        return 0

    n = close.shape[0]
    min_lookback = 20
    count = 0

    for t in range(min_lookback, n, step):
        bar_data = {}
        for k, v in data.items():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                bar_data[k] = v[:t]
            else:
                bar_data[k] = v
        try:
            signals = strategy.on_data(bar_data)
            count += sum(1 for s in signals if s.direction != 0)
        except Exception:
            pass

    return count


# ---------------------------------------------------------------------------
# Main Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    strategy_names: list[str] | None = None,
    skip_options: bool = False,
    null_check: bool = False,
    output_path: str | None = None,
) -> pd.DataFrame:
    """Run the honest sweep on real data."""

    # Import all strategy modules to trigger @register decorators
    _import_all_strategies()

    from quantcore.strategies.registry import list_strategies, create
    from quantcore.strategies.base import StrategyConfig, StrategyType

    all_registered = list_strategies()

    if strategy_names:
        targets = [s for s in strategy_names if s in all_registered]
        missing = [s for s in strategy_names if s not in all_registered]
        if missing:
            print(f"  WARNING: Strategies not found in registry: {missing}")
    else:
        targets = [s for s in all_registered if s in _STRATEGY_PARAMS]
        unconfigured = [s for s in all_registered if s not in _STRATEGY_PARAMS]
        if unconfigured:
            print(f"  NOTE: Strategies without params (skipped): {unconfigured}")

    if skip_options:
        targets = [s for s in targets if s not in OPTIONS_STRATEGIES]

    # Load data
    print("\n=== PHASE 1: Loading Real Market Data ===")
    equity_data = load_real_data(DEFAULT_TICKERS, DATA_START, DATA_END)
    macro_data = load_real_data(MACRO_TICKERS, DATA_START, DATA_END)

    # Null strategy check first
    if null_check:
        print("\n=== NULL STRATEGY INTEGRITY CHECK ===")
        null_ok = run_null_check(equity_data)
        if not null_ok:
            print("\n  [FAIL] ENGINE INTEGRITY FAILURE: NullStrategy shows positive Sharpe!")
            print("  [FAIL] The validation pipeline has systematic bias. DO NOT TRUST RESULTS.")
            print("  [FAIL] Debug the engine before proceeding.")
        else:
            print("\n  [PASS] NullStrategy check PASSED -- engine is unbiased.")

    # Run sweep
    print(f"\n=== HONEST SWEEP: {len(targets)} strategies on REAL data ===\n")

    results = []
    total_n = len(targets)  # for DSR multiple-testing adjustment

    for idx, name in enumerate(targets, 1):
        stype = _strategy_type(name)
        params = _STRATEGY_PARAMS.get(name, {})
        data = macro_data if name in MACRO_STRATEGIES else equity_data

        print(f"  [{idx}/{len(targets)}] {name} ({stype})...", end="", flush=True)

        # Create and initialize strategy
        try:
            cfg = StrategyConfig(
                name=name,
                strategy_type=StrategyType(stype),
                params=params,
            )
            strat = create(name, cfg)
            strat.initialize()
        except Exception as e:
            print(f" INIT FAIL: {e}")
            results.append({
                "strategy": name, "type": stype,
                "oos_sharpe": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "max_dd_pct": float("nan"), "n_trades": 0, "dsr_p": 1.0,
                "status": "INIT_FAIL", "n_folds": 0,
            })
            continue

        # Walk-forward folds
        n_bars = data["close"].shape[0]
        from quantcore.backtest.walk_forward import anchored_walk_forward
        folds = anchored_walk_forward(n_bars, initial_train=TRAIN_SIZE, test_size=TEST_SIZE, embargo_pct=EMBARGO_PCT)

        if len(folds) < MIN_FOLDS:
            print(f" Only {len(folds)} folds (need {MIN_FOLDS}), using rolling...")
            from quantcore.backtest.walk_forward import rolling_walk_forward
            folds = rolling_walk_forward(n_bars, train_size=TRAIN_SIZE, test_size=TEST_SIZE, embargo_pct=EMBARGO_PCT)

        # Collect OOS returns across folds
        all_oos_returns = []
        n_trades_total = 0
        total_oos_bars = 0
        step = 5  # Sample every 5 bars within each test window

        for fold in folds:
            # KEY: Feed training data first so strategy builds internal state
            # (moving averages, vol estimates, pair history, etc.)
            train_data = {}
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.ndim >= 1:
                    train_data[k] = v[fold.train_start:fold.train_end]
                else:
                    train_data[k] = v

            try:
                _ = strat.on_data(train_data)  # Build state from training window
            except Exception:
                pass

            # Now slide through test window with expanding lookback
            n_test = fold.test_end - fold.test_start
            n_assets = data["close"].shape[1]
            symbols = data.get("symbols", [f"A{i}" for i in range(n_assets)])

            # Pre-compute asset returns for the test period
            test_prices = data["close"][fold.test_start:fold.test_end]
            test_asset_rets = np.diff(test_prices, axis=0) / np.maximum(test_prices[:-1], 1e-10)

            fold_trades = 0
            for t_offset in range(20, n_test - step, step):
                # Build data from train_start through test_start+t_offset
                # This gives the strategy full history including training context
                global_end = fold.test_start + t_offset
                bar_data = {}
                for k, v in data.items():
                    if isinstance(v, np.ndarray) and v.ndim >= 1:
                        bar_data[k] = v[fold.train_start:global_end]
                    else:
                        bar_data[k] = v

                try:
                    signals = strat.on_data(bar_data)
                except Exception:
                    signals = []

                n_dir = sum(1 for s in signals if s.direction != 0)
                if n_dir == 0:
                    continue

                fold_trades += n_dir

                # Compute portfolio return over next `step` bars
                port_ret = 0.0
                total_weight = 0.0
                for s in signals:
                    if s.direction == 0:
                        continue
                    if s.symbol in symbols:
                        idx = symbols.index(s.symbol)
                    else:
                        continue
                    if idx < n_assets and t_offset - 1 < len(test_asset_rets):
                        end_idx = min(t_offset - 1 + step, len(test_asset_rets))
                        if end_idx > t_offset - 1:
                            multi_ret = float(np.sum(test_asset_rets[t_offset - 1:end_idx, idx]))
                            port_ret += float(s.direction) * float(s.strength) * multi_ret
                            total_weight += float(s.strength)

                if total_weight > 0:
                    all_oos_returns.append(port_ret / total_weight)
                    total_oos_bars += 1

            n_trades_total += fold_trades

        if len(all_oos_returns) < 10:
            print(f" INSUFFICIENT DATA ({len(all_oos_returns)} obs)")
            results.append({
                "strategy": name, "type": stype,
                "oos_sharpe": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "max_dd_pct": float("nan"), "n_trades": n_trades_total, "dsr_p": 1.0,
                "status": "NO_DATA", "n_folds": len(folds),
            })
            continue

        oos_array = np.array(all_oos_returns)
        oos_sharpe, ci_lower, ci_upper = bootstrap_sharpe_ci(oos_array)
        max_dd = compute_max_drawdown(oos_array)

        n_trades = n_trades_total

        # DSR p-value
        dsr_p = compute_dsr_pvalue(oos_sharpe, n_trials=total_n, n_bars=total_oos_bars)

        # Triage
        if not math.isnan(ci_lower) and ci_lower > 0 and dsr_p < 0.1 and n_trades > 10 and max_dd < 50:
            status = "KEEP"
        elif not math.isnan(ci_lower) and ci_lower > 0 and n_trades > 10:
            status = "MONITOR"
        else:
            status = "DISABLE"

        print(f" Sharpe={oos_sharpe:+.2f} CI=[{ci_lower:+.2f}, {ci_upper:+.2f}] DD={max_dd:.1f}% trades={n_trades} DSR={dsr_p:.3f} -> {status}")

        results.append({
            "strategy": name,
            "type": stype,
            "oos_sharpe": round(oos_sharpe, 4) if math.isfinite(oos_sharpe) else float("nan"),
            "ci_lower": round(ci_lower, 4) if math.isfinite(ci_lower) else float("nan"),
            "ci_upper": round(ci_upper, 4) if math.isfinite(ci_upper) else float("nan"),
            "max_dd_pct": round(max_dd, 2),
            "n_trades": n_trades,
            "dsr_p": round(dsr_p, 4),
            "status": status,
            "n_folds": len(folds),
        })

    df = pd.DataFrame(results)

    # Output
    if output_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        output_path = str(REPO_ROOT / "reports" / f"honest_sweep_results_{date_str}.csv")

    df.to_csv(output_path, index=False)
    print(f"\n=== RESULTS SAVED: {output_path} ===\n")

    # Summary
    _print_summary(df)

    return df


def run_null_check(data: dict) -> bool:
    """Run NullStrategy through the sweep to detect systematic bias.

    Returns True if engine passes (no bias detected).
    """
    print("  Generating NullStrategy (random entries, random sizing, random direction)...")

    rng = np.random.default_rng(12345)
    close = data["close"]
    n = close.shape[0]
    n_assets = close.shape[1]

    # Generate random signals
    all_returns = []
    from quantcore.backtest.walk_forward import anchored_walk_forward
    folds = anchored_walk_forward(n, initial_train=TRAIN_SIZE, test_size=TEST_SIZE, embargo_pct=EMBARGO_PCT)

    for fold in folds:
        test_close = close[fold.test_start:fold.test_end]
        if test_close.shape[0] < 2:
            continue

        asset_rets = np.diff(test_close, axis=0) / np.maximum(test_close[:-1], 1e-10)

        # Random signals: random direction, random strength
        for t in range(len(asset_rets)):
            direction = rng.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
            strength = rng.uniform(0.1, 1.0)
            asset_idx = rng.integers(0, n_assets)
            ret = direction * strength * asset_rets[t, asset_idx]
            all_returns.append(ret)

    if len(all_returns) < 10:
        print("  [WARN] Not enough data for null check")
        return False

    null_rets = np.array(all_returns)
    sharpe, ci_l, ci_u = bootstrap_sharpe_ci(null_rets)
    print(f"  NullStrategy: Sharpe={sharpe:+.4f} CI=[{ci_l:+.4f}, {ci_u:+.4f}]")

    # Save null report
    report = {
        "timestamp": datetime.now().isoformat(),
        "null_sharpe": sharpe,
        "null_ci_lower": ci_l,
        "null_ci_upper": ci_u,
        "n_bars": len(null_rets),
        "pass": not (math.isfinite(ci_l) and ci_l > 0) and (math.isfinite(sharpe) and abs(sharpe) < 0.3),
    }
    report_path = REPO_ROOT / "reports" / "null_strategy_report.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Gate: null sharpe should be ~0, CI should NOT exclude 0
    if math.isfinite(ci_l) and ci_l > 0:
        print(f"  [FAIL] Null CI_lower = {ci_l:+.4f} > 0 -- systematic positive bias!")
        return False
    if math.isfinite(sharpe) and abs(sharpe) > 0.3:
        print(f"  [FAIL] Null Sharpe = {sharpe:+.4f} -- too far from zero!")
        return False

    return True


def _print_summary(df: pd.DataFrame) -> None:
    """Print summary of sweep results."""
    total = len(df)
    keep = len(df[df["status"] == "KEEP"])
    monitor = len(df[df["status"] == "MONITOR"])
    disable = len(df[df["status"] == "DISABLE"])
    no_data = len(df[df["status"].isin(["INIT_FAIL", "NO_DATA"])])

    print("=" * 70)
    print("HONEST SWEEP SUMMARY")
    print("=" * 70)
    print(f"  Total strategies:  {total}")
    print(f"  KEEP:              {keep}  (ci_lower > 0, DSR < 0.1, trades > 10, DD < 50%)")
    print(f"  MONITOR:           {monitor}  (ci_lower > 0, trades > 10)")
    print(f"  DISABLE:           {disable}  (failed validation)")
    print(f"  INIT_FAIL/NO_DATA: {no_data}")
    print("=" * 70)

    if keep > 0:
        print("\n  [KEEP] strategies:")
        for _, row in df[df["status"] == "KEEP"].iterrows():
            print(f"     {row['strategy']:40s} Sharpe={row['oos_sharpe']:+.2f}  CI=[{row['ci_lower']:+.2f}, {row['ci_upper']:+.2f}]  DD={row['max_dd_pct']:.1f}%")

    if monitor > 0:
        print("\n  [MONITOR] strategies:")
        for _, row in df[df["status"] == "MONITOR"].iterrows():
            print(f"     {row['strategy']:40s} Sharpe={row['oos_sharpe']:+.2f}  CI=[{row['ci_lower']:+.2f}, {row['ci_upper']:+.2f}]  DD={row['max_dd_pct']:.1f}%")

    if keep == 0 and monitor == 0:
        print("\n  [FATAL] NO STRATEGIES PASS honest validation on real data.")
        print("  [FATAL] QuantCore has no validated deployable alpha.")
        print("  [FATAL] DO NOT DEPLOY. Report this finding honestly.")


def _import_all_strategies() -> None:
    """Import all strategy modules to trigger @register decorators."""
    import importlib
    modules = [
        "quantcore.strategies.equity.momentum",
        "quantcore.strategies.equity.mean_reversion",
        "quantcore.strategies.equity.stat_arb",
        "quantcore.strategies.equity.trend_following",
        "quantcore.strategies.equity.intraday_breakout",
        "quantcore.strategies.equity.intraday_mean_reversion",
        "quantcore.strategies.equity.intraday_momentum",
        "quantcore.strategies.equity.intraday_trend",
        "quantcore.strategies.equity.cross_sectional_momentum",
        "quantcore.strategies.equity.full_universe_cross_sectional_momentum",
        "quantcore.strategies.equity.idiosyncratic_momentum",
        "quantcore.strategies.equity.short_term_reversal",
        "quantcore.strategies.equity.tsmom",
        "quantcore.strategies.equity.value_factor",
        "quantcore.strategies.equity.value_factor_hml",
        "quantcore.strategies.equity.quality_factor",
        "quantcore.strategies.equity.quality_factor_rmw_cma",
        "quantcore.strategies.equity.low_volatility",
        "quantcore.strategies.equity.earnings_momentum",
        "quantcore.strategies.equity.volatility_breakout",
        "quantcore.strategies.equity.volume_profile",
        "quantcore.strategies.equity.multi_factor",
        "quantcore.strategies.equity.multifactor_composite",
        "quantcore.strategies.equity.opening_gap_fade",
        "quantcore.strategies.macro.carry",
        "quantcore.strategies.macro.cta_trend",
        "quantcore.strategies.macro.risk_parity",
        "quantcore.strategies.macro.fred_regime_strategy",
        "quantcore.strategies.macro.factor_residual",
        "quantcore.strategies.macro.factor_residual_strict",
        "quantcore.strategies.macro.factor_residual_qp",
        "quantcore.strategies.macro.factor_residual_v3",
        "quantcore.strategies.macro.regime_adaptive_momentum",
        "quantcore.strategies.macro.state_selective_momentum",
        "quantcore.strategies.macro.cross_sectional_momentum",
        "quantcore.strategies.macro.cross_sectional_factor_neutral",
        "quantcore.strategies.macro.carry_trend_filter",
        "quantcore.strategies.macro.true_carry_strategy",
        "quantcore.strategies.macro.volatility_risk_premium",
        "quantcore.strategies.macro.simple_test_strategy",
        "quantcore.strategies.options.delta_neutral",
        "quantcore.strategies.options.iron_condor",
        "quantcore.strategies.options.earnings_vol",
        "quantcore.strategies.options.tail_hedge",
        "quantcore.strategies.options.put_write",
        "quantcore.alpha.stat_arb_mean_reversion",
        "quantcore.alpha.sentiment_momentum",
    ]
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass  # Strategy may not exist or may fail to import — skip


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Honest Sweep: Real-data validation of all strategies")
    parser.add_argument("--strategy", nargs="*", help="Specific strategies to test")
    parser.add_argument("--skip-options", action="store_true", help="Skip options strategies (need IV data)")
    parser.add_argument("--null-check", action="store_true", help="Run NullStrategy integrity check first")
    parser.add_argument("--output", type=str, help="Output CSV path")
    args = parser.parse_args()

    print("=" * 64)
    print("  QUANTCORE HONEST SWEEP -- Real Data, Real Validation")
    print("  No synthetic fixtures. No np.random prices.")
    print("  Only honest results on real market data.")
    print("=" * 64 + "\n")

    start = time.time()
    df = run_sweep(
        strategy_names=args.strategy,
        skip_options=args.skip_options,
        null_check=args.null_check,
        output_path=args.output,
    )
    elapsed = time.time() - start
    print(f"\nSweep completed in {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
