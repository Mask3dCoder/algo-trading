#!/usr/bin/env python3
"""HMM Regime Overlay Backtest — test regime-conditional dynamic allocation.

Compares static Sharpe-proportional weights (57/39/4) against HMM-determined
regime-conditional weights for the 3 surviving strategies from the honest sweep.

In-sample:  2010-01-01 to 2020-12-31 (HMM training + per-regime Sharpe lookup)
OOS:        2021-01-01 to 2024-12-31 (regime-weighted portfolio backtest)

Usage:
    python scripts/hmm_regime_backtest.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# ─── Constants ───────────────────────────────────────────────────────────────

MACRO_ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"]
BLUECHIP = [
    "AAPL", "MSFT", "JNJ", "PFE", "PG", "KO", "PEP", "WMT", "T", "VZ",
    "XOM", "CVX", "MRK", "ABT", "BMY", "LLY", "IBM", "CSCO", "INTC",
    "BA", "GE", "MMM", "DIS", "HD", "NKE", "MCD", "SBUX", "YUM",
    "WFC", "BAC", "C", "USB", "PNC", "TFC", "COF", "SCHW",
    "JPM", "GS", "MS", "BLK", "AXP", "V", "MA", "COST", "PCAR",
    "ADM", "SHW", "CLX", "GIS", "KMB", "SYY", "HSY", "MKC",
    "EMR", "ETN", "PH", "ROK", "DOV", "ITW", "CAT", "DE", "HON",
    "UPS", "FDX", "RTX", "LMT", "NOC", "GD", "TDG",
    "AON", "MMC", "AJG", "TRV", "ALL", "MET", "PRU",
    "AVB", "EQR", "EXR", "PSA", "AMT", "EQIX", "CCI",
    "NSC", "UNP", "CSX", "F",
    "CMCSA", "ORCL", "ACN", "INFY", "ADI", "TXN", "QCOM",
]

DATA_START = "2010-01-01"
DATA_END = "2024-12-31"
IS_END = "2020-12-31"
OOS_START = "2021-01-01"

STATIC_WEIGHTS = {
    "macro_cross_sectional_momentum": 0.571,
    "equity_tsmom": 0.391,
    "equity_cross_sectional_momentum": 0.038,
}

COST_PARAMS = {
    "spread_bps": 5.0,
    "impact_bps": 10.0,
    "commission_per_share": 0.005,
}


# ─── Data Loading ────────────────────────────────────────────────────────────


def load_ohlcv_wide(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    """Load OHLCV and pivot to wide DataFrames."""
    from quantcore.data.yfinance_loader import load_multiple

    _log.info("Loading OHLCV for %d tickers (%s to %s)...", len(tickers), start, end)
    data = load_multiple(tickers, start=start, end=end, use_cache=True)

    fields = ["close", "high", "low", "open", "volume"]
    wide: dict[str, dict[str, pd.Series]] = {f: {} for f in fields}

    for sym, df in data.items():
        if df.empty:
            continue
        df = df.copy()
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        for f in fields:
            if f in df.columns:
                wide[f][sym] = df[f]

    result = {}
    for f in fields:
        if not wide[f]:
            continue
        frame = pd.DataFrame(wide[f]).sort_index().dropna(how="all")
        result[f] = frame

    _log.info("Loaded: %s", {k: v.shape for k, v in result.items()})
    return result


# ─── Feature Engineering ────────────────────────────────────────────────────


def compute_hmm_features(close: pd.DataFrame) -> pd.DataFrame:
    """Compute HMM observation features from ETF close prices.

    Features (per ETF, then averaged across universe):
      - daily log return
      - 20-day realized volatility
      - rolling Hurst exponent (20-day)
      - 14-day ADX proxy (return-based, no HLC)

    Returns: DataFrame indexed by date with feature columns.
    """
    from quantcore.regime.features import hurst_exponent, realized_volatility

    log_ret = np.log(close / close.shift(1))

    # Per-ETF features, then cross-sectional mean
    features = pd.DataFrame(index=close.index)

    # 1. Mean daily log return across ETFs
    features["mean_log_return"] = log_ret.mean(axis=1)

    # 2. 20-day realized vol (annualized, mean across ETFs)
    vol_20 = log_ret.rolling(20).std() * np.sqrt(252)
    features["realized_vol_20d"] = vol_20.mean(axis=1)

    # 3. Rolling Hurst (20-day window, computed on mean return)
    #    Use vectorized R/S on the mean return series
    mean_ret = features["mean_log_return"].values
    hurst_vals = np.full(len(mean_ret), 0.5)
    window = 60  # Need more than 20 for reliable Hurst
    for i in range(window, len(mean_ret)):
        segment = mean_ret[i - window:i]
        if np.all(np.isfinite(segment)) and np.std(segment) > 1e-12:
            try:
                hurst_vals[i] = hurst_exponent(segment, max_lag=20)
            except Exception:
                pass
    features["hurst"] = hurst_vals

    # 4. Rolling correlation between SPY and TLT (regime indicator)
    if "SPY" in log_ret.columns and "TLT" in log_ret.columns:
        features["spy_tlt_corr"] = (
            log_ret["SPY"].rolling(42).corr(log_ret["TLT"])
        )

    # 5. Vol-of-vol (20-day vol of 20-day vol)
    features["vol_of_vol"] = vol_20.mean(axis=1).rolling(20).std()

    # 6. Cross-sectional dispersion (return spread)
    features["return_dispersion"] = log_ret.std(axis=1)

    return features.dropna()


# ─── HMM Training ───────────────────────────────────────────────────────────


def train_hmm(
    features: pd.DataFrame,
    n_states: int = 3,
    n_restarts: int = 5,
) -> tuple:
    """Train HMM on in-sample features. Returns (result, features_is)."""
    from quantcore.regime.hmm_engine import fit_hmm

    X = features.values.astype(np.float64)
    _log.info("Training HMM: %d samples x %d features, %d states, %d restarts",
              X.shape[0], X.shape[1], n_states, n_restarts)

    t0 = time.time()
    result = fit_hmm(X, n_states=n_states, covar_type="diag", n_restarts=n_restarts)
    t1 = time.time()

    _log.info("HMM fitted in %.1fs: LL=%.2f, converged=%s, BIC=%.2f",
              t1 - t0, result.log_likelihood, result.converged, result.bic)

    return result


# ─── Regime-Conditional Sharpe Lookup ───────────────────────────────────────


def compute_regime_sharpes(
    strategy_returns: dict[str, pd.Series],
    regime_states: np.ndarray,
    regime_dates: pd.DatetimeIndex,
    n_states: int,
) -> dict[int, dict[str, float]]:
    """Compute annualized Sharpe per strategy per regime from in-sample data.

    Returns: {regime_id: {strategy_name: annualized_sharpe}}
    """
    regime_series = pd.Series(regime_states, index=regime_dates)

    result = {}
    for regime in range(n_states):
        mask = regime_series == regime
        n_days = mask.sum()
        result[regime] = {}

        for name, rets in strategy_returns.items():
            # Align
            common = rets.index.intersection(regime_dates)
            r = rets.loc[common]
            m = mask.loc[common]

            regime_rets = r[m]
            if len(regime_rets) < 20:
                result[regime][name] = 0.0
                continue

            ann_ret = regime_rets.mean() * 252
            ann_vol = regime_rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0
            result[regime][name] = sharpe

    return result


def compute_regime_vols(
    strategy_returns: dict[str, pd.Series],
    regime_states: np.ndarray,
    regime_dates: pd.DatetimeIndex,
    n_states: int,
) -> dict[int, dict[str, float]]:
    """Compute annualized vol per strategy per regime from in-sample data."""
    regime_series = pd.Series(regime_states, index=regime_dates)

    result = {}
    for regime in range(n_states):
        mask = regime_series == regime
        result[regime] = {}

        for name, rets in strategy_returns.items():
            common = rets.index.intersection(regime_dates)
            r = rets.loc[common]
            m = mask.loc[common]
            regime_rets = r[m]
            if len(regime_rets) < 20:
                result[regime][name] = 0.15  # default
                continue
            result[regime][name] = regime_rets.std() * np.sqrt(252)

    return result


# ─── Dynamic Weight Computation ─────────────────────────────────────────────


def regime_weights_for_month(
    state_probs: np.ndarray,
    regime_sharpes: dict[int, dict[str, float]],
    strategy_names: list[str],
    max_weight: float = 0.5,
) -> dict[str, float]:
    """Compute regime-conditional portfolio weights for a single month.

    1. Use state_probs to get expected Sharpe = Σ_i prob_i * sharpe_i
    2. Normalize positive Sharpes; if all negative, equal weight
    3. Cap at max_weight
    """
    expected_sharpes = {}
    for name in strategy_names:
        exp_s = sum(
            state_probs[i] * regime_sharpes.get(i, {}).get(name, 0.0)
            for i in range(len(state_probs))
        )
        expected_sharpes[name] = exp_s

    # Filter to positive
    positive = {k: v for k, v in expected_sharpes.items() if v > 0}

    if not positive:
        # All negative → equal weight
        w = 1.0 / len(strategy_names)
        return {name: w for name in strategy_names}

    total = sum(positive.values())
    weights = {}
    for name in strategy_names:
        if name in positive:
            weights[name] = positive[name] / total
        else:
            weights[name] = 0.0

    # Cap at max_weight and renormalize
    capped = {k: min(v, max_weight) for k, v in weights.items()}
    total_capped = sum(capped.values())
    if total_capped > 0:
        capped = {k: v / total_capped for k, v in capped.items()}

    return capped


# ─── Strategy Return Generation ─────────────────────────────────────────────


def run_strategy_returns(
    spec: dict,
    ohlcv: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    training_start: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    """Run a single strategy and return (equity, daily_returns)."""
    from scripts.run_honest_sweep import HonestStrategyWrapper, _build_config, _find_strategy_class

    StratClass = _find_strategy_class(spec["module"], spec["class_hint"])
    if StratClass is None:
        _log.error("Could not find strategy class for %s", spec["name"])
        return pd.Series(dtype=float), pd.Series(dtype=float)

    config = _build_config(spec["name"], spec["type"], spec.get("params", {}))
    try:
        instance = StratClass(config)
    except Exception as exc:
        _log.error("Strategy init failed for %s: %s", spec["name"], exc)
        return pd.Series(dtype=float), pd.Series(dtype=float)

    wrapper = HonestStrategyWrapper(
        strategy=instance,
        config=config,
        ohlcv=ohlcv,
        **COST_PARAMS,
    )

    try:
        equity, returns, turnover, logs = wrapper.run_returns_backtest(
            start=start,
            end=end,
            training_start=training_start,
        )
        return equity, returns
    except Exception as exc:
        _log.error("Backtest failed for %s: %s", spec["name"], exc)
        return pd.Series(dtype=float), pd.Series(dtype=float)


# ─── Portfolio Backtest ─────────────────────────────────────────────────────


def backtest_weighted_portfolio(
    strategy_returns: dict[str, pd.Series],
    weights_over_time: pd.DataFrame,
    initial_capital: float = 1_000_000.0,
) -> tuple[pd.Series, pd.Series]:
    """Combine strategy returns with time-varying weights.

    Args:
        strategy_returns: {name: daily_returns_series}
        weights_over_time: DataFrame indexed by date, columns=strategy names,
                          values=weight (should sum to ~1 per row)

    Returns:
        (equity_curve, daily_returns)
    """
    # Align all
    common_dates = weights_over_time.index
    for name, rets in strategy_returns.items():
        common_dates = common_dates.intersection(rets.index)
    common_dates = common_dates.sort_values()

    weights = weights_over_time.loc[common_dates]
    rets_df = pd.DataFrame(
        {name: strategy_returns[name].reindex(common_dates, fill_value=0.0)
         for name in strategy_returns}
    )

    # Normalize weights per row
    row_sums = weights.sum(axis=1).replace(0, 1)
    weights = weights.div(row_sums, axis=0)

    # Portfolio return = Σ weight_i * return_i
    port_returns = (weights * rets_df).sum(axis=1)

    # Equity curve
    equity = initial_capital * (1 + port_returns).cumprod()

    return equity, port_returns


def compute_metrics(returns: pd.Series) -> dict:
    """Compute standard portfolio metrics."""
    if len(returns) < 2:
        return {"sharpe": 0, "max_dd_pct": 0, "annual_return_pct": 0, "total_return_pct": 0, "win_rate_pct": 0}

    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0

    equity = (1 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100

    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
    n_years = len(returns) / 252
    annual_ret = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / max(n_years, 0.01)) - 1) * 100

    win_rate = (returns > 0).sum() / len(returns) * 100

    return {
        "sharpe": round(sharpe, 4),
        "max_dd_pct": round(max_dd, 2),
        "annual_return_pct": round(annual_ret, 2),
        "total_return_pct": round(total_ret, 2),
        "win_rate_pct": round(win_rate, 2),
    }


# ─── Main ───────────────────────────────────────────────────────────────────


def main():
    t_start = time.time()

    # ── Step 1: Load data ────────────────────────────────────────────────
    _log.info("=" * 70)
    _log.info("STEP 1: Loading data")
    _log.info("=" * 70)

    # Load macro ETF data for HMM features
    macro_ohlcv = load_ohlcv_wide(MACRO_ETFS, DATA_START, DATA_END)
    macro_close = macro_ohlcv["close"]

    # Load full universe for strategies (macro + bluechip)
    all_tickers = list(set(MACRO_ETFS + BLUECHIP))
    full_ohlcv = load_ohlcv_wide(all_tickers, DATA_START, DATA_END)

    # ── Step 2: Compute features & train HMM ────────────────────────────
    _log.info("=" * 70)
    _log.info("STEP 2: Computing features and training HMM")
    _log.info("=" * 70)

    features = compute_hmm_features(macro_close)
    features_is = features.loc[:IS_END]
    _log.info("In-sample features: %s", features_is.shape)

    hmm_result = train_hmm(features_is, n_states=3, n_restarts=5)

    # Decode in-sample states
    from quantcore.regime.hmm_engine import decode_states

    state_labels = ["bear", "sideways", "bull"]
    interp = decode_states(hmm_result, state_labels=state_labels)
    _log.info("In-sample regime interpretation:")
    for sid, info in interp.items():
        _log.info("  State %d (%s): mean_ret=%.4f, vol=%.4f, occupancy=%.1f%%, self_trans=%.3f",
                  sid, info["label"], info["mean_return"], info["volatility"],
                  info["occupancy_pct"], info["self_transition_prob"])

    # Regime distribution in-sample
    is_states = hmm_result.states
    for s in range(3):
        count = (is_states == s).sum()
        _log.info("  IS regime %d: %d days (%.1f%%)", s, count, count / len(is_states) * 100)

    # ── Step 3: Run strategies for in-sample regime Sharpe lookup ────────
    _log.info("=" * 70)
    _log.info("STEP 3: Computing per-regime strategy Sharpes (in-sample)")
    _log.info("=" * 70)

    strategy_specs = [
        {
            "name": "macro_cross_sectional_momentum",
            "module": "quantcore.strategies.macro.cross_sectional_momentum",
            "class_hint": "CrossSectionalMomentum",
            "type": "macro",
            "universe": "macro",
            "params": {},
        },
        {
            "name": "equity_tsmom",
            "module": "quantcore.strategies.equity.tsmom",
            "class_hint": "TSMOM",
            "type": "equity",
            "universe": "bluechip",
            "params": {},
        },
        {
            "name": "equity_cross_sectional_momentum",
            "module": "quantcore.strategies.equity.cross_sectional_momentum",
            "class_hint": "CrossSectionalMomentum",
            "type": "equity",
            "universe": "bluechip",
            "params": {},
        },
    ]

    # Run each strategy for full period
    strategy_returns_all: dict[str, pd.Series] = {}
    is_start = pd.Timestamp(DATA_START)

    for spec in strategy_specs:
        _log.info("Running %s...", spec["name"])
        # Use macro-only ohlcv for macro strategies, full for equity
        ohlcv = macro_ohlcv if spec.get("universe") == "macro" else full_ohlcv

        _, rets = run_strategy_returns(
            spec=spec,
            ohlcv=ohlcv,
            start=pd.Timestamp(DATA_START),
            end=pd.Timestamp(DATA_END),
            training_start=is_start,
        )
        if len(rets) > 0:
            strategy_returns_all[spec["name"]] = rets
            _log.info("  Got %d days of returns", len(rets))
        else:
            _log.warning("  No returns for %s", spec["name"])

    # Compute in-sample per-regime Sharpes
    is_dates = features_is.index
    regime_sharpes = compute_regime_sharpes(
        strategy_returns_all, is_states, is_dates, n_states=3
    )
    regime_vols = compute_regime_vols(
        strategy_returns_all, is_states, is_dates, n_states=3
    )

    _log.info("Per-regime annualized Sharpes:")
    for regime in range(3):
        label = state_labels[regime] if regime < len(state_labels) else f"state_{regime}"
        sharpes_str = ", ".join(
            f"{name}={regime_sharpes[regime].get(name, 0):.3f}"
            for name in strategy_returns_all
        )
        _log.info("  %s: %s", label, sharpes_str)

    _log.info("Per-regime annualized Volatility:")
    for regime in range(3):
        label = state_labels[regime] if regime < len(state_labels) else f"state_{regime}"
        vols_str = ", ".join(
            f"{name}={regime_vols[regime].get(name, 0):.3f}"
            for name in strategy_returns_all
        )
        _log.info("  %s: %s", label, vols_str)

    # ── Step 4: Compute OOS regime weights ───────────────────────────────
    _log.info("=" * 70)
    _log.info("STEP 4: Computing OOS regime-conditional weights")
    _log.info("=" * 70)

    features_oos = features.loc[OOS_START:]
    _log.info("OOS features: %s", features_oos.shape)

    # Online forward filtering for OOS (no look-ahead)
    from quantcore.regime.hmm_engine import predict_online, _compute_log_B

    params = hmm_result.params
    log_pi = np.log(np.maximum(params.pi, 1e-300))

    # Initialize with last in-sample forward variable
    from quantcore.regime.hmm_engine import forward_log
    log_B_is = _compute_log_B(features_is.values.astype(np.float64), params)
    log_A = np.log(np.maximum(params.A, 1e-300))
    log_alpha_is, _ = forward_log(log_B_is, log_A, log_pi)

    # Start OOS from the last in-sample alpha
    log_alpha_prev = log_alpha_is[-1].copy()

    strategy_names = list(strategy_returns_all.keys())
    oos_dates = features_oos.index

    # Monthly regime probabilities
    monthly_weights: list[dict] = []
    current_month = None
    month_probs: list[np.ndarray] = []

    for i, date in enumerate(oos_dates):
        x_t = features_oos.iloc[i].values.astype(np.float64)

        # Online forward update
        _, state_probs, log_alpha_prev = predict_online(x_t, log_alpha_prev, params)

        # Accumulate by month
        month_key = (date.year, date.month)
        if month_key != current_month:
            if month_probs and current_month is not None:
                # Average state probs for the month
                avg_probs = np.mean(month_probs, axis=0)
                weights = regime_weights_for_month(
                    avg_probs, regime_sharpes, strategy_names
                )
                weights["_month"] = current_month
                weights["_regime"] = int(np.argmax(avg_probs))
                weights["_prob_bull"] = float(avg_probs[2]) if len(avg_probs) > 2 else 0
                weights["_prob_sideways"] = float(avg_probs[1]) if len(avg_probs) > 1 else 0
                weights["_prob_bear"] = float(avg_probs[0])
                monthly_weights.append(weights)

            current_month = month_key
            month_probs = [state_probs]
        else:
            month_probs.append(state_probs)

    # Last month
    if month_probs:
        avg_probs = np.mean(month_probs, axis=0)
        weights = regime_weights_for_month(avg_probs, regime_sharpes, strategy_names)
        weights["_month"] = current_month
        weights["_regime"] = int(np.argmax(avg_probs))
        weights["_prob_bull"] = float(avg_probs[2]) if len(avg_probs) > 2 else 0
        weights["_prob_sideways"] = float(avg_probs[1]) if len(avg_probs) > 1 else 0
        weights["_prob_bear"] = float(avg_probs[0])
        monthly_weights.append(weights)

    _log.info("Computed %d monthly regime weight sets", len(monthly_weights))

    # Build weight DataFrames
    # HMM weights: expand monthly to daily
    hmm_weight_records = []
    for mw in monthly_weights:
        year, month = mw["_month"]
        # Get all trading days in this month
        month_mask = (oos_dates.year == year) & (oos_dates.month == month)
        month_dates = oos_dates[month_mask]
        for d in month_dates:
            row = {name: mw.get(name, 0.0) for name in strategy_names}
            row["_regime"] = mw["_regime"]
            row["_prob_bull"] = mw["_prob_bull"]
            hmm_weight_records.append((d, row))

    if not hmm_weight_records:
        _log.error("No OOS weight records generated")
        return

    hmm_weight_df = pd.DataFrame(
        [r[1] for r in hmm_weight_records],
        index=pd.DatetimeIndex([r[0] for r in hmm_weight_records]),
    )

    # Static weights
    static_weight_df = pd.DataFrame(
        {name: STATIC_WEIGHTS.get(name, 0.0) for name in strategy_names},
        index=hmm_weight_df.index,
    )

    # ── Step 5: Backtest both portfolios ─────────────────────────────────
    _log.info("=" * 70)
    _log.info("STEP 5: Backtesting portfolios")
    _log.info("=" * 70)

    # Trim strategy returns to OOS period
    oos_strategy_returns = {}
    for name, rets in strategy_returns_all.items():
        oos_rets = rets.reindex(hmm_weight_df.index, fill_value=0.0)
        oos_strategy_returns[name] = oos_rets

    # Static portfolio
    static_eq, static_rets = backtest_weighted_portfolio(
        oos_strategy_returns, static_weight_df
    )
    static_metrics = compute_metrics(static_rets)

    # HMM regime portfolio
    regime_cols = [c for c in hmm_weight_df.columns if not c.startswith("_")]
    hmm_eq, hmm_rets = backtest_weighted_portfolio(
        oos_strategy_returns, hmm_weight_df[regime_cols]
    )
    hmm_metrics = compute_metrics(hmm_rets)

    # ── Step 6: Report ───────────────────────────────────────────────────
    _log.info("=" * 70)
    _log.info("RESULTS")
    _log.info("=" * 70)

    regime_map = {0: "bear", 1: "sideways", 2: "bull"}

    _log.info("")
    _log.info("| Metric              | Static Weights | HMM Regime Weights |")
    _log.info("|---------------------|---------------:|-------------------:|")
    _log.info("| OOS Sharpe          | %+.4f        | %+.4f            |",
              static_metrics["sharpe"], hmm_metrics["sharpe"])
    _log.info("| MaxDD               | %.2f%%          | %.2f%%            |",
              static_metrics["max_dd_pct"], hmm_metrics["max_dd_pct"])
    _log.info("| Annual Return       | %.2f%%          | %.2f%%            |",
              static_metrics["annual_return_pct"], hmm_metrics["annual_return_pct"])
    _log.info("| Total Return        | %.2f%%         | %.2f%%           |",
              static_metrics["total_return_pct"], hmm_metrics["total_return_pct"])
    _log.info("| Win Rate            | %.2f%%          | %.2f%%            |",
              static_metrics["win_rate_pct"], hmm_metrics["win_rate_pct"])

    # Regime distribution OOS
    _log.info("")
    _log.info("OOS Regime Distribution:")
    regime_col = hmm_weight_df["_regime"] if "_regime" in hmm_weight_df.columns else None
    if regime_col is not None:
        for regime_id in sorted(regime_col.unique()):
            count = (regime_col == regime_id).sum()
            label = regime_map.get(regime_id, f"state_{regime_id}")
            _log.info("  %s: %d days (%.1f%%)", label, count, count / len(regime_col) * 100)

    # Monthly weight evolution
    _log.info("")
    _log.info("Monthly HMM Weights (sample):")
    for mw in monthly_weights[:6]:
        year, month = mw["_month"]
        regime_label = regime_map.get(mw["_regime"], f"state_{mw['_regime']}")
        w_str = ", ".join(f"{name}={mw.get(name, 0):.2f}" for name in strategy_names)
        _log.info("  %d-%02d [%s]: %s", year, month, regime_label, w_str)
    _log.info("  ...")

    # Decision
    _log.info("")
    sharpe_improvement = hmm_metrics["sharpe"] - static_metrics["sharpe"]
    dd_improvement = static_metrics["max_dd_pct"] - hmm_metrics["max_dd_pct"]

    if sharpe_improvement > 0.1 and dd_improvement > 0:
        decision = "INTEGRATE — HMM overlay improves Sharpe by {:+.3f} and reduces DD by {:.1f}pp".format(
            sharpe_improvement, dd_improvement
        )
    elif sharpe_improvement > 0.05:
        decision = "MONITOR — marginal improvement (Sharpe {:+.3f}), run 30-day paper test".format(
            sharpe_improvement
        )
    else:
        decision = "SHELVE — no meaningful improvement (Sharpe {:+.3f}), static weights preferred".format(
            sharpe_improvement
        )

    _log.info("DECISION: %s", decision)

    # ── Write report ─────────────────────────────────────────────────────
    report = {
        "metadata": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "is_period": f"{DATA_START} to {IS_END}",
            "oos_period": f"{OOS_START} to {DATA_END}",
            "hmm_states": 3,
            "hmm_covar_type": "diag",
            "hmm_n_restarts": 5,
            "hmm_log_likelihood": hmm_result.log_likelihood,
            "hmm_converged": hmm_result.converged,
            "hmm_bic": hmm_result.bic,
            "static_weights": STATIC_WEIGHTS,
            "cost_model": COST_PARAMS,
        },
        "regime_interpretation": {
            str(k): v for k, v in interp.items()
        },
        "per_regime_sharpes": {
            regime_map.get(k, f"state_{k}"): v for k, v in regime_sharpes.items()
        },
        "per_regime_vols": {
            regime_map.get(k, f"state_{k}"): v for k, v in regime_vols.items()
        },
        "results": {
            "static": static_metrics,
            "hmm_regime": hmm_metrics,
        },
        "decision": decision,
        "monthly_weights": [
            {k: v for k, v in mw.items() if not k.startswith("_")}
            | {"month": f"{mw['_month'][0]}-{mw['_month'][1]:02d}",
               "regime": regime_map.get(mw["_regime"], str(mw["_regime"]))}
            for mw in monthly_weights
        ],
    }

    report_path = REPORTS_DIR / "hmm_regime_overlay_backtest.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    _log.info("JSON report: %s", report_path)

    # Markdown report
    md_lines = [
        "# HMM Regime Overlay Backtest Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Configuration",
        "",
        f"- In-sample: {DATA_START} to {IS_END}",
        f"- OOS: {OOS_START} to {DATA_END}",
        f"- HMM: 3 states, diag covariance, 5 restarts",
        f"- Log-likelihood: {hmm_result.log_likelihood:.2f}",
        f"- Converged: {hmm_result.converged}",
        f"- BIC: {hmm_result.bic:.2f}",
        "",
        "## Regime Interpretation (In-Sample)",
        "",
        "| State | Label | Mean Return | Volatility | Occupancy | Self-Transition |",
        "|------:|-------|------------:|-----------:|----------:|----------------:|",
    ]
    for sid, info in interp.items():
        md_lines.append(
            f"| {sid} | {info['label']} | {info['mean_return']:.4f} | "
            f"{info['volatility']:.4f} | {info['occupancy_pct']:.1f}% | "
            f"{info['self_transition_prob']:.3f} |"
        )

    md_lines += [
        "",
        "## Per-Regime Strategy Sharpe (In-Sample Annualized)",
        "",
        "| Regime | " + " | ".join(strategy_names) + " |",
        "|--------|" + "|".join(["---:"] * len(strategy_names)) + "|",
    ]
    for regime in range(3):
        label = regime_map.get(regime, f"state_{regime}")
        vals = " | ".join(f"{regime_sharpes[regime].get(n, 0):.3f}" for n in strategy_names)
        md_lines.append(f"| {label} | {vals} |")

    md_lines += [
        "",
        "## Results Comparison",
        "",
        "| Metric | Static Weights | HMM Regime Weights | Delta |",
        "|--------|---------------:|-------------------:|------:|",
        f"| OOS Sharpe | {static_metrics['sharpe']:+.4f} | {hmm_metrics['sharpe']:+.4f} | {sharpe_improvement:+.4f} |",
        f"| MaxDD | {static_metrics['max_dd_pct']:.2f}% | {hmm_metrics['max_dd_pct']:.2f}% | {dd_improvement:+.2f}pp |",
        f"| Annual Return | {static_metrics['annual_return_pct']:.2f}% | {hmm_metrics['annual_return_pct']:.2f}% | {hmm_metrics['annual_return_pct'] - static_metrics['annual_return_pct']:+.2f}pp |",
        f"| Total Return | {static_metrics['total_return_pct']:.2f}% | {hmm_metrics['total_return_pct']:.2f}% | {hmm_metrics['total_return_pct'] - static_metrics['total_return_pct']:+.2f}pp |",
        f"| Win Rate | {static_metrics['win_rate_pct']:.2f}% | {hmm_metrics['win_rate_pct']:.2f}% | {hmm_metrics['win_rate_pct'] - static_metrics['win_rate_pct']:+.2f}pp |",
        "",
        "## OOS Regime Distribution",
        "",
    ]
    if regime_col is not None:
        for regime_id in sorted(regime_col.unique()):
            count = (regime_col == regime_id).sum()
            label = regime_map.get(regime_id, f"state_{regime_id}")
            md_lines.append(f"- **{label}**: {count} days ({count / len(regime_col) * 100:.1f}%)")

    md_lines += [
        "",
        "## Decision",
        "",
        f"**{decision}**",
        "",
    ]

    if sharpe_improvement > 0.1:
        md_lines += [
            "The HMM regime overlay meaningfully improves risk-adjusted returns.",
            "Recommendation: integrate into the live allocation after a 30-day paper trading run.",
            "",
            "### Integration Steps",
            "1. Add `hmm_regime_detect` to the live runner's monthly rebalance loop",
            "2. Compute online forward probabilities (no look-ahead)",
            "3. Replace static weights with regime-conditional weights",
            "4. Monitor regime stability (if regime flips every few days, increase smoothing)",
        ]
    elif sharpe_improvement > 0.05:
        md_lines += [
            "The HMM regime overlay shows marginal improvement.",
            "Recommendation: run a 30-day paper test to validate before integration.",
        ]
    else:
        md_lines += [
            "The HMM regime overlay does not meaningfully improve over static weights.",
            "Possible causes:",
            "- The 3 strategies already have built-in regime filters (vol percentile)",
            "- The OOS period (2021-2024) may not contain a clear regime shift",
            "- The HMM features may not capture the right regime signal for these strategies",
            "",
            "Recommendation: shelve until a regime-aware strategy (e.g., CTA trend) is added.",
        ]

    md_path = REPORTS_DIR / "hmm_regime_overlay_backtest.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))
    _log.info("Markdown report: %s", md_path)

    t_end = time.time()
    _log.info("Total runtime: %.1f minutes", (t_end - t_start) / 60)


if __name__ == "__main__":
    main()
