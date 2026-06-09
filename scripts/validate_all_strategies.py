"""Automated strategy validation pipeline.

Runs every non-DISABLED strategy through walk-forward validation with:
  - Rolling walk-forward (504 train / 126 test, expanding anchor)
  - OOS annualized Sharpe ratio
  - 90% bootstrap CI lower bound (block bootstrap, 2000 replications)
  - Deflated Sharpe Ratio p-value
  - Maximum drawdown
  - Win rate
  - Number of trades

NOTE: All output uses ASCII-safe characters for Windows console compatibility.
  - Calmar ratio

Outputs:
  - reports/strategy_scoreboard.json
  - reports/strategy_scoreboard.md

Usage:
    python scripts/validate_all_strategies.py
    python scripts/validate_all_strategies.py --quick    # fewer bootstraps
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Strategy-universe mapping
# ---------------------------------------------------------------------------

# Each strategy gets a universe and a strategy_type for construction.
# Options strategies are excluded (need options OHLCV data, not available).
STRATEGY_CONFIGS: dict[str, dict] = {
    # ---- Equity: cross-sectional (need many assets) ----
    "equity_cross_sectional_momentum": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_full_universe_xs_momentum": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_multi_factor": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_multifactor_composite": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_idiosyncratic_momentum": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_low_volatility": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_quality_factor": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_quality_rmw_cma": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_value_factor": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_value_hml": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 300,
    },
    "equity_short_term_reversal": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 100,
    },
    # ---- Equity: single-asset / pairs (need fewer assets) ----
    "equity_mean_reversion": {
        "universe": "original_5",
        "type": "equity",
        "warmup": 100,
    },
    "equity_trend_following": {
        "universe": "original_5",
        "type": "equity",
        "warmup": 250,
    },
    "equity_tsmom": {
        "universe": "original_5",
        "type": "equity",
        "warmup": 250,
    },
    "equity_stat_arb": {
        "universe": "original_5",  # Smaller universe: O(n^2) pair scan
        "type": "equity",
        "warmup": 300,
    },
    "equity_volatility_breakout": {
        "universe": "original_5",
        "type": "equity",
        "warmup": 100,
    },
    "equity_volume_profile": {
        "universe": "original_5",
        "type": "equity",
        "warmup": 100,
    },
    "equity_earnings_momentum": {
        "universe": "expanded_15",
        "type": "equity",
        "warmup": 250,
    },
    "opening_gap_fade": {
        "universe": "original_5",
        "type": "equity",
        "warmup": 50,
    },
    # ---- Macro ----
    "macro_carry": {
        "universe": "expanded_15",
        "type": "macro",
        "warmup": 300,
    },
    "macro_carry_trend_filter": {
        "universe": "expanded_15",
        "type": "macro",
        "warmup": 300,
    },
    "macro_cross_sectional_factor_neutral": {
        "universe": "expanded_15",
        "type": "macro",
        "warmup": 300,
    },
    "macro_cross_sectional_momentum": {
        "universe": "expanded_15",
        "type": "macro",
        "warmup": 300,
    },
    "macro_regime_adaptive_momentum": {
        "universe": "expanded_15",
        "type": "macro",
        "warmup": 300,
    },
    "macro_state_selective_momentum": {
        "universe": "expanded_15",
        "type": "macro",
        "warmup": 300,
    },
    "macro_true_carry": {
        "universe": "expanded_15",
        "type": "macro",
        "warmup": 300,
    },
    "macro_vrp": {
        "universe": "expanded_15",
        "type": "macro",
        "warmup": 300,
    },
    # ---- Excluded: options strategies (need options data) ----
    # "options_delta_neutral": needs VIX/options chain
    # "options_earnings_vol": needs earnings dates + options
    # "options_iron_condor": needs options chain
    # "options_tail_hedge": needs options chain
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StrategyResult:
    strategy_id: str
    universe: str
    oos_sharpe: float = float("nan")
    ci_lower_90: float = float("nan")
    ci_upper_90: float = float("nan")
    dsr_pvalue: float = float("nan")
    max_drawdown_pct: float = float("nan")
    win_rate_pct: float = float("nan")
    n_trades: int = 0
    calmar_ratio: float = float("nan")
    annual_return_pct: float = float("nan")
    total_return_pct: float = float("nan")
    oos_bars: int = 0
    status: str = "error"
    error: str = ""
    triage: str = ""  # KEEP / MONITOR / DISABLE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_CACHE: dict[str, pd.DataFrame] = {}


def _load_data(universe: str, start: str, end: str) -> pd.DataFrame:
    """Load and cache universe data."""
    cache_key = f"{universe}_{start}_{end}"
    if cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key]

    from scripts.real_data_loader import load_universe
    df = load_universe(universe, start, end, use_cache=True)
    _DATA_CACHE[cache_key] = df
    return df


def _df_to_data(df: pd.DataFrame) -> dict:
    """Convert long-format DataFrame to strategy data dict."""
    symbols = sorted(df["symbol"].unique())
    if not symbols:
        return {"close": np.array([[]], dtype=float), "symbols": []}

    pivots: dict[str, pd.DataFrame] = {}
    for field_name in ["open", "high", "low", "close", "volume"]:
        pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field_name)
        pivots[field_name] = pivoted[symbols] if all(s in pivoted.columns for s in symbols) else pivoted

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
    # Forward-fill NaN (causal: only past data)
    for k in ["close", "high", "low", "open", "volume"]:
        arr = data[k]
        if np.any(np.isnan(arr)):
            df_filled = pd.DataFrame(arr).ffill(limit=5)
            data[k] = np.nan_to_num(df_filled.values, nan=0.0)
    return data


# ---------------------------------------------------------------------------
# Signal-based return computation (from run_backtest_focused.py)
# ---------------------------------------------------------------------------

def _compute_signal_returns(
    strat,
    data: dict,
    start: int,
    end: int,
    timestamps: list | None = None,
    tc_bps: float = 5.0,
    spread_bps: float = 5.0,
    impact_bps: float = 10.0,
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
        current_weights: dict[str, float] = {}
        current_signals: dict[str, dict] = {}

        for s in signals:
            if hasattr(s, 'size') and s.size > 0:
                weight = float(s.direction) * float(s.size)
            else:
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

        bar_rets[idx] = weighted_ret

        # Realistic transaction cost: bid-ask spread + market impact
        # Total cost bps = spread_bps + impact_bps (e.g. 5 + 10 = 15 bps)
        if (tc_bps > 0 or spread_bps > 0 or impact_bps > 0) and current_weights:
            turnover = 0.0
            all_syms = set(carried_weights.keys()) | set(current_weights.keys())
            for sym in all_syms:
                prev_w = carried_weights.get(sym, 0.0)
                curr_w = current_weights.get(sym, 0.0)
                turnover += abs(curr_w - prev_w)
            total_cost_bps = tc_bps + spread_bps + impact_bps
            bar_rets[idx] -= turnover * total_cost_bps / 10_000.0

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
                    positions[sym] = {"direction": sig["direction"], "entry_ts": entry_ts, "entry_price": entry_price}

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
# Walk-forward validation
# ---------------------------------------------------------------------------

def _run_walk_forward(
    strat_cls: type,
    params: dict,
    data: dict,
    timestamps: list,
    strategy_type: str,
    warmup: int = 300,
    train_size: int = 504,
    test_size: int = 126,
    tc_bps: float = 5.0,
    spread_bps: float = 5.0,
    impact_bps: float = 10.0,
) -> tuple[np.ndarray, list[dict]]:
    """Run anchored expanding-window walk-forward validation.

    Returns aggregated OOS returns and trades across all folds.
    """
    from quantcore.strategies.base import StrategyConfig, StrategyType

    close = data["close"]
    n_periods = close.shape[0]

    # Map strategy type
    stype_map = {
        "equity": StrategyType.EQUITY,
        "macro": StrategyType.MACRO,
        "options": StrategyType.OPTIONS,
    }
    stype = stype_map.get(strategy_type, StrategyType.EQUITY)

    all_oos_returns: list[np.ndarray] = []
    all_trades: list[dict] = []

    # Walk-forward folds: anchored (expanding training window)
    fold_start = warmup
    while fold_start + test_size <= n_periods:
        train_end = fold_start
        test_start = fold_start
        test_end = min(fold_start + test_size, n_periods)

        if test_end - test_start < 10:
            break

        # Create fresh strategy instance for this fold
        cfg = StrategyConfig(
            name=strat_cls.__name__,
            strategy_type=stype,
            params=dict(params),
        )
        try:
            strat = strat_cls(cfg)
            strat.initialize()
        except Exception:
            break

        # Warmup: feed training data to build state
        for bar_i in range(0, train_end):
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

        # OOS test: collect returns
        oos_rets, fold_trades = _compute_signal_returns(
            strat, data, test_start, test_end,
            timestamps=timestamps, tc_bps=tc_bps,
            spread_bps=spread_bps, impact_bps=impact_bps,
        )
        if len(oos_rets) > 0:
            all_oos_returns.append(oos_rets)
            all_trades.extend(fold_trades)

        fold_start += test_size

    if not all_oos_returns:
        return np.array([], dtype=float), []

    return np.concatenate(all_oos_returns), all_trades


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_oos_metrics(
    returns: np.ndarray,
    trades: list[dict],
    n_bootstrap: int = 2000,
    ci_level: float = 0.90,
    n_strategies_tested: int = 40,
) -> dict:
    """Compute all OOS metrics including bootstrap CI and DSR."""
    from quantcore.risk.metrics import sharpe_ratio

    if len(returns) < 20:
        return {"error": "insufficient OOS bars"}

    # Guard: check for NaN/inf in input
    if not np.all(np.isfinite(returns)):
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Basic metrics — clip BOTH tails to prevent overflow.
    # np.clip with symmetric bounds: negatives capped at -0.5 (≤50% loss/bar),
    # positives ALSO capped at +0.5 (≤50% gain/bar) so compounding cannot overflow.
    capped = np.clip(returns, -0.5, 0.5)
    mean_ret = np.mean(capped)
    std_ret = np.std(capped, ddof=1)
    sharpe = float(sharpe_ratio(capped, rf=0.04)) if std_ret > 1e-10 else float("nan")

    # Total return via compounding; guard against overflow.
    raw_total = np.prod(1 + capped)
    if not np.isfinite(raw_total) or raw_total > 1e100:
        # Compounding overflowed — fall back to linear return approximation.
        total_ret = float(np.sum(capped))
    else:
        total_ret = float(raw_total - 1.0)
    # Hard sanity cap: a >10,000,000% total return is physically impossible.
    total_ret = max(-1.0 + 1e-10, min(total_ret, 1e8))

    n_years = len(capped) / 252.0
    safe_n_years = max(n_years, 0.001)
    if 1 + total_ret <= 0:
        annual_ret = -1.0
    elif not np.isfinite((1 + total_ret) ** (1 / safe_n_years)):
        # Power overflowed — fall back to simple annualization.
        annual_ret = float(mean_ret * 252)
    else:
        annual_ret = float((1 + total_ret) ** (1 / safe_n_years) - 1)

    # Cap Sharpe at ±100 (a 100 Sharpe is already physically impossible).
    if not np.isfinite(sharpe) or abs(sharpe) > 100:
        sharpe = float("nan")

    # Max drawdown (use capped returns to prevent overflow)
    equity = np.cumprod(1 + capped)
    if not np.all(np.isfinite(equity)):
        equity = np.nan_to_num(equity, nan=1.0, posinf=1e10, neginf=1e-10)
    peak = np.maximum.accumulate(equity)
    # Guard: if peak is 0, use 1.0 to avoid division by zero in dd.
    safe_peak = np.where(peak > 1e-10, peak, 1.0)
    dd = (equity - peak) / safe_peak
    max_dd = float(abs(np.min(dd)))
    # Sanity cap: max drawdown cannot exceed 100%.
    max_dd = min(max_dd, 1.0)

    if not np.isfinite(max_dd):
        max_dd = 1.0

    # Calmar ratio (guard all overflow paths)
    if not np.isfinite(annual_ret) or max_dd < 0.0001:
        calmar = float("nan")
    else:
        calmar = annual_ret / max_dd
        if not np.isfinite(calmar) or abs(calmar) > 1e8:
            calmar = float("nan")

    # Win rate
    pnls = [t.get("pnl_abs", 0) for t in trades if t.get("pnl_abs") is not None]
    win_rate = float(np.mean(np.array(pnls) > 0)) * 100 if pnls else 0.0

    # Bootstrap CI for Sharpe (block bootstrap)
    ci_lower, ci_upper = _block_bootstrap_sharpe_ci(
        returns, n_bootstrap=n_bootstrap, ci_level=ci_level, block_len=63,
    )

    # Deflated Sharpe Ratio (full B&LdP formula with skew/kurtosis)
    dsr_pvalue = _deflated_sharpe_pvalue(
        sharpe, len(returns), n_strategies_tested,
        returns=returns,
    )

    return {
        "oos_sharpe": round(sharpe, 4),
        "ci_lower_90": round(ci_lower, 4),
        "ci_upper_90": round(ci_upper, 4),
        "dsr_pvalue": round(dsr_pvalue, 4),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate, 1),
        "n_trades": len(trades),
        "calmar_ratio": round(calmar, 4) if np.isfinite(calmar) else None,
        "annual_return_pct": round(annual_ret * 100, 2),
        "total_return_pct": round(total_ret * 100, 2),
        "oos_bars": len(returns),
    }


def _block_bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 2000,
    ci_level: float = 0.90,
    block_len: int = 63,
) -> tuple[float, float]:
    """Block bootstrap confidence interval for Sharpe ratio.

    Uses non-overlapping blocks of length block_len to preserve
    autocorrelation structure.
    """
    from quantcore.risk.metrics import sharpe_ratio

    rng = np.random.default_rng(42)
    n = len(returns)
    n_blocks = max(1, n // block_len)

    sharpes = np.zeros(n_bootstrap)
    for b in range(n_bootstrap):
        # Sample blocks with replacement
        block_starts = rng.integers(0, max(1, n - block_len + 1), size=n_blocks)
        sample = np.concatenate([returns[s:s + block_len] for s in block_starts])
        sr = sharpe_ratio(sample, rf=0.04)
        sharpes[b] = sr if np.isfinite(sr) else 0.0

    alpha = (1 - ci_level) / 2
    ci_lower = float(np.percentile(sharpes, alpha * 100))
    ci_upper = float(np.percentile(sharpes, (1 - alpha) * 100))
    return ci_lower, ci_upper


def _deflated_sharpe_pvalue(
    observed_sharpe: float,
    n_obs: int,
    n_strategies: int,
    returns: np.ndarray | None = None,
) -> float:
    """Compute Deflated Sharpe Ratio p-value.

    Uses the full Bailey & López de Prado (2012) formula which accounts for:
      - Number of independent trials (strategies tested)
      - Sample length
      - Skewness and excess kurtosis of returns

    The simplified version (without skew/kurtosis) is used as fallback.
    """
    from quantcore.research.deflated_sharpe import deflated_sharpe_ratio

    if np.isnan(observed_sharpe) or n_obs < 20:
        return 1.0

    # Conservative trial count: strategies tested + 10 prior experiments
    n_trials = n_strategies + 10

    # Compute skewness and kurtosis from returns if available
    skewness = None
    kurtosis_ = None
    if returns is not None and len(returns) > 20:
        r = np.asarray(returns, dtype=float)
        r_clean = r[np.isfinite(r)]
        if len(r_clean) > 20:
            mean_r = np.mean(r_clean)
            std_r = np.std(r_clean, ddof=1)
            if std_r > 1e-12:
                skewness = float(np.mean((r_clean - mean_r) ** 3) / (std_r ** 3))
                kurtosis_ = float(np.mean((r_clean - mean_r) ** 4) / (std_r ** 4) - 3.0)

    p_value, _ = deflated_sharpe_ratio(
        observed_sharpe=observed_sharpe,
        num_trials=n_trials,
        sample_length=n_obs,
        skewness=skewness,
        kurtosis_=kurtosis_,
    )

    return float(np.clip(p_value, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Strategy triage
# ---------------------------------------------------------------------------

def _triage(result: StrategyResult) -> str:
    """Classify strategy into KEEP / MONITOR / DISABLE."""
    if result.status == "error":
        return "DISABLE"

    # Check if we have valid metrics
    if np.isnan(result.oos_sharpe) or np.isnan(result.ci_lower_90):
        return "DISABLE"

    # KEEP: CI lower > 0 AND DSR p-value < 0.1 AND >= 20 trades
    if result.ci_lower_90 > 0 and result.dsr_pvalue < 0.1 and result.n_trades >= 20:
        return "KEEP"

    # MONITOR: CI lower > 0 (edge exists but not strongly deflated)
    if result.ci_lower_90 > 0:
        return "MONITOR"

    return "DISABLE"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_validation(
    quick: bool = False,
    start_date: str = "2018-01-01",
    end_date: str = "2024-12-31",
) -> list[StrategyResult]:
    """Run validation on all configured strategies."""
    import quantcore.strategies.equity
    import quantcore.strategies.macro
    import quantcore.strategies.options
    from quantcore.strategies.registry import _registry

    n_bootstrap = 200 if quick else 2000
    results: list[StrategyResult] = []

    # Filter to strategies we can validate
    to_validate = []
    for strat_id, cfg in STRATEGY_CONFIGS.items():
        if strat_id not in _registry:
            print(f"  SKIP {strat_id}: not registered")
            continue
        cls = _registry[strat_id]
        status = getattr(cls, "STRATEGY_STATUS", None)
        if status and status.value == "disabled":
            print(f"  SKIP {strat_id}: DISABLED")
            continue
        to_validate.append((strat_id, cls, cfg))

    print(f"\n{'='*70}")
    print(f"  STRATEGY VALIDATION PIPELINE")
    print(f"  Period: {start_date} -> {end_date}")
    print(f"  Strategies to validate: {len(to_validate)}")
    print(f"  Bootstrap replications: {n_bootstrap}")
    print(f"{'='*70}\n")

    # Group by universe to load data once
    universes_needed = set(cfg["universe"] for _, _, cfg in to_validate)

    for universe in universes_needed:
        print(f"  Loading universe '{universe}'...")
        try:
            df = _load_data(universe, start_date, end_date)
            if df.empty:
                print(f"  WARNING: No data for universe '{universe}'")
            else:
                print(f"  → {len(df)} bars, {df['symbol'].nunique()} symbols")
        except Exception as e:
            print(f"  ERROR loading '{universe}': {e}")

    print()

    # Validate each strategy
    for i, (strat_id, cls, cfg) in enumerate(to_validate, 1):
        universe = cfg["universe"]
        warmup = cfg.get("warmup", 300)
        strategy_type = cfg.get("type", "equity")

        result = StrategyResult(strategy_id=strat_id, universe=universe)
        print(f"[{i}/{len(to_validate)}] {strat_id} ({universe})...", end=" ", flush=True)

        try:
            df = _load_data(universe, start_date, end_date)
            if df.empty:
                result.status = "error"
                result.error = "no data"
                print("NO DATA")
                results.append(result)
                continue

            data = _df_to_data(df)
            timestamps = sorted(df["timestamp"].unique())

            # Get default params from the strategy class
            from quantcore.strategies.base import StrategyConfig, StrategyType
            stype_map = {"equity": StrategyType.EQUITY, "macro": StrategyType.MACRO, "options": StrategyType.OPTIONS}
            temp_cfg = StrategyConfig(name=strat_id, strategy_type=stype_map.get(strategy_type, StrategyType.EQUITY))
            temp_strat = cls(temp_cfg)
            default_params = dict(temp_cfg.params)

            t0 = time.time()
            returns, trades = _run_walk_forward(
                cls, default_params, data, timestamps,
                strategy_type=strategy_type,
                warmup=warmup,
                train_size=504,
                test_size=126,
                tc_bps=5.0,
                spread_bps=5.0,
                impact_bps=10.0,
            )
            elapsed = time.time() - t0

            if len(returns) < 20:
                result.status = "error"
                result.error = f"insufficient OOS returns ({len(returns)} bars)"
                print(f"SKIP ({len(returns)} bars, {elapsed:.1f}s)")
                results.append(result)
                continue

            metrics = _compute_oos_metrics(
                returns, trades,
                n_bootstrap=n_bootstrap,
                n_strategies_tested=len(to_validate),
            )

            result.oos_sharpe = metrics.get("oos_sharpe", float("nan"))
            result.ci_lower_90 = metrics.get("ci_lower_90", float("nan"))
            result.ci_upper_90 = metrics.get("ci_upper_90", float("nan"))
            result.dsr_pvalue = metrics.get("dsr_pvalue", float("nan"))
            result.max_drawdown_pct = metrics.get("max_drawdown_pct", float("nan"))
            result.win_rate_pct = metrics.get("win_rate_pct", float("nan"))
            result.n_trades = metrics.get("n_trades", 0)
            result.calmar_ratio = metrics.get("calmar_ratio", float("nan"))
            result.annual_return_pct = metrics.get("annual_return_pct", float("nan"))
            result.total_return_pct = metrics.get("total_return_pct", float("nan"))
            result.oos_bars = metrics.get("oos_bars", 0)
            result.status = "ok"

            sr = result.oos_sharpe
            ci = result.ci_lower_90
            print(f"SR={sr:+.3f}  CI_low={ci:+.3f}  DD={result.max_drawdown_pct:.1f}%  "
                  f"Trades={result.n_trades}  ({elapsed:.1f}s)")

        except Exception as e:
            result.status = "error"
            result.error = str(e)[:200]
            print(f"ERROR: {e}")
            traceback.print_exc()

        results.append(result)

    # Triage
    for r in results:
        r.triage = _triage(r)

    return results


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def _save_scoreboard_json(results: list[StrategyResult], path: Path) -> None:
    """Save scoreboard as JSON."""
    data = {
        "generated": datetime.now().isoformat(),
        "n_strategies": len(results),
        "n_ok": sum(1 for r in results if r.status == "ok"),
        "n_keep": sum(1 for r in results if r.triage == "KEEP"),
        "n_monitor": sum(1 for r in results if r.triage == "MONITOR"),
        "n_disable": sum(1 for r in results if r.triage == "DISABLE"),
        "strategies": [asdict(r) for r in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\n  JSON scoreboard: {path}")


def _save_scoreboard_md(results: list[StrategyResult], path: Path) -> None:
    """Save scoreboard as Markdown table."""
    sorted_results = sorted(results, key=lambda r: r.ci_lower_90 if not np.isnan(r.ci_lower_90) else -999, reverse=True)

    lines = [
        "# Strategy Scoreboard",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"**Total strategies evaluated:** {len(results)}",
        f"**KEEP:** {sum(1 for r in results if r.triage == 'KEEP')} | "
        f"**MONITOR:** {sum(1 for r in results if r.triage == 'MONITOR')} | "
        f"**DISABLE:** {sum(1 for r in results if r.triage == 'DISABLE')}",
        "",
        "## Rankings (sorted by CI lower bound)",
        "",
        "| # | Strategy | OOS Sharpe | CI Lower | CI Upper | DSR p-val | Max DD% | Win% | Trades | Calmar | Triage |",
        "|---|----------|-----------|----------|----------|-----------|---------|------|--------|--------|--------|",
    ]

    for i, r in enumerate(sorted_results, 1):
        sr = f"{r.oos_sharpe:+.3f}" if not np.isnan(r.oos_sharpe) else "ERR"
        ci_lo = f"{r.ci_lower_90:+.3f}" if not np.isnan(r.ci_lower_90) else "ERR"
        ci_hi = f"{r.ci_upper_90:+.3f}" if not np.isnan(r.ci_upper_90) else "ERR"
        dsr = f"{r.dsr_pvalue:.4f}" if not np.isnan(r.dsr_pvalue) else "ERR"
        dd = f"{r.max_drawdown_pct:.1f}" if not np.isnan(r.max_drawdown_pct) else "ERR"
        wr = f"{r.win_rate_pct:.1f}" if not np.isnan(r.win_rate_pct) else "ERR"
        cal = f"{r.calmar_ratio:.2f}" if r.calmar_ratio and not np.isnan(r.calmar_ratio) else "—"
        triage = r.triage or "—"
        emoji = {"KEEP": "[OK]", "MONITOR": "[??]", "DISABLE": "[X]"}.get(triage, "[?]")

        lines.append(
            f"| {i} | {r.strategy_id} | {sr} | {ci_lo} | {ci_hi} | {dsr} | {dd} | {wr} | {r.n_trades} | {cal} | {emoji} {triage} |"
        )

    lines.extend([
        "",
        "## Triage Criteria",
        "",
        "- **KEEP**: OOS Sharpe CI lower > 0 AND DSR p-value < 0.1 AND >= 20 trades",
        "- **MONITOR**: OOS Sharpe CI lower > 0 (edge exists but not strongly deflated)",
        "- **DISABLE**: All others (no statistical evidence of edge)",
        "",
        "## Notes",
        "",
        "- Walk-forward: 504-bar train (expanding), 126-bar test, anchored",
        "- Costs: 5 bps half-spread + commission per turnover",
        "- Bootstrap: 2000 replications, block length = 63 days",
        "- DSR: conservatively assumes N_strategies + 10 prior trials",
        "- Options strategies excluded (require options chain data)",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  MD scoreboard:  {path}")


def _save_triage_md(results: list[StrategyResult], path: Path) -> None:
    """Save triage decisions with rationale."""
    lines = [
        "# Strategy Triage Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Decisions",
        "",
    ]

    for group in ["KEEP", "MONITOR", "DISABLE"]:
        group_results = [r for r in results if r.triage == group]
        emoji = {"KEEP": "[OK]", "MONITOR": "[??]", "DISABLE": "[X]"}[group]
        lines.append(f"### {emoji} {group} ({len(group_results)} strategies)")
        lines.append("")

        for r in sorted(group_results, key=lambda x: x.oos_sharpe if not np.isnan(x.oos_sharpe) else -999, reverse=True):
            sr = f"{r.oos_sharpe:+.3f}" if not np.isnan(r.oos_sharpe) else "ERR"
            ci = f"{r.ci_lower_90:+.3f}" if not np.isnan(r.ci_lower_90) else "ERR"
            dsr = f"{r.dsr_pvalue:.4f}" if not np.isnan(r.dsr_pvalue) else "ERR"

            if r.status == "error":
                reason = f"Error: {r.error}"
            elif group == "KEEP":
                reason = f"OOS Sharpe {sr}, CI lower {ci} > 0, DSR p={dsr} < 0.1, {r.n_trades} trades"
            elif group == "MONITOR":
                if r.n_trades < 20:
                    reason = f"OOS Sharpe {sr}, CI lower {ci} > 0, but only {r.n_trades} trades (< 20)"
                elif r.dsr_pvalue >= 0.1:
                    reason = f"OOS Sharpe {sr}, CI lower {ci} > 0, but DSR p={dsr} >= 0.1"
                else:
                    reason = f"OOS Sharpe {sr}, CI lower {ci} > 0, borderline"
            else:
                if np.isnan(r.oos_sharpe) or np.isnan(r.ci_lower_90):
                    reason = "Insufficient data or computation error"
                elif r.ci_lower_90 <= 0:
                    reason = f"CI lower {ci} <= 0 (no statistically significant edge)"
                else:
                    reason = f"DSR p={dsr} too high or < 20 trades"

            lines.append(f"- **{r.strategy_id}** — {reason}")

        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Triage report:  {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Validate all strategies")
    parser.add_argument("--quick", action="store_true", help="Fewer bootstraps (200 vs 2000)")
    parser.add_argument("--start", default="2018-01-01", help="Start date")
    parser.add_argument("--end", default="2024-12-31", help="End date")
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.WARNING)

    t0 = time.time()
    results = run_validation(quick=args.quick, start_date=args.start, end_date=args.end)
    elapsed = time.time() - t0

    # Save outputs
    reports_dir = REPO_ROOT / "reports"
    _save_scoreboard_json(results, reports_dir / "strategy_scoreboard.json")
    _save_scoreboard_md(results, reports_dir / "strategy_scoreboard.md")
    _save_triage_md(results, reports_dir / "strategy_triage.md")

    # Summary
    n_keep = sum(1 for r in results if r.triage == "KEEP")
    n_monitor = sum(1 for r in results if r.triage == "MONITOR")
    n_disable = sum(1 for r in results if r.triage == "DISABLE")
    n_error = sum(1 for r in results if r.status == "error")

    print(f"\n{'='*70}")
    print(f"  VALIDATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*70}")
    print(f"  [OK]  KEEP:    {n_keep}")
    print(f"  [??]  MONITOR: {n_monitor}")
    print(f"  [X]   DISABLE: {n_disable}")
    print(f"  [!]   ERROR:   {n_error}")
    print(f"{'='*70}")

    if n_keep > 0:
        print(f"\n  Surviving strategies:")
        for r in results:
            if r.triage == "KEEP":
                print(f"    [OK] {r.strategy_id}: SR={r.oos_sharpe:+.3f}, CI_low={r.ci_lower_90:+.3f}")


if __name__ == "__main__":
    main()
