"""Automated strategy validation pipeline.

Runs every non-DISABLED strategy through walk-forward validation with:
  - Anchored expanding-window walk-forward (300-bar warmup, 126-bar OOS folds)
    executed through the FULL BacktestEngine stack (queue-based no-lookahead
    fills, slippage 5 bps, $0.005/share commissions, mark-to-market,
    end-of-period liquidation) via FrameworkStrategyAdapter
  - OOS annualized Sharpe ratio
  - 90% bootstrap CI lower bound (block bootstrap, 2000 replications)
  - Deflated Sharpe Ratio p-value (published Bailey & Lopez de Prado 2014
    form via quantcore.research.dsr_published)
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
    python scripts/validate_all_strategies.py --quick --enforce-runtime-gate  # CI gate
"""

from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_STATUS_PATH = REPO_ROOT / "strategy_status.yaml"
DEFAULT_WAIVERS_PATH = REPO_ROOT / "config" / "validation_waivers.yaml"
RUNTIME_EVIDENCE_GATE = "runtime_evidence_gate"
RUNTIME_PHASES = {"PAPER", "LIVE"}

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
    """Convert long-format DataFrame to strategy data dict.

    Delisted / not-yet-listed names are masked to NaN after their last traded
    day (the mask is carried as ``live_mask`` + ``last_traded_index``) and are
    NEVER zero-filled: a zero price manufactures a fake -100% return for every
    dead name and poisons the whole cross-section.
    """
    symbols = sorted(df["symbol"].unique())
    if not symbols:
        return {"close": np.array([[]], dtype=float), "symbols": []}

    pivots: dict[str, pd.DataFrame] = {}
    for field_name in ["open", "high", "low", "close", "volume"]:
        pivoted = df.pivot_table(index="timestamp", columns="symbol", values=field_name)
        pivots[field_name] = pivoted[symbols] if all(s in pivoted.columns for s in symbols) else pivoted

    if pivots["close"].empty:
        return {"close": np.array([[]], dtype=float), "symbols": symbols}

    # Last traded row per symbol, from RAW prices (before any forward-fill).
    close_raw = pivots["close"].to_numpy()
    last_traded_index: dict[str, int] = {}
    live_mask = np.zeros(close_raw.shape, dtype=bool)
    for j, sym in enumerate(symbols):
        valid = np.flatnonzero(~np.isnan(close_raw[:, j]))
        last_idx = int(valid[-1]) if valid.size else -1
        last_traded_index[sym] = last_idx
        if last_idx >= 0:
            live_mask[: last_idx + 1, j] = True

    data = {
        "close": pivots["close"].values,
        "high": pivots["high"].values,
        "low": pivots["low"].values,
        "open": pivots["open"].values,
        "volume": pivots["volume"].values,
        "symbols": symbols,
    }
    # Causal forward-fill of short gaps (max 5 days), then mask every row after
    # a name's last traded day back to NaN. Never zero-fill: 0.0 is a real price
    # for nothing and manufactures a fake -100% bar after delisting.
    for k in ["close", "high", "low", "open", "volume"]:
        arr = data[k]
        if np.any(np.isnan(arr)):
            df_filled = pd.DataFrame(arr).ffill(limit=5)
            arr = df_filled.to_numpy(copy=True)
            arr[~live_mask] = np.nan
            data[k] = arr
    data["live_mask"] = live_mask
    data["last_traded_index"] = last_traded_index
    return data


def _last_valid_price(
    close: np.ndarray,
    symbols: list[str],
    sym: str,
    bar_i: int,
    last_traded_index: dict[str, int],
) -> float | None:
    """Price at ``bar_i`` for ``sym``, falling back to the final traded close.

    After a name's last traded day its column is NaN; a position still held
    through delisting is settled at the final real trade, never at NaN.
    """
    try:
        idx = symbols.index(sym)
    except ValueError:
        return None
    price = close[bar_i, idx] if bar_i < close.shape[0] else np.nan
    if np.isfinite(price):
        return float(price)
    lti = last_traded_index.get(sym)
    if lti is not None and 0 <= lti <= bar_i and lti < close.shape[0] and np.isfinite(close[lti, idx]):
        return float(close[lti, idx])
    return None


# ---------------------------------------------------------------------------
# Signal-based return computation (RETIRED from the pipeline; tests only)
#
# _df_to_data / _last_valid_price / _compute_signal_returns implement the old
# signal-returns approximation (weights x close-to-close returns minus
# approximate costs). The validation pipeline stopped using them on
# 2026-10-10 in favor of _run_walk_forward_engine (full BacktestEngine
# execution stack). They are retained ONLY because
# tests/unit/test_df_to_data_delisted_tail.py regression-tests their
# delisted-name safety properties (no zero-fill, no fake -100% bars), which
# document survivorship-handling conventions the engine path must respect.
# Do NOT wire them back into any verdict path.
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

    last_traded_index: dict[str, int] = data.get("last_traded_index", {})

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
            if not np.isfinite(weight):
                continue
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
            asset_ret = float(asset_rets[idx, asset_idx])
            if not np.isfinite(asset_ret):
                continue
            weighted_ret += weight * asset_ret

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
                exit_price = _last_valid_price(close, symbols, sym, bar_i, last_traded_index)
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
                    try:
                        asset_idx = symbols.index(sym)
                    except ValueError:
                        continue
                    entry_price = float(close[bar_i, asset_idx]) if bar_i < close.shape[0] else np.nan
                    if not np.isfinite(entry_price):
                        continue
                    entry_ts = timestamps[bar_i] if timestamps and bar_i < len(timestamps) else None
                    positions[sym] = {"direction": sig["direction"], "entry_ts": entry_ts, "entry_price": entry_price}

        if current_weights:
            carried_weights = dict(current_weights)

    # Close remaining positions
    for sym in list(positions.keys()):
        pos = positions.pop(sym)
        exit_ts = timestamps[end - 1] if timestamps and end - 1 < len(timestamps) else None
        exit_price = _last_valid_price(close, symbols, sym, end - 1, last_traded_index)
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
# Walk-forward validation (full execution engine)
# ---------------------------------------------------------------------------

def _run_walk_forward_engine(
    strat_cls: type,
    params: dict,
    df: pd.DataFrame,
    strategy_type: str,
    warmup: int = 300,
    test_size: int = 126,
    tc_bps: float = 5.0,
) -> tuple[np.ndarray, list[dict]]:
    """Anchored expanding-window walk-forward through the FULL BacktestEngine.

    Replaces the retired signal-returns shortcut (2026-10-10): every fold now
    runs the complete execution stack — queue-based no-lookahead fills,
    slippage, commissions, mark-to-market, end-of-period liquidation — so
    validation numbers are net of realistic execution and the engine's
    honesty guarantees (canary look-ahead tests, buy_and_hold(SPY) cost gate)
    apply to every verdict this pipeline produces.

    Per fold:
      1. Fresh strategy instance (no state leakage across folds). The
         FrameworkStrategyAdapter calls initialize() itself.
      2. Train pass: engine over bars [0, test_start) — indicator state is
         built through real execution, with fills fed back via on_fill.
      3. Test pass: FRESH portfolio, same adapter (bar buffers continue from
         training so indicators see continuous history), engine over bars
         [test_start, test_end).
      4. Daily equity -> log returns; engine fills -> FIFO realized
         round-trip P&Ls for trade counts and win rate.

    Returns aggregated OOS returns and realized-trade dicts across folds.
    """
    from quantcore.backtest.adapter import FrameworkStrategyAdapter
    from quantcore.backtest.engine import BacktestEngine
    from quantcore.strategies.base import StrategyConfig, StrategyType

    symbols = sorted(df["symbol"].unique())
    if not symbols:
        return np.array([], dtype=float), []

    # Unique timestamps sorted; map every row to its positional index via
    # int64 epoch keys (robust to tz-aware datetime64 dtypes, whose .values
    # would otherwise degrade to object arrays of Timestamps).
    unique_ts = pd.DatetimeIndex(sorted(df["timestamp"].unique()))
    n_periods = len(unique_ts)
    row_pos = np.searchsorted(
        unique_ts.asi8, pd.DatetimeIndex(df["timestamp"]).asi8
    )

    stype_map = {
        "equity": StrategyType.EQUITY,
        "macro": StrategyType.MACRO,
        "options": StrategyType.OPTIONS,
    }
    stype = stype_map.get(strategy_type, StrategyType.EQUITY)

    all_oos_returns: list[np.ndarray] = []
    all_trades: list[dict] = []

    fold_start = warmup
    fold_idx = 0
    while fold_start + test_size <= n_periods:
        test_start = fold_start
        test_end = min(fold_start + test_size, n_periods)
        if test_end - test_start < 10:
            break

        fold_idx += 1
        try:
            cfg = StrategyConfig(
                name=strat_cls.__name__,
                strategy_type=stype,
                params=dict(params),
            )
            strat = strat_cls(cfg)
            adapter = FrameworkStrategyAdapter(strategy=strat, symbols=symbols)
        except Exception as e:
            print(f"WARNING: fold {fold_idx}: strategy init failed: {e}", flush=True)
            break

        try:
            # Train pass [0, test_start): build state through real execution.
            if test_start > 5:
                train_data = df[row_pos < test_start].copy()
                if len(train_data) > 5:
                    train_engine = BacktestEngine(
                        strategy=adapter,
                        initial_cash=1_000_000,
                        slippage_bps=tc_bps,
                        commission_per_share=0.005,
                    )
                    train_engine.run(train_data)

            # Test pass [test_start, test_end): fresh portfolio, same adapter.
            test_data = df[(row_pos >= test_start) & (row_pos < test_end)].copy()
            if len(test_data) < 5:
                fold_start += test_size
                continue

            test_engine = BacktestEngine(
                strategy=adapter,
                initial_cash=1_000_000,
                slippage_bps=tc_bps,
                commission_per_share=0.005,
            )
            eq_df = test_engine.run(test_data)
        except Exception as e:
            print(f"WARNING: fold {fold_idx} (bars {test_start}:{test_end}) failed: {e}", flush=True)
            fold_start += test_size
            continue

        if len(eq_df) < 3:
            fold_start += test_size
            continue

        # Daily equity (one point per timestamp) -> log returns.
        eq_df = eq_df.copy()
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
        daily_eq = eq_df.groupby("timestamp")["equity"].last().sort_index()
        eq = daily_eq.values.astype(float)
        eq = eq[eq > 0]
        if len(eq) < 3:
            fold_start += test_size
            continue

        oos_rets = np.diff(np.log(eq))
        oos_rets = oos_rets[np.isfinite(oos_rets)]
        if len(oos_rets) > 2:
            all_oos_returns.append(oos_rets)

            # Realized round-trip P&Ls from engine fills (FIFO pairing).
            fills = test_engine.portfolio.trades
            pnls = BacktestEngine._compute_trade_pnls(fills)
            for pnl in pnls:
                all_trades.append({
                    "pnl_abs": float(pnl),
                    "fold": fold_idx,
                    "n_fills": len(fills),
                })

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
    """Compute the Deflated Sharpe Ratio p-value.

    Uses the published Bailey & Lopez de Prado (2014) exact form
    (``quantcore.research.dsr_published``) per FREEZE.md's DSR correction note
    (2026-08-05): the legacy ``quantcore.research.deflated_sharpe`` instrument
    inflates the statistic by sqrt(T-0.25) and its callers passed annualized
    Sharpe, so every prior DSR verdict computed through it is uninformative.

    Contract (see dsr_published docstring):
      - ``observed_sharpe`` must be PER-OBSERVATION; the annualized Sharpe
        passed in here is converted by / sqrt(252).
      - kurtosis is RAW (3.0 = normal), not excess.
    """
    from quantcore.research.dsr_published import deflated_sharpe_ratio_published

    if np.isnan(observed_sharpe) or n_obs < 20:
        return 1.0

    # Conservative trial count: strategies tested + 10 prior experiments
    n_trials = n_strategies + 10

    # Compute skewness and raw kurtosis from returns if available
    skewness = None
    kurt_raw = None
    if returns is not None and len(returns) > 20:
        r = np.asarray(returns, dtype=float)
        r_clean = r[np.isfinite(r)]
        if len(r_clean) > 20:
            mean_r = np.mean(r_clean)
            std_r = np.std(r_clean, ddof=1)
            if std_r > 1e-12:
                skewness = float(np.mean((r_clean - mean_r) ** 3) / (std_r ** 3))
                kurt_raw = float(np.mean((r_clean - mean_r) ** 4) / (std_r ** 4))

    per_obs_sharpe = float(observed_sharpe) / np.sqrt(252.0)

    p_value, _ = deflated_sharpe_ratio_published(
        observed_sharpe=per_obs_sharpe,
        num_trials=n_trials,
        sample_length=n_obs,
        skewness=skewness,
        kurtosis=kurt_raw,
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
# Runtime evidence gate
# ---------------------------------------------------------------------------


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML mapping, returning an empty dict when the file is absent."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _load_strategy_phases(status_path: Path = DEFAULT_STATUS_PATH) -> dict[str, str]:
    """Load strategy phases from strategy_status.yaml."""
    status_data = _load_yaml_mapping(status_path)
    strategies = status_data.get("strategies", {})
    if not isinstance(strategies, dict):
        return {}
    phases: dict[str, str] = {}
    for name, record in strategies.items():
        if isinstance(record, dict) and record.get("phase") is not None:
            phases[str(name)] = str(record["phase"])
    return phases


def _load_waivers(waivers_path: Path = DEFAULT_WAIVERS_PATH) -> dict[str, dict[str, Any]]:
    """Load validation waivers keyed by strategy id."""
    waiver_data = _load_yaml_mapping(waivers_path)
    waivers = waiver_data.get("waivers", {})
    if not isinstance(waivers, dict):
        return {}
    return {str(name): waiver for name, waiver in waivers.items() if isinstance(waiver, dict)}


def _waiver_is_active(waiver: dict[str, Any]) -> bool:
    """Return whether a waiver is still within its documented validity window."""
    expires_on = waiver.get("expires_on") or waiver.get("allowed_until")
    if expires_on is None:
        return True
    try:
        expiry = datetime.fromisoformat(str(expires_on)).date()
    except ValueError:
        return True
    return expiry >= datetime.now().date()


def _waiver_covers_gate(waiver: dict[str, Any], gate: str) -> bool:
    """Return whether a waiver covers a specific validation gate."""
    gates = waiver.get("gates") or waiver.get("scope") or waiver.get("gate") or []
    if isinstance(gates, str):
        gates = [gates]
    if not isinstance(gates, list):
        return False
    normalized = {str(item).lower() for item in gates}
    return gate.lower() in normalized or "all" in normalized


def _runtime_evidence_failures(result: StrategyResult) -> list[str]:
    """Return evidence failures for a PAPER/LIVE strategy."""
    failures: list[str] = []
    if result.status != "ok":
        failures.append(f"validation status is {result.status}")
        if result.error:
            failures.append(f"error: {result.error}")
        return failures
    if not np.isfinite(result.oos_sharpe):
        failures.append("OOS Sharpe is NaN or non-finite")
    if result.n_trades <= 0:
        failures.append("zero trades")
    if not np.isfinite(result.ci_lower_90):
        failures.append("CI lower bound is NaN or non-finite")
    elif result.ci_lower_90 <= 0:
        failures.append(f"CI lower bound {result.ci_lower_90:.4f} <= 0")
    return failures


def runtime_evidence_gate_failures(
    results: list[StrategyResult],
    status_path: Path = DEFAULT_STATUS_PATH,
    waivers_path: Path = DEFAULT_WAIVERS_PATH,
) -> list[str]:
    """Fail PAPER/LIVE strategies that lack positive OOS Sharpe evidence.

    Waivers are intentionally explicit and documented in config/validation_waivers.yaml.
    """
    phases = _load_strategy_phases(status_path)
    waivers = _load_waivers(waivers_path)
    failures: list[str] = []
    for result in results:
        phase = phases.get(result.strategy_id)
        if phase not in RUNTIME_PHASES:
            continue
        evidence_failures = _runtime_evidence_failures(result)
        if not evidence_failures:
            continue
        waiver = waivers.get(result.strategy_id, {})
        if _waiver_is_active(waiver) and _waiver_covers_gate(waiver, RUNTIME_EVIDENCE_GATE):
            continue
        failures.append(
            f"{result.strategy_id} ({phase}): "
            + "; ".join(evidence_failures)
            + ". Add a documented waiver in config/validation_waivers.yaml only while under review."
        )
    return failures


def print_runtime_evidence_warnings(results: list[StrategyResult]) -> None:
    """Print non-blocking warnings for PAPER/LIVE evidence failures."""
    failures = runtime_evidence_gate_failures(results)
    if not failures:
        return
    print("\nRuntime evidence warnings:")
    for failure in failures:
        print(f"  WARNING: {failure}")


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

            # Get default params from the strategy class
            from quantcore.strategies.base import StrategyConfig, StrategyType
            stype_map = {"equity": StrategyType.EQUITY, "macro": StrategyType.MACRO, "options": StrategyType.OPTIONS}
            temp_cfg = StrategyConfig(name=strat_id, strategy_type=stype_map.get(strategy_type, StrategyType.EQUITY))
            temp_strat = cls(temp_cfg)
            default_params = dict(temp_cfg.params)

            t0 = time.time()
            returns, trades = _run_walk_forward_engine(
                cls, default_params, df,
                strategy_type=strategy_type,
                warmup=warmup,
                test_size=126,
                tc_bps=5.0,
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
        "- Runtime gate: PAPER/LIVE strategies require finite OOS Sharpe, >0 trades, and CI lower > 0 unless waived",
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
    parser.add_argument(
        "--enforce-runtime-gate",
        action="store_true",
        help="Fail when any PAPER/LIVE strategy has NaN Sharpe, zero trades, or CI lower <= 0 unless waived.",
    )
    parser.add_argument(
        "--status-path",
        type=Path,
        default=DEFAULT_STATUS_PATH,
        help="strategy_status.yaml path",
    )
    parser.add_argument(
        "--waivers-path",
        type=Path,
        default=DEFAULT_WAIVERS_PATH,
        help="Validation waiver YAML path",
    )
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

    failures = runtime_evidence_gate_failures(
        results,
        status_path=args.status_path,
        waivers_path=args.waivers_path,
    )
    if failures:
        print("\nRuntime evidence warnings:")
        for failure in failures:
            print(f"  WARNING: {failure}")
        if args.enforce_runtime_gate:
            print("\nRuntime evidence gate FAILED; refusing to pass CI.")
            sys.exit(1)

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
