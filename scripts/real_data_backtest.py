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
from quantcore.risk.metrics import sharpe_ratio, sortino_ratio, drawdown_stats, bootstrap_ci
# Import strategy modules to trigger @register decorators
import quantcore.strategies.macro.cta_trend  # noqa: F401
import quantcore.strategies.macro.risk_parity  # noqa: F401
import quantcore.strategies.macro.factor_residual  # noqa: F401
import quantcore.strategies.macro.factor_residual_v3  # noqa: F401
import quantcore.strategies.macro.cross_sectional_factor_neutral  # noqa: F401
# DISABLED: macro_cross_sectional_momentum_v2 (prospective trial #1, REJECTED)
# import quantcore.strategies.macro.cross_sectional_momentum_v2  # noqa: F401

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
    "macro_factor_residual": "expanded_15",
    "macro_cross_sectional_factor_neutral": "ETF_EXPANDED",
    # DISABLED per prospective_trial #1 (REJECTED):
    # "macro_cross_sectional_momentum_v2": "expanded_15",
    "macro_factor_residual_v3": "expanded_15",
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
        "alignment_filter": True,
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
    "macro_factor_residual": {
        "rebalance_freq": 5,
        "reg_window": 252,
        "ridge_alpha": 1e-6,
        "resid_mom_window": 63,
        "resid_rev_weight": 0.3,
        "vol_lookback": 20,
        "min_weight": 0.01,
    },
    # DISABLED per prospective_trial #1 (REJECTED):
    # "macro_cross_sectional_momentum_v2" params removed
    "macro_factor_residual_v3": {
        "rebalance_freq": 5,
        "reg_window": 252,
        "ridge_alpha": 1e-6,
        "residual_window": 63,
        "vol_lookback": 20,
        "min_weight": 0.01,
    },
    "macro_cross_sectional_factor_neutral": {
        "rebalance_freq": 21,
        "reg_lookback": 252,
        "pred_window": 20,
        "ridge_alpha": 1.0,
        "max_position": 0.03,
        "target_vol": 0.15,
        "vol_lookback": 20,
        "zscore_cutoff": 1.5,
        "percentile_top": 0.30,
        "factor_symbols": "SPY,TLT,GLD,DBC,EFA,VNQ,LQD,UUP",
    },
}

# ---------------------------------------------------------------------------
# Pass criteria (min OOS Sharpe per regime)
# ---------------------------------------------------------------------------

PASS_CRITERIA: dict[tuple[str, str], float] = {
    # Full period (walk-forward)
    ("macro_cta_trend", "full"): 0.50,
    ("macro_risk_parity", "full"): 0.30,
    ("macro_factor_residual", "full"): 0.50,
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
    ("macro_factor_residual", "gfc"): 0.30,
    ("macro_factor_residual", "covid"): 0.30,
    ("macro_factor_residual", "rate_hikes"): 0.30,
    ("macro_factor_residual", "q4_vol"): -0.10,
    ("macro_factor_residual", "bull"): 0.30,
    # CSM v2: disabled per prospective_trial #1 (REJECTED)
    # ("macro_cross_sectional_momentum_v2", "full"): 0.30,
    # Factor residual v3
    ("macro_factor_residual_v3", "full"): 0.50,
    ("macro_factor_residual_v3", "gfc"): 0.30,
    ("macro_factor_residual_v3", "covid"): 0.30,
    ("macro_factor_residual_v3", "rate_hikes"): 0.30,
    ("macro_factor_residual_v3", "q4_vol"): -0.10,
    ("macro_factor_residual_v3", "bull"): 0.30,
    # New Phase 4 cross-sectional factor-neutral stat-arb
    ("macro_cross_sectional_factor_neutral", "full"): 0.30,
    ("macro_cross_sectional_factor_neutral", "gfc"): 0.00,
    ("macro_cross_sectional_factor_neutral", "covid"): 0.00,
    ("macro_cross_sectional_factor_neutral", "rate_hikes"): -0.10,
    ("macro_cross_sectional_factor_neutral", "q4_vol"): -0.20,
    ("macro_cross_sectional_factor_neutral", "bull"): 0.20,
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
    ("macro_factor_residual", "full"): -0.10,
    ("macro_factor_residual", "gfc"): -0.50,
    ("macro_factor_residual", "covid"): -0.20,
    ("macro_factor_residual", "rate_hikes"): -0.20,
    ("macro_factor_residual", "bull"): -0.10,
    ("macro_factor_residual_v3", "full"): -0.10,
    ("macro_factor_residual_v3", "gfc"): -0.50,
    ("macro_factor_residual_v3", "covid"): -0.20,
    ("macro_factor_residual_v3", "rate_hikes"): -0.20,
    ("macro_factor_residual_v3", "bull"): -0.10,
    # New Phase 4 cross-sectional factor-neutral stat-arb
    ("macro_cross_sectional_factor_neutral", "full"): -0.10,
    ("macro_cross_sectional_factor_neutral", "gfc"): -0.50,
    ("macro_cross_sectional_factor_neutral", "covid"): -0.20,
    ("macro_cross_sectional_factor_neutral", "rate_hikes"): -0.20,
    ("macro_cross_sectional_factor_neutral", "q4_vol"): -0.30,
    ("macro_cross_sectional_factor_neutral", "bull"): -0.10,
}

# ---------------------------------------------------------------------------
# DataFrame -> numpy data dict conversion
# ---------------------------------------------------------------------------


def _causal_ffill_2d(arr: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Bounded causal forward-fill along the time axis (axis=0).

    Each bar only uses information available at the same or earlier
    timestamps (no look-ahead). A bounded gap prevents stale prices
    from propagating across long periods (e.g. delistings, halted
    trading).

    Args:
        arr: 2D array with shape (time, assets).
        max_gap: Maximum number of consecutive NaN bars to fill. NaN
            values beyond this limit are left as NaN.

    Returns:
        (filled_arr, fill_mask) where fill_mask is True for every
        value that was imputed by the forward-fill.
    """
    df = pd.DataFrame(arr)
    filled = df.ffill(limit=max_gap)
    # mask of values that were NaN in the original but are now valid
    fill_mask = df.isna().values & (~filled.isna().values)

    # --- sanity: every true in fill_mask must be behind a non-NaN ---
    for col in range(fill_mask.shape[1]):
        if not fill_mask[:, col].any():
            continue
        # find every row that was filled
        filled_rows = np.where(fill_mask[:, col])[0]
        for row in filled_rows:
            # Before `row`, there must be at least one non-NaN within max_gap+1 rows
            search_start = max(0, row - max_gap)
            segment = arr[search_start:row, col]
            if np.all(np.isnan(segment)):
                raise ValueError(
                    f"_causal_ffill_2d non-causal detected: col {col}, "
                    f"row {row} was filled with no prior non-NaN within max_gap={max_gap}. "
                    f"Impossible under forward-fill."
                )

    return filled.values, fill_mask


def _verify_no_future_in_fill(
    original: np.ndarray,
    filled: np.ndarray,
    mask: np.ndarray,
) -> None:
    """Verify that every filled cell never borrows a value from a later row."""
    for col in range(mask.shape[1]):
        col_mask = mask[:, col]
        if not col_mask.any():
            continue
        filled_rows = np.where(col_mask)[0]
        for row in filled_rows:
            # The value that appears at (row, col) must already appear
            # at some prior row <= row that was non-NaN in the original.
            val = filled[row, col]
            if row == 0:
                # nothing before; filled[0,col] should equal itself if no prev value
                continue
            prior = original[: row + 1, col]
            non_nan_prior = prior[~np.isnan(prior)]
            if len(non_nan_prior) == 0:
                # still NaN -- this would mean the fill didn't work
                if not np.isnan(val):
                    raise ValueError(
                        f"Non-NaN filled value {val} at col={col} row={row} "
                        f"with no prior non-NaN. This is impossible under causal ffill."
                    )
                continue
            if not np.isclose(val, non_nan_prior[-1]):
                raise ValueError(
                    f"Filled value {val} at col={col} row={row} does not match "
                    f"the prior non-NaN value {non_nan_prior[-1]}. Potential future leak."
                )


def _sanity_check_prices(data: dict) -> None:
    """Validate that no zero or negative prices remain after filling.

    Raises:
        ValueError: If any close/open/high/low are <= 0.
    """
    for field in ("close", "open", "high", "low"):
        arr = data.get(field)
        if arr is None:
            continue
        if np.any(arr <= 0):
            bad_idx = np.argwhere(arr <= 0)[0]
            raise ValueError(
                f"{field} contains non-positive price at bar {bad_idx[0]}, "
                f"asset {bad_idx[1]} (value={arr[tuple(bad_idx)]}). "
                f"This usually means a symbol has no valid price history and "
                f"the entire column was zero-filled."
            )


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
            df_filled, fill_mask = _causal_ffill_2d(arr, max_gap=5)
            # --- enforce no future leak ---
            _verify_no_future_in_fill(arr, df_filled, fill_mask)
            fill_pct = fill_mask.sum() / fill_mask.size
            if fill_pct > 0.005:
                log.warning(
                    "_df_to_data: %s filled %.2f%% of cells (max_gap=5). "
                    "High fill rate suggests data quality issues.", k, fill_pct * 100
                )
            # Zero-fill any leading NaN (e.g. before a symbol's first valid bar)
            data[k] = np.nan_to_num(df_filled, nan=0.0)

    _sanity_check_prices(data)
    return data


# ---------------------------------------------------------------------------
# Signal-based return computation
# ---------------------------------------------------------------------------


def _holding_days(entry_ts, exit_ts) -> float:
    """Calculate holding period in days between two timestamps."""
    if entry_ts is None or exit_ts is None:
        return 0.0
    try:
        from datetime import timedelta
        e = entry_ts if hasattr(entry_ts, 'timestamp') else entry_ts
        x = exit_ts if hasattr(exit_ts, 'timestamp') else exit_ts
        # Convert to pandas Timestamp for consistent handling
        import pandas as pd
        e = pd.Timestamp(e)
        x = pd.Timestamp(x)
        return (x - e).total_seconds() / 86400.0  # 24*3600
    except Exception:
        return 0.0


def _compute_signal_returns(
    strat: Any,
    data: dict,
    start: int,
    end: int,
    tc_bps: float = 0.0,
    timestamps: list = None,
    track_trades: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """Compute bar-level signal-weighted returns over data[start:end).

    Supports transaction costs via tc_bps (turnover cost in basis points).
    When tc_bps > 0, tracks per-symbol position weights and applies a
    cost proportional to absolute weight changes between consecutive bars.

    If track_trades is True, also returns a list of trade dicts with:
    - timestamp_entry, timestamp_exit, position_direction, entry_price, exit_price,
      pnl_abs, pnl_pct, holding_days, signal_type

    Returns:
        bar_returns if track_trades=False (backward compatible)
        (bar_returns, trades_list) if track_trades=True
    """
    LOSS_THRESHOLD = -0.0075  # -0.75% unrealized loss trigers exit (Phase 1.7)
    trades = []
    if track_trades:
        # position tracker: {symbol: {direction, entry_ts, entry_price, signal_type}}
        positions = {}
        # Instrumentation (Phase 1.7)
        early_exit_count = 0
        total_loss_pre_exit = 0.0
        loss_exit_pnl_list = []

    if end - start < 2:
        return np.array([], dtype=float), trades

    close = data["close"]
    symbols = data["symbols"]
    n_assets = len(symbols)

    if n_assets == 0:
        return np.array([], dtype=float), trades

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

        # Set timestamp for diagnostic tracking
        if timestamps is not None and bar_i < len(timestamps):
            ts = timestamps[bar_i]
            if hasattr(strat, 'set_timestamp'):
                strat.set_timestamp(ts)
            bar_data["timestamp"] = ts

        try:
            signals = strat.on_data(bar_data) or []
        except Exception as e:
            import traceback
            log.error(" strategy signal error on_data: %s", e)
            traceback.print_exc()
            signals = []

        weighted_ret = 0.0
        total_weight = 0.0
        current_weights: dict[str, float] = {}
        current_signals: dict[str, dict] = {}  # symbol -> signal metadata

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
        elif carried_weights:
            # On non-rebalance days, use carried weights from last signal
            for sym, weight in carried_weights.items():
                try:
                    asset_idx = symbols.index(sym)
                except (ValueError, IndexError):
                    continue
                weighted_ret += weight * float(asset_rets[idx, asset_idx])
                total_weight += abs(weight)
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

        # Track trades if enabled
        if track_trades:
            for sym in list(positions.keys()):
                pos = positions[sym]
                exit_ts = timestamps[bar_i] if timestamps else None
                exit_price = close[bar_i, symbols.index(sym)] if bar_i < close.shape[0] else None
                
                # Phase 1.7: Loss-based exit acceleration
                if exit_price is not None and pos["entry_price"]:
                    current_pnl_pct = (exit_price - pos["entry_price"]) / pos["entry_price"] * pos["direction"]
                    # Cut losing trades fast (Step 1)
                    if current_pnl_pct < LOSS_THRESHOLD:
                        # Don't cut if profitable (Step 3)
                        if current_pnl_pct > 0:
                            pass  # Profitable - let it run
                        else:
                            # Loss-based exit: close position
                            early_exit_count += 1
                            total_loss_pre_exit += current_pnl_pct
                            loss_exit_pnl_list.append(current_pnl_pct)
                            holding_days = _holding_days(pos["entry_ts"], exit_ts)
                            pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"]
                            trades.append({
                                "timestamp_entry": pos["entry_ts"],
                                "timestamp_exit": exit_ts,
                                "position_direction": pos["direction"],
                                "entry_price": pos["entry_price"],
                                "exit_price": exit_price,
                                "pnl_abs": pnl_abs,
                                "pnl_pct": current_pnl_pct,
                                "holding_days": holding_days,
                                "signal_type": pos["signal_type"],
                                "symbol": sym,
                                "exit_reason": "loss_threshold"
                            })
                            positions.pop(sym)
                            continue
                
                # Normal exit logic (signal-driven)
                if sym not in current_signals or current_signals[sym]["direction"] != pos["direction"]:
                    # Close position
                    holding_days = _holding_days(pos["entry_ts"], exit_ts)
                    pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
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
                        "signal_type": pos["signal_type"],
                        "symbol": sym,
                        "exit_reason": "signal"
                    })
                    positions.pop(sym)
            
            # Open new positions
            for sym in current_signals:
                if sym not in positions:
                    sig = current_signals[sym]
                    entry_price = close[bar_i, symbols.index(sym)] if bar_i < close.shape[0] else None
                    entry_ts = timestamps[bar_i] if timestamps else None
                    confirms = sig["metadata"].get("confirms", 0)
                    signal_type = "fast_only" if confirms == 0 else "confirmed"
                    positions[sym] = {
                        "direction": sig["direction"],
                        "entry_ts": entry_ts,
                        "entry_price": entry_price,
                        "signal_type": signal_type,
                    }
            for sym in current_signals:
                if sym not in positions:
                    # New position opened
                    sig = current_signals[sym]
                    entry_price = close[bar_i, symbols.index(sym)] if bar_i < close.shape[0] else None
                    entry_ts = timestamps[bar_i] if timestamps else None
                    confirms = sig["metadata"].get("confirms", 0)
                    signal_type = "fast_only" if confirms == 0 else "confirmed"
                    positions[sym] = {
                        "direction": sig["direction"],
                        "entry_ts": entry_ts,
                        "entry_price": entry_price,
                        "signal_type": signal_type,
                    }

        # Update carried weights only when new signals arrive
        if current_weights:
            carried_weights = dict(current_weights)

    # Close any remaining positions at end of period
    if track_trades:
        for sym in list(positions.keys()):
            pos = positions.pop(sym)
            exit_ts = timestamps[end - 1] if timestamps else None
            exit_price = close[end - 1, symbols.index(sym)] if end - 1 < close.shape[0] else None
            holding_days = _holding_days(pos["entry_ts"], exit_ts)
            pnl_abs = (exit_price - pos["entry_price"]) * pos["direction"] if exit_price and pos["entry_price"] else 0.0
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
                "signal_type": pos["signal_type"],
                "symbol": sym,
            })

    # Instrumentation output (Phase 1.7)
    if track_trades:
        avg_loss_exit = np.mean(loss_exit_pnl_list) if loss_exit_pnl_list else 0.0
        print(f"[INSTRUMENTATION] early_exit_count={early_exit_count}, "
              f"avg_loss_pre_exit={avg_loss_exit:.4f}, "
              f"total_loss_exit={total_loss_pre_exit:.4f}")

    return (bar_rets, trades) if track_trades else bar_rets



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

    # Bootstrap 90% CI for Sharpe ratio
    if len(oos_returns) >= 30:
        ci_lo, ci_hi = bootstrap_ci(oos_returns, sharpe_ratio, n_boot=1000, ci=0.90)
        result["sharpe_ci_90"] = (ci_lo, ci_hi)
    else:
        result["sharpe_ci_90"] = (None, None)

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

    # Bootstrap 90% CI for Sharpe ratio
    if len(oos_rets) >= 30:
        ci_lo, ci_hi = bootstrap_ci(oos_rets, sharpe_ratio, n_boot=1000, ci=0.90)
        result["sharpe_ci_90"] = (ci_lo, ci_hi)
    else:
        result["sharpe_ci_90"] = (None, None)
    result["oos_returns"] = oos_rets.tolist()

    return result


# ---------------------------------------------------------------------------
# V2 BacktestEngine-based validation (full execution stack)
# ---------------------------------------------------------------------------
# These functions replace the _compute_signal_returns() path with
# BacktestEngine.run(), which enforces look-ahead bias prevention,
# transaction costs, position sizing, risk limits, drawdown tracking,
# and the FillSimulator — the honest measurement infrastructure.


def run_full_period_backtest_v2(
    strategy_id: str,
    df: pd.DataFrame,
    tc_bps: float = 5.0,
) -> dict[str, Any]:
    """Run anchored walk-forward on the full period using BacktestEngine.run().

    Uses the full execution stack (FillSimulator, risk limits, position
    sizing, drawdown tracking) instead of the signal-only _compute_signal_returns.
    """
    from quantcore.backtest.engine import BacktestEngine

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

    symbols = sorted(df["symbol"].unique())
    if len(symbols) == 0:
        result["error"] = "No symbols in data"
        return result

    timestamps = sorted(df["timestamp"].unique())
    n_bars = len(timestamps)

    if n_bars < 100:
        result["error"] = f"Insufficient data: {n_bars} bars"
        return result

    initial_train = min(200, n_bars // 3)
    test_size = min(50, max(20, n_bars // 10))
    embargo = max(1, int(0.02 * test_size))

    if initial_train + test_size + embargo >= n_bars:
        initial_train = max(50, n_bars // 4)
        test_size = max(10, n_bars // 8)

    if initial_train + test_size >= n_bars:
        result["error"] = f"Insufficient bars ({n_bars}) for walk-forward"
        return result

    try:
        folds = anchored_walk_forward(
            n_bars,
            initial_train=initial_train,
            test_size=test_size,
        )
    except Exception as e:
        result["error"] = f"Fold generation failed: {e}"
        return result

    fold_results: list[dict] = []
    all_oos_rets: list[np.ndarray] = []

    for fold in folds:
        if fold.train_end > n_bars or fold.test_end > n_bars:
            continue
        if fold.test_end - fold.test_start < 5:
            continue

        try:
            strat = _setup_strategy(strategy_id)

            # In-sample: build strategy state via BacktestEngine
            train_ts = timestamps[fold.train_start:fold.train_end]
            train_data = df[df["timestamp"].isin(train_ts)].copy()
            if len(train_data) > 5:
                engine = BacktestEngine(
                    strategy=strat,
                    initial_cash=1_000_000,
                    slippage_bps=tc_bps,
                    commission_per_share=0.005,
                )
                engine.run(train_data)

            # Out-of-sample: collect returns via BacktestEngine
            test_ts = timestamps[fold.test_start:fold.test_end]
            test_data = df[df["timestamp"].isin(test_ts)].copy()
            if len(test_data) < 5:
                continue

            engine = BacktestEngine(
                strategy=strat,
                initial_cash=1_000_000,
                slippage_bps=tc_bps,
                commission_per_share=0.005,
            )
            eq_df = engine.run(test_data)

            if len(eq_df) < 3:
                continue

            # Compute true *daily* returns: take the last equity value
            # for each calendar timestamp (one per day after all symbols
            # processed), then compute log-diffs. This avoids treating
            # intra-day symbol-by-symbol equity updates as trades.
            eq_df = eq_df.copy()
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
            daily_eq = eq_df.groupby("timestamp")["equity"].last().sort_index()
            if len(daily_eq) < 3:
                continue

            eq = daily_eq.values.astype(float)
            eq = eq[eq > 0]
            if len(eq) < 3:
                continue

            oos_rets = np.diff(np.log(eq))
            oos_rets = oos_rets[np.isfinite(oos_rets)]

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
            log.debug("Fold (%d, %d) failed in v2: %s", fold.test_start, fold.test_end, e)
            continue

    if not all_oos_rets:
        result["error"] = "No folds completed successfully"
        return result

    oos_returns = np.concatenate(all_oos_rets)

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

    if len(oos_returns) >= 30:
        ci_lo, ci_hi = bootstrap_ci(oos_returns, sharpe_ratio, n_boot=1000, ci=0.90)
        result["sharpe_ci_90"] = (ci_lo, ci_hi)
    else:
        result["sharpe_ci_90"] = (None, None)

    return result


def run_regime_backtest_v2(
    strategy_id: str,
    df: pd.DataFrame,
    warmup_end_idx: int,
    tc_bps: float = 5.0,
) -> dict[str, Any]:
    """Run single train/test split using BacktestEngine.run() full execution stack.

    warmup_end_idx is the bar index where warmup ends and OOS evaluation begins.
    Bars [0, warmup_end_idx) warm up the strategy via BacktestEngine.
    Bars [warmup_end_idx, end) generate OOS returns.
    """
    from quantcore.backtest.engine import BacktestEngine

    result: dict[str, Any] = {
        "oos_sharpe": None,
        "oos_sortino": None,
        "max_dd": 0.0,
        "win_rate": 0.0,
        "n_oos_bars": 0,
        "error": None,
    }

    symbols = sorted(df["symbol"].unique())
    if len(symbols) == 0:
        result["error"] = "No symbols in data"
        return result

    timestamps = sorted(df["timestamp"].unique())
    n_bars = len(timestamps)

    if n_bars < warmup_end_idx + 3:
        result["error"] = f"Insufficient data: {n_bars} bars, warmup end {warmup_end_idx}"
        return result

    try:
        strat = _setup_strategy(strategy_id)

        # Warmup: build strategy state via BacktestEngine
        if warmup_end_idx > 10:
            warmup_ts = timestamps[:warmup_end_idx]
            warmup_data = df[df["timestamp"].isin(warmup_ts)].copy()
            if len(warmup_data) > 5:
                engine = BacktestEngine(
                    strategy=strat,
                    initial_cash=1_000_000,
                    slippage_bps=tc_bps,
                    commission_per_share=0.005,
                )
                engine.run(warmup_data)

        # OOS: collect returns via BacktestEngine
        test_ts = timestamps[warmup_end_idx:n_bars]
        test_data = df[df["timestamp"].isin(test_ts)].copy()
        if len(test_data) < 5:
            result["error"] = "Insufficient test data"
            return result

        engine = BacktestEngine(
            strategy=strat,
            initial_cash=1_000_000,
            slippage_bps=tc_bps,
            commission_per_share=0.005,
        )
        eq_df = engine.run(test_data)

        if len(eq_df) < 3:
            result["error"] = "BacktestEngine produced insufficient rows"
            return result

        # Aggregate to daily equity (one data point per calendar day)
        eq_df = eq_df.copy()
        eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
        daily_eq = eq_df.groupby("timestamp")["equity"].last().sort_index()
        if len(daily_eq) < 3:
            result["error"] = "BacktestEngine produced insufficient daily rows"
            return result

        eq = daily_eq.values.astype(float)
        eq = eq[eq > 0]
        if len(eq) < 3:
            result["error"] = "Equity curve has insufficient positive values"
            return result

        oos_rets = np.diff(np.log(eq))
        oos_rets = oos_rets[np.isfinite(oos_rets)]

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

    if len(oos_rets) >= 30:
        ci_lo, ci_hi = bootstrap_ci(oos_rets, sharpe_ratio, n_boot=1000, ci=0.90)
        result["sharpe_ci_90"] = (ci_lo, ci_hi)
    else:
        result["sharpe_ci_90"] = (None, None)
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
    use_v2: bool = True,
) -> list[dict[str, Any]]:
    """Run Phase A validation across all strategy/period combinations.

    use_v2 (default True): Use BacktestEngine.run() full execution stack
    with look-ahead bias prevention, FillSimulator, and transaction costs.
    Falls back to _compute_signal_returns() when BacktestEngine is unavailable.

    Returns list of result dicts (one per strategy-period pair).
    """
    if strategies is None:
        strategies = list(STRATEGY_PARAMS.keys())

    # Resolve v2 availability
    if use_v2:
        try:
            from quantcore.backtest.engine import BacktestEngine  # noqa: F401
        except ImportError:
            log.warning("BacktestEngine not available, falling back to _compute_signal_returns")
            use_v2 = False

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

            if use_v2:
                result = run_full_period_backtest_v2(strategy_id, df, tc_bps=5.0)
            else:
                result = run_full_period_backtest(strategy_id, df, tc_bps=5.0)
            result["strategy"] = strategy_id
            result["period"] = "full"
            result["period_label"] = "Full Period (2018-2024)"
            result["engine"] = "BacktestEngine" if use_v2 else "signal_only"

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

            if use_v2:
                result = run_regime_backtest_v2(strategy_id, df, warmup_end_idx, tc_bps=5.0)
            else:
                result = run_regime_backtest(strategy_id, df, warmup_end_idx, tc_bps=5.0)
            result["strategy"] = strategy_id
            result["period"] = regime_id
            result["period_label"] = regime_info["label"]
            result["engine"] = "BacktestEngine" if use_v2 else "signal_only"

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
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy _compute_signal_returns path instead of BacktestEngine")

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
        use_v2=not args.legacy,
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
