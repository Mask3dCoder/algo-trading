#!/usr/bin/env python
"""Phase 3.1: Honest walk-forward re-validation using strategy-native backtests.

Uses the PROVEN backtest approach for each strategy type:
  - Options strategies: standalone BSM simulation (delta_neutral_realistic.py approach)
  - Macro strategies: signal-based with causal slices (run_backtest_focused.py approach)

All strategies validated with:
  - Anchored expanding window (504-day train, 126-day test, 1% embargo)
  - 2018-2024 data period
  - DSR p-value (trials=20)
  - Bootstrap CI(90%)

Triage:
  KEEP    — CI_lower > 0 AND DSR p-value < 0.1
  MONITOR — CI_lower > 0 but DSR p-value >= 0.1
  DISABLE — everything else
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from quantcore.backtest.walk_forward import anchored_walk_forward
from quantcore.research.deflated_sharpe import deflated_sharpe_ratio
from quantcore.strategies.base import StrategyConfig

# Register strategies
import quantcore.strategies.macro.cross_sectional_momentum  # noqa: F401
import quantcore.strategies.macro.risk_parity  # noqa: F401
from quantcore.strategies.registry import get as get_strategy_class

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPORTS_DIR = REPO_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TRAIN_SIZE = 504
TEST_SIZE = 126
EMBARGO_PCT = 0.01
NUM_TRIALS = 20
START_DATE = "2018-01-01"
END_DATE = "2024-12-31"
INITIAL_CASH = 1_000_000.0
TC_BPS = 10.0  # 10bps round-trip


# ---------------------------------------------------------------------------
# BSM helpers (from alpha/delta_neutral_realistic.py)
# ---------------------------------------------------------------------------

def _bsm_put(S, K, T, r, sigma):
    if T < 1e-10 or sigma < 1e-10:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))

def _bsm_call(S, K, T, r, sigma):
    if T < 1e-10 or sigma < 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))

def _bsm_delta(S, K, T, r, sigma, is_call):
    if T < 1e-10 or sigma < 1e-10:
        return 1.0 if (is_call and S > K) or (not is_call and S < K) else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1)) if is_call else float(norm.cdf(d1) - 1.0)

def _bsm_gamma(S, K, T, r, sigma):
    if T < 1e-10 or sigma < 1e-10 or S < 1e-10:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    """Load SPY + VIX data for options strategies, and multi-asset for macro."""
    import yfinance as yf

    # SPY + VIX for options
    spy = yf.download("SPY", start=START_DATE, end=END_DATE, progress=False)
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE, progress=False)

    if isinstance(spy.columns, pd.MultiIndex):
        spy_close = spy["Close"].iloc[:, 0] if len(spy["Close"].shape) > 1 else spy["Close"]
    else:
        spy_close = spy["Close"]
    if isinstance(vix.columns, pd.MultiIndex):
        vix_close = vix["Close"].iloc[:, 0] if len(vix["Close"].shape) > 1 else vix["Close"]
    else:
        vix_close = vix["Close"]

    spy_close = spy_close.dropna()
    vix_close = vix_close.dropna()

    # Multi-asset for macro strategies
    macro_symbols = ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"]
    from quantcore.data.yfinance_loader import load_combined
    macro_df = load_combined(macro_symbols, start=START_DATE, end=END_DATE)

    return spy_close, vix_close, macro_df


def df_to_data(df: pd.DataFrame) -> dict:
    symbols = sorted(df["symbol"].unique())
    pivots = {}
    for f in ["open", "high", "low", "close", "volume"]:
        if f in df.columns:
            pivoted = df.pivot_table(index="timestamp", columns="symbol", values=f)
            pivoted = pivoted.reindex(columns=symbols)
            pivots[f] = pivoted.values
    close_df = pd.DataFrame(pivots["close"])
    close_filled = close_df.ffill(limit=5).fillna(0).values
    return {
        "close": close_filled,
        "open": pivots.get("open", close_filled),
        "high": pivots.get("high", close_filled),
        "low": pivots.get("low", close_filled),
        "volume": pivots.get("volume", np.ones_like(close_filled)),
        "symbols": symbols,
    }


# ---------------------------------------------------------------------------
# Options: Delta Neutral VRP backtest (standalone BSM simulation)
# ---------------------------------------------------------------------------

def backtest_delta_neutral_vrp(
    spy_close: pd.Series, vix_close: pd.Series,
    start_idx: int, end_idx: int,
    params: dict | None = None,
) -> np.ndarray:
    """Walk-forward backtest of Delta Neutral VRP strategy.

    Returns daily returns array for the test window [start_idx:end_idx].
    """
    params = params or {}
    ivr_entry = params.get("ivr_entry", 50)
    ivr_exit = params.get("ivr_exit", 25)
    dte_min = params.get("dte_min", 30)
    dte_max = params.get("dte_max", 60)
    delta_target = params.get("delta_target", 0.3)
    risk_per_trade = params.get("risk_per_trade", 0.01)

    # Align data
    common_idx = spy_close.index.intersection(vix_close.index)
    spy = spy_close.loc[common_idx]
    vix = vix_close.loc[common_idx]
    n = len(spy)

    if end_idx > n:
        end_idx = n
    if start_idx < 60:
        start_idx = 60

    daily_returns = np.zeros(end_idx - start_idx)
    equity = INITIAL_CASH
    positions = []  # list of {strike, entry_credit, contracts, entry_date, expiry, entry_spot, entry_iv}

    for i in range(start_idx, end_idx):
        spot = float(spy.iloc[i])
        vix_val = float(vix.iloc[i])
        iv = vix_val / 100.0

        # IV Rank calculation (rolling 252-day)
        lookback = min(i, 252)
        vix_window = vix.iloc[i-lookback:i+1].values
        iv_min, iv_max = np.nanmin(vix_window), np.nanmax(vix_window)
        ivr = ((vix_val - iv_min) / (iv_max - iv_min) * 100) if (iv_max - iv_min) > 0.1 else 55.0

        # Mark-to-market existing positions
        daily_pnl = 0.0
        remaining_positions = []
        for pos in positions:
            T = (pos["expiry"] - spy.index[i]).days / 365.0
            if T <= 0:
                # Expire: close at intrinsic
                put_val = max(pos["strike"] - spot, 0.0)
                close_cost = put_val * pos["contracts"] * 100
                daily_pnl += pos["entry_credit"] * pos["contracts"] * 100 - close_cost
            else:
                # Mark-to-market with bid/ask haircut
                mid_put = _bsm_put(spot, pos["strike"], T, 0.04, iv)
                # 10% bid-ask spread on options
                buy_back = mid_put * 1.10
                mtm_liability = buy_back * pos["contracts"] * 100
                entry_value = pos["entry_credit"] * pos["contracts"] * 100
                # P&L = entry credit - current liability
                daily_pnl += 0  # We track total P&L at close
                remaining_positions.append(pos)

                # Close at 50% profit
                if mid_put < pos["entry_credit"] * 0.5:
                    close_cost = mid_put * 1.10 * pos["contracts"] * 100
                    daily_pnl += pos["entry_credit"] * pos["contracts"] * 100 - close_cost
                    remaining_positions = [p for p in remaining_positions if p is not pos]

        positions = remaining_positions

        # Entry: sell puts when IVR > threshold and VIX > 15
        if ivr >= ivr_entry and vix_val > 15.0 and len(positions) < 3:
            dte = (dte_min + dte_max) // 2
            T = dte / 365.0
            strike = spot * 0.95  # 5% OTM
            mid_put = _bsm_put(spot, strike, T, 0.04, iv)
            # Sell at 95% of mid (bid-ask haircut)
            credit = mid_put * 0.95
            # Size: risk 1% of equity
            max_loss = strike * 100  # per contract
            contracts = max(1, int(equity * risk_per_trade / max_loss))
            # Cap at reasonable size
            contracts = min(contracts, 10)

            positions.append({
                "strike": strike,
                "entry_credit": credit,
                "contracts": contracts,
                "entry_date": spy.index[i],
                "expiry": spy.index[i] + pd.Timedelta(days=dte),
                "entry_spot": spot,
                "entry_iv": iv,
            })
            # Upfront credit received
            daily_pnl += credit * contracts * 100

        # Close positions if IVR drops below exit threshold
        if ivr < ivr_exit:
            for pos in list(positions):
                T = max((pos["expiry"] - spy.index[i]).days / 365.0, 0.001)
                mid_put = _bsm_put(spot, pos["strike"], T, 0.04, iv)
                buy_back = mid_put * 1.10
                close_cost = buy_back * pos["contracts"] * 100
                daily_pnl += pos["entry_credit"] * pos["contracts"] * 100 - close_cost
            positions = []

        equity += daily_pnl
        ret_idx = i - start_idx
        if ret_idx < len(daily_returns):
            daily_returns[ret_idx] = daily_pnl / max(equity - daily_pnl, 1.0)

    return daily_returns


# ---------------------------------------------------------------------------
# Options: Iron Condor backtest (standalone BSM simulation)
# ---------------------------------------------------------------------------

def backtest_iron_condor(
    spy_close: pd.Series, vix_close: pd.Series,
    start_idx: int, end_idx: int,
    params: dict | None = None,
) -> np.ndarray:
    """Walk-forward backtest of Iron Condor strategy.

    Uses percentage-based strikes (more reliable than delta approximation):
    - Short put at 95% of spot, long put at 92% of spot (put spread)
    - Short call at 105% of spot, long call at 108% of spot (call spread)
    """
    params = params or {}
    dte_min = params.get("dte_min", 30)
    dte_max = params.get("dte_max", 45)
    ivr_entry = params.get("ivr_entry", 40)
    # Strike offsets as % of spot
    put_short_pct = params.get("put_short_pct", 0.95)  # 5% OTM
    put_long_pct = params.get("put_long_pct", 0.92)    # 8% OTM
    call_short_pct = params.get("call_short_pct", 1.05) # 5% OTM
    call_long_pct = params.get("call_long_pct", 1.08)   # 8% OTM

    common_idx = spy_close.index.intersection(vix_close.index)
    spy = spy_close.loc[common_idx]
    vix = vix_close.loc[common_idx]
    n = len(spy)

    if end_idx > n:
        end_idx = n
    if start_idx < 60:
        start_idx = 60

    daily_returns = np.zeros(end_idx - start_idx)
    equity = INITIAL_CASH
    positions = []

    for i in range(start_idx, end_idx):
        spot = float(spy.iloc[i])
        vix_val = float(vix.iloc[i])
        iv = vix_val / 100.0

        # IV Rank
        lookback = min(i, 252)
        vix_window = vix.iloc[i-lookback:i+1].values
        iv_min, iv_max = np.nanmin(vix_window), np.nanmax(vix_window)
        ivr = ((vix_val - iv_min) / (iv_max - iv_min) * 100) if (iv_max - iv_min) > 0.1 else 55.0

        daily_pnl = 0.0

        # Close expired positions
        remaining = []
        for pos in positions:
            T = (pos["expiry"] - spy.index[i]).days / 365.0
            if T <= 0:
                # At expiration: P&L = credit received - max(0, spread width - credit)
                # For each spread: max loss = width - credit
                # Actual loss depends on where spot is relative to strikes
                put_short_itm = max(pos["put_short_strike"] - spot, 0.0)
                put_long_itm = max(pos["put_long_strike"] - spot, 0.0)
                call_short_itm = max(spot - pos["call_short_strike"], 0.0)
                call_long_itm = max(spot - pos["call_long_strike"], 0.0)

                put_spread_loss = (put_short_itm - put_long_itm) * pos["contracts"] * 100
                call_spread_loss = (call_short_itm - call_long_itm) * pos["contracts"] * 100
                total_loss = put_spread_loss + call_spread_loss
                daily_pnl += pos["net_credit"] * pos["contracts"] * 100 - total_loss
            else:
                remaining.append(pos)
        positions = remaining

        # Entry: Iron condor when IVR > threshold
        if ivr >= ivr_entry and len(positions) < 2:
            dte = (dte_min + dte_max) // 2
            T = dte / 365.0

            # Percentage-based strikes
            put_short_strike = spot * put_short_pct
            put_long_strike = spot * put_long_pct
            call_short_strike = spot * call_short_pct
            call_long_strike = spot * call_long_pct

            # BSM mid prices
            short_put_mid = _bsm_put(spot, put_short_strike, T, 0.04, iv)
            long_put_mid = _bsm_put(spot, put_long_strike, T, 0.04, iv)
            short_call_mid = _bsm_call(spot, call_short_strike, T, 0.04, iv)
            long_call_mid = _bsm_call(spot, call_long_strike, T, 0.04, iv)

            # Net credit = (short put - long put) + (short call - long call) with bid-ask haircut
            put_spread_credit = short_put_mid - long_put_mid
            call_spread_credit = short_call_mid - long_call_mid
            net_credit = (put_spread_credit + call_spread_credit) * 0.90  # 10% haircut

            if net_credit <= 0:
                continue

            # Size: risk 1% of equity per trade
            spread_width = (put_short_strike - put_long_strike)  # = call_long - call_short
            max_loss_per_contract = (spread_width * 100 - net_credit * 100)
            if max_loss_per_contract <= 0:
                continue
            contracts = max(1, int(equity * 0.01 / max_loss_per_contract))
            contracts = min(contracts, 10)

            positions.append({
                "put_short_strike": put_short_strike,
                "put_long_strike": put_long_strike,
                "call_short_strike": call_short_strike,
                "call_long_strike": call_long_strike,
                "net_credit": net_credit,
                "contracts": contracts,
                "expiry": spy.index[i] + pd.Timedelta(days=dte),
            })
            daily_pnl += net_credit * contracts * 100

        # Early exit at 50% of max profit
        remaining_2 = []
        for pos in positions:
            T = max((pos["expiry"] - spy.index[i]).days / 365.0, 0.001)
            sp = _bsm_put(spot, pos["put_short_strike"], T, 0.04, iv)
            lp = _bsm_put(spot, pos["put_long_strike"], T, 0.04, iv)
            sc = _bsm_call(spot, pos["call_short_strike"], T, 0.04, iv)
            lc = _bsm_call(spot, pos["call_long_strike"], T, 0.04, iv)
            current_value = (sp - lp + sc - lc) * 1.10  # buy back at 110% of mid
            if current_value < pos["net_credit"] * 0.5:
                # Close at 50% profit
                daily_pnl += (pos["net_credit"] - current_value) * pos["contracts"] * 100
            else:
                remaining_2.append(pos)
        positions = remaining_2

        equity += daily_pnl
        ret_idx = i - start_idx
        if ret_idx < len(daily_returns):
            daily_returns[ret_idx] = daily_pnl / max(equity - daily_pnl, 1.0)

    return daily_returns


# ---------------------------------------------------------------------------
# Options: Put Write VRP backtest (standalone BSM simulation)
# ---------------------------------------------------------------------------

def backtest_put_write(
    spy_close: pd.Series, vix_close: pd.Series,
    start_idx: int, end_idx: int,
    params: dict | None = None,
) -> np.ndarray:
    """Walk-forward backtest of Put Write VRP strategy.

    Sells OTM puts monthly when VIX > threshold. Collects premium,
    holds to expiration or closes at 50% profit.

    Entry: First trading day of each month, if VIX > 15
    Exit: Hold to expiration (30 DTE) or close at 50% profit
    Sizing: Risk 2% of equity per trade
    """
    params = params or {}
    vix_entry = params.get("vix_entry", 15.0)
    otm_pct = params.get("otm_pct", 0.95)  # 5% OTM
    dte = params.get("dte", 30)
    risk_per_trade = params.get("risk_per_trade", 0.02)

    common_idx = spy_close.index.intersection(vix_close.index)
    spy = spy_close.loc[common_idx]
    vix = vix_close.loc[common_idx]
    n = len(spy)

    if end_idx > n:
        end_idx = n
    if start_idx < 60:
        start_idx = 60

    daily_returns = np.zeros(end_idx - start_idx)
    equity = INITIAL_CASH
    positions = []
    last_entry_month = None

    for i in range(start_idx, end_idx):
        spot = float(spy.iloc[i])
        vix_val = float(vix.iloc[i])
        iv = vix_val / 100.0

        daily_pnl = 0.0

        # Close expired positions
        remaining = []
        for pos in positions:
            T = (pos["expiry"] - spy.index[i]).days / 365.0
            if T <= 0:
                # At expiration: put value = max(K - S, 0)
                put_value = max(pos["strike"] - spot, 0.0)
                # P&L = credit received - put value at expiry
                pnl = (pos["entry_credit"] - put_value) * pos["contracts"] * 100
                daily_pnl += pnl
            else:
                remaining.append(pos)
        positions = remaining

        # Early exit at 50% profit
        remaining_2 = []
        for pos in positions:
            T = max((pos["expiry"] - spy.index[i]).days / 365.0, 0.001)
            current_put_mid = _bsm_put(spot, pos["strike"], T, 0.04, iv)
            # Buy back at 110% of mid (bid-ask spread)
            buy_back_cost = current_put_mid * 1.10
            if buy_back_cost < pos["entry_credit"] * 0.5:
                # Close at 50% profit
                pnl = (pos["entry_credit"] - buy_back_cost) * pos["contracts"] * 100
                daily_pnl += pnl
            else:
                remaining_2.append(pos)
        positions = remaining_2

        # Entry: sell puts on first trading day of each month if VIX > threshold
        current_month = spy.index[i].month
        if (last_entry_month != current_month and
            vix_val > vix_entry and
            len(positions) < 3):

            last_entry_month = current_month
            T = dte / 365.0
            strike = spot * otm_pct  # 5% OTM

            # BSM put price with bid-ask haircut (sell at 95% of mid)
            mid_put = _bsm_put(spot, strike, T, 0.04, iv)
            credit = mid_put * 0.95

            if credit < 0.10:  # minimum premium
                continue

            # Size: risk 2% of equity
            max_loss = strike * 100  # per contract (max loss = strike price)
            contracts = max(1, int(equity * risk_per_trade / max_loss))
            contracts = min(contracts, 10)

            positions.append({
                "strike": strike,
                "entry_credit": credit,
                "contracts": contracts,
                "entry_date": spy.index[i],
                "expiry": spy.index[i] + pd.Timedelta(days=dte),
            })
            daily_pnl += credit * contracts * 100

        equity += daily_pnl
        ret_idx = i - start_idx
        if ret_idx < len(daily_returns):
            daily_returns[ret_idx] = daily_pnl / max(equity - daily_pnl, 1.0)

    return daily_returns


# ---------------------------------------------------------------------------
# Macro: Signal-based return computation (proven approach)
# ---------------------------------------------------------------------------

def backtest_macro_strategy(strategy, data: dict, start_idx: int, end_idx: int) -> np.ndarray:
    """Run macro strategy with causal-slice approach (proven in run_backtest_focused.py)."""
    close = data["close"]
    symbols = data.get("symbols", [])
    n_periods = close.shape[0]
    n_assets = len(symbols)

    if n_assets == 0 or n_periods < 2 or end_idx - start_idx < 2:
        return np.array([])

    start_idx = max(start_idx, 1)
    end_idx = min(end_idx, n_periods)
    if end_idx - start_idx < 2:
        return np.array([])

    asset_rets = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)
    bar_rets = np.zeros(end_idx - start_idx)

    for idx, bar_i in enumerate(range(start_idx, end_idx)):
        if bar_i >= n_periods - 1:
            break
        bar_data = {}
        for k, v in data.items():
            if k == "symbols":
                bar_data[k] = v
            elif isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n_periods:
                bar_data[k] = v[:bar_i + 1]
            else:
                bar_data[k] = v

        try:
            signals = strategy.on_data(bar_data) or []
        except Exception:
            signals = []

        weighted_ret = 0.0
        for s in signals:
            if hasattr(s, "size") and s.size and s.size > 0:
                weight = float(s.direction) * float(s.size)
            else:
                weight = float(s.direction) * float(s.strength) if s.strength else float(s.direction) * 0.1
            if s.direction == 0:
                continue
            try:
                asset_idx = symbols.index(str(s.symbol).upper())
            except (ValueError, IndexError):
                continue
            if bar_i < asset_rets.shape[0] and asset_idx < asset_rets.shape[1]:
                weighted_ret += weight * float(asset_rets[bar_i, asset_idx])

        bar_rets[idx] = weighted_ret

    return bar_rets


# ---------------------------------------------------------------------------
# Walk-forward validation framework
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    strategy_name: str
    display_name: str
    strategy_type: str
    oos_sharpe: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    max_dd_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    dsr_p_value: float = 1.0
    n_folds: int = 0
    n_passing_folds: int = 0
    oos_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    triage: str = "DISABLE"
    triage_reason: str = ""
    risk_parity_weight: float = 0.0


def compute_metrics(returns: np.ndarray) -> dict:
    """Compute standard metrics from a returns array."""
    if len(returns) < 2:
        return {"sharpe": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "max_dd_pct": 0.0,
                "win_rate": 0.0, "profit_factor": 0.0, "dsr_p": 1.0}

    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    sharpe = float((mean_ret / std_ret) * np.sqrt(252)) if std_ret > 1e-10 else 0.0

    # Bootstrap CI(90%)
    n_boot = 1000
    sharpes = []
    n = len(returns)
    for _ in range(n_boot):
        sample = np.random.choice(returns, size=n, replace=True)
        s_mean = np.mean(sample)
        s_std = np.std(sample, ddof=1)
        if s_std > 1e-10:
            sharpes.append((s_mean / s_std) * np.sqrt(252))
    ci_lower = float(np.percentile(sharpes, 5)) if sharpes else 0.0
    ci_upper = float(np.percentile(sharpes, 95)) if sharpes else 0.0

    # Max drawdown
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.maximum(peak, 1e-10)
    max_dd_pct = float(abs(np.min(dd)) * 100)

    # Win rate
    non_zero = returns[returns != 0]
    win_rate = float(np.sum(non_zero > 0) / len(non_zero) * 100) if len(non_zero) > 0 else 0.0

    # Profit factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    # DSR
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurtosis())
    dsr_p, _ = deflated_sharpe_ratio(
        observed_sharpe=sharpe, num_trials=NUM_TRIALS,
        sample_length=len(returns), skewness=skew, kurtosis_=kurt,
    )

    return {
        "sharpe": sharpe, "ci_lower": ci_lower, "ci_upper": ci_upper,
        "max_dd_pct": max_dd_pct, "win_rate": win_rate, "profit_factor": pf,
        "dsr_p": dsr_p,
    }


def validate_options_strategy(
    display_name: str, registry_name: str,
    spy_close: pd.Series, vix_close: pd.Series,
    backtest_fn, params: dict,
) -> ValidationResult:
    """Walk-forward validate an options strategy using standalone BSM simulation."""
    log.info("Validating %s...", display_name)

    result = ValidationResult(
        strategy_name=registry_name, display_name=display_name, strategy_type="OPTIONS",
    )

    n_bars = len(spy_close)
    folds = anchored_walk_forward(n_bars=n_bars, initial_train=TRAIN_SIZE,
                                   test_size=TEST_SIZE, embargo_pct=EMBARGO_PCT)
    if not folds:
        result.triage_reason = f"Insufficient data ({n_bars} bars)"
        return result

    result.n_folds = len(folds)
    all_returns = []

    for fold in folds:
        fold_returns = backtest_fn(spy_close, vix_close, fold.test_start, fold.test_end, params)
        if len(fold_returns) > 0:
            all_returns.extend(fold_returns.tolist())
            if np.sum(fold_returns) > 0:
                result.n_passing_folds += 1

    if not all_returns:
        result.triage_reason = "No OOS returns"
        return result

    result.oos_returns = np.array(all_returns)
    m = compute_metrics(result.oos_returns)
    result.oos_sharpe = m["sharpe"]
    result.ci_lower = m["ci_lower"]
    result.ci_upper = m["ci_upper"]
    result.max_dd_pct = m["max_dd_pct"]
    result.win_rate = m["win_rate"]
    result.profit_factor = m["profit_factor"]
    result.dsr_p_value = m["dsr_p"]

    # Triage
    if result.ci_lower > 0 and result.dsr_p_value < 0.1:
        result.triage = "KEEP"
        result.triage_reason = f"CI_low={result.ci_lower:.2f}>0, DSR p={result.dsr_p_value:.3f}<0.1"
    elif result.ci_lower > 0:
        result.triage = "MONITOR"
        result.triage_reason = f"CI_low={result.ci_lower:.2f}>0, DSR p={result.dsr_p_value:.3f}>=0.1"
    else:
        result.triage = "DISABLE"
        result.triage_reason = f"CI_low={result.ci_lower:.2f}<=0"

    log.info("  %s: Sharpe=%.2f, CI=[%.2f, %.2f], MaxDD=%.1f%%, DSR p=%.3f -> %s",
             display_name, result.oos_sharpe, result.ci_lower, result.ci_upper,
             result.max_dd_pct, result.dsr_p_value, result.triage)
    return result


def validate_macro_strategy(
    display_name: str, registry_name: str,
    macro_data: dict, params: dict,
) -> ValidationResult:
    """Walk-forward validate a macro strategy using signal-based returns."""
    log.info("Validating %s...", display_name)

    result = ValidationResult(
        strategy_name=registry_name, display_name=display_name, strategy_type="MACRO",
    )

    try:
        cls = get_strategy_class(registry_name)
        cfg = StrategyConfig(name=registry_name, strategy_type="MACRO",
                            max_position_notional=INITIAL_CASH, params=params)
    except Exception as e:
        result.triage_reason = f"Strategy creation failed: {e}"
        return result

    close = macro_data["close"]
    n_bars = close.shape[0]
    folds = anchored_walk_forward(n_bars=n_bars, initial_train=TRAIN_SIZE,
                                   test_size=TEST_SIZE, embargo_pct=EMBARGO_PCT)
    if not folds:
        result.triage_reason = f"Insufficient data ({n_bars} bars)"
        return result

    result.n_folds = len(folds)
    all_returns = []

    for fold in folds:
        fold_strategy = cls(cfg)
        fold_returns = backtest_macro_strategy(fold_strategy, macro_data, fold.test_start, fold.test_end)
        if len(fold_returns) > 0:
            all_returns.extend(fold_returns.tolist())
            if np.sum(fold_returns) > 0:
                result.n_passing_folds += 1

    if not all_returns:
        result.triage_reason = "No OOS returns"
        return result

    result.oos_returns = np.array(all_returns)
    m = compute_metrics(result.oos_returns)
    result.oos_sharpe = m["sharpe"]
    result.ci_lower = m["ci_lower"]
    result.ci_upper = m["ci_upper"]
    result.max_dd_pct = m["max_dd_pct"]
    result.win_rate = m["win_rate"]
    result.profit_factor = m["profit_factor"]
    result.dsr_p_value = m["dsr_p"]

    if result.ci_lower > 0 and result.dsr_p_value < 0.1:
        result.triage = "KEEP"
        result.triage_reason = f"CI_low={result.ci_lower:.2f}>0, DSR p={result.dsr_p_value:.3f}<0.1"
    elif result.ci_lower > 0:
        result.triage = "MONITOR"
        result.triage_reason = f"CI_low={result.ci_lower:.2f}>0, DSR p={result.dsr_p_value:.3f}>=0.1"
    else:
        result.triage = "DISABLE"
        result.triage_reason = f"CI_low={result.ci_lower:.2f}<=0"

    log.info("  %s: Sharpe=%.2f, CI=[%.2f, %.2f], MaxDD=%.1f%%, DSR p=%.3f -> %s",
             display_name, result.oos_sharpe, result.ci_lower, result.ci_upper,
             result.max_dd_pct, result.dsr_p_value, result.triage)
    return result


# ---------------------------------------------------------------------------
# Risk parity weights
# ---------------------------------------------------------------------------

def compute_risk_parity_weights(results: list[ValidationResult]) -> dict[str, float]:
    kept = [r for r in results if r.triage in ("KEEP", "MONITOR")]
    if not kept:
        return {}
    vols = {}
    for r in kept:
        if len(r.oos_returns) > 1:
            vol = np.std(r.oos_returns, ddof=1) * np.sqrt(252)
            vols[r.strategy_name] = max(vol, 0.001)
        else:
            vols[r.strategy_name] = 0.20
    inv_vols = {k: 1.0 / v for k, v in vols.items()}
    total_inv = sum(inv_vols.values())
    return {k: v / total_inv for k, v in inv_vols.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 70)
    log.info("PHASE 3.1: Walk-Forward Re-Validation (Strategy-Native Backtests)")
    log.info("=" * 70)

    t0 = time.time()
    spy_close, vix_close, macro_df = load_data()
    macro_data = df_to_data(macro_df)
    log.info("Data loaded in %.1fs: SPY=%d, VIX=%d, Macro=%s",
             time.time()-t0, len(spy_close), len(vix_close), macro_data["close"].shape)

    results = []

    # 1. Options strategies (standalone BSM simulation)
    log.info("-" * 50)
    results.append(validate_options_strategy(
        "Delta Neutral VRP", "options_delta_neutral",
        spy_close, vix_close, backtest_delta_neutral_vrp,
        {"ivr_entry": 50, "ivr_exit": 25, "dte_min": 30, "dte_max": 60, "delta_target": 0.3},
    ))
    results.append(validate_options_strategy(
        "Iron Condor VRP", "options_iron_condor",
        spy_close, vix_close, backtest_iron_condor,
        {"dte_min": 30, "dte_max": 45, "ivr_entry": 40,
         "put_short_pct": 0.95, "put_long_pct": 0.92,
         "call_short_pct": 1.05, "call_long_pct": 1.08},
    ))
    results.append(validate_options_strategy(
        "Put Write VRP", "options_put_write",
        spy_close, vix_close, backtest_put_write,
        {"vix_entry": 15.0, "otm_pct": 0.95, "dte": 30, "risk_per_trade": 0.02},
    ))

    # 2. Macro strategies (signal-based)
    log.info("-" * 50)
    results.append(validate_macro_strategy(
        "Macro Cross-Sectional Momentum", "macro_cross_sectional_momentum",
        macro_data, {"lookback": 60, "top_n": 3, "rebalance_freq": 5},
    ))
    results.append(validate_macro_strategy(
        "Macro Risk Parity", "macro_risk_parity",
        macro_data, {"vol_lookback": 60, "rebalance_freq": 21},
    ))

    # 3. Compute weights
    weights = compute_risk_parity_weights(results)
    for r in results:
        r.risk_parity_weight = weights.get(r.strategy_name, 0.0)

    # 4. Generate report
    lines = [
        "# Phase 3 Validation Report",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Method:** Anchored WF (504-day train, 126-day test, 1% embargo) + strategy-native backtests",
        f"**Period:** {START_DATE} to {END_DATE}",
        f"**DSR Trials:** {NUM_TRIALS}",
        "",
        "| Strategy | Type | OOS Sharpe | CI(90%) Lower | MaxDD | Win% | PF | DSR p | Triage |",
        "|----------|------|-----------|---------------|-------|------|-----|-------|--------|",
    ]
    for r in sorted(results, key=lambda x: x.triage):
        pf = f"{r.profit_factor:.1f}" if r.profit_factor < 100 else "inf"
        dsr = f"{r.dsr_p_value:.3f}" if r.dsr_p_value < 1.0 else "N/A"
        emoji = {"KEEP": "PASS", "MONITOR": "WARN", "DISABLE": "FAIL"}.get(r.triage, "")
        lines.append(f"| {r.display_name} | {r.strategy_type} | {r.oos_sharpe:.2f} | {r.ci_lower:.2f} | "
                     f"{r.max_dd_pct:.1f}% | {r.win_rate:.0f}% | {pf} | {dsr} | {emoji} {r.triage} |")

    lines.extend(["", "## Risk-Parity Weights", ""])
    for name, w in sorted(weights.items(), key=lambda x: -x[1]):
        display = next((r.display_name for r in results if r.strategy_name == name), name)
        lines.append(f"- **{display}:** {w:.1%}")
    if not weights:
        lines.append("No strategies passed triage gate.")

    report = "\n".join(lines)
    (REPORTS_DIR / "phase3_validation.md").write_text(report, encoding="utf-8")

    # 5. Save JSON
    json_results = []
    for r in results:
        json_results.append({
            "strategy_name": r.strategy_name, "display_name": r.display_name,
            "oos_sharpe": r.oos_sharpe, "ci_lower": r.ci_lower, "ci_upper": r.ci_upper,
            "max_dd_pct": r.max_dd_pct, "win_rate": r.win_rate, "profit_factor": r.profit_factor,
            "dsr_p_value": r.dsr_p_value, "triage": r.triage, "triage_reason": r.triage_reason,
            "risk_parity_weight": r.risk_parity_weight,
        })
    (REPORTS_DIR / "phase3_validation.json").write_text(
        json.dumps(json_results, indent=2, default=str), encoding="utf-8")

    # 6. Summary
    log.info("=" * 70)
    for r in results:
        tag = {"KEEP": "PASS", "MONITOR": "WARN", "DISABLE": "FAIL"}.get(r.triage, "")
        log.info("  [%s] %s — Sharpe=%.2f, CI_low=%.2f, DSR p=%.3f, Weight=%.1f%%",
                 tag, r.display_name, r.oos_sharpe, r.ci_lower, r.dsr_p_value,
                 r.risk_parity_weight * 100)

    kept = sum(1 for r in results if r.triage == "KEEP")
    mon = sum(1 for r in results if r.triage == "MONITOR")
    dis = sum(1 for r in results if r.triage == "DISABLE")
    log.info("KEEP: %d | MONITOR: %d | DISABLE: %d", kept, mon, dis)

    return results, weights


if __name__ == "__main__":
    main()
