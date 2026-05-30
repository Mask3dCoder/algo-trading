"""Iron Condor strategy targeting the Volatility Risk Premium (VRP).

Sells OTM put spreads and OTM call spreads on SPY when implied volatility
is elevated relative to realized volatility.  The strategy captures the
systematic tendency of options to overprice future volatility.

Designed for honest walk-forward validation via ``run_returns_backtest()``.

Two pricing modes:

* **Synthetic** (default) — BSM model pricing with ``generate_synthetic_iv``.
* **Real data** — Pass an ``OptionsDataLoader`` (Alpaca or Polygon) via the
  ``options_loader`` kwarg.  The strategy will use actual option chain data
  for contract selection, entry/exit pricing, and mark-to-market.  Falls
  back to BSM when the loader returns no data for a given date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from quantcore.data.options_loader import OptionsDataLoader, PolygonOptionsDataLoader

_log = logging.getLogger(__name__)

# ── BSM helpers ──────────────────────────────────────────────────────────────


def _bsm_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-10 or sigma < 1e-10:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bsm_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-10 or sigma < 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _strike_from_delta(
    S: float, T: float, r: float, sigma: float, delta_target: float, is_call: bool
) -> float:
    """Invert BSM to find the strike for a given absolute delta."""
    if T < 1e-10 or sigma < 1e-10:
        return S
    target = delta_target
    if not is_call:
        target = 1.0 - target
    target = np.clip(target, 0.001, 0.999)
    try:
        d1_target = norm.ppf(target)
        K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
        return float(K)
    except (ValueError, OverflowError):
        return S * (1 + sigma * np.sqrt(T) * (0.5 if is_call else -0.5))


# ── Synthetic IV model ───────────────────────────────────────────────────────


def generate_synthetic_iv(
    close: pd.Series,
    lookback: int = 60,
    vix_series: pd.Series | None = None,
    use_enhanced: bool = True,
) -> pd.Series:
    """Generate realistic IV from realized vol with mean-reversion and VRP premium.

    Parameters
    ----------
    close : pd.Series
        Daily close prices (index = DatetimeIndex).
    lookback : int
        Window for realized vol computation (default 60).
    vix_series : pd.Series | None
        Optional VIX series for more realistic ATM IV levels.
    use_enhanced : bool
        If True (default), use the enhanced vol surface model with
        VIX-dependent VRP scaling.  If False, use the flat premium model.
    """
    if use_enhanced:
        from quantcore.data.vol_surface import VolatilitySurface, generate_enhanced_synthetic_iv
        return generate_enhanced_synthetic_iv(
            close, lookback=lookback, vix_series=vix_series, use_skew=True,
        )

    # Legacy flat premium model (kept for comparison)
    log_returns = np.log(close / close.shift(1)).dropna()
    realized_vol = log_returns.rolling(lookback).std() * np.sqrt(252)
    iv_premium = 0.15  # vol risk premium
    iv = realized_vol + iv_premium
    long_run_iv = 0.20
    mean_reversion = 0.02
    iv_smooth = iv.copy()
    for i in range(1, len(iv_smooth)):
        if np.isnan(iv_smooth.iloc[i]):
            iv_smooth.iloc[i] = iv_smooth.iloc[i - 1] if not np.isnan(iv_smooth.iloc[i - 1]) else long_run_iv
        else:
            iv_smooth.iloc[i] = iv_smooth.iloc[i - 1] + mean_reversion * (iv_smooth.iloc[i] - iv_smooth.iloc[i - 1])
    iv_smooth = iv_smooth.clip(lower=0.08, upper=0.80)
    iv_smooth = iv_smooth.fillna(long_run_iv)
    iv_smooth = iv_smooth.reindex(close.index, method="ffill").fillna(long_run_iv)
    return iv_smooth


# ── IV Rank ──────────────────────────────────────────────────────────────────


def compute_ivr(iv_series: pd.Series, idx: int, lookback: int = 252) -> float:
    start = max(0, idx - lookback)
    window = iv_series.iloc[start : idx + 1].dropna()
    if len(window) < 20:
        return 50.0
    iv_min, iv_max = window.min(), window.max()
    if iv_max - iv_min < 1e-10:
        return 50.0
    return float((window.iloc[-1] - iv_min) / (iv_max - iv_min) * 100)


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass
class CondorTrade:
    """A single iron condor position."""
    entry_date: date
    expiry: date
    short_put: float
    long_put: float
    short_call: float
    long_call: float
    credit: float        # per-share credit received
    contracts: int
    entry_spot: float
    entry_iv: float
    pnl: float = 0.0
    closed: bool = False
    close_date: date | None = None
    close_reason: str = ""
    # Contract identifiers for real-data mode
    sp_contract: dict | None = field(default=None, repr=False)
    lp_contract: dict | None = field(default=None, repr=False)
    sc_contract: dict | None = field(default=None, repr=False)
    lc_contract: dict | None = field(default=None, repr=False)


# ── Strategy ─────────────────────────────────────────────────────────────────


class IronCondorVRP:
    """Iron Condor strategy targeting the Volatility Risk Premium.

    Stores full-period close and IV data at construction.  The
    ``run_returns_backtest`` method slices to the requested window
    and simulates the iron condor lifecycle day-by-day.

    Parameters
    ----------
    close : pd.Series
        Daily close prices for the underlying (index = DatetimeIndex).
    iv : pd.Series
        Daily implied volatility (same index).  If ``None`` and no
        *options_loader* is given, synthetic IV is generated from
        realized vol with a 15% VRP premium.
    options_loader : OptionsDataLoader | PolygonOptionsDataLoader | None
        If provided, uses real options chain data for contract selection,
        entry/exit pricing, and mark-to-market.  Falls back to BSM when
        the loader has no data for a given date.
    underlying : str
        Ticker symbol (default "SPY").  Used to query the options loader.
    delta_short : float
        Target delta for short strikes (default 0.16).
    delta_long : float
        Target delta for long strikes (default 0.10).
    spread_width : float
        Width of each wing in underlying points (default 5).  Used only
        in BSM mode; real data mode selects strikes by delta.
    dte_entry_min / dte_entry_max : int
        DTE range at entry (default 30–45).
    dte_exit : int
        Time-stop exit DTE (default 21).
    profit_target_pct : float
        Fraction of max profit to take (default 0.50).
    stop_loss_mult : float
        Stop-loss as multiple of credit received (default 2.0).
    risk_per_trade_pct : float
        Fraction of equity to risk per trade (default 0.02).
    entry_interval_days : int
        Minimum days between new entries (default 30).
    ivr_entry : float
        IV rank percentile threshold for entry (default 40).
    risk_free_rate : float
        Annual risk-free rate for BSM (default 0.04).
    """

    def __init__(
        self,
        close: pd.Series,
        iv: pd.Series | None = None,
        *,
        options_loader: OptionsDataLoader | PolygonOptionsDataLoader | None = None,
        underlying: str = "SPY",
        delta_short: float = 0.16,
        delta_long: float = 0.10,
        spread_width: float = 5.0,
        dte_entry_min: int = 30,
        dte_entry_max: int = 45,
        dte_exit: int = 21,
        profit_target_pct: float = 0.50,
        stop_loss_mult: float = 2.0,
        risk_per_trade_pct: float = 0.02,
        entry_interval_days: int = 30,
        ivr_entry: float = 40.0,
        risk_free_rate: float = 0.04,
        use_enhanced_iv: bool = True,
        vix_series: pd.Series | None = None,
    ) -> None:
        self._close = close
        self._dates = close.index
        self._options_loader = options_loader
        self._underlying = underlying

        # IV series: prefer explicit > loader > synthetic
        if iv is not None:
            self._iv = iv
        elif options_loader is not None:
            self._iv = self._build_real_iv_series(close, options_loader)
        else:
            self._iv = generate_synthetic_iv(
                close, vix_series=vix_series, use_enhanced=use_enhanced_iv,
            )

        self.delta_short = delta_short
        self.delta_long = delta_long
        self.spread_width = spread_width
        self.dte_entry_min = dte_entry_min
        self.dte_entry_max = dte_entry_max
        self.dte_exit = dte_exit
        self.profit_target_pct = profit_target_pct
        self.stop_loss_mult = stop_loss_mult
        self.risk_per_trade_pct = risk_per_trade_pct
        self.entry_interval_days = entry_interval_days
        self.ivr_entry = ivr_entry
        self.r = risk_free_rate

    @staticmethod
    def _build_real_iv_series(
        close: pd.Series,
        loader: OptionsDataLoader | PolygonOptionsDataLoader,
        lookback: int = 60,
    ) -> pd.Series:
        """Build a daily IV series from ATM options via *loader*.

        Dates where the loader has no data are forward-filled.  The
        synthetic IV model is used as a fallback for the warmup period.
        """
        _log.info("Building real ATM IV series from options data...")
        synthetic = generate_synthetic_iv(close, lookback=lookback)
        iv_values: list[float] = []
        loaded = 0
        for dt_idx in close.index:
            dt = dt_idx.date() if hasattr(dt_idx, "date") else dt_idx
            try:
                atm_iv = loader.get_atm_iv(close.name or "SPY", dt)
            except Exception:
                atm_iv = None
            if atm_iv is not None and atm_iv > 0:
                iv_values.append(atm_iv)
                loaded += 1
            else:
                # Forward fill or use synthetic
                iv_values.append(iv_values[-1] if iv_values else float(synthetic.iloc[len(iv_values)]))
        _log.info("Loaded real ATM IV for %d / %d dates", loaded, len(close))
        result = pd.Series(iv_values, index=close.index, dtype=float)
        # Fill any remaining NaN with synthetic
        result = result.fillna(synthetic)
        return result

    # ── Walk-forward interface ───────────────────────────────────────────

    def run_returns_backtest(
        self,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        training_start: pd.Timestamp,
        initial_capital: float = 1_000_000.0,
        spread_bps: float = 5.0,
        impact_bps: float = 10.0,
        commission_per_contract: float = 0.65,
    ) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
        """Run the iron condor backtest over [start, end].

        This is the interface expected by ``WalkForwardBacktester``.

        Returns
        -------
        equity : pd.Series
            Daily equity curve (index = dates in [start, end]).
        daily_returns : pd.Series
            Daily simple returns.
        turnover : pd.Series
            Daily turnover (non-zero on entry/exit days).
        logs : dict
            Trade-level logs.
        """
        # Slice data to the requested window (plus warmup for IVR)
        warmup = pd.Timedelta(days=300)
        warmup_start = start - warmup
        mask_warmup = (self._dates >= warmup_start) & (self._dates <= end)
        close_w = self._close[mask_warmup]
        iv_w = self._iv[mask_warmup]

        if len(close_w) < 60:
            empty = pd.Series(dtype=float)
            return empty, empty, empty, {}

        # Run the simulation on the warmup+test window
        trades, eq_curve, daily_rets, daily_turnover = self._simulate(
            close_w, iv_w, capital=initial_capital,
            commission_per_contract=commission_per_contract,
            slippage_pct=spread_bps / 100.0,
            options_loader=self._options_loader,
        )

        # Trim to [start, end] only
        eq_dates = [d for d, _ in eq_curve]
        eq_values = [v for _, v in eq_curve]
        eq_series = pd.Series(eq_values, index=eq_dates, dtype=float)
        ret_series = pd.Series(daily_rets, index=eq_dates, dtype=float)
        turn_series = pd.Series(daily_turnover, index=eq_dates, dtype=float)

        # Filter to OOS window
        start_date = start.date() if hasattr(start, "date") else start
        end_date = end.date() if hasattr(end, "date") else end

        eq_oos = eq_series[(eq_series.index >= start_date) & (eq_series.index <= end_date)]
        ret_oos = ret_series[(ret_series.index >= start_date) & (ret_series.index <= end_date)]
        turn_oos = turn_series[(turn_series.index >= start_date) & (turn_series.index <= end_date)]

        # Rebase equity to start at initial_capital
        if len(eq_oos) > 0:
            eq_oos = eq_oos / eq_oos.iloc[0] * initial_capital

        return eq_oos, ret_oos, turn_oos, {"trades": trades}

    # ── Internal simulation ──────────────────────────────────────────────

    # ── Real-data helpers ──────────────────────────────────────────────

    def _real_fill_price(
        self,
        loader: OptionsDataLoader | PolygonOptionsDataLoader,
        dt: date,
        contract: dict,
        side: str,
    ) -> float | None:
        """Get an execution price for *contract* on *dt*.

        Parameters
        ----------
        side : str
            ``"buy"`` → use ask; ``"sell"`` → use bid.
        """
        if loader is None:
            return None
        chain = loader.load_chain(self._underlying, dt)
        if chain.empty:
            return None

        chain = chain.copy()
        if "contract_type" in chain.columns:
            chain["contract_type"] = chain["contract_type"].str.lower().replace(
                {"c": "call", "p": "put"}
            )
        chain["_exp_date"] = pd.to_datetime(chain["expiration"]).dt.date

        mask = (
            (chain["strike"] == contract["strike"])
            & (chain["_exp_date"] == contract["expiration"])
            & (chain["contract_type"] == contract["option_type"])
        )
        matches = chain[mask]
        if matches.empty:
            return None

        row = matches.iloc[0]
        bid = row.get("bid")
        ask = row.get("ask")
        if side == "buy":
            if pd.notna(ask) and float(ask) > 0:
                return float(ask)
            if pd.notna(bid) and pd.notna(ask):
                return float((bid + ask) / 2)
        else:  # sell
            if pd.notna(bid) and float(bid) > 0:
                return float(bid)
            if pd.notna(bid) and pd.notna(ask):
                return float((bid + ask) / 2)
        # Fallback to last_price
        if pd.notna(row.get("last_price")) and float(row["last_price"]) > 0:
            return float(row["last_price"])
        return None

    def _real_mid_price(
        self,
        loader: OptionsDataLoader | PolygonOptionsDataLoader,
        dt: date,
        contract: dict,
    ) -> float | None:
        """Get the bid/ask midpoint for *contract* on *dt*."""
        if loader is None:
            return None
        return loader.get_mid_price(
            self._underlying,
            dt,
            contract["strike"],
            contract["expiration"],
            contract["option_type"],
        )

    def _entry_from_real_data(
        self,
        loader: OptionsDataLoader | PolygonOptionsDataLoader,
        dt: date,
        S: float,
        dte_entry: int,
    ) -> tuple[float, dict | None, dict | None, dict | None, dict | None] | None:
        """Select contracts and compute net credit from real chain data.

        Returns (net_credit, sp_contract, lp_contract, sc_contract, lc_contract)
        or ``None`` if the loader has no usable data.
        """
        min_dte = self.dte_entry_min
        max_dte = self.dte_entry_max

        # Short put: delta_target = delta_short
        sp = loader.get_option_by_delta(
            self._underlying, dt, self.delta_short, "put", min_dte, max_dte,
        )
        if sp is None:
            return None
        sp_exp = pd.to_datetime(sp["expiration"]).date()
        sp_strike = float(sp["strike"])

        # Long put: delta_target = delta_long, same expiry, lower strike
        lp = loader.get_option_by_delta(
            self._underlying, dt, self.delta_long, "put", min_dte, max_dte,
        )
        if lp is None:
            return None
        lp_strike = float(lp["strike"])
        if lp_strike >= sp_strike:
            # Long put must be lower strike than short put
            return None

        # Short call: delta_target = delta_short
        sc = loader.get_option_by_delta(
            self._underlying, dt, self.delta_short, "call", min_dte, max_dte,
        )
        if sc is None:
            return None
        sc_strike = float(sc["strike"])

        # Long call: delta_target = delta_long, same expiry, higher strike
        lc = loader.get_option_by_delta(
            self._underlying, dt, self.delta_long, "call", min_dte, max_dte,
        )
        if lc is None:
            return None
        lc_strike = float(lc["strike"])
        if lc_strike <= sc_strike:
            return None

        sc_exp = pd.to_datetime(sc["expiration"]).date()
        lc_exp = pd.to_datetime(lc["expiration"]).date()

        # Pricing: sell at bid, buy at ask
        sp_price = self._real_fill_price(loader, dt,
            {"strike": sp_strike, "expiration": sp_exp, "option_type": "put"}, "sell")
        lp_price = self._real_fill_price(loader, dt,
            {"strike": lp_strike, "expiration": pd.to_datetime(lp["expiration"]).date(), "option_type": "put"}, "buy")
        sc_price = self._real_fill_price(loader, dt,
            {"strike": sc_strike, "expiration": sc_exp, "option_type": "call"}, "sell")
        lc_price = self._real_fill_price(loader, dt,
            {"strike": lc_strike, "expiration": lc_exp, "option_type": "call"}, "buy")

        if any(p is None for p in [sp_price, lp_price, sc_price, lc_price]):
            return None

        net_credit = sp_price - lp_price + sc_price - lc_price
        if net_credit <= 0.01:
            return None

        sp_c = {"strike": sp_strike, "expiration": sp_exp, "option_type": "put",
                "option_symbol": sp.get("option_symbol", "")}
        lp_c = {"strike": lp_strike, "expiration": pd.to_datetime(lp["expiration"]).date(),
                "option_type": "put", "option_symbol": lp.get("option_symbol", "")}
        sc_c = {"strike": sc_strike, "expiration": sc_exp, "option_type": "call",
                "option_symbol": sc.get("option_symbol", "")}
        lc_c = {"strike": lc_strike, "expiration": lc_exp, "option_type": "call",
                "option_symbol": lc.get("option_symbol", "")}

        return net_credit, sp_c, lp_c, sc_c, lc_c

    def _real_unrealized_pnl(
        self,
        loader: OptionsDataLoader | PolygonOptionsDataLoader,
        dt: date,
        pos: CondorTrade,
    ) -> float | None:
        """Compute mark-to-market P&L using real chain data.

        To close the iron condor we need to *buy back* the short legs (at ask)
        and *sell* the long legs (at bid).  The cost to close is therefore
        ``buy_sp + buy_lc - sell_lp - sell_sc``  (the signs flip from entry).

        Alternatively, compute it as ``entry_credit - cost_to_close`` where
        cost_to_close uses bid/ask appropriately.

        Returns the unrealized P&L in dollars, or ``None`` if data is missing.
        """
        def _close_price(contract: dict, was_sold: bool) -> float | None:
            """Price to close a leg.  If originally sold → buy back at ask.
            If originally bought → sell at bid."""
            side = "buy" if was_sold else "sell"
            return self._real_fill_price(loader, dt, contract, side)

        # Short legs were sold at entry → buy back at ask
        sp_now = _close_price(pos.sp_contract, True)
        sc_now = _close_price(pos.sc_contract, True)
        # Long legs were bought at entry → sell at bid
        lp_now = _close_price(pos.lp_contract, False)
        lc_now = _close_price(pos.lc_contract, False)

        if any(p is None for p in [sp_now, sc_now, lp_now, lc_now]):
            return None

        # Cost to close: buy back shorts, sell longs
        cost_to_close = sp_now + sc_now - lp_now - lc_now
        return (pos.credit - cost_to_close) * 100 * pos.contracts

    # ── Core simulation ──────────────────────────────────────────────

    def _simulate(
        self,
        close: pd.Series,
        iv: pd.Series,
        *,
        capital: float,
        commission_per_contract: float,
        slippage_pct: float,
        options_loader: OptionsDataLoader | PolygonOptionsDataLoader | None = None,
    ) -> tuple[list[CondorTrade], list[tuple[date, float]], list[float], list[float]]:
        """Core simulation loop.  Returns (trades, equity_curve, daily_returns, daily_turnover).

        When *options_loader* is provided, contract selection and pricing
        use real options chain data.  Falls back to BSM when the loader
        returns no data for a given date.
        """
        trades: list[CondorTrade] = []
        equity_curve: list[tuple[date, float]] = []
        daily_returns: list[float] = []
        daily_turnover: list[float] = []
        active: list[CondorTrade] = []
        last_entry_date: date | None = None

        close_arr = close.values
        iv_arr = iv.values
        dates = close.index
        n = len(close)
        use_real = options_loader is not None

        equity = capital
        prev_equity = equity

        for i in range(60, n):
            S = float(close_arr[i])
            sigma = float(iv_arr[i])
            dt = dates[i].date() if hasattr(dates[i], "date") else dates[i]
            ivr = compute_ivr(iv, i)
            daily_pnl = 0.0
            turnover_today = 0.0

            # ── Update active positions ──
            for pos in list(active):
                if pos.closed:
                    continue

                dte = (pos.expiry - dt).days

                # Expiration
                if dte <= 0:
                    put_val = max(pos.short_put - S, 0) - max(pos.long_put - S, 0)
                    call_val = max(S - pos.short_call, 0) - max(S - pos.long_call, 0)
                    pos.pnl = (pos.credit - put_val - call_val) * 100 * pos.contracts
                    pos.pnl -= commission_per_contract * pos.contracts * 2
                    pos.closed = True
                    pos.close_date = dt
                    pos.close_reason = "expiration"
                    daily_pnl += pos.pnl
                    turnover_today += abs(pos.pnl)
                    continue

                T = dte / 252.0

                # Mark-to-market — prefer real data
                unrealized: float | None = None
                if use_real and pos.sp_contract is not None:
                    unrealized = self._real_unrealized_pnl(options_loader, dt, pos)

                if unrealized is None:
                    # BSM fallback
                    sp_val = _bsm_put(S, pos.short_put, T, self.r, sigma)
                    lp_val = _bsm_put(S, pos.long_put, T, self.r, sigma)
                    sc_val = _bsm_call(S, pos.short_call, T, self.r, sigma)
                    lc_val = _bsm_call(S, pos.long_call, T, self.r, sigma)
                    current_cost = sp_val - lp_val + sc_val - lc_val
                    unrealized = (pos.credit - current_cost) * 100 * pos.contracts

                max_profit = pos.credit * 100 * pos.contracts

                # Profit target
                if unrealized >= self.profit_target_pct * max_profit:
                    pos.pnl = unrealized - commission_per_contract * pos.contracts * 2
                    pos.closed = True
                    pos.close_date = dt
                    pos.close_reason = "profit_target"
                    daily_pnl += pos.pnl
                    turnover_today += abs(pos.pnl)
                    continue

                # Stop-loss: underlying touched short strike
                if S <= pos.short_put or S >= pos.short_call:
                    stop_loss_amount = max_profit * self.stop_loss_mult
                    if unrealized <= -stop_loss_amount:
                        pos.pnl = unrealized - commission_per_contract * pos.contracts * 2
                        pos.closed = True
                        pos.close_date = dt
                        pos.close_reason = "stop_loss"
                        daily_pnl += pos.pnl
                        turnover_today += abs(pos.pnl)
                        continue

                # Time exit
                if dte <= self.dte_exit:
                    pos.pnl = unrealized - commission_per_contract * pos.contracts * 2
                    pos.closed = True
                    pos.close_date = dt
                    pos.close_reason = "time_exit"
                    daily_pnl += pos.pnl
                    turnover_today += abs(pos.pnl)

            # ── Entry logic ──
            can_enter = True
            if last_entry_date is not None:
                if (dt - last_entry_date).days < self.entry_interval_days:
                    can_enter = False

            active_open = [p for p in active if not p.closed]
            if can_enter and ivr > self.ivr_entry and len(active_open) < 3:
                dte_entry = (self.dte_entry_min + self.dte_entry_max) // 2
                T_entry = dte_entry / 252.0
                expiry_date = dt + timedelta(days=dte_entry)

                # Try real data first
                real_result = None
                if use_real:
                    real_result = self._entry_from_real_data(options_loader, dt, S, dte_entry)

                if real_result is not None:
                    net_credit, sp_c, lp_c, sc_c, lc_c = real_result
                    short_put_k = sp_c["strike"]
                    long_put_k = lp_c["strike"]
                    short_call_k = sc_c["strike"]
                    long_call_k = lc_c["strike"]
                else:
                    # BSM fallback
                    long_put_k = _strike_from_delta(S, T_entry, self.r, sigma, self.delta_short, False)
                    short_put_k = long_put_k + self.spread_width
                    short_call_k = _strike_from_delta(S, T_entry, self.r, sigma, self.delta_short, True)
                    long_call_k = short_call_k + self.spread_width

                    sp_prem = _bsm_put(S, short_put_k, T_entry, self.r, sigma)
                    lp_prem = _bsm_put(S, long_put_k, T_entry, self.r, sigma)
                    sc_prem = _bsm_call(S, short_call_k, T_entry, self.r, sigma)
                    lc_prem = _bsm_call(S, long_call_k, T_entry, self.r, sigma)
                    net_credit = sp_prem - lp_prem + sc_prem - lc_prem

                    if net_credit <= 0.01:
                        continue

                    net_credit *= (1.0 - slippage_pct)
                    sp_c = lp_c = sc_c = lc_c = None

                risk_amount = equity * self.risk_per_trade_pct
                wing_width = min(short_put_k - long_put_k, long_call_k - short_call_k)
                max_risk_per_contract = (wing_width - net_credit) * 100
                if max_risk_per_contract <= 0:
                    max_risk_per_contract = net_credit * 100
                contracts = max(1, int(risk_amount / max_risk_per_contract))

                entry_commission = commission_per_contract * contracts * 2

                pos = CondorTrade(
                    entry_date=dt,
                    expiry=expiry_date,
                    short_put=short_put_k,
                    long_put=long_put_k,
                    short_call=short_call_k,
                    long_call=long_call_k,
                    credit=net_credit,
                    contracts=contracts,
                    entry_spot=S,
                    entry_iv=sigma,
                    sp_contract=sp_c,
                    lp_contract=lp_c,
                    sc_contract=sc_c,
                    lc_contract=lc_c,
                )
                active.append(pos)
                last_entry_date = dt
                daily_pnl -= entry_commission
                turnover_today += entry_commission

            # ── Collect closed trades ──
            newly_closed = [p for p in active if p.closed]
            trades.extend(newly_closed)
            active = [p for p in active if not p.closed]

            equity += daily_pnl
            equity_curve.append((dt, equity))
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            daily_returns.append(daily_ret)
            daily_turnover.append(turnover_today / equity if equity > 0 else 0.0)
            prev_equity = equity

        # Close remaining at end — fold into the last day's P&L
        if active:
            S_final = float(close_arr[-1])
            dt_final = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]
            eob_pnl = 0.0
            for pos in active:
                if not pos.closed:
                    put_val = max(pos.short_put - S_final, 0) - max(pos.long_put - S_final, 0)
                    call_val = max(S_final - pos.short_call, 0) - max(S_final - pos.long_call, 0)
                    pos.pnl = (pos.credit - put_val - call_val) * 100 * pos.contracts
                    pos.pnl -= 0.65 * pos.contracts * 2
                    pos.closed = True
                    pos.close_date = dt_final
                    pos.close_reason = "end_of_backtest"
                    eob_pnl += pos.pnl
            trades.extend(active)
            # Update the last equity curve entry and return
            if equity_curve:
                equity += eob_pnl
                equity_curve[-1] = (dt_final, equity)
                if daily_returns:
                    daily_returns[-1] = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

        return trades, equity_curve, daily_returns, daily_turnover
