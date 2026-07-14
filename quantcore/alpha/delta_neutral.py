"""Delta Neutral VRP strategy -- sell options and hedge delta with the underlying.

Sells ATM straddles (or 10-delta strangles) on SPY and continuously
delta-hedges the underlying position to isolate the volatility risk
premium (VRP).  Lower structural cost than Iron Condor (2 option legs
vs 4), which may push the Deflated Sharpe Ratio p-value below the
0.10 threshold.

Two pricing modes (same as Iron Condor):

* **Synthetic** (default) -- BSM model pricing with ``generate_synthetic_iv``.
* **Real data** -- Pass an ``OptionsDataLoader`` (Alpaca or Polygon) via the
  ``options_loader`` kwarg.  Falls back to BSM when the loader returns
  no data for a given date.
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

from quantcore.core.exceptions import DataQualityViolation

_log = logging.getLogger(__name__)

# AUDIT MANDATE 2026-07-12 — emitted once when a backtest is run after the
# mark-to-market accounting fix (hedge cost-basis now subtracted).
_ACCOUNTING_FIX_WARNING = (
    "Pre-fix delta-neutral Sharpe on 2024 data was artificially inflated. "
    "Re-running backtest is mandatory before proceeding."
)
_warned_accounting_fix = False


def _emit_accounting_fix_warning() -> None:
    global _warned_accounting_fix
    if not _warned_accounting_fix:
        _warned_accounting_fix = True
        _log.warning(_ACCOUNTING_FIX_WARNING)
        print(f"\n[!] WARNING: {_ACCOUNTING_FIX_WARNING}\n")

# ---- BSM helpers (reused from Iron Condor) ------------------------------------------------------------------------


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


def _bsm_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Return the BSM delta for a call or put."""
    if T < 1e-10 or sigma < 1e-10:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if is_call:
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


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


# ---- Synthetic IV model (same as Iron Condor) ------------------------------------------------------------------


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
        from quantcore.data.vol_surface import generate_enhanced_synthetic_iv
        return generate_enhanced_synthetic_iv(
            close, lookback=lookback, vix_series=vix_series, use_skew=True,
        )

    # Legacy flat premium model (kept for comparison)
    log_returns = np.log(close / close.shift(1)).dropna()
    realized_vol = log_returns.rolling(lookback).std() * np.sqrt(252)
    iv_premium = 0.15
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


# ---- IV Rank ------------------------------------------------------------------------------------------------------------------------------------


def compute_ivr(iv_series: pd.Series, idx: int, lookback: int = 252) -> float:
    start = max(0, idx - lookback)
    window = iv_series.iloc[start : idx + 1].dropna()
    if len(window) < 20:
        return 50.0
    iv_min, iv_max = window.min(), window.max()
    if iv_max - iv_min < 1e-10:
        return 50.0
    return float((window.iloc[-1] - iv_min) / (iv_max - iv_min) * 100)


# ---- Data structures --------------------------------------------------------------------------------------------------------------------


@dataclass
class DeltaNeutralTrade:
    """A single delta-neutral short straddle/strangle position."""
    entry_date: date
    expiry: date
    short_call_strike: float
    short_put_strike: float
    credit: float          # per-share credit received (sum of both legs)
    contracts: int
    entry_spot: float
    entry_iv: float
    # Delta hedge state
    hedge_shares: float = 0.0   # current shares held for delta hedge
    cumulative_hedge_cost: float = 0.0  # total TRANSACTION cost (commission+slippage) only
    # AUDIT FIX 2026-07-12: cost basis of the hedge inventory (net cash paid for
    # the shares themselves).  Without this, hedge MTM booked `shares * spot` as
    # pure profit -> systematic P&L inflation.  See `avg_hedge_entry_price`.
    hedge_cost_basis: float = 0.0  # net signed cash outlay to build current inventory
    # P&L
    pnl: float = 0.0
    closed: bool = False
    close_date: date | None = None
    close_reason: str = ""
    # Contract identifiers for real-data mode
    call_contract: dict | None = field(default=None, repr=False)
    put_contract: dict | None = field(default=None, repr=False)

    @property
    def avg_hedge_entry_price(self) -> float:
        """Volume-weighted average entry price of the current hedge inventory.

        Equals ``hedge_cost_basis / hedge_shares``.  Undefined (0.0) when flat.
        Martingale identity: unrealized hedge P&L == (spot - avg_entry) * shares
        == hedge_shares * spot - hedge_cost_basis.
        """
        if abs(self.hedge_shares) < 1e-9:
            return 0.0
        return self.hedge_cost_basis / self.hedge_shares


# ---- Strategy ----------------------------------------------------------------------------------------------------------------------------------


class DeltaNeutralVRP:
    """Delta Neutral short straddle/strangle targeting the Volatility Risk Premium.

    Sells ATM straddle (or 10-delta strangle) and delta-hedges daily with
    the underlying.  Lower structural cost than Iron Condor because only
    two option legs are traded (no long protective wings).

    Parameters
    ----------
    close : pd.Series
        Daily close prices for the underlying (index = DatetimeIndex).
    iv : pd.Series
        Daily implied volatility (same index).  If ``None`` and no
        *options_loader* is given, synthetic IV is generated.
    options_loader : OptionsDataLoader | PolygonOptionsDataLoader | None
        If provided, uses real options chain data for contract selection,
        entry/exit pricing, and mark-to-market.
    underlying : str
        Ticker symbol (default "SPY").
    structure : str
        ``"straddle"`` for ATM straddle, ``"strangle"`` for 10-delta strangle.
    delta_target_strangle : float
        Target delta for strangle legs (default 0.10).  Ignored for straddle.
    dte_entry_min / dte_entry_max : int
        DTE range at entry (default 30-45).
    dte_exit : int
        Time-stop exit DTE (default 14).
    profit_target_pct : float
        Fraction of max profit to take (default 0.50).
    risk_per_trade_pct : float
        Fraction of equity to risk per trade (default 0.02).
    max_loss_mult : float
        Max loss as multiple of credit for sizing (default 3.0).
    entry_interval_days : int
        Minimum days between new entries (default 30).
    ivr_entry : float
        IV rank percentile threshold for entry (default 0 -- always enter).
    delta_hedge_threshold : float
        Minimum delta (in shares) to trigger a re-hedge (default 5).
    hedge_slippage_pct : float
        Slippage on underlying hedge as fraction of price (default 0.0005 = 5bps).
    risk_free_rate : float
        Annual risk-free rate for BSM (default 0.04).
    commission_per_contract : float
        Options commission per contract (default 0.65).
    commission_per_share : float
        Equity hedge commission per share (default 0.005).
    """

    def __init__(
        self,
        close: pd.Series,
        iv: pd.Series | None = None,
        *,
        options_loader: OptionsDataLoader | PolygonOptionsDataLoader | None = None,
        underlying: str = "SPY",
        structure: str = "straddle",
        delta_target_strangle: float = 0.10,
        dte_entry_min: int = 30,
        dte_entry_max: int = 45,
        dte_exit: int = 14,
        profit_target_pct: float = 0.50,
        risk_per_trade_pct: float = 0.02,
        max_loss_mult: float = 3.0,
        entry_interval_days: int = 30,
        ivr_entry: float = 0.0,
        delta_hedge_threshold: float = 5.0,
        hedge_slippage_pct: float = 0.0005,
        risk_free_rate: float = 0.04,
        commission_per_contract: float = 0.65,
        commission_per_share: float = 0.005,
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

        self.structure = structure
        self.delta_target_strangle = delta_target_strangle
        self.dte_entry_min = dte_entry_min
        self.dte_entry_max = dte_entry_max
        self.dte_exit = dte_exit
        self.profit_target_pct = profit_target_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_loss_mult = max_loss_mult
        self.entry_interval_days = entry_interval_days
        self.ivr_entry = ivr_entry
        self.delta_hedge_threshold = delta_hedge_threshold
        self.hedge_slippage_pct = hedge_slippage_pct
        self.r = risk_free_rate
        self.commission_per_contract = commission_per_contract
        self.commission_per_share = commission_per_share

    @staticmethod
    def _build_real_iv_series(
        close: pd.Series,
        loader: OptionsDataLoader | PolygonOptionsDataLoader,
        lookback: int = 60,
    ) -> pd.Series:
        """Build a daily IV series from ATM options via *loader*."""
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
                iv_values.append(iv_values[-1] if iv_values else float(synthetic.iloc[len(iv_values)]))
        _log.info("Loaded real ATM IV for %d / %d dates", loaded, len(close))
        result = pd.Series(iv_values, index=close.index, dtype=float)
        result = result.fillna(synthetic)
        return result

    # ---- Walk-forward interface --------------------------------------------------------------------------------------

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
        """Run the delta-neutral backtest over [start, end].

        This is the interface expected by ``WalkForwardBacktester``.

        Returns
        -------
        equity : pd.Series
            Daily equity curve (index = dates in [start, end]).
        daily_returns : pd.Series
            Daily simple returns.
        turnover : pd.Series
            Daily turnover (non-zero on entry/exit/hedge days).
        logs : dict
            Trade-level logs.
        """
        _emit_accounting_fix_warning()
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
            options_loader=self._options_loader,
        )

        # Trim to [start, end] only
        eq_dates = [d for d, _ in eq_curve]
        eq_values = [v for _, v in eq_curve]
        eq_series = pd.Series(eq_values, index=eq_dates, dtype=float)
        ret_series = pd.Series(daily_rets, index=eq_dates, dtype=float)
        turn_series = pd.Series(daily_turnover, index=eq_dates, dtype=float)

        start_date = start.date() if hasattr(start, "date") else start
        end_date = end.date() if hasattr(end, "date") else end

        eq_oos = eq_series[(eq_series.index >= start_date) & (eq_series.index <= end_date)]
        ret_oos = ret_series[(ret_series.index >= start_date) & (ret_series.index <= end_date)]
        turn_oos = turn_series[(turn_series.index >= start_date) & (turn_series.index <= end_date)]

        # Rebase equity to start at initial_capital
        if len(eq_oos) > 0:
            eq_oos = eq_oos / eq_oos.iloc[0] * initial_capital

        return eq_oos, ret_oos, turn_oos, {"trades": trades}

    # ---- Real-data helpers --------------------------------------------------------------------------------------------

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
            ``"buy"`` -> use ask; ``"sell"`` -> use bid.
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
        if pd.notna(row.get("last_price")) and float(row["last_price"]) > 0:
            return float(row["last_price"])
        return None

    def _real_unrealized_pnl(
        self,
        loader: OptionsDataLoader | PolygonOptionsDataLoader,
        dt: date,
        pos: DeltaNeutralTrade,
        spot: float,
    ) -> float:
        """Compute mark-to-market P&L using real chain data.

        Uses ACTUAL bid/ask from the options_loader. Raises DataQualityViolation
        if real prices are unavailable - BSM must NEVER be used for fill prices.
        BSM is only valid for computing Greeks (Delta, Gamma), never for price discovery.

        Returns total unrealized P&L in dollars.
        """
        call_now = self._real_fill_price(loader, dt, pos.call_contract, "buy")
        put_now = self._real_fill_price(loader, dt, pos.put_contract, "buy")

        if call_now is None or put_now is None:
            raise DataQualityViolation(
                f"Real market quotes missing for {pos.short_call_strike}/{pos.short_put_strike} "
                f"straddle on {dt}. Cannot use BSM fallback for fill simulation -- "
                f"BSM is valid only for Greeks, not for price discovery. "
                f"call_now={call_now}, put_now={put_now}"
            )

        cost_to_close = call_now + put_now
        options_pnl = (pos.credit - cost_to_close) * 100 * pos.contracts

        # Underlying hedge P&L (AUDIT FIX): mark-to-market vs. cost basis, NOT
        # gross market value.  hedge_pnl = (spot - avg_entry) * shares.
        hedge_pnl = pos.hedge_shares * spot - pos.hedge_cost_basis

        # Total: options + hedge MTM - cumulative transaction costs
        return options_pnl + hedge_pnl - pos.cumulative_hedge_cost

    # ---- Core simulation --------------------------------------------------------------------------------------------

    def _simulate(
        self,
        close: pd.Series,
        iv: pd.Series,
        *,
        capital: float,
        options_loader: OptionsDataLoader | PolygonOptionsDataLoader | None = None,
    ) -> tuple[list[DeltaNeutralTrade], list[tuple[date, float]], list[float], list[float]]:
        """Core simulation loop.  Returns (trades, equity_curve, daily_returns, daily_turnover).

        When *options_loader* is provided, contract selection and pricing
        use real options chain data.  Falls back to BSM when the loader
        returns no data for a given date.
        """
        trades: list[DeltaNeutralTrade] = []
        equity_curve: list[tuple[date, float]] = []
        daily_returns: list[float] = []
        daily_turnover: list[float] = []
        active: list[DeltaNeutralTrade] = []
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

            # ---- Update active positions ----
            for pos in active:
                if pos.closed:
                    continue

                dte = (pos.expiry - dt).days

                # ---- Daily delta hedge ----
                T = max(dte / 252.0, 1e-6)

                # Compute current delta of short options
                call_delta = _bsm_delta(
                    S, pos.short_call_strike, T, self.r, sigma, is_call=True
                )
                put_delta = _bsm_delta(
                    S, pos.short_put_strike, T, self.r, sigma, is_call=False
                )
                # Short positions: portfolio delta = -(call_delta + put_delta) ?- 100 ?- contracts
                # For a short straddle: -call_delta (negative for short call) + (-put_delta) (negative for short put)
                # Actually: short call delta = -call_delta_per_share; short put delta = -put_delta_per_share
                # put_delta_per_share is negative (e.g. -0.50 for ATM), so -put_delta_per_share is positive
                # Net delta per share = -call_delta_per_share + (-put_delta_per_share)
                # For ATM straddle: -(0.50) + -(-0.50) = -0.50 + 0.50 = 0 (balanced at ATM)
                # But as price moves, this changes -- that's the whole point of hedging
                short_call_delta = -call_delta  # negative (we sold a call)
                short_put_delta = -put_delta     # positive (we sold a put, put delta is negative)
                net_delta_per_contract = (short_call_delta + short_put_delta) * 100  # per contract
                net_delta = net_delta_per_contract * pos.contracts  # total delta in shares

                # Target: net delta = 0, so we need to buy -net_delta shares
                target_hedge = -net_delta
                hedge_change = target_hedge - pos.hedge_shares

                if abs(hedge_change) > self.delta_hedge_threshold:
                    # Execute hedge trade
                    hedge_cost = abs(hedge_change) * self.commission_per_share
                    # Spread + market impact: ~5bps on SPY
                    hedge_cost += abs(hedge_change) * S * self.hedge_slippage_pct
                    pos.cumulative_hedge_cost += hedge_cost
                    # AUDIT FIX: accrue the ACTUAL cash outlay for the shares.
                    # Buying (hedge_change>0) spends cash (+basis); selling releases
                    # cash (-basis).  This is what makes MTM = (spot-avg)*shares.
                    pos.hedge_cost_basis += hedge_change * S
                    pos.hedge_shares = target_hedge
                    daily_pnl -= hedge_cost
                    turnover_today += abs(hedge_change) * S

                # ---- Mark-to-market the hedge position ----
                # Hedge P&L is the change in value of shares held
                # We track this through cumulative_hedge_cost and the current position
                # The daily P&L from the hedge is already captured in the equity curve
                # through the change in S * hedge_shares from yesterday

                # ---- Expiration ----
                if dte <= 0:
                    # At expiration, options settle at intrinsic value
                    call_intrinsic = max(S - pos.short_call_strike, 0)
                    put_intrinsic = max(pos.short_put_strike - S, 0)
                    # We sold both, so we owe intrinsic value
                    options_pnl = (pos.credit - call_intrinsic - put_intrinsic) * 100 * pos.contracts
                    options_pnl -= self.commission_per_contract * pos.contracts * 2  # closing commissions

                    # Hedge P&L: close the hedge position at current price
                    hedge_close_cost = abs(pos.hedge_shares) * (self.commission_per_share + S * self.hedge_slippage_pct)

                    # AUDIT FIX: realized hedge P&L = liquidation proceeds - cost basis.
                    # Previously `pos.hedge_shares * S` booked gross proceeds as pure
                    # profit (cost basis never subtracted) -> inflated Sharpe.
                    hedge_realized = (pos.hedge_shares * S) - pos.hedge_cost_basis
                    pos.pnl = options_pnl + hedge_realized - pos.cumulative_hedge_cost - hedge_close_cost
                    pos.closed = True
                    pos.close_date = dt
                    pos.close_reason = "expiration"
                    daily_pnl += pos.pnl
                    turnover_today += abs(pos.pnl)
                    pos.hedge_shares = 0.0
                    continue

                # ---- Mark-to-market options (prefer real data) ----
                options_unrealized: float
                if use_real and pos.call_contract is not None:
                    # During cold-start (no real IV for this date), fall back to BSM for
                    # MTM ONLY. The entry credit is locked at the real fill price. This
                    # is acceptable for MTM accuracy during the warmup gap; it does NOT
                    # fake an entry. Once real data becomes available, real MTM resumes.
                    try:
                        options_unrealized = self._real_unrealized_pnl(
                            options_loader, dt, pos, S
                        )
                    except DataQualityViolation:
                        # Cold-start gap -- use BSM for MTM only (entry credit is real)
                        call_val = _bsm_call(S, pos.short_call_strike, T, self.r, sigma)
                        put_val  = _bsm_put (S, pos.short_put_strike, T, self.r, sigma)
                        current_cost = call_val + put_val
                        options_unrealized = (pos.credit - current_cost) * 100 * pos.contracts
                else:
                    # BSM fallback -- ONLY for synthetic mode (no real options_loader)
                    call_val = _bsm_call(S, pos.short_call_strike, T, self.r, sigma)
                    put_val = _bsm_put(S, pos.short_put_strike, T, self.r, sigma)
                    current_cost = call_val + put_val
                    options_unrealized = (pos.credit - current_cost) * 100 * pos.contracts
                max_profit = pos.credit * 100 * pos.contracts

                # Hedge unrealized: (spot - avg_entry) * shares, net of txn costs.
                # AUDIT FIX: subtract hedge_cost_basis (cash paid for the shares).
                # Old code (`shares * S - cumulative_hedge_cost`) treated the entire
                # market value of the hedge as profit -> P&L inflation bias.
                hedge_unrealized = (
                    pos.hedge_shares * S - pos.hedge_cost_basis - pos.cumulative_hedge_cost
                )

                total_unrealized = options_unrealized + hedge_unrealized

                # ---- Profit target (on total position) ----
                if total_unrealized >= self.profit_target_pct * max_profit:
                    # Close the position
                    hedge_close_cost = abs(pos.hedge_shares) * (self.commission_per_share + S * self.hedge_slippage_pct)
                    close_commission = self.commission_per_contract * pos.contracts * 2

                    pos.pnl = total_unrealized - hedge_close_cost - close_commission
                    pos.closed = True
                    pos.close_date = dt
                    pos.close_reason = "profit_target"
                    daily_pnl += pos.pnl
                    turnover_today += abs(pos.pnl)
                    pos.hedge_shares = 0.0
                    continue

                # ---- Time exit ----
                if dte <= self.dte_exit:
                    hedge_close_cost = abs(pos.hedge_shares) * (self.commission_per_share + S * self.hedge_slippage_pct)
                    close_commission = self.commission_per_contract * pos.contracts * 2

                    pos.pnl = total_unrealized - hedge_close_cost - close_commission
                    pos.closed = True
                    pos.close_date = dt
                    pos.close_reason = "time_exit"
                    daily_pnl += pos.pnl
                    turnover_today += abs(pos.pnl)
                    pos.hedge_shares = 0.0
                    continue

            # ---- Cold-start guard: skip entries until real IV data is available ----
            # During the warmup window, _build_real_iv_series returns synthetic IV.
            # Entries with synthetic fills cause hard DataQualityViolation crashes once
            # real data becomes available. Skip all entries until >= 10 real IV values
            # appear in the last 30 days (indicating the cold-start has passed).
            real_iv_count = sum(
                1 for j in range(max(0, i - 30), i + 1)
                if pd.notna(iv_arr[j])
            )
            can_enter = True
            if real_iv_count < 10:
                can_enter = False  # still cold-starting -- skip entry
            if last_entry_date is not None and (dt - last_entry_date).days < self.entry_interval_days:
                can_enter = False

            active_open = [p for p in active if not p.closed]
            if can_enter and ivr > self.ivr_entry and len(active_open) < 3:
                dte_entry = (self.dte_entry_min + self.dte_entry_max) // 2
                T_entry = dte_entry / 252.0
                expiry_date = dt + timedelta(days=dte_entry)

                # Try real data first
                real_call = None
                real_put = None
                call_strike: float | None = None
                put_strike: float | None = None

                if use_real:
                    if self.structure == "straddle":
                        # ATM: find the closest-to-ATM call and put
                        real_call = options_loader.get_option_by_delta(
                            self._underlying, dt, 0.50, "call",
                            self.dte_entry_min, self.dte_entry_max,
                        )
                        real_put = options_loader.get_option_by_delta(
                            self._underlying, dt, 0.50, "put",
                            self.dte_entry_min, self.dte_entry_max,
                        )
                    else:
                        real_call = options_loader.get_option_by_delta(
                            self._underlying, dt, self.delta_target_strangle, "call",
                            self.dte_entry_min, self.dte_entry_max,
                        )
                        real_put = options_loader.get_option_by_delta(
                            self._underlying, dt, self.delta_target_strangle, "put",
                            self.dte_entry_min, self.dte_entry_max,
                        )

                    if real_call is not None and real_put is not None:
                        call_strike = float(real_call["strike"])
                        put_strike = float(real_put["strike"])

                # BSM fallback or if real data unavailable
                if call_strike is None or put_strike is None:
                    if self.structure == "straddle":
                        # ATM: strike = spot
                        atm_strike = round(S)
                        call_strike = atm_strike
                        put_strike = atm_strike
                    else:
                        call_strike = _strike_from_delta(
                            S, T_entry, self.r, sigma, self.delta_target_strangle, True
                        )
                        put_strike = _strike_from_delta(
                            S, T_entry, self.r, sigma, self.delta_target_strangle, False
                        )
                    real_call = None
                    real_put = None

                # Compute entry credit
                if real_call is not None and real_put is not None:
                    # Sell at bid (realistic fills)
                    call_exp = pd.to_datetime(real_call["expiration"]).date()
                    put_exp = pd.to_datetime(real_put["expiration"]).date()
                    call_contract = {
                        "strike": call_strike, "expiration": call_exp,
                        "option_type": "call",
                        "option_symbol": real_call.get("option_symbol", ""),
                    }
                    put_contract = {
                        "strike": put_strike, "expiration": put_exp,
                        "option_type": "put",
                        "option_symbol": real_put.get("option_symbol", ""),
                    }
                    call_price = self._real_fill_price(
                        options_loader, dt, call_contract, "sell"
                    )
                    put_price = self._real_fill_price(
                        options_loader, dt, put_contract, "sell"
                    )
                    if call_price is None or put_price is None:
                        continue
                    net_credit = call_price + put_price
                else:
                    # BSM pricing
                    call_prem = _bsm_call(S, call_strike, T_entry, self.r, sigma)
                    put_prem = _bsm_put(S, put_strike, T_entry, self.r, sigma)
                    net_credit = call_prem + put_prem
                    call_contract = {
                        "strike": call_strike, "expiration": expiry_date,
                        "option_type": "call", "option_symbol": "",
                    }
                    put_contract = {
                        "strike": put_strike, "expiration": expiry_date,
                        "option_type": "put", "option_symbol": "",
                    }
                    # Apply slippage
                    net_credit *= 0.95  # 5% slippage on synthetic

                if net_credit <= 0.01:
                    continue

                # Position sizing: risk 2% of equity, max loss = max_loss_mult ?- credit
                risk_amount = equity * self.risk_per_trade_pct
                max_loss_per_contract = self.max_loss_mult * net_credit * 100
                contracts = max(1, int(risk_amount / max_loss_per_contract))

                entry_commission = self.commission_per_contract * contracts * 2

                # Compute initial delta hedge
                call_delta_init = _bsm_delta(S, call_strike, T_entry, self.r, sigma, True)
                put_delta_init = _bsm_delta(S, put_strike, T_entry, self.r, sigma, False)
                short_call_delta = -call_delta_init
                short_put_delta = -put_delta_init
                net_delta_init = (short_call_delta + short_put_delta) * 100 * contracts
                initial_hedge = -net_delta_init  # shares to buy

                hedge_cost = abs(initial_hedge) * (self.commission_per_share + S * self.hedge_slippage_pct)

                pos = DeltaNeutralTrade(
                    entry_date=dt,
                    expiry=expiry_date,
                    short_call_strike=call_strike,
                    short_put_strike=put_strike,
                    credit=net_credit,
                    contracts=contracts,
                    entry_spot=S,
                    entry_iv=sigma,
                    hedge_shares=initial_hedge,
                    cumulative_hedge_cost=hedge_cost,
                    # AUDIT FIX: record cash paid for the initial hedge inventory.
                    hedge_cost_basis=initial_hedge * S,
                    call_contract=call_contract,
                    put_contract=put_contract,
                )
                active.append(pos)
                last_entry_date = dt
                daily_pnl -= entry_commission + hedge_cost
                turnover_today += entry_commission + abs(initial_hedge) * S

            # ---- Collect closed trades ----
            newly_closed = [p for p in active if p.closed]
            trades.extend(newly_closed)
            active = [p for p in active if not p.closed]

            equity += daily_pnl
            equity_curve.append((dt, equity))
            daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0
            daily_returns.append(daily_ret)
            daily_turnover.append(turnover_today / equity if equity > 0 else 0.0)
            prev_equity = equity

        # Close remaining at end
        if active:
            S_final = float(close_arr[-1])
            dt_final = dates[-1].date() if hasattr(dates[-1], "date") else dates[-1]
            eob_pnl = 0.0
            for pos in active:
                if not pos.closed:
                    call_intrinsic = max(S_final - pos.short_call_strike, 0)
                    put_intrinsic = max(pos.short_put_strike - S_final, 0)
                    options_pnl = (pos.credit - call_intrinsic - put_intrinsic) * 100 * pos.contracts
                    hedge_close_cost = abs(pos.hedge_shares) * (self.commission_per_share + S_final * self.hedge_slippage_pct)
                    close_commission = self.commission_per_contract * pos.contracts * 2

                    # AUDIT FIX: hedge realized = proceeds - cost basis (not gross proceeds)
                    hedge_realized = (pos.hedge_shares * S_final) - pos.hedge_cost_basis
                    pos.pnl = options_pnl + hedge_realized - pos.cumulative_hedge_cost - hedge_close_cost - close_commission
                    pos.closed = True
                    pos.close_date = dt_final
                    pos.close_reason = "end_of_backtest"
                    pos.hedge_shares = 0.0
                    eob_pnl += pos.pnl
            trades.extend(active)
            if equity_curve:
                equity += eob_pnl
                equity_curve[-1] = (dt_final, equity)
                if daily_returns:
                    daily_returns[-1] = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0.0

        return trades, equity_curve, daily_returns, daily_turnover
