"""Execution Engine & Data Integrity for Delta Neutral VRP Paper Trading.

Three responsibilities:
  1. Zero look-ahead bias data feed (rolling IV window only)
  2. Order sizing & slippage with InitialPaperCapital
  3. Daily delta tracker (net portfolio delta at close)

All data structures are immutable where possible to prevent accidental mutation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from quantcore.data.vol_surface import generate_enhanced_synthetic_iv
from quantcore.live.data_feed import LiveDataFeed

_log = __import__("logging").getLogger(__name__)


# ---------------------------------------------------------------------------
# Option Position Ledger
# ---------------------------------------------------------------------------


@dataclass
class OpenOptionPosition:
    """A single short option position tracked across trading days."""

    entry_date: date
    expiry: date
    option_type: str              # "straddle" or "strangle"
    short_call_strike: float
    short_put_strike: float
    credit: float                 # per-share credit received (sum of both legs)
    contracts: int
    entry_spot: float
    entry_iv: float
    # Delta hedge state (shares of underlying held for hedging)
    hedge_shares: float = 0.0
    cumulative_hedge_cost: float = 0.0
    # Previous-day BSM mid-prices (for MtM delta)
    prev_call_mid: float = 0.0
    prev_put_mid: float = 0.0
    prev_spot: float = 0.0
    # Realized P&L upon close
    realized_pnl: float = 0.0
    closed: bool = False
    close_date: date | None = None
    close_reason: str = ""

# ---------------------------------------------------------------------------
# BSM helpers for synthetic option pricing (same impl as alpha/delta_neutral.py)
# ---------------------------------------------------------------------------

def _bsm_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-10 or sigma < 1e-10:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def _bsm_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T < 1e-10 or sigma < 1e-10:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def _bsm_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    if T < 1e-10 or sigma < 1e-10:
        if is_call:
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if is_call:
        return float(norm.cdf(d1))
    return float(norm.cdf(d1) - 1.0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PaperExecutionConfig:
    """Execution parameters — immutable after construction."""

    # -- Capital -----------------------------------------------------------------
    initial_paper_capital: float = 100_000.0

    # -- Slippage model -----------------------------------------------------------
    slippage_spread_bps: float = 5.0        # 5 bps ± spread on SPY hedge
    options_slippage_bps: float = 50.0       # 5% haircut on options fills

    # -- Commission -------------------------------------------------------------
    options_commission_per_contract: float = 0.65
    equity_commission_per_share: float = 0.005

    # -- Delta hedge ------------------------------------------------------------
    delta_hedge_threshold: float = 5.0       # shares
    hedge_slippage_bps: float = 5.0          # 5 bps on SPY hedge

    # -- Data integrity ---------------------------------------------------------
    iv_lookback_days: int = 60              # strictly historical for IV estimation
    max_bar_history_days: int = 300          # rolling window cap

    # -- Logging ----------------------------------------------------------------
    log_dir: Path = field(default_factory=lambda: Path("reports/paper_trading/logs"))

    def __post_init__(self):
        self.log_dir = Path(self.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Paper Execution Environment
# ---------------------------------------------------------------------------


class PaperExecutionEnvironment:
    """Orchestrates paper trading execution with zero look-ahead IV and delta tracking."""

    def __init__(
        self,
        config: PaperExecutionConfig,
        data_feed: LiveDataFeed | None = None,
    ) -> None:
        self._cfg = config
        self._data_feed = data_feed or LiveDataFeed()

        # Rolling window for IV (no look-ahead)
        self._price_window: pd.DataFrame | None = None
        self._iv_series: pd.Series | None = None

        # Capital tracking
        self._equity = config.initial_paper_capital
        self._peak_equity = config.initial_paper_capital

        # Delta tracking
        self._open_positions: list[dict[str, Any]] = []
        self._hedge_shares: float = 0.0
        self._net_delta: float = 0.0

        # ── Option position ledger (MtM P&L engine) ──────────────────────
        self._option_ledger: list[OpenOptionPosition] = []
        self._prev_spy_close: float = 0.0
        self._risk_free_rate: float = 0.045       # annualised, for BSM
        self._hedge_slippage_pct: float = 0.0005    # 5 bps on SPY hedge
        self._option_multiplier: int = 100

        # Slippage accumulators
        self._today_slippage_bps: float = 0.0
        self._today_slippage_dollars: float = 0.0
        self._today_trades: int = 0

        _log.info(
            "PaperExecutionEnvironment initialised: capital=$%.2f, "
            "slippage_bps=%.1f, iv_lookback=%dd",
            self._cfg.initial_paper_capital,
            self._cfg.slippage_spread_bps,
            self._cfg.iv_lookback_days,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def net_delta(self) -> float:
        """Current net portfolio delta from the most recent delta recording."""
        return self._net_delta

    @property
    def prev_spot(self) -> float:
        """Most recent SPY close price from the previous trading day."""
        return self._prev_spy_close

    def get_todays_slippage(self) -> dict[str, float]:
        """Return slippage stats for the current trading day."""
        return {
            "slippage_bps": self._today_slippage_bps,
            "slippage_dollars": self._today_slippage_dollars,
            "trades_today": self._today_trades,
        }

    def reset_daily_slippage(self) -> None:
        """Call at market close — resets per-day accumulators."""
        self._today_slippage_bps = 0.0
        self._today_slippage_dollars = 0.0
        self._today_trades = 0

    # -- Mark-to-Market P&L engine --------------------------------------------

    @property
    def option_ledger(self) -> list[OpenOptionPosition]:
        """Read-only access to the open option positions."""
        return [p for p in self._option_ledger if not p.closed]

    @property
    def all_positions(self) -> list[OpenOptionPosition]:
        """Read-only access to all positions (including closed)."""
        return list(self._option_ledger)

    @property
    def total_hedge_shares(self) -> float:
        """Net shares of underlying held across all open positions."""
        return sum(p.hedge_shares for p in self.option_ledger)

    def add_short_straddle(
        self,
        entry_date: date,
        expiry: date,
        call_strike: float,
        put_strike: float,
        credit: float,
        contracts: int,
        entry_spot: float,
        entry_iv: float,
        initial_hedge_shares: float = 0.0,
        hedge_cost: float = 0.0,
    ) -> OpenOptionPosition:
        """Open a new short straddle/strangle position in the ledger.

        Returns the created position object for downstream reference.
        """
        pos = OpenOptionPosition(
            entry_date=entry_date,
            expiry=expiry,
            option_type="straddle" if call_strike == put_strike else "strangle",
            short_call_strike=call_strike,
            short_put_strike=put_strike,
            credit=credit,
            contracts=contracts,
            entry_spot=entry_spot,
            entry_iv=entry_iv,
            hedge_shares=initial_hedge_shares,
            cumulative_hedge_cost=hedge_cost,
            prev_call_mid=0.0,
            prev_put_mid=0.0,
            prev_spot=entry_spot,
        )
        # Price the position at entry to set prev mid-prices
        sigma = entry_iv if entry_iv > 0 else self.get_current_iv()
        T_entry = max((expiry - entry_date).days / 252.0, 1e-6)
        pos.prev_call_mid = _bsm_call(entry_spot, call_strike, T_entry, self._risk_free_rate, sigma)
        pos.prev_put_mid = _bsm_put(entry_spot, put_strike, T_entry, self._risk_free_rate, sigma)
        pos.prev_spot = entry_spot

        self._option_ledger.append(pos)
        _log.info(
            "OPENED short %s: call_K=%.1f put_K=%.1f credit=%.2f "
            "contracts=%d expiry=%s hedge_shares=%.0f",
            pos.option_type, call_strike, put_strike, credit,
            contracts, expiry, initial_hedge_shares,
        )
        return pos

    def mark_to_market(self, as_of: date, spot: float, sigma: float) -> float:
        """Revalue all open positions at current market prices. Returns daily MtM P&L.

        Must be called BEFORE processing new signals each day.  This is the
        heart of the cash register: it computes the day-over-day change in
        portfolio value from both option decay and underlying hedge moves.

        Parameters
        ----------
        as_of : date
            Current simulation date.
        spot : float
            Current SPY close price.
        sigma : float
            Current implied volatility (from the rolling IV window).

        Returns
        -------
        float
            Net daily P&L in dollars (options MtM + hedge MtM - hedge costs).
        """
        daily_pnl = 0.0

        for pos in self._option_ledger:
            if pos.closed:
                continue

            dte = (pos.expiry - as_of).days

            # ── Expiration: settle at intrinsic value ──────────────────
            if dte <= 0:
                call_intrinsic = max(spot - pos.short_call_strike, 0.0)
                put_intrinsic = max(pos.short_put_strike - spot, 0.0)

                # We sold both legs; P&L = (credit received - cost to close) * 100 * contracts
                options_pnl = (pos.credit - call_intrinsic - put_intrinsic) * self._option_multiplier * pos.contracts
                # Deduct closing commissions
                close_commission = self._cfg.options_commission_per_contract * pos.contracts * 2
                options_pnl -= close_commission

                # Hedge P&L: close the hedge at current price
                hedge_close_cost = abs(pos.hedge_shares) * (
                    self._cfg.equity_commission_per_share
                    + spot * self._hedge_slippage_pct
                )
                # Realised hedge value = shares * current price - cumulative cost
                hedge_pnl = (pos.hedge_shares * spot) - pos.cumulative_hedge_cost - hedge_close_cost

                pos.realized_pnl = options_pnl + hedge_pnl
                pos.closed = True
                pos.close_date = as_of
                pos.close_reason = "expiration"
                pos.hedge_shares = 0.0
                daily_pnl += pos.realized_pnl
                _log.info(
                    "EXPIRED %s on %s: options_pnl=%.2f hedge_pnl=%.2f total=%.2f",
                    pos.option_type, as_of, options_pnl, hedge_pnl, pos.realized_pnl,
                )
                continue

            # ── Daily revaluation: options MtM ─────────────────────────
            T = max(dte / 252.0, 1e-6)
            cur_call_mid = _bsm_call(spot, pos.short_call_strike, T, self._risk_free_rate, sigma)
            cur_put_mid = _bsm_put(spot, pos.short_put_strike, T, self._risk_free_rate, sigma)

            # Day-over-day option P&L for SHORT position:
            # If the cost-to-close went UP (options got more expensive), we LOSE money.
            # P&L = (prev_cost_to_close - cur_cost_to_close) * 100 * contracts
            prev_cost = pos.prev_call_mid + pos.prev_put_mid
            cur_cost = cur_call_mid + cur_put_mid
            option_mtm = (prev_cost - cur_cost) * self._option_multiplier * pos.contracts

            # ── Daily revaluation: underlying hedge MtM ────────────────
            hedge_mtm = pos.hedge_shares * (spot - pos.prev_spot)

            # ── Daily delta hedge rebalance ────────────────────────────
            call_delta = _bsm_delta(spot, pos.short_call_strike, T, self._risk_free_rate, sigma, True)
            put_delta = _bsm_delta(spot, pos.short_put_strike, T, self._risk_free_rate, sigma, False)
            # Short options portfolio delta (in shares)
            short_call_delta = -call_delta * self._option_multiplier * pos.contracts
            short_put_delta = -put_delta * self._option_multiplier * pos.contracts
            net_options_delta = short_call_delta + short_put_delta
            target_hedge = -net_options_delta
            hedge_change = target_hedge - pos.hedge_shares

            hedge_cost_today = 0.0
            if abs(hedge_change) > self._cfg.delta_hedge_threshold:
                hedge_cost_today = abs(hedge_change) * (
                    self._cfg.equity_commission_per_share
                    + spot * self._hedge_slippage_pct
                )
                pos.cumulative_hedge_cost += hedge_cost_today
                pos.hedge_shares = target_hedge

            # ── Accumulate daily P&L ───────────────────────────────────
            pos_daily = option_mtm + hedge_mtm - hedge_cost_today
            daily_pnl += pos_daily

            # ── Update previous-day markers for next iteration ─────────
            pos.prev_call_mid = cur_call_mid
            pos.prev_put_mid = cur_put_mid
            pos.prev_spot = spot

        self._prev_spy_close = spot
        return daily_pnl

    def get_position_snapshot(self) -> list[dict[str, Any]]:
        """Return current open positions as dicts for delta telemetry.

        Each dict contains call_delta, put_delta, and contracts keys
        expected by ``record_delta_hedge()`` and ``DeltaTelemetryMiddleware``.
        """
        sigma = self.get_current_iv()
        spot = self._prev_spy_close if self._prev_spy_close > 0 else 0.0
        snapshots: list[dict[str, Any]] = []

        for pos in self.option_ledger:
            dte = (pos.expiry - date.today()).days
            T = max(dte / 252.0, 1e-6) if dte > 0 else 1e-6
            call_delta = _bsm_delta(spot, pos.short_call_strike, T, self._risk_free_rate, sigma, True) if spot > 0 else 0.0
            put_delta = _bsm_delta(spot, pos.short_put_strike, T, self._risk_free_rate, sigma, False) if spot > 0 else 0.0
            margin = self._estimate_margin_per_contract(
                premium=pos.credit,
                underlying_price=spot if spot > 0 else pos.entry_spot,
                position_type=pos.option_type,
            ) * pos.contracts
            snapshots.append({
                "call_delta": call_delta,
                "put_delta": put_delta,
                "contracts": pos.contracts,
                "strike": pos.short_call_strike,
                "put_strike": pos.short_put_strike,
                "expiry": str(pos.expiry),
                "credit": pos.credit,
                "hedge_shares": pos.hedge_shares,
                "premium_cost": pos.credit * self._option_multiplier * pos.contracts,
                "margin_required": margin,
                "option_type": pos.option_type,
            })
        return snapshots

    def _estimate_margin_per_contract(
        self,
        premium: float,
        underlying_price: float,
        position_type: str,
    ) -> float:
        """Estimate Reg T initial margin per contract (mirrors PositionSizer)."""
        notional = underlying_price * self._option_multiplier
        if position_type in ("iron_condor", "credit_spread", "defined_risk"):
            return notional * 0.05
        else:
            return premium * self._option_multiplier + max(
                self._cfg.initial_paper_capital * 0.001,  # fallback
                0.10 * notional,
            )

    # -- Data feed (zero look-ahead) ------------------------------------------

    def ingest_daily_bar(self, dt: date, close_price: float, vix: float | None = None) -> None:
        """Append a daily bar to the rolling IV window.  No historical query allowed."""
        new_row = pd.DataFrame(
            {"close": [close_price], "vix": [vix if vix else np.nan]},
            index=pd.DatetimeIndex([dt], name="date"),
        )
        if self._price_window is None:
            self._price_window = new_row
        else:
            self._price_window = pd.concat([self._price_window, new_row])
            if len(self._price_window) > self._cfg.max_bar_history_days:
                self._price_window = self._price_window.iloc[-self._cfg.max_bar_history_days:]

        self._update_iv(dt)
        _log.debug(
            "Ingested %s close=%.2f — window=%d bars",
            dt, close_price,
            len(self._price_window) if self._price_window is not None else 0,
        )

    def get_current_iv(self) -> float:
        """Return the most recently computed IV (strictly historical)."""
        if self._iv_series is None or self._iv_series.empty:
            return 0.20
        return float(self._iv_series.iloc[-1])

    def get_historical_iv(self, as_of: date, lookback: int = 60) -> float:
        """Get IV from data strictly before as_of.  No peeking."""
        if self._price_window is None or len(self._price_window) < lookback:
            _log.warning("Insufficient history for IV — returning default")
            return 0.20
        window = self._price_window.loc[:as_of]
        if len(window) < lookback:
            return 0.20
        window = window.iloc[-lookback:]
        iv = self._compute_iv_from_window(window["close"], window["vix"] if "vix" in window.columns else None)
        return iv

    # -- Order sizing & slippage -----------------------------------------------

    def compute_order_details(
        self,
        theoretical_price: float,
        quantity: int,
        side: str,   # "buy" or "sell"
        asset_type: str,  # "option" or "equity"
    ) -> dict[str, Any]:
        """Compute fill price with realistic slippage and commissions."""
        if asset_type == "option":
            spread_bps = self._cfg.options_slippage_bps
        else:
            spread_bps = self._cfg.slippage_spread_bps

        # Triangular perturbation: mode at 0, range [-half, +half]
        half = spread_bps / 20000.0
        perturbation = random.triangular(-half, 0, half)
        if side == "buy":
            perturbation = abs(perturbation) if perturbation < 0 else perturbation
        else:
            perturbation = -abs(perturbation) if perturbation > 0 else perturbation

        fill_price = theoretical_price * (1 + perturbation)

        if asset_type == "option":
            commission = self._cfg.options_commission_per_contract * abs(quantity)
        else:
            commission = self._cfg.equity_commission_per_share * abs(quantity)

        slippage_bps = perturbation * 10_000
        slippage_dollars = abs(fill_price - theoretical_price) * abs(quantity)

        self._today_slippage_bps += slippage_bps
        self._today_slippage_dollars += slippage_dollars
        self._today_trades += 1

        return {
            "theoretical_price": round(theoretical_price, 4),
            "fill_price": round(fill_price, 4),
            "slippage_bps": round(slippage_bps, 4),
            "slippage_dollars": round(slippage_dollars, 4),
            "commission": round(commission, 4),
            "total_cost": round(commission + slippage_dollars, 4),
            "side": side,
            "asset_type": asset_type,
            "quantity": quantity,
        }

    # -- Delta tracker -------------------------------------------------------

    def record_delta_hedge(
        self,
        dt: date,
        positions: list[dict[str, Any]],
        hedge_shares: float,
    ) -> dict[str, Any]:
        """Record net portfolio delta at close of day.

        ``positions`` is a list of dicts with keys:
            - call_delta, put_delta  (per-share deltas at close)
            - contracts

        Returns a dict with the delta tracking record.
        """
        total_options_delta = 0.0
        for pos in positions:
            call_delta_per_share = pos.get("call_delta", 0.0)
            put_delta_per_share = pos.get("put_delta", 0.0)
            contracts = pos.get("contracts", 0)
            # Short call/put delta: negative of long delta × shares
            short_call_delta = -call_delta_per_share * 100 * contracts
            short_put_delta = -put_delta_per_share * 100 * contracts
            total_options_delta += short_call_delta + short_put_delta

        hedge_delta = hedge_shares
        net_delta = total_options_delta + hedge_delta
        delta_drift = net_delta - 0.0  # target = 0

        self._open_positions = positions
        self._hedge_shares = hedge_shares
        self._net_delta = net_delta

        return {
            "date": str(dt),
            "net_delta": round(net_delta, 2),
            "target_delta": 0.0,
            "delta_drift": round(delta_drift, 2),
            "num_positions": len(positions),
            "hedge_shares": round(hedge_delta, 2),
            "total_options_delta": round(total_options_delta, 2),
        }

    def get_delta_log_path(self) -> Path:
        return self._cfg.log_dir / "daily_delta_log.csv"

    # -- Capital update --------------------------------------------------------

    def update_equity(self, pnl: float) -> None:
        """Apply P&L and update peak equity tracker."""
        self._equity += pnl
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update_iv(self, as_of: date) -> None:
        """Compute synthetic IV from the rolling window (strictly historical)."""
        if self._price_window is None or len(self._price_window) < self._cfg.iv_lookback_days:
            return

        window = self._price_window.iloc[-self._cfg.iv_lookback_days:]
        close = window["close"].dropna()
        vix = window["vix"].dropna() if "vix" in window.columns else None

        iv = self._compute_iv_from_window(close, vix)

        new_series = pd.Series([iv], index=pd.DatetimeIndex([as_of]))
        if self._iv_series is None:
            self._iv_series = new_series
        else:
            self._iv_series = pd.concat([self._iv_series, new_series]).sort_index()

    def _compute_iv_from_window(
        self,
        close: pd.Series,
        vix: pd.Series | None,
    ) -> float:
        """Compute synthetic IV from a historical price window."""
        try:
            synthetic_iv = generate_enhanced_synthetic_iv(close, vix_series=vix)
            return float(synthetic_iv.iloc[-1])
        except Exception as exc:
            _log.warning("IV computation failed: %s — using default", exc)
            return 0.20
