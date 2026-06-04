"""Put Write VRP strategy — sells OTM cash-secured puts on SPY.

Entry: Monthly (or ~21 trading days), when:
  1. VRP = IV – realized_vol  >  min_vrp (default 3%)
  2. Trend filter: position reduced 50% if SPY < 200-DMA
  3. Vol regime filter: position reduced 50% if VIX > 30

Pricing: Black-Scholes via enhanced VolatilitySurface with put-skew.
  - Receive 95% of theoretical mid (bid-ask haircut on entry)
  - Pay 105% of theoretical mid on exit (buy to close)
  - Commission: $0.65 / contract

Sizing: Fractional Kelly (0.25×) capped at 5% notional, with a
99th-percentile overnight gap overlay.  Cash-secured notional per
contract = strike × multiplier (100).

Exit: 50% of max credit, 21 DTE hard stop, or 2× credit stop-loss.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from quantcore.data.vol_surface import VolatilitySurface
from quantcore.math_models.options.bsm import bsm_put
from quantcore.portfolio.kelly_sizing import compute_kelly_from_returns
from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal position tracking
# ---------------------------------------------------------------------------


@dataclass
class _ShortPutPosition:
    """State for one short-put leg."""

    entry_bar: int
    contracts: int
    strike: float
    dte_at_entry: int
    entry_credit: float          # per-share credit actually received
    credit_after_costs: float    # per-share after spread + commission
    current_credit_mkt: float    # per-share mark-to-market
    iv_at_entry: float
    spot_at_entry: float
    trend_filter_active: bool    # was 200-DMA filter on at entry?
    vol_filter_active: bool      # was VIX > 30 filter on at entry?
    contract_id: str = ""        # unique leg identifier: SPY_{strike}_{dte}DTE_PUT


# ---------------------------------------------------------------------------
# PutWriteVRPStrategy
# ---------------------------------------------------------------------------


@register("options_put_write")
class PutWriteVRPStrategy(Strategy):
    """Sell OTM puts on SPY to collect the volatility risk premium.

    Config params (legacy + new):
        vix_entry:          Minimum VIX level (default 15)
        otm_pct:            Strike as fraction of spot (default 0.95 = 5% OTM)
        dte:                Days to expiration at entry (default 30)
        risk_per_trade:     Legacy scalar — overridden by Kelly + gap overlay
        max_positions:      Max concurrent short puts (default 3)
        use_enhanced_iv:    Use VolatilitySurface (default True)
        min_vrp:            Minimum VRP to enter, decimal (default 0.03)
        profit_take_pct:    Take-profit fraction of max credit (default 0.50)
        dte_exit:           Hard exit when DTE <= this (default 21)
        stop_loss_mult:     Stop when put value = credit × this (default 2.0)
        kelly_fraction:     Kelly multiplier (default 0.25)
        max_notional_pct:   Cap position as % of equity (default 0.05)
        gap_percentile:     Historical gap percentile for overlay (default 0.99)
        commission_per_contract:  $ per contract (default 0.65)
        bid_ask_spread_pct: Spread haircut on premium (default 0.05 = 5%)
        risk_free_rate:     For BSM (default 0.045)
        multiplier:         Option contract multiplier (default 100)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.vix_entry = float(config.params.get("vix_entry", 15.0))
        self.otm_pct = float(config.params.get("otm_pct", 0.95))
        self.dte = int(config.params.get("dte", 30))
        self.risk_per_trade = float(config.params.get("risk_per_trade", 0.02))
        self.max_positions = int(config.params.get("max_positions", 3))

        # New Phase-2 parameters
        self.use_enhanced_iv = bool(config.params.get("use_enhanced_iv", True))
        self.min_vrp = float(config.params.get("min_vrp", 0.03))
        self.profit_take_pct = float(config.params.get("profit_take_pct", 0.50))
        self.dte_exit = int(config.params.get("dte_exit", 21))
        self.stop_loss_mult = float(config.params.get("stop_loss_mult", 2.0))
        self.kelly_fraction = float(config.params.get("kelly_fraction", 0.25))
        self.max_notional_pct = float(config.params.get("max_notional_pct", 0.05))
        self.gap_percentile = float(config.params.get("gap_percentile", 0.99))
        self.commission_per_contract = float(config.params.get("commission_per_contract", 0.65))
        self.bid_ask_spread_pct = float(config.params.get("bid_ask_spread_pct", 0.05))
        self.risk_free_rate = float(config.params.get("risk_free_rate", 0.045))
        self.multiplier = int(config.params.get("multiplier", 100))
        self.ma_lookback = int(config.params.get("ma_lookback", 200))

        # State
        self._last_entry_bar: int | None = None
        self._active_positions: list[_ShortPutPosition] = []
        self._bar_count = 0
        self._return_history: list[float] = []
        self._trade_history: list[dict] = []

    # ------------------------------------------------------------------
    # on_data — main entry point
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        """Generate entry and exit signals for short puts.

        Expected data keys:
            close          : np.ndarray (n_periods, n_assets)
            implied_vol    : np.ndarray (n_periods, 1) — VIX proxy, optional
            symbols        : list[str]
            vix_series     : np.ndarray (n_periods,) — raw VIX, optional
            current_date   : date  — injected by runner, optional
        """
        close = data.get("close")
        if close is None or close.size == 0:
            return []

        n_periods = close.shape[0]
        if n_periods < 2:
            return []

        symbols = data.get("symbols", [])
        if not symbols:
            return []

        self._bar_count = n_periods
        spy_idx = self._spy_index(symbols)
        spot = float(close[-1, spy_idx])

        # ---- Realized vol (21-day) -------------------------------------
        rv = self._realized_vol(close[:, spy_idx], window=21)

        # ---- VIX / ATM IV -----------------------------------------------
        vix_val, iv_decimal = self._extract_vix(data, close, spy_idx, n_periods)

        # ---- VRP gate ----------------------------------------------------
        vrp = iv_decimal - rv
        if vrp < self.min_vrp:
            # VRP too thin — no entry, but still check exits
            return self._check_exits(spot, iv_decimal, close[:, spy_idx])

        # ---- Trend filter (200-DMA) --------------------------------------
        trend_ok = True
        ma200 = self._sma(close[:, spy_idx], self.ma_lookback)
        if ma200 is not None and spot < ma200 * 0.98:  # 2% buffer
            trend_ok = False

        # ---- Vol regime filter (VIX > 30) --------------------------------
        vol_ok = vix_val <= 30.0

        # ---- Enhanced vol surface (put skew) ----------------------------
        put_iv = iv_decimal
        if self.use_enhanced_iv:
            try:
                surface = VolatilitySurface(
                    atm_iv=iv_decimal,
                    vix_level=vix_val,
                )
                put_iv = surface.iv_at_strike_dte(
                    moneyness=self.otm_pct,
                    dte=float(self.dte),
                )
            except Exception:
                _log.warning("VolSurface failed — falling back to ATM IV")
                put_iv = iv_decimal

        # ---- BSM pricing -------------------------------------------------
        strike = spot * self.otm_pct
        T = self.dte / 365.0
        theoretical_mid = float(
            bsm_put(
                S=spot,
                K=strike,
                T=T,
                r=self.risk_free_rate,
                sigma=put_iv,
            )
        )

        # Bid-ask haircut: receive 95% of mid when selling
        entry_credit = theoretical_mid * (1.0 - self.bid_ask_spread_pct)

        # Per-contract economics
        credit_per_contract = entry_credit * self.multiplier
        commission = self.commission_per_contract
        net_credit_per_contract = credit_per_contract - commission
        if net_credit_per_contract <= 0:
            _log.warning("Net credit <= 0 after costs — skipping")
            return self._check_exits(spot, iv_decimal, close[:, spy_idx])

        # ---- Position sizing ---------------------------------------------
        # 1. Fractional Kelly from strategy return history
        contracts_kelly = self._kelly_contracts(
            net_credit_per_contract,
            spot,
            equity=data.get("equity", 100_000.0),
        )

        # 2. Max notional cap (cash-secured)
        equity = float(data.get("equity", 100_000.0))
        max_notional = equity * self.max_notional_pct
        max_contracts_notional = int(max_notional // (strike * self.multiplier))

        # 3. Gap overlay: 99th-percentile overnight gap
        gap_contracts = self._gap_overlay_contracts(
            close[:, spy_idx],
            strike,
            equity,
            spot,
        )

        # 4. Trend / vol regime reduction
        contracts = min(contracts_kelly, max_contracts_notional, gap_contracts)
        contracts = max(0, contracts)
        if not trend_ok:
            contracts = int(contracts * 0.5)
        if not vol_ok:
            contracts = int(contracts * 0.5)
        contracts = max(0, contracts)

        # ---- Entry cadence (monthly ~21 bars) ----------------------------
        signals: list[TradeSignal] = []
        current_bar = n_periods - 1
        should_enter = False
        if self._last_entry_bar is None or (current_bar - self._last_entry_bar) >= 21:
            if vix_val > self.vix_entry and len(self._active_positions) < self.max_positions:
                should_enter = contracts > 0

        # ---- Check exits first (priority) --------------------------------
        exit_signals = self._check_exits(spot, iv_decimal, close[:, spy_idx])
        signals.extend(exit_signals)

        # ---- Emit entry signal if passed all gates -------------------------
        if should_enter and contracts > 0:
            # Entry signal
            self._last_entry_bar = current_bar
            total_notional = contracts * strike * self.multiplier
            contract_id = f"SPY_{strike}_{self.dte}DTE_PUT"

            pos = _ShortPutPosition(
                entry_bar=current_bar,
                contracts=contracts,
                strike=strike,
                dte_at_entry=self.dte,
                entry_credit=entry_credit,
                credit_after_costs=net_credit_per_contract,
                current_credit_mkt=entry_credit,
                iv_at_entry=put_iv,
                spot_at_entry=spot,
                trend_filter_active=not trend_ok,
                vol_filter_active=not vol_ok,
                contract_id=contract_id,
            )
            self._active_positions.append(pos)

            # Strength = scaled VRP confidence
            strength = min(1.0, vrp / 0.06)
            strength = max(0.3, strength)

            signals.append(
                TradeSignal(
                    symbol="SPY",
                    direction=-1,  # short put = sell premium
                    instrument_type="option",
                    strength=strength,
                    size=total_notional,  # cash-secured notional
                    metadata={
                        "strategy": "put_write_vrp",
                        "instrument": "OPTION",
                        "option_type": "PUT",
                        "action": "SELL_TO_OPEN",
                        "contract_id": contract_id,
                        "underlying": "SPY",
                        "spot": spot,
                        "strike": strike,
                        "dte": self.dte,
                        "iv": put_iv,
                        "vix": vix_val,
                        "vrp": vrp,
                        "theoretical_mid": theoretical_mid,
                        "entry_credit": entry_credit,
                        "net_credit_per_contract": net_credit_per_contract,
                        "contracts": contracts,
                        "cash_secured_notional": total_notional,
                        "trend_filter": not trend_ok,
                        "vol_filter": not vol_ok,
                    },
                )
            )

        return signals

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

    def _check_exits(
        self,
        spot: float,
        iv: float,
        spy_prices: NDArray,
    ) -> list[TradeSignal]:
        """Evaluate all active positions for exit conditions."""
        signals: list[TradeSignal] = []
        positions_to_remove: list[_ShortPutPosition] = []

        for pos in self._active_positions:
            # Decay DTE by elapsed bars (approximate: 1 bar ≈ 1 trading day)
            elapsed = self._bar_count - pos.entry_bar
            dte_now = max(0, pos.dte_at_entry - elapsed)

            # Mark-to-market: current theoretical value of the short put
            if dte_now > 0:
                T_now = dte_now / 365.0
                current_theo = float(
                    bsm_put(
                        S=spot,
                        K=pos.strike,
                        T=T_now,
                        r=self.risk_free_rate,
                        sigma=iv,
                    )
                )
                # To close, pay 105% of mid (spread)
                pos.current_credit_mkt = current_theo * (1.0 + self.bid_ask_spread_pct)
            else:
                # Expiration
                pos.current_credit_mkt = max(0.0, pos.strike - spot)

            # 1. Profit target: retain ≥ 50% of max credit
            #    Short put: profit = entry_credit - current_value
            profit = (pos.entry_credit - pos.current_credit_mkt) * pos.contracts * self.multiplier
            max_profit_potential = pos.credit_after_costs * pos.contracts
            if max_profit_potential > 0 and profit >= max_profit_potential * self.profit_take_pct:
                signals.append(self._exit_signal(pos, "profit_target", spot, dte_now))
                positions_to_remove.append(pos)
                continue

            # 2. DTE hard stop
            if dte_now <= self.dte_exit:
                signals.append(self._exit_signal(pos, "dte_exit", spot, dte_now))
                positions_to_remove.append(pos)
                continue

            # 3. Stop loss: put value ≥ 2× entry credit
            if pos.current_credit_mkt >= pos.entry_credit * self.stop_loss_mult:
                signals.append(self._exit_signal(pos, "stop_loss", spot, dte_now))
                positions_to_remove.append(pos)
                continue

        for p in positions_to_remove:
            self._active_positions.remove(p)

        return signals

    def _exit_signal(
        self,
        pos: _ShortPutPosition,
        reason: str,
        spot: float,
        dte_now: int,
    ) -> TradeSignal:
        """Build a BUY_TO_CLOSE signal."""
        close_cost = pos.current_credit_mkt * pos.contracts * self.multiplier
        close_cost += self.commission_per_contract * pos.contracts
        pnl_estimate = (
            pos.entry_credit * pos.contracts * self.multiplier - close_cost
        )
        self._record_trade(pnl_estimate)

        return TradeSignal(
            symbol="SPY",
            direction=0,  # 0 = close / flatten (runner interprets as close position)
            instrument_type="option",
            strength=1.0,
            size=0.0,
            metadata={
                "strategy": "put_write_vrp",
                "instrument": "OPTION",
                "option_type": "PUT",
                "action": "BUY_TO_CLOSE",
                "underlying": "SPY",
                "spot": spot,
                "strike": pos.strike,
                "dte": dte_now,
                "contracts": pos.contracts,
                "contract_id": f"SPY_{pos.strike}_{pos.dte_at_entry}DTE_PUT",
                "exit_reason": reason,
                "entry_credit": pos.entry_credit,
                "close_credit": pos.current_credit_mkt,
                "pnl_estimate": pnl_estimate,
            },
        )

    # ------------------------------------------------------------------
    # Sizing helpers
    # ------------------------------------------------------------------

    def _kelly_contracts(
        self,
        net_credit_per_contract: float,
        spot: float,
        equity: float,
    ) -> int:
        """Fractional Kelly position size in contracts."""
        if not self._return_history:
            # No history yet — use fixed fraction of equity
            notional = equity * self.risk_per_trade
            return max(1, int(notional // (spot * self.otm_pct * self.multiplier)))

        # Use strategy's own daily return history as proxy
        returns = np.array(self._return_history[-252:])
        stats = compute_kelly_from_returns(
            returns,
            fraction=self.kelly_fraction,
            min_observations=30,
        )
        kelly_pct = max(0.0, min(stats.fractional_kelly, 0.20))
        notional = equity * kelly_pct
        return max(0, int(notional // (spot * self.otm_pct * self.multiplier)))

    def _gap_overlay_contracts(
        self,
        prices: NDArray,
        strike: float,
        equity: float,
        spot: float,
    ) -> int:
        """Size so that a gap through the strike costs ≤ 5% of equity."""
        if prices.shape[0] < 22:
            return 9999  # disable overlay until enough data

        log_rets = np.log(prices[1:] / prices[:-1])
        overnight_gaps = np.abs(log_rets)
        gap_threshold = float(
            np.quantile(overnight_gaps, self.gap_percentile)
        )
        # Minimum gap floor: at least 3% to avoid OTM put buffer disabling
        gap_threshold = max(gap_threshold, 0.03)

        # Distance from spot to strike
        distance_to_strike = (spot - strike) / spot  # positive for OTM puts
        if distance_to_strike <= 0:
            # Put is ATM or ITM — max constraint
            max_loss_per_contract = strike * self.multiplier
            max_tolerable = equity * self.max_notional_pct
            contracts = max(0, int(max_tolerable // max_loss_per_contract))
            return max(contracts, 1)

        # Gap needed to reach strike
        if gap_threshold <= distance_to_strike:
            return 9999  # even 99th %ile gap doesn't reach strike

        # Worst-case spot after gap
        adverse_spot = spot * (1.0 - gap_threshold)
        loss_per_contract = max(0.0, strike - adverse_spot) * self.multiplier
        if loss_per_contract <= 0:
            return 9999  # gap doesn't cause loss at strike

        max_tolerable = equity * self.max_notional_pct
        return max(0, int(max_tolerable // loss_per_contract))

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    def _spy_index(self, symbols: list[str]) -> int:
        for i, sym in enumerate(symbols):
            if "SPY" in str(sym).upper():
                return i
        return 0

    def _realized_vol(self, prices: NDArray, window: int = 21) -> float:
        if prices.shape[0] <= window:
            return 0.15
        log_rets = np.log(prices[1:] / prices[:-1])
        log_rets = log_rets[~np.isnan(log_rets)]
        if len(log_rets) < window:
            return 0.15
        rv = float(np.std(log_rets[-window:]) * np.sqrt(252))
        return max(0.05, rv)

    def _extract_vix(
        self,
        data: dict[str, NDArray],
        close: NDArray,
        spy_idx: int,
        n_periods: int,
    ) -> tuple[float, float]:
        """Return (vix_as_pct, iv_as_decimal)."""
        iv_array = data.get("implied_vol")
        vix_series = data.get("vix_series")

        # Prefer raw vix_series (from runner)
        if vix_series is not None and vix_series.size > 0:
            last_vix = float(vix_series[-1])
            return last_vix, last_vix / 100.0

        # Fallback: implied_vol array
        if iv_array is not None and iv_array.size > 0:
            if iv_array.ndim == 0:
                iv_val = iv_array.item()
            else:
                iv_val = iv_array[-1]
            if isinstance(iv_val, np.ndarray):
                iv_val = iv_val.item() if iv_val.size == 1 else iv_val[0]
            vix_val = float(iv_val) * 100  # convert from decimal to percentage
            return vix_val, float(iv_val)

        # Final fallback: estimate from realized vol
        rv = self._realized_vol(close[:, spy_idx], window=21)
        if n_periods > 21:
            log_rets = np.diff(np.log(close[:, spy_idx]))
            log_rets = log_rets[~np.isnan(log_rets)]
            if len(log_rets) > 21:
                rv = float(np.std(log_rets[-21:]) * np.sqrt(252))
                iv_decimal = rv * 1.2
                return iv_decimal * 100, iv_decimal
        return 20.0, 0.20

    def _sma(self, prices: NDArray, window: int) -> float | None:
        if prices.shape[0] < window:
            return None
        return float(np.mean(prices[-window:]))

    # ------------------------------------------------------------------
    # Trade recording (for Kelly + backtest)
    # ------------------------------------------------------------------

    def _record_trade(self, pnl_estimate: float) -> None:
        self._trade_history.append({"pnl_pct": pnl_estimate})
        # Keep rolling return history for Kelly (simplified: use PnL as proxy)
        # In a real system, we'd get daily mark-to-market from the broker
        if self._trade_history:
            daily_ret = pnl_estimate / 100_000.0  # normalized by nominal equity
            self._return_history.append(daily_ret)
            if len(self._return_history) > 252 * 3:
                self._return_history = self._return_history[-252 * 3:]
