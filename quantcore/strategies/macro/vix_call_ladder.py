"""VIX Call Ladder — long-volatility hedge for the short-vol portfolio.

Buys 1-month VIX calls at 110% of current VIX, rolling monthly.
Cost target: 1.0–1.5% of NAV annually.
Size: Covers 50% of the notional exposure from options_delta_neutral + options_put_write.

Rationale:
The portfolio is 90%+ short-volatility (VRP). A vol spike event would be
catastrophic. This hedge provides crisis insurance at a known premium cost.

Backtest logic (standalone and portfolio-integrated):
- Entry: Buy VIX calls when VIX < threshold (to avoid buying at peaks)
- Strike: 1.10 × current_VIX
- DTE: 30 days
- Roll: 5 days before expiration
- Cost drag: Track actual premium paid vs. NAV
- Payout: When VIX > strike (typically during crisis), mark-to-market premium
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@dataclass
class _VIXCallLeg:
    """State for a single VIX call leg."""
    entry_bar: int
    vix_at_entry: float
    strike: float              # 1.10 × vix_at_entry
    dte: int                   # 30
    premium_paid: float          # Cost in $ (per 1000 notional, scaled later)
    contracts: int
    contract_id: str = ""      # VIX_{strike}_{dte}DTE_CALL


@register("vix_call_ladder")
class VIXCallLadderStrategy(Strategy):
    """Long-volatility hedge using VIX call ladders.

    Parameters (config):
        vix_strike_mult:       Strike as fraction of VIX (default 1.10)
        dte:                   Days to expiration (default 30)
        roll_dte:              Days before expiry to roll (default 5)
        max_cost_ann_pct:      Max annual cost as % of NAV (default 1.5%)
        target_cover_pct:      Notional coverage target (default 50% of short-vol exposure)
        vix_entry_threshold:   Only buy calls when VIX below this (default 25)
        contract_multiplier:   Notional per contract (default 1000)

    Cost tracking:
        total_premium_paid: Sum of all premiums paid (annual cost)
        max_payout:         Theoretical max if VIX → infinity
        current_hedge_ratio: Actual coverage / target coverage
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.vix_strike_mult = float(config.params.get("vix_strike_mult", 1.10))
        self.dte = int(config.params.get("dte", 30))
        self.roll_dte = int(config.params.get("roll_dte", 5))
        self.max_cost_ann_pct = float(config.params.get("max_cost_ann_pct", 0.015))
        self.target_cover_pct = float(config.params.get("target_cover_pct", 0.50))
        self.vix_entry_threshold = float(config.params.get("vix_entry_threshold", 25.0))
        self.contract_multiplier = int(config.params.get("contract_multiplier", 1000))

        # State
        self._active_legs: list[_VIXCallLeg] = []
        self._bar_count = 0
        self._total_premium_paid: float = 0.0
        self._total_payouts: float = 0.0
        self._hedge_history: list[dict] = []

    # ── Data helpers ────────────────────────────────────────────────────────

    def _extract_vix(self, data: dict[str, NDArray]) -> tuple[float, float]:
        """Return (vix_as_pct, vix_decimal)."""
        iv = data.get("implied_vol")
        vix_s = data.get("vix_series")

        if vix_s is not None and vix_s.size > 0:
            last_vix = float(vix_s[-1])
            return last_vix, last_vix / 100.0

        if iv is not None and iv.size > 0:
            if iv.ndim == 0:
                iv_val = iv.item()
            else:
                iv_val = iv[-1]
            if isinstance(iv_val, np.ndarray):
                iv_val = iv_val.item()
            vix_val = float(iv_val) * 100 if float(iv_val) < 2 else float(iv_val)
            return vix_val, float(iv_val)

        return 20.0, 0.20

    def _short_vol_exposure(self, data: dict[str, NDArray]) -> float:
        """Estimate current short-vol notional exposure from portfolio state.

        In production this comes from the portfolio aggregator.
        Fallback: use data['total_short_vol_notional'] if injected by runner.
        """
        short_vol = data.get("total_short_vol_notional")
        if short_vol is not None:
            if hasattr(short_vol, "item"):
                return float(short_vol.item())
            return float(short_vol)
        # Fallback: estimate from equity × target exposure
        equity = data.get("equity", np.array([1_000_000.0]))
        if hasattr(equity, "shape") and equity.ndim > 0:
            equity = float(equity[-1]) if equity.size > 0 else 1_000_000.0
        return float(equity) * 0.90  # 90% of portfolio in short vol

    def _compute_premium(self, vix: float, strike: float, dte: int) -> float:
        """Estimates premium per contract using a simple approximation.

        VIX call ≈ intrinsic + time value ≈ max(0, VIX - strike) + IV premium
        For OTM calls (strike > VIX), premium ≈ time value ≈ f(VIX, strike, T)

        Approximate using BSM-like approach with VIX as underlying.
        Actual would use VIX futures, but for backtest this is adequate.
        """
        from scipy.stats import norm
        T = dte / 252.0
        r = 0.045
        sigma = vix / 100.0  # VIX itself as vol

        # BSM call price with VIX as underlying
        d1 = (np.log(vix / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        call_price = vix * norm.cdf(d1) - strike * norm.cdf(d2) * np.exp(-r * T)
        call_price = max(call_price, 0.01 * vix)  # Minimum 1% of VIX as premium
        return float(call_price) * self.contract_multiplier

    # ── Main entry point ──────────────────────────────────────────────────────

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        # Get current VIX
        vix_pct, vix_dec = self._extract_vix(data)

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["VIX"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols

        # Total short-vol exposure to hedge
        short_vol_exposure = self._short_vol_exposure(data)
        target_hedge_notional = short_vol_exposure * self.target_cover_pct

        # Get current equity for cost cap
        equity_arr = data.get("equity")
        if equity_arr is not None:
            equity = float(equity_arr[-1]) if equity_arr.size > 0 else 1_000_000.0
        else:
            equity = 1_000_000.0

        # ── Expire / roll existing legs ──────────────────────────────────
        for leg in list(self._active_legs):
            elapsed = self._bar_count - leg.entry_bar
            dte_now = max(0, leg.dte - elapsed)

            if dte_now <= self.roll_dte:
                # Roll: close old, open new (handled as signals)
                signals.append(TradeSignal(
                    symbol=symbol,
                    direction=0,
                    instrument_type="option",
                    strength=0.0,
                    size=0.0,
                    metadata={
                        "action": "vix_call_roll",
                        "old_strike": leg.strike,
                        "old_vix": leg.vix_at_entry,
                        "old_dte": leg.dte,
                        "premium_paid": leg.premium_paid,
                        "reason": f"roll_dte={dte_now}",
                    },
                ))
                self._active_legs.remove(leg)

        # ── Check annual cost cap ────────────────────────────────────────
        # Cost budget per bar (assuming ~252 bars/year)
        annual_budget = equity * self.max_cost_ann_pct
        # Approximate elapsed since start of year
        cost_used = self._total_premium_paid
        cost_remaining = annual_budget - cost_used

        # ── Enter new leg if coverage needed and budget remains ──────────
        current_hedge_notional = sum(l.contracts * l.strike * self.contract_multiplier for l in self._active_legs)
        coverage_gap = target_hedge_notional - current_hedge_notional

        # Entry gate: VIX below threshold (cheap insurance)
        vix_cheap = vix_pct < self.vix_entry_threshold

        # Don't add legs if we're over budget or coverage is adequate
        if vix_cheap and coverage_gap > 0 and cost_remaining > 0:
            # Compute new leg
            strike = vix_pct * self.vix_strike_mult
            premium = self._compute_premium(vix_pct, strike, self.dte)

            if premium > 0:
                # Contracts needed to fill coverage gap
                notional_per_contract = strike * self.contract_multiplier
                needed_contracts = int(coverage_gap // notional_per_contract)
                # Also cap by remaining budget
                max_contracts_by_budget = int(cost_remaining // premium)
                contracts = min(needed_contracts, max_contracts_by_budget)
                contracts = max(1, contracts)

                total_premium = premium * contracts
                self._total_premium_paid += total_premium

                leg = _VIXCallLeg(
                    entry_bar=self._bar_count,
                    vix_at_entry=vix_pct,
                    strike=strike,
                    dte=self.dte,
                    premium_paid=total_premium,
                    contracts=contracts,
                    contract_id=f"VIX_{strike}_{self.dte}DTE_CALL",
                )
                self._active_legs.append(leg)

                signals.append(TradeSignal(
                    symbol=symbol,
                    direction=1,  # Long VIX call
                    instrument_type="option",
                    strength=1.0,
                    size=contracts * notional_per_contract,
                    metadata={
                        "action": "vix_call_entry",
                        "strike": strike,
                        "vix_at_entry": vix_pct,
                        "dte": self.dte,
                        "premium_paid": total_premium,
                        "contracts": contracts,
                        "coverage_gap": coverage_gap,
                        "cost_remaining": cost_remaining,
                        "hedge_ratio": current_hedge_notional / max(target_hedge_notional, 1),
                        "annual_budget_used_pct": cost_used / max(annual_budget, 1),
                    },
                ))

        self._bar_count += 1
        return signals

    def hedge_effectiveness(
        self,
        portfolio_returns: NDArray,
        hedge_returns: NDArray,
    ) -> dict:
        """Measure hedge effectiveness: correlation and tail-risk reduction."""
        if len(portfolio_returns) < 2 or len(hedge_returns) < 2:
            return {"correlation": np.nan, "tail_reduction": np.nan}

        corr = float(np.corrcoef(portfolio_returns, hedge_returns)[0, 1])

        # Tail reduction: how much does hedge reduce worst 1% of returns?
        port_tail = np.percentile(portfolio_returns, 1)
        hedge_tail = np.percentile(hedge_returns, 99)
        combined = portfolio_returns + hedge_returns
        combined_tail = np.percentile(combined, 1)

        tail_reduction = (port_tail - combined_tail) / max(abs(port_tail), 1e-10)

        return {
            "correlation": corr,
            "tail_reduction": max(0, tail_reduction),
            "port_tail_1pct": port_tail,
            "hedge_tail_99pct": hedge_tail,
            "combined_tail_1pct": combined_tail,
        }