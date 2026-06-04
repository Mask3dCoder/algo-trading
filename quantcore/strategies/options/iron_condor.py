"""Iron condor options strategy: 16-delta wings, 30-45 DTE, with adjustments.

Sells OTM put spread + OTM call spread when IV rank is elevated.
Monitors for wing adjustments when price approaches short strikes.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    TradeSignal,
)
from quantcore.strategies.registry import register


@register("options_iron_condor")
class IronCondorStrategy(Strategy):
    """Iron condor: sell 16-delta wings, 30-45 DTE, with roll/adjust logic.

    Config params:
        delta_short: short strike delta target (default 16)
        delta_wing_width: spread width in deltas (default 10)
        dte_entry_min: min DTE to open (default 30)
        dte_entry_max: max DTE to open (default 45)
        ivr_entry: IV rank threshold to enter (default 40)
        ivr_exit: IV rank threshold to exit (default 20)
        profit_target_pct: close at % of max profit (default 50)
        max_loss_pct: close at % of max loss (default 200)
        adjustment_threshold: adjust when underlying within X% of short strike (default 3)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.delta_short = float(config.params.get("delta_short", 16))
        self.delta_wing_width = float(config.params.get("delta_wing_width", 10))
        self.dte_entry_min = int(config.params.get("dte_entry_min", 30))
        self.dte_entry_max = int(config.params.get("dte_entry_max", 45))
        self.ivr_entry = float(config.params.get("ivr_entry", 40))
        self.ivr_exit = float(config.params.get("ivr_exit", 20))
        self.profit_target_pct = float(config.params.get("profit_target_pct", 50))
        self.max_loss_pct = float(config.params.get("max_loss_pct", 200))
        self.adjustment_threshold = float(config.params.get("adjustment_threshold", 3))
        self._active_condors: list[dict] = []

    def _delta_to_strike(
        self,
        S: float,
        T: float,
        sigma: float,
        delta_target: float,
        is_call: bool = True,
    ) -> float:
        """Approximate the strike corresponding to a given delta using BSM inversion."""
        from scipy.stats import norm

        r = 0.05
        if T < 1e-10 or sigma < 1e-10:
            return S

        if is_call:
            # Call delta = N(d1), N(d1) = delta_target/100
            target = delta_target / 100.0
            try:
                d1_target = norm.ppf(np.clip(target, 0.001, 0.999))
                K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
                return K
            except (ValueError, OverflowError):
                return S * (1 + sigma * np.sqrt(T) * 0.5)
        else:
            # Put delta = N(d1) - 1, so N(d1) = 1 - delta_target/100
            target = 1.0 - delta_target / 100.0
            try:
                d1_target = norm.ppf(np.clip(target, 0.001, 0.999))
                K = S * np.exp(-d1_target * sigma * np.sqrt(T) + (r + 0.5 * sigma**2) * T)
                return K
            except (ValueError, OverflowError):
                return S * (1 - sigma * np.sqrt(T) * 0.5)

    def _compute_ivr(self, iv_history: NDArray) -> float:
        valid = iv_history[~np.isnan(iv_history)]
        if len(valid) < 20:
            return 50.0
        iv_min, iv_max = np.min(valid), np.max(valid)
        if iv_max - iv_min < 1e-10:
            return 50.0
        return float((iv_history[-1] - iv_min) / (iv_max - iv_min) * 100)

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        iv = data.get("implied_vol")

        if close is None:
            return []

        if close.ndim == 2:
            close = close[:, 0]
        if iv is not None and iv.ndim == 2:
            iv = iv[:, 0]

        n = len(close)
        if n < 20:
            return []

        IV = iv if iv is not None else np.full(n, 0.3)
        S = float(close[-1])
        sigma = float(IV[-1])

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["asset_0"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols

        iv_rank = self._compute_ivr(IV[-252:] if n >= 252 else IV)

        # Entry: open iron condor when IV rank is elevated
        if iv_rank > self.ivr_entry and len(self._active_condors) == 0:
            dte = (self.dte_entry_min + self.dte_entry_max) // 2
            T = dte / 252.0

            # Compute strikes from deltas
            short_put_strike = self._delta_to_strike(S, T, sigma, self.delta_short, is_call=False)
            long_put_strike = self._delta_to_strike(
                S, T, sigma, self.delta_short + self.delta_wing_width, is_call=False,
            )
            short_call_strike = self._delta_to_strike(S, T, sigma, self.delta_short, is_call=True)
            long_call_strike = self._delta_to_strike(
                S, T, sigma, self.delta_short + self.delta_wing_width, is_call=True,
            )

            # Net premium (approximate: short strikes - long strikes, normalized)
            put_spread_width = short_put_strike - long_put_strike
            call_spread_width = long_call_strike - short_call_strike
            max_profit = min(put_spread_width, call_spread_width)

            self._active_condors.append({
                "symbol": symbol,
                "short_put": short_put_strike,
                "long_put": long_put_strike,
                "short_call": short_call_strike,
                "long_call": long_call_strike,
                "dte": dte,
                "entry_price": S,
                "entry_iv": sigma,
                "max_profit": max_profit,
                "entry_ivr": iv_rank,
            })

            signals.append(TradeSignal(
                symbol=symbol,
                direction=0,  # Delta-neutral
                instrument_type="option",
                strength=0.7,
                size=self.config.max_position_notional * self.config.risk_budget_pct,
                metadata={
                    "action": "iron_condor_entry",
                    "short_put": short_put_strike,
                    "short_call": short_call_strike,
                    "iv_rank": iv_rank,
                    "dte": dte,
                },
            ))

        # Monitor and adjust existing condors
        for condor in list(self._active_condors):
            condor["dte"] -= 1

            # Check if price approaching short strikes
            dist_to_put = (S - condor["short_put"]) / S * 100
            dist_to_call = (condor["short_call"] - S) / S * 100

            # Adjustment: roll threatened side
            if dist_to_put < self.adjustment_threshold:
                # Roll put spread down and out
                new_dte = max(condor["dte"] + 7, self.dte_entry_min)
                T_new = new_dte / 252.0
                new_short_put = self._delta_to_strike(
                    S, T_new, sigma, self.delta_short, is_call=False,
                )
                signals.append(TradeSignal(
                    symbol=symbol,
                    direction=0,
                    instrument_type="option",
                    strength=0.4,
                    size=0.0,
                    metadata={
                        "action": "adjust_put_side",
                        "old_strike": condor["short_put"],
                        "new_strike": new_short_put,
                        "dist_pct": dist_to_put,
                    },
                ))
                condor["short_put"] = new_short_put

            elif dist_to_call < self.adjustment_threshold:
                new_dte = max(condor["dte"] + 7, self.dte_entry_min)
                T_new = new_dte / 252.0
                new_short_call = self._delta_to_strike(
                    S, T_new, sigma, self.delta_short, is_call=True,
                )
                signals.append(TradeSignal(
                    symbol=symbol,
                    direction=0,
                    instrument_type="option",
                    strength=0.4,
                    size=0.0,
                    metadata={
                        "action": "adjust_call_side",
                        "old_strike": condor["short_call"],
                        "new_strike": new_short_call,
                        "dist_pct": dist_to_call,
                    },
                ))
                condor["short_call"] = new_short_call

            # Exit conditions
            exit_reason = None
            if condor["dte"] <= 7:
                exit_reason = "near_expiry"
            elif iv_rank < self.ivr_exit:
                exit_reason = "ivr_low"

            if exit_reason:
                signals.append(TradeSignal(
                    symbol=symbol, direction=0, instrument_type="option", strength=0.0, size=0.0,
                    metadata={"action": "iron_condor_exit", "reason": exit_reason},
                ))
                self._active_condors.remove(condor)

        return signals
