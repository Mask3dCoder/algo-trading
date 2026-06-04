"""Tail hedge strategy: systematic SPX put buying for crash protection.

Buys OTM SPX puts (5-10% OTM, 60-90 DTE) when crash risk indicators
are elevated. Rolls before expiry. Targets convex payoff in tail events.
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


@register("options_tail_hedge")
class TailHedgeStrategy(Strategy):
    """SPX tail hedge: systematic OTM put buying when crash risk is elevated.

    Config params:
        otm_pct: how far OTM for puts (default 10)
        dte_entry: DTE at entry (default 90)
        dte_roll: roll when DTE reaches this (default 30)
        notional_budget_pct: % of portfolio for hedge budget (default 1)
        skew_entry_threshold: put skew level to enter/increase (default 1.5)
        vol_entry_threshold: VIX level to enter/increase (default 25)
        correlation_threshold: cross-asset correlation level (default 0.7)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.otm_pct = float(config.params.get("otm_pct", 10))
        self.dte_entry = int(config.params.get("dte_entry", 90))
        self.dte_roll = int(config.params.get("dte_roll", 30))
        self.notional_budget_pct = float(config.params.get("notional_budget_pct", 1))
        self.skew_entry_threshold = float(config.params.get("skew_entry_threshold", 1.5))
        self.vol_entry_threshold = float(config.params.get("vol_entry_threshold", 25))
        self.correlation_threshold = float(config.params.get("correlation_threshold", 0.7))
        self._active_puts: list[dict] = []

    def _compute_skew_proxy(self, iv: NDArray, window: int = 21) -> float:
        """Approximate put skew as IV change / price change ratio."""
        if len(iv) < window + 1:
            return 1.0
        recent_iv = iv[-window:]
        iv_trend = recent_iv[-1] - recent_iv[0]
        return abs(iv_trend) / np.std(recent_iv) + 1.0  # Normalized around 1

    def _compute_correlation_proxy(self, close: NDArray, benchmark: NDArray | None = None) -> float:
        """Cross-asset correlation proxy (uses autocorrelation as proxy)."""
        if len(close) < 60:
            return 0.5
        rets = np.diff(close[-61:]) / close[-61:-1]
        if np.std(rets) < 1e-10:
            return 0.0
        # Autocorrelation as proxy for systematic risk
        autocorr = np.corrcoef(rets[:-1], rets[1:])[0, 1]
        if np.isnan(autocorr):
            return 0.0
        return abs(autocorr)

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []
        if close.ndim == 2:
            close = close[:, 0]

        iv = data.get("implied_vol")
        if iv is not None and iv.ndim == 2:
            iv = iv[:, 0]

        n = len(close)
        if n < 60:
            return []

        IV = iv if iv is not None else np.full(n, 0.2)
        S = float(close[-1])
        sigma = float(IV[-1])

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["SPX"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols

        # Risk indicators
        vix_proxy = sigma * 100  # IV as % = VIX-like
        skew_proxy = self._compute_skew_proxy(IV)
        corr_proxy = self._compute_correlation_proxy(close)

        # Crash risk score (0-1)
        risk_score = 0.0
        risk_score += min(vix_proxy / self.vol_entry_threshold, 1.0) * 0.4
        risk_score += min(skew_proxy / self.skew_entry_threshold, 1.0) * 0.3
        risk_score += min(corr_proxy / self.correlation_threshold, 1.0) * 0.3

        # Entry: buy puts when crash risk is elevated
        if risk_score > 0.5 and len(self._active_puts) == 0:
            strike = S * (1 - self.otm_pct / 100)
            notional = self.config.max_position_notional * self.notional_budget_pct / 100
            n_puts = max(1, int(notional / (S * 100)))  # 100x multiplier

            self._active_puts.append({
                "symbol": symbol,
                "strike": strike,
                "dte": self.dte_entry,
                "entry_spot": S,
                "n_puts": n_puts,
            })

            signals.append(TradeSignal(
                symbol=symbol,
                direction=-1,  # Long puts = short delta
                instrument_type="option",
                strength=min(risk_score, 1.0),
                size=notional,
                stop_loss=None,
                metadata={
                    "action": "tail_hedge_entry",
                    "strike": strike,
                    "dte": self.dte_entry,
                    "risk_score": risk_score,
                    "otm_pct": self.otm_pct,
                    "n_puts": n_puts,
                },
            ))

        # Roll: when DTE gets low
        for pos in list(self._active_puts):
            pos["dte"] -= 1

            if pos["dte"] <= self.dte_roll:
                # Roll to new expiry
                new_strike = S * (1 - self.otm_pct / 100)
                signals.append(TradeSignal(
                    symbol=symbol,
                    direction=-1,
                    instrument_type="option",
                    strength=0.5,
                    size=self.config.max_position_notional * self.notional_budget_pct / 100,
                    metadata={
                        "action": "tail_hedge_roll",
                        "old_strike": pos["strike"],
                        "new_strike": new_strike,
                        "old_dte": pos["dte"],
                        "new_dte": self.dte_entry,
                    },
                ))
                pos["strike"] = new_strike
                pos["dte"] = self.dte_entry
                pos["entry_spot"] = S

            # Exit: crash risk subsided
            if risk_score < 0.3:
                signals.append(TradeSignal(
                    symbol=symbol, direction=0, instrument_type="option", strength=0.0, size=0.0,
                    metadata={
                        "action": "tail_hedge_exit",
                        "reason": "risk_subsided",
                        "risk_score": risk_score,
                    },
                ))
                self._active_puts.remove(pos)

        return signals
