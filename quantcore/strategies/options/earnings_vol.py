"""Earnings volatility strategy: straddle/strangle around earnings events.

Buys ATM straddle before earnings when implied move is underpriced
relative to historical realized moves. Exits after announcement.
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


@register("options_earnings_vol")
class EarningsVolStrategy(Strategy):
    """Earnings volatility: buy straddle when implied move < historical average.

    Config params:
        lookback_earnings: number of past earnings to average (default 8)
        implied_vs_historical_ratio: enter when implied/historical < this (default 0.85)
        days_before_entry: days before earnings to enter (default 5)
        days_after_exit: days after earnings to exit (default 1)
        max_position_pct: max position as % of portfolio (default 2)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback_earnings = int(config.params.get("lookback_earnings", 8))
        self.implied_vs_historical_ratio = float(
            config.params.get("implied_vs_historical_ratio", 0.85)
        )
        self.days_before_entry = int(config.params.get("days_before_entry", 5))
        self.days_after_exit = int(config.params.get("days_after_exit", 1))
        self.max_position_pct = float(config.params.get("max_position_pct", 2))
        self._pending_earnings: dict[str, dict] = {}

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

        signals: list[TradeSignal] = []
        symbols = data.get("symbols", ["asset_0"])
        symbol = symbols[0] if isinstance(symbols, list) else symbols
        IV = iv if iv is not None else np.full(n, 0.3)

        # Historical earnings moves (simulated from large daily moves)
        daily_rets = np.diff(close) / close[:-1]
        abs_rets = np.abs(daily_rets[-252:])
        threshold = np.percentile(abs_rets, 95)  # Top 5% = earnings-like
        earnings_moves = abs_rets[abs_rets > threshold]
        avg_historical_move = np.mean(earnings_moves) if len(earnings_moves) > 0 else 0.03

        # Implied move from IV: IV * sqrt(DTE/252)
        dte = self.days_before_entry
        implied_move = float(IV[-1]) * np.sqrt(dte / 252)

        # Check if implied move is underpriced
        ratio = implied_move / avg_historical_move if avg_historical_move > 1e-10 else 1.0

        # Simulate upcoming earnings event (in real system, this comes from calendar)
        has_upcoming_earnings = data.get("earnings_upcoming")
        if has_upcoming_earnings is not None:
            if has_upcoming_earnings.ndim > 0:
                has_upcoming_earnings = bool(has_upcoming_earnings[-1])
            else:
                has_upcoming_earnings = bool(has_upcoming_earnings)

        # Entry: buy straddle if implied < historical and earnings approaching
        if ratio < self.implied_vs_historical_ratio and has_upcoming_earnings:
            straddle_cost = 2 * implied_move * float(close[-1])  # Approx ATM straddle ≈ 2 * implied_move * S
            position_size = (
                self.config.max_position_notional
                * self.max_position_pct
                / 100
            )

            signals.append(TradeSignal(
                symbol=symbol,
                direction=1,  # Long straddle = long vol
                strength=0.7,
                size=position_size,
                metadata={
                    "action": "earnings_straddle_entry",
                    "implied_move": float(implied_move),
                    "historical_move": float(avg_historical_move),
                    "ratio": float(ratio),
                    "straddle_cost": float(straddle_cost),
                    "dte": dte,
                },
            ))
            self._pending_earnings[symbol] = {
                "entry_price": float(close[-1]),
                "entry_iv": float(IV[-1]),
                "days_after": 0,
            }

        # Exit: after earnings announcement
        for sym, pos in list(self._pending_earnings.items()):
            pos["days_after"] += 1
            if pos["days_after"] >= self.days_after_exit:
                # Realized move vs expected
                if pos["entry_price"] > 0:
                    realized_move = abs(float(close[-1]) - pos["entry_price"]) / pos["entry_price"]
                else:
                    realized_move = 0.0

                signals.append(TradeSignal(
                    symbol=sym, direction=0, strength=0.0, size=0.0,
                    metadata={
                        "action": "earnings_straddle_exit",
                        "realized_move": float(realized_move),
                        "implied_move_at_entry": pos.get("entry_iv", 0) * np.sqrt(self.days_before_entry / 252),
                    },
                ))
                del self._pending_earnings[sym]

        return signals
