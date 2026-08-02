"""Simple buy-and-hold strategy for sanity checking the backtest engine."""

from __future__ import annotations

from quantcore.backtest.engine import SignalData
from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("simple_buy_hold")
class SimpleBuyHoldStrategy(Strategy):
    """Buy and hold SPY. For testing the backtest engine."""

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self._has_position = False

    def on_bar(self, bar, portfolio) -> list[SignalData]:
        """Event-driven entry point."""
        sym = bar.symbol

        # Only trade SPY
        if sym != "SPY":
            return []

        # Buy on first bar
        if not self._has_position:
            self._has_position = True
            return [SignalData(
                symbol="SPY",
                direction="long",
                strength=1.0,
                strategy_id="simple_buy_hold",
                notional_size=float(self.config.max_position_notional or 100000),
            )]

        return []

    def on_data(self, data: dict) -> list[TradeSignal]:
        """Batch data entry point."""
        return []

    def on_fill(self, *args, **kwargs) -> None:
        """No-op — accepts both backtest (fill, portfolio) and live
        (symbol, fill_price, fill_qty, side) signatures.

        The engine calls ``on_fill(fill, portfolio)`` (its own Protocol) while
        the base ``Strategy`` class exposes the live ``on_fill(symbol, price,
        qty, side)``. This strategy doesn't react to fills, so it must tolerate
        either interface rather than inherit the mismatched base default.
        """
        return None
