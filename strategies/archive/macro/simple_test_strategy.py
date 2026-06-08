"""Simple buy-and-hold strategy for sanity checking the backtest engine."""

from __future__ import annotations

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register
from quantcore.backtest.engine import SignalData


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
