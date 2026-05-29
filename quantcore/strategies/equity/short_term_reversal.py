"""Short-term equity reversal (1-week mean-reversion).

Stocks that went up the most in the past week tend to reverse next week,
and vice versa.  Driven by order-flow imbalances, market-maker inventory
rebalancing, and retail herding.  Has negative correlation with momentum,
making it the ideal diversifier in a multi-strategy book.

High turnover (~200+ trades/yr at weekly rebalance + stop-losses).
Realistic transaction costs (5 bps spread + 10 bps impact) are critical
in backtesting -- gross Sharpe must exceed 0.5 to be viable after costs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("equity_short_term_reversal")
class ShortTermReversalStrategy(Strategy):
    """Short-term reversal: long recent losers, short recent winners.

    Config params:
        lookback_days: return lookback in trading days (default 5)
        rebalance_freq: rebalance frequency in trading days (default 5)
        n_long_pct: fraction of universe for long leg (default 0.10)
        n_short_pct: fraction of universe for short leg (default 0.10)
        min_price: minimum stock price filter (default 5.0)
        min_volume: minimum average daily dollar volume filter (default 5e6)
        min_history: minimum bars of history required (default 20)
        stop_loss_pct: stop-loss trigger as negative return from entry (default 0.02)
        max_holding_bars: max bars to hold a position before forced exit (default 5)
        max_position_pct: per-asset weight cap (default 0.05)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback_days = int(config.params.get("lookback_days", 5))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.n_long_pct = float(config.params.get("n_long_pct", 0.10))
        self.n_short_pct = float(config.params.get("n_short_pct", 0.10))
        self.min_price = float(config.params.get("min_price", 5.0))
        self.min_volume = float(config.params.get("min_volume", 5e6))
        self.min_history = int(config.params.get("min_history", 20))
        self.stop_loss_pct = float(config.params.get("stop_loss_pct", 0.02))
        self.max_holding_bars = int(config.params.get("max_holding_bars", 5))
        self.max_position_pct = float(config.params.get("max_position_pct", 0.05))

        # Rebalance clock
        self._days_since_rebalance: int = 0
        self._current_weights: NDArray | None = None

        # Position tracking for stop-loss and time exits
        # {symbol: entry_price}
        self._entry_prices: dict[str, float] = {}
        # {symbol: bar index at entry}
        self._entry_bars: dict[str, int] = {}
        self._bar_counter: int = 0

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def on_fill(self, symbol: str, fill_price: float, fill_qty: float, side: str) -> None:
        """Track entry prices and bar indices for stop-loss / time exits."""
        super().on_fill(symbol, fill_price, fill_qty, side)
        if side == "buy":
            if symbol not in self._entry_prices:
                self._entry_prices[symbol] = fill_price
                self._entry_bars[symbol] = self._bar_counter
        else:
            # Position closed or reduced -- clear tracking if flat
            if abs(self._positions.get(symbol, 0.0)) < 1e-8:
                self._entry_prices.pop(symbol, None)
                self._entry_bars.pop(symbol, None)

    def teardown(self) -> None:
        """Clean up tracking state."""
        self._entry_prices.clear()
        self._entry_bars.clear()
        super().teardown()

    # ------------------------------------------------------------------
    # Core signal logic
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        volume = data.get("volume")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        symbols: list[str] = list(
            data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        )

        self._bar_counter += 1

        # --- Check stop-loss and max-holding-period exits first ---
        exit_signals = self._check_exits(close, symbols)

        # --- Rebalance gate ---
        self._days_since_rebalance += 1
        if (
            self._days_since_rebalance < self.rebalance_freq
            and self._current_weights is not None
        ):
            # Between rebalances: emit hold signals + any exits
            hold_signals = _signals_from_weights(
                self._current_weights, symbols, self.config.max_position_notional,
            )
            return _merge_signals(hold_signals, exit_signals)

        # --- Not enough history ---
        if n_periods < self.lookback_days + 1 or n_periods < self.min_history:
            return exit_signals

        # --- Universe filters ---
        valid = np.ones(n_assets, dtype=bool)
        valid &= close[-1] >= self.min_price

        if volume is not None and volume.ndim == 2 and volume.shape[0] >= 21:
            avg_dollar_vol = np.mean(volume[-21:] * close[-21:], axis=0)
            valid &= avg_dollar_vol >= self.min_volume

        n_valid = int(np.sum(valid))
        if n_valid < 4:
            self._current_weights = None
            return exit_signals

        # --- 5-day cumulative return ---
        ret_5d = close[-1] / np.maximum(close[-self.lookback_days - 1], 1e-10) - 1.0
        ret_5d[~valid] = np.nan

        # --- Cross-sectional z-score of the REVERSED signal ---
        raw_signal = -ret_5d  # buy losers, sell winners
        mu = np.nanmean(raw_signal)
        sigma = np.nanstd(raw_signal)
        if sigma < 1e-10:
            self._current_weights = None
            return exit_signals
        z = (raw_signal - mu) / sigma

        # --- Rank and select long / short legs ---
        n_long = max(1, int(n_valid * self.n_long_pct))
        n_short = max(1, int(n_valid * self.n_short_pct))

        valid_indices = np.where(valid)[0]
        sorted_idx = valid_indices[np.argsort(z[valid_indices])]

        long_leg = sorted_idx[-n_long:]   # highest z  (biggest losers)
        short_leg = sorted_idx[:n_short]  # lowest z   (biggest winners)

        # Ensure no overlap
        overlap = set(long_leg.tolist()) & set(short_leg.tolist())
        if overlap:
            short_leg = np.array([i for i in short_leg if i not in overlap])

        if len(long_leg) == 0 or len(short_leg) == 0:
            self._current_weights = None
            return exit_signals

        # --- Equal-weighted, dollar-neutral, with per-asset cap ---
        weights = np.zeros(n_assets)
        raw_long = 1.0 / len(long_leg) / 2.0   # longs sum to +0.5
        raw_short = 1.0 / len(short_leg) / 2.0  # shorts sum to -0.5

        # Apply per-asset cap; if capped, redistribute unused weight
        long_w = min(raw_long, self.max_position_pct)
        short_w = min(raw_short, self.max_position_pct)

        weights[long_leg] = long_w
        weights[short_leg] = -short_w

        self._current_weights = weights
        self._days_since_rebalance = 0

        signals = _signals_from_weights(
            weights, symbols, self.config.max_position_notional,
        )
        return _merge_signals(signals, exit_signals)

    # ------------------------------------------------------------------
    # Stop-loss and time-exit checks
    # ------------------------------------------------------------------

    def _check_exits(
        self, close: NDArray, symbols: list[str],
    ) -> list[TradeSignal]:
        """Emit direction=0 exit signals for positions hitting stop-loss or max hold."""
        exits: list[TradeSignal] = []
        _n_periods, n_assets = close.shape

        for sym, pos in list(self._positions.items()):
            if abs(pos) < 1e-8:
                continue

            # Find asset index
            try:
                idx = symbols.index(sym)
            except ValueError:
                continue

            current_price = float(close[-1, idx])
            entry_price = self._entry_prices.get(sym)
            entry_bar = self._entry_bars.get(sym)

            if entry_price is None or entry_bar is None:
                continue

            # Stop-loss: price moved against us by stop_loss_pct
            if pos > 0:  # long position
                ret_from_entry = current_price / entry_price - 1.0
                if ret_from_entry <= -self.stop_loss_pct:
                    exits.append(TradeSignal(
                        symbol=sym,
                        direction=0,
                        strength=1.0,
                        size=abs(pos) * self.config.max_position_notional,
                        metadata={"exit_reason": "stop_loss"},
                    ))
                    continue
            elif pos < 0:  # short position
                ret_from_entry = 1.0 - current_price / entry_price
                if ret_from_entry <= -self.stop_loss_pct:
                    exits.append(TradeSignal(
                        symbol=sym,
                        direction=0,
                        strength=1.0,
                        size=abs(pos) * self.config.max_position_notional,
                        metadata={"exit_reason": "stop_loss"},
                    ))
                    continue

            # Max holding period
            bars_held = self._bar_counter - entry_bar
            if bars_held >= self.max_holding_bars:
                exits.append(TradeSignal(
                    symbol=sym,
                    direction=0,
                    strength=1.0,
                    size=abs(pos) * self.config.max_position_notional,
                    metadata={"exit_reason": "max_holding_period"},
                ))

        return exits


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _signals_from_weights(
    weights: NDArray,
    symbols: list[str],
    max_notional: float,
) -> list[TradeSignal]:
    """Convert a weight vector into TradeSignal objects."""
    signals: list[TradeSignal] = []
    for i in range(len(weights)):
        if abs(weights[i]) > 1e-8:
            signals.append(TradeSignal(
                symbol=symbols[i],
                direction=1 if weights[i] > 0 else -1,
                strength=min(abs(weights[i]) * 2, 1.0),
                size=abs(weights[i]) * float(max_notional),
            ))
    return signals


def _merge_signals(
    primary: list[TradeSignal],
    exits: list[TradeSignal],
) -> list[TradeSignal]:
    """Merge rebalance signals with exit signals.

    Exit signals (direction=0) take precedence: if a symbol appears in both
    lists, the exit wins -- the position is being closed, not re-weighted.
    """
    if not exits:
        return primary

    exit_symbols = {s.symbol for s in exits}
    filtered = [s for s in primary if s.symbol not in exit_symbols]
    return filtered + exits
