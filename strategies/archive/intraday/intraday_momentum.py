"""Intraday cross-sectional momentum: relative strength on 5-30 min bars.

Ranks symbols by recent cumulative return, goes long winners / short losers,
weights by inverse volatility. Designed for resampled 5-min or 15-min bars.
Differently from the daily momentum strategy, this uses bar-based lookbacks
and more frequent rebalancing with tight intraday stops.
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


@register("intraday_momentum")
class IntradayMomentumStrategy(Strategy):
    """Cross-sectional intraday momentum with inverse-vol weighting.

    Config params:
        lookback_bars: bars for momentum calculation (default 12)
        skip_bars: most recent bars to skip (default 1)
        n_long: number of long positions (default 3)
        n_short: number of short positions (default 3)
        vol_lookback: bars for volatility estimation (default 20)
        rebalance_freq: bars between rebalances (default 5)
        stop_loss_pct: stop loss % from entry price (default 0.3)
        take_profit_pct: take profit % from entry price (default 0.6)
        min_strength: minimum momentum score to enter (default 0.15)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback_bars = int(config.params.get("lookback_bars", 12))
        self.skip_bars = int(config.params.get("skip_bars", 1))
        self.n_long = int(config.params.get("n_long", 3))
        self.n_short = int(config.params.get("n_short", 3))
        self.vol_lookback = int(config.params.get("vol_lookback", 20))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.stop_loss_pct = float(config.params.get("stop_loss_pct", 0.3))
        self.take_profit_pct = float(config.params.get("take_profit_pct", 0.6))
        self.min_strength = float(config.params.get("min_strength", 0.15))

        self._bars_since_rebalance = 0
        self._current_weights: NDArray | None = None
        self._entry_price: dict[str, float] = {}
        self._entry_direction: dict[str, int] = {}
        self._bars_held: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        signals: list[TradeSignal] = []

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # --- Exit checks (risk management) ---
        for sym, pos in list(self._positions.items()):
            if abs(pos) < 1e-8:
                self._entry_price.pop(sym, None)
                self._entry_direction.pop(sym, None)
                self._bars_held.pop(sym, None)
                continue

            if sym not in symbols:
                continue

            col = symbols.index(sym)
            price = float(close[-1, min(col, n_assets - 1)])
            entry_px = self._entry_price.get(sym, price)
            direction = self._entry_direction.get(sym, 0)
            self._bars_held[sym] = self._bars_held.get(sym, 0) + 1

            exit_reason = None

            if direction > 0:
                if price <= entry_px * (1 - self.stop_loss_pct / 100):
                    exit_reason = "stop_loss"
                elif price >= entry_px * (1 + self.take_profit_pct / 100):
                    exit_reason = "take_profit"
            elif direction < 0:
                if price >= entry_px * (1 + self.stop_loss_pct / 100):
                    exit_reason = "stop_loss"
                elif price <= entry_px * (1 - self.take_profit_pct / 100):
                    exit_reason = "take_profit"

            if exit_reason:
                signals.append(TradeSignal(
                    symbol=sym, direction=0, strength=0.0, size=0.0,
                    metadata={
                        "action": "intraday_exit",
                        "reason": exit_reason,
                        "bars_held": self._bars_held.get(sym, 0),
                    },
                ))
                self._entry_price.pop(sym, None)
                self._entry_direction.pop(sym, None)
                self._bars_held.pop(sym, None)

        # --- Rebalance check ---
        self._bars_since_rebalance += 1
        if self._bars_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return signals

        min_bars = max(self.lookback_bars, self.vol_lookback) + 1
        if n_periods < min_bars:
            return signals

        # Compute cumulative returns over lookback (excluding skip)
        end_idx = n_periods - self.skip_bars
        start_idx = n_periods - self.lookback_bars
        if end_idx <= start_idx:
            return signals

        cum_returns = np.ones(n_assets)
        for i in range(start_idx + 1, end_idx):
            rets = (close[i] - close[i - 1]) / close[i - 1]
            cum_returns *= (1.0 + rets)
        cum_returns -= 1.0

        # Volatility over vol_lookback
        vol_start = max(0, n_periods - self.vol_lookback)
        daily_rets = np.diff(close[vol_start:], axis=0) / close[vol_start:-1]
        vols = np.std(daily_rets, axis=0, ddof=1)
        vols[vols < 1e-10] = 1e-10

        # Rank by cumulative return
        ranks = np.argsort(np.argsort(cum_returns))

        n_long_actual = min(self.n_long, n_assets)
        n_short_actual = min(self.n_short, n_assets - n_long_actual)
        long_mask = ranks >= (n_assets - n_long_actual)
        short_mask = ranks < n_short_actual

        # Inverse-vol weights
        inv_vol = 1.0 / vols
        inv_vol[~long_mask & ~short_mask] = 0.0

        total_inv_vol = np.sum(inv_vol[long_mask]) + np.sum(inv_vol[short_mask])
        if total_inv_vol < 1e-10:
            self._current_weights = None
            return signals

        weights = np.zeros(n_assets)
        if np.any(long_mask):
            weights[long_mask] = inv_vol[long_mask] / np.sum(inv_vol[long_mask]) / 2.0
        if np.any(short_mask):
            weights[short_mask] = -inv_vol[short_mask] / np.sum(inv_vol[short_mask]) / 2.0

        self._current_weights = weights
        self._bars_since_rebalance = 0

        # Emit signals for non-zero weights
        for i in range(n_assets):
            w = weights[i]
            if abs(w) < self.min_strength:
                continue

            symbol = symbols[i] if i < len(symbols) else f"asset_{i}"
            price = float(close[-1, i])
            direction = 1 if w > 0 else -1

            size = abs(w) * float(self.config.max_position_notional) * float(self.config.risk_budget_pct)

            stop_loss = price * (1 - direction * self.stop_loss_pct / 100.0)
            take_profit = price * (1 + direction * self.take_profit_pct / 100.0)

            signals.append(TradeSignal(
                symbol=symbol,
                direction=direction,
                strength=float(abs(w)),
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "cum_return": float(cum_returns[i]),
                    "vol": float(vols[i]),
                    "rank": int(ranks[i]),
                },
            ))

            self._entry_price[symbol] = price
            self._entry_direction[symbol] = direction
            self._bars_held[symbol] = 0

        return signals
