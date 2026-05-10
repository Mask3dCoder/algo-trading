"""Post-earnings announcement drift (PEAD): SUE ranking, underreaction drift.

Ball & Brown (1968) and Bernard & Thomas (1989): markets underreact to
earnings surprises, causing drift lasting 1-3 months. Strategy ranks by
standardised unexpected earnings (SUE), longs positive surprises and
shorts negative ones, holding for the drift window.

Uses actual earnings surprise data when provided; when unavailable,
proxies SUE via price gaps (large overnight/opens relative to volatility)
which cluster around earnings events.
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


@register("equity_earnings_momentum")
class EarningsMomentumStrategy(Strategy):
    """Post-earnings announcement drift: SUE ranking, 1-3 month drift.

    Config params:
        lookback: window for SUE proxy (default 63)
        n_long: number of long positions (default 5)
        n_short: number of short positions (default 5)
        hold_days: holding period for drift (default 42)
        vol_lookback: window for volatility (default 63)
        rebalance_freq: rebalance in trading days (default 21)
        sue_z_threshold: minimum SUE z-score to trade (default 0.5)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback = int(config.params.get("lookback", 63))
        self.n_long = int(config.params.get("n_long", 5))
        self.n_short = int(config.params.get("n_short", 5))
        self.hold_days = int(config.params.get("hold_days", 42))
        self.vol_lookback = int(config.params.get("vol_lookback", 63))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.sue_z_threshold = float(config.params.get("sue_z_threshold", 0.5))
        self._days_since_rebalance = 0
        self._entry_days: dict[str, int] = {}
        self._current_weights: NDArray | None = None

    def _sue_proxy(self, close: NDArray) -> NDArray:
        """SUE proxy: standardised unusual return (SUR).

        Large price gaps relative to recent vol proxy earnings surprises.
        SUR = recent_return / realised_vol, z-scored against history.
        """
        n_periods, n_assets = close.shape
        sue = np.zeros(n_assets)
        lb = min(self.lookback, n_periods)

        for a in range(n_assets):
            prices = close[-lb:, a]
            valid = prices[~np.isnan(prices)]
            if len(valid) < 20:
                continue
            rets = np.diff(valid) / valid[:-1]
            recent_ret = np.mean(rets[-5:])  # 1-week return
            hist_vol = np.std(rets, ddof=1)
            if hist_vol > 1e-10:
                sue[a] = recent_ret / hist_vol
            else:
                sue[a] = 0.0

        return sue

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        signals: list[TradeSignal] = []

        # Check exits for held positions
        for sym in list(self._entry_days):
            self._entry_days[sym] += 1
            if self._entry_days[sym] >= self.hold_days:
                signals.append(TradeSignal(
                    symbol=sym, direction=0, strength=0.0, size=0.0,
                    metadata={"action": "pead_exit", "days_held": self._entry_days[sym]},
                ))
                self._entry_days.pop(sym)

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return signals

        if n_periods < self.lookback:
            return signals

        # Use actual SUE data when available
        earnings_surprise = data.get("earnings_surprise")
        if earnings_surprise is not None and earnings_surprise.ndim >= 1:
            if earnings_surprise.ndim == 2:
                sue = earnings_surprise[-1]
            else:
                sue = earnings_surprise
        else:
            sue = self._sue_proxy(close)

        # Z-score SUE across assets
        sue_std = np.std(sue, ddof=1)
        if sue_std > 1e-10:
            sue_z = (sue - np.mean(sue)) / sue_std
        else:
            sue_z = np.zeros_like(sue)

        # Filter by minimum surprise magnitude
        n_long_actual = min(self.n_long, n_assets)
        n_short_actual = min(self.n_short, n_assets - n_long_actual)

        # Rank by SUE
        ranks = np.argsort(np.argsort(sue_z))

        # Vol for inverse-vol weighing
        vol_start = max(0, n_periods - self.vol_lookback)
        daily_rets = np.zeros((n_periods - vol_start, n_assets))
        for i in range(vol_start + 1, n_periods):
            daily_rets[i - vol_start] = close[i] / close[i - 1] - 1.0
        vols = np.std(daily_rets, axis=0, ddof=1)
        vols[vols < 1e-10] = 1e-10

        long_mask = ranks >= (n_assets - n_long_actual)  # Top SUE
        short_mask = ranks < n_short_actual               # Bottom SUE

        # Only trade if SUE exceeds threshold
        long_mask &= sue_z > self.sue_z_threshold
        short_mask &= sue_z < -self.sue_z_threshold

        inv_vol = 1.0 / vols
        inv_vol[~long_mask & ~short_mask] = 0.0
        total_inv_vol = np.sum(inv_vol[long_mask]) + np.sum(inv_vol[short_mask])
        if total_inv_vol < 1e-10:
            return signals

        weights = np.zeros(n_assets)
        if np.any(long_mask):
            weights[long_mask] = inv_vol[long_mask] / np.sum(inv_vol[long_mask]) / 2.0
        if np.any(short_mask):
            weights[short_mask] = -inv_vol[short_mask] / np.sum(inv_vol[short_mask]) / 2.0

        self._current_weights = weights
        self._days_since_rebalance = 0

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        for i in range(n_assets):
            if abs(weights[i]) > 1e-8:
                direction = 1 if weights[i] > 0 else -1
                sym = symbols[i]
                signals.append(TradeSignal(
                    symbol=sym,
                    direction=direction,
                    strength=min(abs(weights[i]) * 2, 1.0),
                    size=abs(weights[i]) * float(self.config.max_position_notional),
                    metadata={"sue_z": float(sue_z[i]), "sue_rank": int(ranks[i])},
                ))
                if sym not in self._entry_days:
                    self._entry_days[sym] = 0

        return signals
