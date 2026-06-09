"""Cross-sectional 12-1 month momentum strategy with inverse-volatility weighting.

Classic Jegadeesh & Titman (1993) momentum: rank assets by past 12-month return
skipping the most recent 1 month (to avoid short-term reversal), then weight
winners inversely by their recent volatility.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    StrategyStatus,
    TradeSignal,
)
from quantcore.strategies.registry import register


@register("equity_momentum")
class MomentumStrategy(Strategy):
    """Cross-sectional momentum: long winners, short losers, inv-vol weighted.

    .. deprecated::
    STRATEGY_STATUS = StrategyStatus.DISABLED

    Forensic audit (2026-05-09): Classic 12-1 momentum edge has decayed
    post-2000 as it became widely known and arbitraged.  OOS validation
    not completed (insufficient universe diversity).  Retained for reference.

    Config params:
        lookback_months: total lookback in months (default 12)
        skip_months: recent months to skip (default 1)
        n_long: number of long positions (default 5)
        n_short: number of short positions (default 5)
        vol_lookback: months for volatility estimation (default 6)
        rebalance_freq: rebalance frequency in trading days (default 21)
    """

    STRATEGY_STATUS: StrategyStatus = StrategyStatus.DISABLED

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.lookback_months = int(config.params.get("lookback_months", 12))
        self.skip_months = int(config.params.get("skip_months", 1))
        self.n_long = int(config.params.get("n_long", 5))
        self.n_short = int(config.params.get("n_short", 5))
        self.vol_lookback = int(config.params.get("vol_lookback", 6))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self._days_since_rebalance = 0
        self._current_weights: NDArray | None = None

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        signals: list[TradeSignal] = []

        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq and self._current_weights is not None:
            return signals

        lookback_days = self.lookback_months * 21
        skip_days = self.skip_months * 21
        vol_days = self.vol_lookback * 21

        if n_periods < lookback_days + 1:
            return signals

        # Cumulative returns over (lookback - skip) window
        end_idx = n_periods - skip_days
        start_idx = n_periods - lookback_days
        if end_idx <= start_idx:
            return signals

        cum_returns = np.ones(n_assets)
        for i in range(start_idx, end_idx):
            rets = close[i] / close[i - 1] - 1.0
            cum_returns *= (1.0 + rets)
        cum_returns -= 1.0

        # Volatility over vol_lookback
        vol_start = max(0, n_periods - vol_days)
        daily_rets = np.zeros((n_periods - vol_start, n_assets))
        for i in range(vol_start + 1, n_periods):
            daily_rets[i - vol_start] = close[i] / close[i - 1] - 1.0
        vols = np.std(daily_rets, axis=0, ddof=1)
        vols[vols < 1e-10] = 1e-10

        # Rank by cumulative return
        ranks = np.argsort(np.argsort(cum_returns))

        # Select top n_long and bottom n_short (mutually exclusive)
        n_long_actual = min(self.n_long, n_assets)
        n_short_actual = min(self.n_short, n_assets - n_long_actual)
        long_mask = ranks >= (n_assets - n_long_actual)
        short_mask = ranks < n_short_actual

        # Inverse-vol weights
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

        # Emit signals for non-zero weights
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        for i in range(n_assets):
            if abs(weights[i]) > 1e-8:
                direction = 1 if weights[i] > 0 else -1
                signals.append(TradeSignal(
                    symbol=symbols[i],
                    direction=direction,
                    strength=min(abs(weights[i]) * 2, 1.0),
                    size=abs(weights[i]) * float(self.config.max_position_notional),
                ))

        return signals
