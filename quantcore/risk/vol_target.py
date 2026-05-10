"""Volatility targeting: scale portfolio leverage to hit a target volatility.

Provides both a stateful VolTarget tracker (for live/incremental use) and a
standalone compute_leverage() function (for one-shot or batch use).

Leverage range is [0.5, 2.0] — below 1.0 reduces exposure in high-vol periods,
above 1.0 scales up in calm regimes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass
class VolTargetConfig:
    target_vol: float = 0.12
    short_window: int = 20
    long_window: int | None = 60
    min_leverage: float = 0.5
    max_leverage: float = 2.0
    use_ewma: bool = True
    ewma_decay: float = 0.94
    min_observations: int = 10


@dataclass
class VolTargetState:
    leverage: float
    short_vol: float
    long_vol: float | None
    effective_vol: float = 0.0
    n_observations: int = 0


class VolTarget:
    """Stateful volatility targeting tracker.

    Call update() each bar with the portfolio return. Call compute_leverage()
    to get the current leverage multiplier.
    """

    def __init__(self, config: VolTargetConfig | None = None) -> None:
        self.config = config or VolTargetConfig()
        self._returns: list[float] = []
        self._ewma_var: float | None = None

    def update(self, portfolio_return: float) -> None:
        self._returns.append(portfolio_return)
        max_len = max(self.config.short_window, self.config.long_window or 0) + 10
        if len(self._returns) > max_len:
            self._returns = self._returns[-max_len:]
        if self.config.use_ewma:
            self._update_ewma(portfolio_return)

    def _update_ewma(self, r: float) -> None:
        d = self.config.ewma_decay
        if self._ewma_var is None:
            self._ewma_var = r * r
        else:
            self._ewma_var = d * self._ewma_var + (1.0 - d) * r * r

    def compute_leverage(self) -> float:
        rets = np.array(self._returns, dtype=float)
        n = len(rets)
        if n < self.config.min_observations:
            return 1.0

        short_vol = self._realized_vol(rets, self.config.short_window)

        long_vol: float | None = None
        if self.config.long_window and n >= self.config.long_window:
            long_vol = self._realized_vol(rets, self.config.long_window)

        effective_vol = short_vol
        if long_vol is not None and long_vol > 0:
            effective_vol = 0.6 * short_vol + 0.4 * long_vol

        if self.config.use_ewma and self._ewma_var is not None and n >= 20:
            ewma_vol = np.sqrt(self._ewma_var * 252)
            if ewma_vol > 0:
                effective_vol = 0.5 * effective_vol + 0.5 * ewma_vol

        if effective_vol <= 1e-10:
            return self.config.max_leverage

        leverage = self.config.target_vol / effective_vol
        return float(np.clip(leverage, self.config.min_leverage, self.config.max_leverage))

    @property
    def state(self) -> VolTargetState:
        rets = np.array(self._returns, dtype=float)
        n = len(rets)
        short_vol = self._realized_vol(rets, self.config.short_window) if n >= self.config.min_observations else 0.0
        long_vol = None
        if self.config.long_window and n >= self.config.long_window:
            long_vol = self._realized_vol(rets, self.config.long_window)
        return VolTargetState(
            leverage=self.compute_leverage(),
            short_vol=short_vol,
            long_vol=long_vol,
            effective_vol=0.0,
            n_observations=n,
        )

    def reset(self) -> None:
        self._returns.clear()
        self._ewma_var = None

    @staticmethod
    def _realized_vol(rets: NDArray[np.floating], window: int) -> float:
        if len(rets) < 2:
            return 0.0
        segment = rets[-window:] if window < len(rets) else rets
        return float(np.std(segment, ddof=1) * np.sqrt(252))


def compute_leverage(
    returns: pd.Series,
    target_vol: float = 0.12,
    short_window: int = 20,
    min_leverage: float = 0.5,
    max_leverage: float = 2.0,
) -> float:
    """Compute portfolio leverage from a return series.

    Args:
        returns: Portfolio return series (daily).
        target_vol: Target annualized volatility.
        short_window: Realized vol lookback in days.
        min_leverage: Minimum leverage multiplier.
        max_leverage: Maximum leverage multiplier.

    Returns:
        Leverage multiplier clamped to [min_leverage, max_leverage].
    """
    r = returns.dropna().values
    if len(r) < 10:
        return 1.0

    realized = float(np.std(r[-short_window:], ddof=1) * np.sqrt(252))
    if realized <= 1e-10:
        return max_leverage

    lev = target_vol / realized
    return float(np.clip(lev, min_leverage, max_leverage))
