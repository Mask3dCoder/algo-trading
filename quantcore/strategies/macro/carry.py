"""Cross-asset carry strategy: long high-yield, short low-yield assets.

Computes carry (roll yield + income yield) for each asset, ranks cross-sectionally,
and goes long top carry / short bottom carry with vol-targeted sizing.
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


@register("macro_carry")
class CarryStrategy(Strategy):
    """Cross-asset carry: long high carry, short low carry, vol-targeted.

    Config params:
        n_long: number of long positions (default 3)
        n_short: number of short positions (default 3)
        vol_target: annualized vol target per asset (default 0.10)
        vol_lookback: vol estimation lookback in days (default 60)
        carry_smoothing: EMA smoothing for carry signal (default 21)
        rebalance_freq: rebalancing frequency in days (default 21)
        carry_adjustment: vol-adjust carry (default True)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.n_long = int(config.params.get("n_long", 3))
        self.n_short = int(config.params.get("n_short", 3))
        self.vol_target = float(config.params.get("vol_target", 0.10))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))
        self.carry_smoothing = int(config.params.get("carry_smoothing", 21))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 21))
        self.carry_adjustment = bool(config.params.get("carry_adjustment", True))
        self._days_since_rebalance = 0
        self._smoothed_carry: NDArray | None = None

    def _compute_carry(self, data: dict[str, NDArray]) -> NDArray:
        """Compute carry for each asset.

        For FX: interest rate differential
        For bonds: yield curve roll-down
        For commodities: futures roll yield (approximated by price momentum)
        For equities: dividend yield (approximated by recent returns vs price change)

        Simplified: use daily returns minus overnight risk-free as carry proxy.
        """
        close = data.get("close")
        if close is None or close.ndim != 2:
            return np.array([])

        n_periods, n_assets = close.shape

        # Carry proxy 1: recent total return vs price change (capturing income)
        if n_periods < self.carry_smoothing + 1:
            return np.zeros(n_assets)

        # Simple carry proxy: difference between 21-day return and spot return
        # This captures roll-down / yield in a simplified way
        ret_short = close[-1] / close[-6] - 1.0  # 5-day return
        ret_medium = close[-1] / close[-22] - 1.0  # 21-day return

        # Rate of return decay = carry proxy (higher = more carry)
        # Assets with positive carry show consistent positive returns
        daily_rets = np.diff(close[-self.vol_lookback:], axis=0) / close[-self.vol_lookback:-1]
        mean_daily_ret = np.mean(daily_rets, axis=0) * 252  # Annualized

        # Use yield data if provided
        yields = data.get("yields")
        if yields is not None:
            if yields.ndim == 2:
                carry = yields[-1]
            else:
                carry = yields
        else:
            carry = mean_daily_ret

        return carry

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        if n_assets < self.n_long + self.n_short:
            return []

        signals: list[TradeSignal] = []
        self._days_since_rebalance += 1

        if self._days_since_rebalance < self.rebalance_freq:
            return signals

        # Compute carry
        carry = self._compute_carry(data)
        if len(carry) != n_assets or len(carry) == 0:
            return []

        # Smooth carry with EMA
        alpha = 2.0 / (self.carry_smoothing + 1)
        if self._smoothed_carry is None:
            self._smoothed_carry = carry.copy()
        else:
            self._smoothed_carry = alpha * carry + (1 - alpha) * self._smoothed_carry

        # Vol estimation
        if n_periods > self.vol_lookback + 1:
            rets = np.diff(close[-self.vol_lookback - 1:], axis=0) / close[-self.vol_lookback - 1:-1]
            vols = np.std(rets, axis=0) * np.sqrt(252)
        else:
            vols = np.ones(n_assets) * 0.2

        vols[vols < 1e-10] = 1e-10

        # Optionally adjust carry by volatility (carry-to-vol ratio)
        if self.carry_adjustment:
            adjusted_carry = self._smoothed_carry / vols
        else:
            adjusted_carry = self._smoothed_carry.copy()

        # Rank and select
        ranks = np.argsort(np.argsort(adjusted_carry))
        long_mask = ranks >= (n_assets - self.n_long)
        short_mask = ranks < self.n_short

        # Vol-targeted sizing
        vol_scales = self.vol_target / vols

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        for i in range(n_assets):
            if long_mask[i]:
                direction = 1
                weight = vol_scales[i] / np.sum(vol_scales[long_mask]) / 2.0
            elif short_mask[i]:
                direction = -1
                weight = vol_scales[i] / np.sum(vol_scales[short_mask]) / 2.0
            else:
                continue

            signals.append(TradeSignal(
                symbol=symbols[i],
                direction=direction,
                strength=min(abs(weight) * n_assets, 1.0),
                size=abs(weight) * float(self.config.max_position_notional),
                metadata={
                    "carry": float(self._smoothed_carry[i]),
                    "vol": float(vols[i]),
                    "carry_to_vol": float(adjusted_carry[i]),
                    "rank": int(ranks[i]),
                },
            ))

        self._days_since_rebalance = 0
        return signals
