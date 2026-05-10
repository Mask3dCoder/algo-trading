"""True carry strategy: carry-driven direction with trend as binary gate.

Core hypothesis:
  Real carry (dividend yield, term premium, roll yield) provides a persistent
  directional edge independent of price momentum. Trend is used ONLY as a
  binary participation gate to avoid entering against the current direction.

Signal sources (pre-computed, passed via data["carry"]):
  Equities (SPY, EFA): z-score of TTM dividend yield
  Bonds (TLT, IEF):    z-score of term premium (^TNX - ^IRX)
  Commodities (GLD, DBC): z-score of roll-yield proxy (21d - 252d return)

Decision logic (strict binary gate - NO blending):
  if sign(carry_signal) == sign(trend_filter):
      take position sized by carry_signal * vol_target / realized_vol
  else:
      position = 0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register


@register("macro_true_carry")
class TrueCarryStrategy(Strategy):
    """Carry-driven strategy with trend as binary gate.

    Config params:
        vol_target: annualized vol target (default 0.15)
        vol_lookback: window for realized vol estimation (default 60)
        rebalance_freq: rebalance interval in days (default 5)
        min_position: minimum strength to emit signal (default 0.01)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        self.vol_target = float(config.params.get("vol_target", 0.15))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.min_position = float(config.params.get("min_position", 0.01))

        self._days_since_rebalance = 0

    # ------------------------------------------------------------------
    # Volatility estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _annualized_vol(prices: NDArray, window: int) -> float:
        if len(prices) < window + 2:
            return 0.0
        prev = prices[-window - 1:-1]
        daily_rets = np.diff(prices[-window - 1:]) / np.maximum(prev, 1e-10)
        vol = float(np.std(daily_rets, ddof=1)) * np.sqrt(252)
        return max(vol, 1e-10)

    # ------------------------------------------------------------------
    # Trend filter (binary gate, NOT signal)
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_sign(prices: NDArray) -> float:
        """Binary trend direction: sign of 63d return / vol Z-score."""
        if len(prices) < 65:
            return 0.0

        total_ret = (prices[-1] / prices[-64]) - 1.0
        prev = prices[-64:-1]
        daily_rets = np.diff(prices[-64:]) / np.maximum(prev, 1e-10)
        vol = float(np.std(daily_rets, ddof=1))

        if vol < 1e-10:
            return 0.0

        return np.sign(total_ret / vol)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        carry = data.get("carry")

        if close is None or carry is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)
            carry = carry.reshape(-1, 1)

        n_periods, n_assets = close.shape

        # Need enough history for trend + vol
        if n_periods < max(65, self.vol_lookback + 5):
            return []

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # Rebalance gate
        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq:
            return []

        signals: list[TradeSignal] = []

        for i in range(n_assets):
            prices = close[:, i]
            carry_history = carry[:, i]

            # Current carry z-score (from pre-computed series)
            carry_z = carry_history[-1]

            # Skip if carry is neutral
            if abs(carry_z) < 0.01:
                continue

            # Direction from carry ONLY
            direction = np.sign(carry_z)

            # Trend filter: ONLY binary gate, NOT blended
            tsign = self._trend_sign(prices)
            if tsign == 0.0:
                continue
            if direction != tsign:
                continue

            # Position sizing: carry_z gives conviction, vol targets risk
            asset_vol = self._annualized_vol(prices, self.vol_lookback)
            if asset_vol < 1e-10:
                continue

            vol_scale = self.vol_target / asset_vol
            position = float(carry_z * vol_scale)

            if abs(position) < self.min_position:
                continue

            strength = min(abs(position), 1.0)

            signals.append(TradeSignal(
                symbol=symbols[i],
                direction=1 if position > 0 else -1,
                strength=strength,
                size=abs(position) * float(self.config.max_position_notional),
                metadata={
                    "carry_z": round(carry_z, 4),
                    "carry_type": "equity_dividend_yield"
                    if symbols[i] in ("SPY", "EFA")
                    else "bond_term_premium"
                    if symbols[i] in ("TLT", "IEF")
                    else "commodity_roll_yield",
                    "trend_sign": int(tsign),
                    "vol_scale": round(vol_scale, 4),
                    "position_z": round(position, 4),
                },
            ))

        self._days_since_rebalance = 0
        return signals
