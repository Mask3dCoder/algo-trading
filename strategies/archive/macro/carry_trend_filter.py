"""Carry-driven directional strategy with trend as binary participation gate.

Core hypothesis:
  Carry provides persistent directional edge via structural yield/drift.
  Trend is used ONLY to filter out bad carry trades (disagreement = flat).
  No blending. No signal averaging. No regime detection.

Asset-specific carry proxies (computed from OHLCV data only):
  Equities (SPY, EFA): trailing 126d total return minus short-rate proxy
    (approximates equity risk premium / dividend yield bias)
  Bonds (TLT, IEF): -trailing 63d return
    (bond prices rise when yields fall; negative return = yields up = positive carry)
  Commodities (GLD, DBC): 63d return minus 252d return
    (futures curve slope proxy: backwardation = positive carry)

Decision logic (strict binary gate):
  if sign(carry_signal) == sign(trend_filter):
      take position with vol-targeted sizing scaled by |carry_signal|
  else:
      position = 0

No blending. No partial weighting. No regime detection.
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


# Asset type mapping for carry computation
ASSET_CARRY_TYPE: dict[str, str] = {
    "SPY": "equity",
    "EFA": "equity",
    "TLT": "bond",
    "IEF": "bond",
    "GLD": "commodity",
    "DBC": "commodity",
}


@register("macro_carry_trend_filter")
class CarryTrendFilterStrategy(Strategy):
    """Carry-driven strategy with trend as a binary participation gate.

    Config params:
        vol_target: annualized vol target (default 0.15)
        vol_lookback: vol estimation window for sizing (default 60)
        rebalance_freq: rebalance interval in days (default 5)
        min_position: minimum abs position to generate signal (default 0.05)
        carry_equity_window: lookback for equity carry drift (default 126)
        short_rate: risk-free rate proxy for equity carry (default 0.03)
        use_conviction_scaling: scale position by |carry_signal| (default True)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        self.vol_target = float(config.params.get("vol_target", 0.15))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.min_position = float(config.params.get("min_position", 0.05))
        self.carry_equity_window = int(config.params.get("carry_equity_window", 126))
        self.short_rate = float(config.params.get("short_rate", 0.03))
        self.use_conviction_scaling = bool(
            config.params.get("use_conviction_scaling", True)
        )

        self._days_since_rebalance = 0

    # ------------------------------------------------------------------
    # Carry computation (asset-specific)
    # ------------------------------------------------------------------

    def _compute_carry(self, prices: NDArray, asset_type: str) -> float:
        """Raw carry proxy. Not yet normalized by vol."""
        if asset_type == "equity":
            if len(prices) < self.carry_equity_window + 2:
                return 0.0
            total_ret = (prices[-1] / prices[-self.carry_equity_window - 1]) - 1.0
            return total_ret - self.short_rate

        elif asset_type == "bond":
            if len(prices) < 65:
                return 0.0
            ret_63 = (prices[-1] / prices[-64]) - 1.0
            return -ret_63

        elif asset_type == "commodity":
            if len(prices) < 254:
                return 0.0
            ret_63 = (prices[-1] / prices[-64]) - 1.0
            ret_252 = (prices[-1] / prices[-253]) - 1.0
            return ret_63 - ret_252

        return 0.0

    # ------------------------------------------------------------------
    # Volatility estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _annualized_vol(prices: NDArray, window: int) -> float:
        """Annualized vol over trailing `window` daily returns."""
        if len(prices) < window + 2:
            return 0.0
        prev = prices[-window - 1:-1]
        daily_rets = np.diff(prices[-window - 1:]) / np.maximum(prev, 1e-10)
        vol = float(np.std(daily_rets, ddof=1)) * np.sqrt(252)
        return max(vol, 1e-10)

    # ------------------------------------------------------------------
    # Trend filter (Z-score, not direction)
    # ------------------------------------------------------------------

    @staticmethod
    def _trend_zscore(prices: NDArray) -> tuple[float, float]:
        """Trend Z-score = total_return_63d / daily_vol_63d.

        Returns (zscore, sign). Zscore can be compared with carry_signal.
        """
        if len(prices) < 65:
            return (0.0, 0.0)

        total_ret = (prices[-1] / prices[-64]) - 1.0
        prev = prices[-64:-1]
        daily_rets = np.diff(prices[-64:]) / np.maximum(prev, 1e-10)
        vol = float(np.std(daily_rets, ddof=1))

        if vol < 1e-10:
            return (0.0, 0.0)

        z = total_ret / vol
        return (z, np.sign(z))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape

        # Must have at least trend (65) + vol_lookback (60) bars
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
            sym = symbols[i]
            asset_type = ASSET_CARRY_TYPE.get(sym, "equity")

            # 1. Annualized vol (for normalization AND sizing)
            asset_vol = self._annualized_vol(prices, self.vol_lookback)
            if asset_vol < 1e-10:
                continue

            # 2. Carry signal (normalized by vol → comparable across assets)
            carry_raw = self._compute_carry(prices, asset_type)
            carry_signal = carry_raw / asset_vol  # unitless, ~Sharpe-like

            # 3. Direction from carry ONLY
            direction = np.sign(carry_signal)
            if direction == 0.0:
                continue

            # 4. Trend filter (binary gate, NOT blended)
            trend_z, trend_sign = self._trend_zscore(prices)
            if trend_sign == 0.0:
                continue
            if direction != trend_sign:
                continue

            # 5. Position sizing: vol target with conviction scaling
            vol_scale = self.vol_target / asset_vol
            position = float(direction * vol_scale)

            if self.use_conviction_scaling:
                position *= min(abs(carry_signal), 3.0)

            if abs(position) < self.min_position:
                continue

            strength = min(abs(position), 1.0)

            signals.append(TradeSignal(
                symbol=sym,
                direction=1 if position > 0 else -1,
                strength=strength,
                size=abs(position) * float(self.config.max_position_notional),
                metadata={
                    "asset_type": asset_type,
                    "carry_raw": round(carry_raw, 6),
                    "carry_signal": round(carry_signal, 4),
                    "trend_z": round(trend_z, 4),
                    "position": round(position, 4),
                    "vol_scale": round(vol_scale, 4),
                },
            ))

        self._days_since_rebalance = 0
        return signals
