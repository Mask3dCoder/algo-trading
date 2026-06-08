"""Regime-adaptive momentum: dual-horizon trend with regime-based blending.

Core innovation:
  final_signal = (1 - regime_score) * signal_long + regime_score * signal_short

During stable/trending regimes (regime_score ≈ 0):
  → dominated by long-term momentum (200d) — captures persistent trends
During transition/crisis regimes (regime_score ≈ 1):
  → dominated by short-term momentum (21d) — fast reaction, low latency

Regime score uses vol ratio (short-term vol / long-term vol) + trailing drawdown.
No complex models, no parameter tuning — designed for falsifiability.
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


@register("macro_regime_adaptive_momentum")
class RegimeAdaptiveMomentumStrategy(Strategy):
    """Dual-horizon momentum with regime-adaptive horizon blending.

    Config params:
        short_window: short-term momentum horizon (default 21)
        long_window: long-term momentum horizon (default 200)
        vol_short_window: window for short-term vol estimate (default 20)
        vol_long_window: window for long-term vol estimate (default 100)
        vol_ratio_scale: vol ratio above 1.0 maps to component [0,1]
            (default 1.0 — ratio of 2.0 → component = 1.0)
        dd_threshold: drawdown pct that maps to component = 1.0 (default 0.15)
        regime_vol_weight: weight of vol component in regime score (default 0.5)
        regime_dd_weight: weight of drawdown component (default 0.5)
        vol_target: annualized vol target (default 0.15)
        vol_lookback: vol estimation window (default 60)
        rebalance_freq: rebalance interval in days (default 5)
        min_position: minimum abs position to generate signal (default 0.05)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        # Momentum horizons
        self.short_window = int(config.params.get("short_window", 21))
        self.long_window = int(config.params.get("long_window", 200))

        # Regime detection
        self.vol_short_window = int(config.params.get("vol_short_window", 20))
        self.vol_long_window = int(config.params.get("vol_long_window", 100))
        self.vol_ratio_scale = float(config.params.get("vol_ratio_scale", 1.0))
        self.dd_threshold = float(config.params.get("dd_threshold", 0.15))
        self.regime_vol_weight = float(config.params.get("regime_vol_weight", 0.5))
        self.regime_dd_weight = float(config.params.get("regime_dd_weight", 0.5))

        # Vol targeting
        self.vol_target = float(config.params.get("vol_target", 0.15))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))

        # Rebalance
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.min_position = float(config.params.get("min_position", 0.05))

        # State
        self._days_since_rebalance = 0

    # ------------------------------------------------------------------
    # Regime detection
    # ------------------------------------------------------------------

    def _portfolio_returns(self, close: NDArray) -> NDArray:
        """Equal-weight cross-asset portfolio daily returns."""
        if close.ndim == 1:
            prices = close
        else:
            prices = close.mean(axis=1)
        rets = np.diff(prices) / np.maximum(prices[:-1], 1e-10)
        return rets

    def _compute_regime_score(self, close: NDArray) -> float:
        """Regime score in [0, 1].

        0 = stable/trending — use long-term momentum.
        1 = transition/crisis — use short-term momentum.

        Two components blended:
        1. Vol ratio: short_term_vol / long_term_vol.
           Spikes during crisis entries, decays naturally afterward.
        2. Trailing drawdown: captures how far price is from recent peak.
        """
        min_bars = self.vol_long_window + 2
        if len(close) < min_bars:
            return 0.0

        port_rets = self._portfolio_returns(close)

        # --- Vol ratio component ---
        short_slice = port_rets[-self.vol_short_window:]
        long_slice = port_rets[-self.vol_long_window:]

        short_vol = float(np.std(short_slice, ddof=1)) * np.sqrt(252)
        long_vol = float(np.std(long_slice, ddof=1)) * np.sqrt(252)

        if long_vol < 1e-10:
            vol_component = 0.0
        else:
            vol_ratio = short_vol / long_vol
            vol_component = max(0.0, min(1.0, (vol_ratio - 1.0) / self.vol_ratio_scale))

        # --- Drawdown component ---
        # 63-day trailing max drawdown
        dd_lookback = min(63, len(close))
        prices = close.mean(axis=1) if close.ndim > 1 else close
        peak = float(np.max(prices[-dd_lookback:]))
        current = float(prices[-1])
        dd_pct = (current - peak) / peak if peak > 1e-10 else 0.0
        dd_component = max(0.0, min(1.0, abs(dd_pct) / self.dd_threshold))

        # --- Blend ---
        score = (
            self.regime_vol_weight * vol_component
            + self.regime_dd_weight * dd_component
        )
        return float(min(1.0, score))

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_momentum_signals(
        self, close: NDArray,
    ) -> tuple[NDArray, NDArray]:
        """Per-asset short and long momentum signals.

        Returns (short_signals, long_signals), each array of shape (n_assets,)
        with values in {-1, 0, +1}.
        """
        n = close.shape[1]

        if n == 0:
            return np.array([]), np.array([])

        # Short-term momentum: 21-day return sign
        short_rets = (
            close[-1] - close[-self.short_window - 1]
        ) / np.maximum(close[-self.short_window - 1], 1e-10)
        short_signals = np.sign(short_rets)

        # Long-term momentum: long_window-day return sign
        if close.shape[0] >= self.long_window + 1:
            long_rets = (
                close[-1] - close[-self.long_window - 1]
            ) / np.maximum(close[-self.long_window - 1], 1e-10)
            long_signals = np.sign(long_rets)
        else:
            # Not enough data for long signal
            long_signals = np.zeros(close.shape[1])

        return short_signals, long_signals

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

        # Need enough history for short signal + vol scaling
        # Also check long_window for regime computation
        min_needed = max(self.short_window, self.vol_long_window) + self.vol_lookback + 2
        if n_periods < min_needed:
            return []

        # --- Rebalance gate ---
        self._days_since_rebalance += 1
        if self._days_since_rebalance < self.rebalance_freq:
            return []

        # --- Portfolio-level regime score ---
        regime_score = self._compute_regime_score(close)

        # --- Per-asset momentum ---
        short_sig, long_sig = self._compute_momentum_signals(close)
        if len(short_sig) == 0:
            return []

        # --- Regime-adaptive blending ---
        long_available = n_periods >= self.long_window + 2
        if long_available:
            blended = (1.0 - regime_score) * long_sig + regime_score * short_sig
        else:
            # Not enough history for long signal — fall back to short-only
            blended = short_sig

        # --- Vol scaling ---
        vol_rets = np.diff(close[-self.vol_lookback - 1:], axis=0) / np.maximum(
            close[-self.vol_lookback - 1:-1], 1e-10
        )
        realized_vols = np.std(vol_rets, axis=0, ddof=1) * np.sqrt(252)
        realized_vols[realized_vols < 1e-10] = 1e-10
        vol_scales = self.vol_target / realized_vols

        positions = blended * vol_scales

        # --- Generate signals ---
        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])
        signals: list[TradeSignal] = []

        for i in range(n_assets):
            pos = float(positions[i])
            if abs(pos) < self.min_position:
                continue

            direction = 1 if pos > 0 else -1
            strength = min(abs(pos), 1.0)

            signals.append(TradeSignal(
                symbol=symbols[i],
                direction=direction,
                strength=strength,
                size=abs(pos) * float(self.config.max_position_notional),
                metadata={
                    "regime_score": regime_score,
                    "short_signal": float(short_sig[i]),
                    "long_signal": float(long_sig[i]) if long_available else 0.0,
                    "vol_scale": float(vol_scales[i]),
                    "realized_vol": float(realized_vols[i]),
                },
            ))

        self._days_since_rebalance = 0
        return signals
