"""FRED-based macro regime strategy.

Uses composite FRED signals (yield curve, inflation, financial conditions,
economic momentum) to classify the current macro regime and adjust momentum
trading behavior accordingly.

Regime → behavior mapping:
  RISK-ON (strong growth + easy conditions):
    → medium-term momentum, full vol target
  RISK-OFF (weak growth + stressed conditions):
    → short-term momentum, reduced vol target, defensive bias
  NEUTRAL:
    → medium-term momentum, reduced vol target

No blending.  One regime, one set of parameters.

This is a Strategy ABC implementation that integrates FRED macro signals
with the existing momentum framework.  It does NOT modify any existing
strategy files.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import Strategy, StrategyConfig, TradeSignal
from quantcore.strategies.registry import register

try:
    from quantcore.alpha.fred_signals import (
        economic_momentum,
        financial_conditions,
        inflation_pressure,
        yield_curve_regime,
    )
except ImportError:
    yield_curve_regime = None  # type: ignore[assignment]
    inflation_pressure = None  # type: ignore[assignment]
    financial_conditions = None  # type: ignore[assignment]
    economic_momentum = None  # type: ignore[assignment]


@register("macro_fred_regime")
class FREDRegimeStrategy(Strategy):
    """Macro regime strategy driven by FRED composite signals.

    Config params:
        momentum_window: momentum lookback in bars (default 63)
        vol_target: annualized vol target (default 0.15)
        vol_lookback: vol estimation window (default 60)
        rebalance_freq: rebalance interval in days (default 5)
        min_position: min abs position to emit signal (default 0.05)
        regime_weights: dict of signal weights (default equal)
        yield_curve_threshold: z-score threshold for curve signal (default 0.5)
        inflation_threshold: z-score threshold for inflation signal (default 0.5)
        financial_threshold: z-score threshold for financial conditions (default 0.5)
        economic_threshold: z-score threshold for economic momentum (default 0.5)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        self.momentum_window = int(config.params.get("momentum_window", 63))
        self.vol_target = float(config.params.get("vol_target", 0.15))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.min_position = float(config.params.get("min_position", 0.05))

        self.regime_weights = config.params.get("regime_weights", {
            "yield_curve": 0.30,
            "inflation": 0.20,
            "financial": 0.25,
            "economic": 0.25,
        })

        self.yield_curve_threshold = float(
            config.params.get("yield_curve_threshold", 0.5)
        )
        self.inflation_threshold = float(
            config.params.get("inflation_threshold", 0.5)
        )
        self.financial_threshold = float(
            config.params.get("financial_threshold", 0.5)
        )
        self.economic_threshold = float(
            config.params.get("economic_threshold", 0.5)
        )

        self._days_since_rebalance = 0

        # Regime tracking buffers
        self._regime_history: list[dict] = []
        self._daily_metrics: list[dict] = []

    def _compute_momentum(self, prices: NDArray, window: int) -> float:
        """Continuous momentum signal: total return / vol."""
        if len(prices) < window + 2:
            return 0.0

        segment = prices[-window - 1:]
        daily_rets = np.diff(segment) / np.maximum(segment[:-1], 1e-10)
        total_ret = (prices[-1] / prices[-window - 1]) - 1.0
        vol = float(np.std(daily_rets, ddof=1))

        if vol < 1e-10:
            return 0.0
        return total_ret / vol

    def _classify_regime(
        self,
        yield_signal: float,
        inflation_signal: float,
        financial_signal: float,
        economic_signal: float,
    ) -> str:
        """Classify macro regime from composite signals.

        Returns one of: "risk_on", "risk_off", "neutral".

        Risk-on: strong curve + strong economy + accommodative financial conditions.
        Risk-off: inverted curve + weak economy + stressed financial conditions.
        """
        yield_score = 1.0 if yield_signal > self.yield_curve_threshold else (
            -1.0 if yield_signal < -self.yield_curve_threshold else 0.0
        )
        inflation_score = -1.0 if inflation_signal > self.inflation_threshold else (
            1.0 if inflation_signal < -self.inflation_threshold else 0.0
        )
        financial_score = -1.0 if financial_signal > self.financial_threshold else (
            1.0 if financial_signal < -self.financial_threshold else 0.0
        )
        economic_score = 1.0 if economic_signal > self.economic_threshold else (
            -1.0 if economic_signal < -self.economic_threshold else 0.0
        )

        w = self.regime_weights
        composite = (
            w.get("yield_curve", 0.25) * yield_score
            + w.get("inflation", 0.25) * inflation_score
            + w.get("financial", 0.25) * financial_score
            + w.get("economic", 0.25) * economic_score
        )

        if composite > 0.3:
            return "risk_on"
        elif composite < -0.3:
            return "risk_off"
        return "neutral"

    def _regime_behavior(self, regime: str) -> dict:
        """Map regime to trading parameters.

        Returns dict with:
          - vol_multiplier: scaling factor for vol target
          - window_factor: multiplier for momentum window
          - vol_scale_factor: additional vol scaling for risk-off
        """
        if regime == "risk_on":
            return {"vol_multiplier": 1.0, "window_factor": 1.0, "vol_scale_factor": 1.0}
        elif regime == "risk_off":
            return {"vol_multiplier": 0.5, "window_factor": 0.5, "vol_scale_factor": 0.7}
        return {"vol_multiplier": 0.75, "window_factor": 0.75, "vol_scale_factor": 0.85}

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 1:
            close = close.reshape(-1, 1)

        n_periods, n_assets = close.shape
        min_needed = max(self.momentum_window, self.vol_lookback) + 5
        if n_periods < min_needed:
            return []

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # Increment rebalance counter
        self._days_since_rebalance += 1
        is_rebalance = self._days_since_rebalance >= self.rebalance_freq

        # Compute regime signals from macro data (if available)
        macro_df = data.get("macro_df")
        yield_signal = 0.0
        inflation_signal = 0.0
        financial_signal = 0.0
        economic_signal = 0.0

        if macro_df is not None and yield_curve_regime is not None:
            try:
                yr = yield_curve_regime(macro_df)
                ip = inflation_pressure(macro_df)
                fc = financial_conditions(macro_df)
                em = economic_momentum(macro_df)

                if not yr.empty:
                    yield_signal = float(yr.iloc[-1])
                if not ip.empty:
                    inflation_signal = float(ip.iloc[-1])
                if not fc.empty:
                    financial_signal = float(fc.iloc[-1])
                if not em.empty:
                    economic_signal = float(em.iloc[-1])
            except Exception:
                pass  # Fall back to neutral regime

        regime = self._classify_regime(
            yield_signal, inflation_signal, financial_signal, economic_signal
        )
        behavior = self._regime_behavior(regime)

        # Record daily metrics
        self._daily_metrics.append({
            "regime": regime,
            "yield_curve": yield_signal,
            "inflation": inflation_signal,
            "financial": financial_signal,
            "economic": economic_signal,
        })

        # Vol scaling
        vol_rets = np.diff(close[-self.vol_lookback - 1:], axis=0) / np.maximum(
            close[-self.vol_lookback - 1:-1], 1e-10
        )
        asset_vols = np.std(vol_rets, axis=0, ddof=1) * np.sqrt(252)
        asset_vols[asset_vols < 1e-10] = 1e-10

        if not is_rebalance:
            return []

        # Compute positions
        adjusted_window = int(self.momentum_window * behavior["window_factor"])
        adjusted_vol_target = self.vol_target * behavior["vol_multiplier"]
        raw_positions: list[tuple[str, float, dict]] = []

        for i in range(n_assets):
            prices = close[:, i]
            sym = symbols[i]

            signal = self._compute_momentum(prices, adjusted_window)
            vol_scale = (adjusted_vol_target / asset_vols[i]) * behavior["vol_scale_factor"]
            position = float(signal * vol_scale)

            if abs(position) < self.min_position:
                continue

            raw_positions.append((sym, position, {
                "regime": regime,
                "signal": round(signal, 4),
                "vol_scale": round(vol_scale, 4),
                "yield_curve": round(yield_signal, 4),
                "inflation": round(inflation_signal, 4),
                "financial": round(financial_signal, 4),
                "economic": round(economic_signal, 4),
            }))

        if is_rebalance:
            self._days_since_rebalance = 0

        if not raw_positions:
            return []

        # Normalize to target gross exposure
        target_gross = 2.0
        total_abs = sum(abs(p) for _, p, _ in raw_positions)
        if total_abs < 1e-10:
            return []
        scale = target_gross / total_abs

        # Emit signals with per-name cap
        max_per_name = 0.25
        signals: list[TradeSignal] = []
        for sym, pos, meta in raw_positions:
            scaled = pos * scale
            if abs(scaled) > max_per_name:
                scaled = max_per_name if scaled > 0 else -max_per_name

            direction = 1 if scaled > 0 else -1
            strength = min(abs(scaled), 1.0)
            notional = abs(scaled) * float(self.config.max_position_notional)

            signals.append(TradeSignal(
                symbol=sym,
                direction=direction,
                strength=strength,
                size=notional,
                metadata=meta,
            ))

        return signals
