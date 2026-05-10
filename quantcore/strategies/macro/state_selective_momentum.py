"""State-selective momentum: picks a single directional view based on signal stability.

Core hypothesis:
  Cross-horizon signal conflict causes failure during regime transitions.
  Eliminating blending and instead SELECTING a signal based on market state
  removes the recovery-phase failure mode.

Three states determined by signal instability:
  STABLE     (low instability)   → medium-term momentum (63d)
  TRANSITION (high instability)  → short-term momentum (21d)
  NOISE      (moderate instability) → zero exposure

Instability = |signal_short - signal_medium|, normalized via rolling percentile.
No blending. No averaging. One coherent directional view at all times.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import (
    Strategy,
    StrategyConfig,
    TradeSignal,
)
from quantcore.strategies.registry import register


@register("macro_state_selective_momentum")
class StateSelectiveMomentumStrategy(Strategy):
    """Momentum with state-conditional horizon selection.

    Config params:
        short_window: short-term momentum horizon (default 21)
        medium_window: medium-term momentum horizon (default 63)
        vol_target: annualized vol target (default 0.15)
        vol_lookback: vol estimation window for position sizing (default 60)
        rebalance_freq: rebalance interval in days (default 5)
        min_position: minimum abs position to generate signal (default 0.05)
        instability_buffer: max size of instability percentile buffer (default 252)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)

        # Horizons
        self.short_window = int(config.params.get("short_window", 21))
        self.medium_window = int(config.params.get("medium_window", 63))

        # Position sizing
        self.vol_target = float(config.params.get("vol_target", 0.15))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))

        # Trading controls
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.min_position = float(config.params.get("min_position", 0.05))

        # State
        self._days_since_rebalance = 0

        # Per-asset instability tracking (deques for rolling percentile)
        buf_size = int(config.params.get("instability_buffer", 252))
        self._inst_buffers: dict[str, deque] = {}

        # Store recent signal values for transition-loss analysis
        self._debug_history: dict[str, list[dict]] = {}

    # ------------------------------------------------------------------
    # Momentum signal: continuous, not binary
    #   signal = total_return_over_window / vol_over_window
    #   Unitless: number of standard deviations of return
    # ------------------------------------------------------------------

    def _compute_signal(self, prices: NDArray, window: int) -> float:
        """Continuous momentum: total return / vol, both over `window` bars.

        Returns float in approximate range [-3, +3] in normal conditions.
        No sign() call.
        """
        if len(prices) < window + 2:
            return 0.0

        segment = prices[-window - 1:]       # window + 1 elements
        prev = segment[:-1]                   # window elements
        daily_rets = np.diff(segment) / np.maximum(prev, 1e-10)
        total_ret = (prices[-1] / prices[-window - 1]) - 1.0
        vol = float(np.std(daily_rets, ddof=1))

        if vol < 1e-10:
            return 0.0
        return total_ret / vol

    # ------------------------------------------------------------------
    # Instability: |short - medium|, normalized via rolling percentile
    # ------------------------------------------------------------------

    def _instability_percentile(self, symbol: str, instability: float) -> float:
        """Rolling percentile rank of instability for a symbol.

        Returns 0.5 if buffer has fewer than 10 entries (conservative default
        that maps to the NOISE / moderate-instability zone).
        """
        if symbol not in self._inst_buffers:
            self._inst_buffers[symbol] = deque(
                maxlen=int(self.config.params.get("instability_buffer", 252))
            )

        buf = self._inst_buffers[symbol]
        buf.append(instability)

        if len(buf) < 10:
            return 0.5  # moderate — safe default

        below = sum(1 for v in buf if v < instability)
        return below / len(buf)

    # ------------------------------------------------------------------
    # State classification: three mutually exclusive states
    # ------------------------------------------------------------------

    def _classify_state(self, instability_pctile: float) -> str:
        """Classify market state from instability percentile.

        33rd and 67th percentiles are data-driven (not arbitrary constants)
        because the percentile ranking adapts to market conditions.
        """
        if instability_pctile < 1.0 / 3.0:
            return "stable"
        elif instability_pctile > 2.0 / 3.0:
            return "transition"
        else:
            return "noise"

    # ------------------------------------------------------------------
    # Signal selection: strictly one signal, no blending
    # ------------------------------------------------------------------

    def _select_signal(self, state: str, signal_short: float, signal_medium: float) -> float:
        """Select a single directional view based on state.

        STABLE     -> medium-term signal (trend anchor)
        TRANSITION -> short-term signal (fast reaction)
        NOISE      -> zero (no conviction)
        """
        if state == "stable":
            return signal_medium
        elif state == "transition":
            return signal_short
        else:  # noise
            return 0.0

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

        # Minimum history for medium signal + vol scaling
        min_needed = max(self.medium_window, self.vol_lookback) + 5
        if n_periods < min_needed:
            return []

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # Update rebalance counter every bar
        self._days_since_rebalance += 1
        is_rebalance = self._days_since_rebalance >= self.rebalance_freq

        # Vol scaling data (shared across assets)
        vol_rets = np.diff(close[-self.vol_lookback - 1:], axis=0) / np.maximum(
            close[-self.vol_lookback - 1:-1], 1e-10
        )
        asset_vols = np.std(vol_rets, axis=0, ddof=1) * np.sqrt(252)
        asset_vols[asset_vols < 1e-10] = 1e-10

        signals: list[TradeSignal] = []

        for i in range(n_assets):
            prices = close[:, i]
            sym = symbols[i]

            # 1. Compute per-asset momentum signals (continuous, every bar)
            sig_short = self._compute_signal(prices, self.short_window)
            sig_medium = self._compute_signal(prices, self.medium_window)

            # 2. Instability measurement (every bar — feeds percentile buffer)
            instability = abs(sig_short - sig_medium)
            inst_pctile = self._instability_percentile(sym, instability)

            # Skip trade generation on non-rebalance days
            if not is_rebalance:
                continue

            # 3. State classification
            state = self._classify_state(inst_pctile)

            # 4. Signal selection (not blending)
            final_signal = self._select_signal(state, sig_short, sig_medium)

            # 5. Vol scaling
            vol_scale = self.vol_target / asset_vols[i]
            position = float(final_signal * vol_scale)

            if abs(position) < self.min_position:
                continue

            direction = 1 if position > 0 else -1
            strength = min(abs(position), 1.0)

            signals.append(TradeSignal(
                symbol=sym,
                direction=direction,
                strength=strength,
                size=abs(position) * float(self.config.max_position_notional),
                metadata={
                    "state": state,
                    "instability_pctile": round(inst_pctile, 3),
                    "sig_short": round(sig_short, 4),
                    "sig_medium": round(sig_medium, 4),
                    "final_signal": round(final_signal, 4),
                    "vol_scale": round(vol_scale, 4),
                },
            ))

        if is_rebalance:
            self._days_since_rebalance = 0

        return signals
