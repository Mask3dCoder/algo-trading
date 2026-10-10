"""CTA multi-speed trend following strategy.

Classic CTA approach: combine signals from multiple timeframes (fast, medium, slow)
using weighted average. Applies volatility targeting for position sizing.
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


@register("macro_cta_trend")
class CTATrendStrategy(Strategy):
    """.. deprecated::
    STRATEGY_STATUS = StrategyStatus.DISABLED

    Forensic audit (2026-05-09) confirmed OOS Sharpe = -0.147 — structural
    negative expected alpha after transaction costs.  Parameters (20/50/200-day
    windows, 0.2/0.3/0.5 weights) have been arbitraged away post-2008.
    Do NOT use in automated backtest runs or paper trading.
    Retained as a historical artifact; pending full rebuild under Q2 plan.
    """

    STRATEGY_STATUS: StrategyStatus = StrategyStatus.DISABLED
    """CTA multi-speed trend: combine fast/medium/slow signals with vol targeting.

    Config params:
    fast_window: fast trend window (default 20)
    medium_window: medium trend window (default 50)
    slow_window: slow trend window (default 200)
    fast_weight: weight for fast signal (default 0.2)
    medium_weight: weight for medium signal (default 0.3)
    slow_weight: weight for slow signal (default 0.5)
    vol_target: annualized vol target (default 0.15)
    vol_lookback: vol estimation lookback (default 60)
    rebalance_freq: rebalancing frequency in days (default 5)
    vol_regime_threshold: scale when 20-bar portfolio vol > vol_target * this (default 0.30)
    vol_regime_lookback: vol estimation window for regime detection (default 20)
    min_vol_scale: minimum scale factor during high vol regimes (default 0.20)
    min_signal_strength: minimum |fast_signal| to act on (default 0.0 = no filter)
    alignment_filter: only trade when fast/slow agree in direction (default False)
    """

    def __init__(self, config: StrategyConfig) -> None:
        super().__init__(config)
        self.fast_window = int(config.params.get("fast_window", 20))
        self.medium_window = int(config.params.get("medium_window", 50))
        self.slow_window = int(config.params.get("slow_window", 200))
        self.fast_weight = float(config.params.get("fast_weight", 0.2))
        self.medium_weight = float(config.params.get("medium_weight", 0.3))
        self.slow_weight = float(config.params.get("slow_weight", 0.5))
        self.vol_target = float(config.params.get("vol_target", 0.15))
        self.vol_lookback = int(config.params.get("vol_lookback", 60))
        self.rebalance_freq = int(config.params.get("rebalance_freq", 5))
        self.vol_regime_threshold = float(config.params.get("vol_regime_threshold", 0.30))
        self.vol_regime_lookback = int(config.params.get("vol_regime_lookback", 20))
        self.min_vol_scale = float(config.params.get("min_vol_scale", 0.20))
        self.min_signal_strength = float(config.params.get("min_signal_strength", 0.0))
        self.alignment_filter = bool(config.params.get("alignment_filter", False))
        self.max_vol_scale = float(config.params.get("max_vol_scale", 3.0))
        self._days_since_rebalance = 0
        # Diagnostic state
        self._fast_flips: list[dict] = []  # {timestamp, asset, prev_sign, new_sign}
        self._prev_fast_dir: dict[int, float] = {}  # asset index -> previous fast sign
        self._total_fast_signals = 0
        self._filtered_fast_signals = 0
        self._skipped_opportunities = 0
        self._skipped_signals: list[dict] = []
        self._all_opportunities: list[dict] = []  # All trade opps (executed + skipped)
        self._current_timestamp = None

    def _trend_signal(self, prices: NDArray, window: int) -> float:
        """Compute trend signal: (price - SMA) / SMA."""
        if len(prices) < window:
            return 0.0
        sma = np.mean(prices[-window:])
        if sma < 1e-10:
            return 0.0
        return float((prices[-1] - sma) / sma)

    def _breakout_signal(self, prices: NDArray, window: int) -> float:
        """Donchian channel breakout: position in range."""
        if len(prices) < window:
            return 0.0
        high = np.max(prices[-window:])
        low = np.min(prices[-window:])
        if high - low < 1e-10:
            return 0.0
        return float(2 * (prices[-1] - low) / (high - low) - 1)  # [-1, 1]

    def _momentum_signal(self, prices: NDArray, window: int) -> float:
        """Time-series momentum: sign of past return."""
        if len(prices) < window + 1:
            return 0.0
        ret = prices[-1] / prices[-window - 1] - 1.0
        return float(np.sign(ret))

    def _vol_regime_scale(self, prices: NDArray) -> float:
        """Detect high-vol regimes using 20-bar portfolio vol.

        When 20-bar vol of the portfolio mean return exceeds a threshold
        (vol_target * vol_regime_threshold), shift signal weights from fast
        to slow and reduce position sizes. This directly targets the HIGH_VOL
        regime definition (elevated 20-bar portfolio vol).
        """
        min_needed = self.vol_regime_lookback + 2
        if len(prices) < min_needed:
            return 1.0

        # Portfolio vol via mean return
        if prices.ndim == 2:
            mean_p = np.mean(prices, axis=1)
        else:
            mean_p = prices

        recent = mean_p[-(self.vol_regime_lookback + 1):]
        recent_rets = np.diff(recent) / recent[:-1]
        recent_vol = float(np.std(recent_rets, ddof=1)) * np.sqrt(252)

        trigger = self.vol_target * self.vol_regime_threshold
        if recent_vol <= trigger:
            return 1.0

        # Inverse-square decay above threshold
        ratio = recent_vol / trigger
        scale = max(self.min_vol_scale, 1.0 / ratio ** 2)
        return float(scale)

    def on_data(self, data: dict[str, NDArray]) -> list[TradeSignal]:
        close = data.get("close")
        if close is None:
            return []

        if close.ndim == 2:
            n_periods, n_assets = close.shape
        else:
            close = close.reshape(-1, 1)
            n_periods, n_assets = close.shape

        if n_periods < self.slow_window + 2:
            return []

        signals: list[TradeSignal] = []
        self._days_since_rebalance += 1

        if self._days_since_rebalance < self.rebalance_freq:
            return signals

        symbols = data.get("symbols", [f"asset_{i}" for i in range(n_assets)])

        # Portfolio-level regime detection (once, before per-asset loop)
        regime_scale = self._vol_regime_scale(close)
        in_high_regime = regime_scale < 1.0

        for a in range(n_assets):
            prices = close[:, a]

            # Multi-speed trend signals
            fast_signal = (
                self._trend_signal(prices, self.fast_window) * 0.4
                + self._breakout_signal(prices, self.fast_window) * 0.3
                + self._momentum_signal(prices, self.fast_window) * 0.3
            )
            medium_signal = (
                self._trend_signal(prices, self.medium_window) * 0.4
                + self._breakout_signal(prices, self.medium_window) * 0.3
                + self._momentum_signal(prices, self.medium_window) * 0.3
            )
            slow_signal = (
                self._trend_signal(prices, self.slow_window) * 0.4
                + self._breakout_signal(prices, self.slow_window) * 0.3
                + self._momentum_signal(prices, self.slow_window) * 0.3
            )

            # --- PHASE 2.2 & 2.4: Alignment + Strength Filters ---
            # Track total signals for diagnostics
            self._total_fast_signals += 1

            # Check if fast signal is strong enough
            fast_signal_strong = (abs(fast_signal) >= self.min_signal_strength) if self.min_signal_strength > 0.0 else True

            # Check if fast and slow agree in direction
            alignment_mismatch = (np.sign(fast_signal) != np.sign(slow_signal)) if self.alignment_filter else False

            # Determine if we should skip this trade
            skip_trade = False
            skip_reason = None

            # Alignment filter: skip if fast and slow disagree
            if self.alignment_filter and alignment_mismatch:
                skip_trade = True
                skip_reason = 'alignment_mismatch'

            # Strength filter: skip if fast signal is NOT strong enough
            # (This is the STANDARD strength filter, not the inverted Phase 2.0 logic)
            if self.min_signal_strength > 0.0 and not fast_signal_strong:
                skip_trade = True
                if skip_reason is None:
                    skip_reason = 'weak_signal'
                else:
                    skip_reason = f'{skip_reason}_and_weak_signal'

            # Dynamic weight shift during high-vol regimes
            if in_high_regime:
                shift = (1.0 - regime_scale) * 0.4
                fw = max(0.0, self.fast_weight - shift)
                sw = self.slow_weight + (self.fast_weight - fw)
            else:
                fw = self.fast_weight
                sw = self.slow_weight

            # Track fast signal flips for diagnostic
            fast_dir = np.sign(fast_signal)
            prev_dir = self._prev_fast_dir.get(a)
            if prev_dir is not None and prev_dir != fast_dir and fast_dir != 0:
                self._fast_flips.append({
                    "timestamp": self._current_timestamp,
                    "asset": symbols[a] if a < len(symbols) else f"asset_{a}",
                    "prev_sign": float(prev_dir),
                    "new_sign": float(fast_dir),
                })
            self._prev_fast_dir[a] = fast_dir

            # Weighted combination with dynamic weights
            composite = (
                fw * np.clip(fast_signal, -1, 1)
                + self.medium_weight * np.clip(medium_signal, -1, 1)
                + sw * np.clip(slow_signal, -1, 1)
            )

            # Vol targeting
            if len(prices) > self.vol_lookback + 1:
                rets = np.diff(prices[-self.vol_lookback - 1:]) / prices[-self.vol_lookback - 1:-1]
                realized_vol = np.std(rets) * np.sqrt(252)
            else:
                realized_vol = 0.2

            if realized_vol < 1e-10:
                continue

            vol_scale = self.vol_target / realized_vol
            position = composite * vol_scale * regime_scale

            if abs(position) < 0.05:
                continue

            direction = 1 if position > 0 else -1
            size = abs(position) * float(self.config.max_position_notional) / n_assets

            # --- PHASE 2.2 & 2.4: Trade gating ---
            trade_opp = {
                "timestamp": self._current_timestamp,
                "asset": symbols[a] if a < len(symbols) else f"asset_{a}",
                "direction": direction,
                "fast_signal": float(fast_signal),
                "medium_signal": float(medium_signal),
                "slow_signal": float(slow_signal),
                "composite": float(composite),
                "position": float(position),
                "entry_price": float(prices[-1]) if len(prices) > 0 else None,
                "fast_signal_strong": fast_signal_strong,
                "alignment_mismatch": alignment_mismatch,
                "skip_reason": skip_reason,  # NEW: track why trade was skipped
                "was_executed": False,  # Will be set to True if not skipped
            }

            if skip_trade:
                # Record skipped trade opportunity
                self._skipped_opportunities += 1
                self._skipped_signals.append(trade_opp)
                self._all_opportunities.append(trade_opp)
                continue  # Skip generating TradeSignal (block entry/flip)

            # Mark as executed
            trade_opp["was_executed"] = True
            self._all_opportunities.append(trade_opp)

            signals.append(TradeSignal(
                symbol=symbols[a],
                direction=direction,
                strength=min(abs(position), 1.0),
                size=size,
                metadata={
                    "composite": float(composite),
                    "fast": float(fast_signal),
                    "medium": float(medium_signal),
                    "slow": float(slow_signal),
                    "vol_scale": float(vol_scale),
                    "realized_vol": float(realized_vol),
                    "fast_signal_strong": fast_signal_strong,
                },
            ))

        self._days_since_rebalance = 0
        return signals

    def set_timestamp(self, ts: any) -> None:
        """Set current bar timestamp for diagnostic tracking."""
        self._current_timestamp = ts

    def get_fast_flips(self) -> list[dict]:
        """Return list of fast signal flips for diagnostic analysis."""
        return self._fast_flips.copy()

    def get_skipped_opportunities(self) -> int:
        """Return count of skipped trade opportunities due to weak fast signal."""
        return self._skipped_opportunities

    def get_skipped_signals(self) -> list[dict]:
        """Return list of skipped signal details for PnL analysis."""
        return self._skipped_signals.copy()

    def get_all_opportunities(self) -> list[dict]:
        """Return ALL trade opportunities (executed + skipped) for attribution analysis."""
        return self._all_opportunities.copy()

    def reset_diagnostics(self) -> None:
        """Reset all diagnostic trackers (flips, confirm distributions)."""
        self._fast_flips = []
        self._prev_fast_dir = {}
        self._total_fast_signals = 0
        self._filtered_fast_signals = 0
        self._skipped_opportunities = 0
        self._skipped_signals = []
        self._all_opportunities = []
