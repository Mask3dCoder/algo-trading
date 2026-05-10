"""Strategy monitoring: signal tracking, regime detection, performance drift alerts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from quantcore.dashboard.alerts import Alert, AlertEngine, AlertRule, AlertSeverity
from quantcore.strategies.base import TradeSignal


class SignalQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"


class RegimeLabel(str, Enum):
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"
    NORMAL = "normal"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    HIGH_CORR = "high_corr"


@dataclass
class SignalSnapshot:
    """Record of signals emitted by a strategy at a point in time."""
    timestamp: datetime
    n_signals: int
    avg_strength: float
    max_strength: float
    min_strength: float
    long_pct: float          # fraction of signals that are long
    gross_leverage: float    # sum of abs(strength)
    symbols: list[str] = field(default_factory=list)
    regime: RegimeLabel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyHealth:
    """Current health assessment for a single strategy."""
    strategy_name: str
    signal_quality: SignalQuality = SignalQuality.GOOD
    regime: RegimeLabel | None = None
    regime_scale: float = 1.0
    consecutive_losses: int = 0
    max_drawdown_pct: float = 0.0
    days_since_rebalance: int = 0
    recent_sharpe: float | None = None     # rolling 20-day Sharpe
    signal_count_avg: float | None = None   # avg signals per tick over last N
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegimeDetectorConfig:
    """Configuration for online regime detection."""
    vol_lookback: int = 20
    vol_percentile_high: float = 75.0
    vol_percentile_low: float = 25.0
    trend_lookback: int = 20
    trend_threshold: float = 0.05           # 5% return over lookback
    corr_lookback: int = 20
    corr_threshold: float = 0.70
    min_history: int = 60


class RegimeDetector:
    """Online regime detection for strategy monitoring.

    Maintains a rolling window of returns and classifies the current
    market regime based on volatility percentiles, trend strength,
    and cross-asset correlation.
    """

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        self.config = config or RegimeDetectorConfig()
        self._returns_buffer: list[NDArray] = []
        self._vol_history: list[float] = []
        self._current_regime: RegimeLabel = RegimeLabel.NORMAL

    def update(self, returns: NDArray) -> RegimeLabel:
        """Update with new return vector and return current regime label."""
        self._returns_buffer.append(returns)
        if len(self._returns_buffer) > self.config.min_history:
            self._returns_buffer.pop(0)

        if len(self._returns_buffer) < self.config.vol_lookback + 1:
            return self._current_regime

        arr = np.array(self._returns_buffer)

        # Portfolio-level vol
        portfolio_rets = arr.mean(axis=1)[-self.config.vol_lookback:]
        realized_vol = float(np.std(portfolio_rets, ddof=1)) * np.sqrt(252)
        self._vol_history.append(realized_vol)
        if len(self._vol_history) > 500:
            self._vol_history.pop(0)

        # Classify regime
        if len(self._vol_history) > 20:
            vol_high = float(np.percentile(self._vol_history, self.config.vol_percentile_high))
            vol_low = float(np.percentile(self._vol_history, self.config.vol_percentile_low))

            if realized_vol > vol_high:
                self._current_regime = RegimeLabel.HIGH_VOL
            elif realized_vol < vol_low:
                self._current_regime = RegimeLabel.LOW_VOL
            else:
                self._current_regime = RegimeLabel.NORMAL

        # Override for trend regimes
        if len(arr) > self.config.trend_lookback:
            cumulative = np.sum(arr[-self.config.trend_lookback:])
            if cumulative > self.config.trend_threshold:
                self._current_regime = RegimeLabel.TREND_UP
            elif cumulative < -self.config.trend_threshold:
                self._current_regime = RegimeLabel.TREND_DOWN

        # Override for high correlation
        if arr.shape[1] > 1 and len(arr) >= self.config.corr_lookback:
            block = arr[-self.config.corr_lookback:]
            cm = np.corrcoef(block.T)
            mask = ~np.eye(cm.shape[0], dtype=bool)
            mean_corr = float(np.mean(cm[mask]))
            if mean_corr > self.config.corr_threshold:
                self._current_regime = RegimeLabel.HIGH_CORR

        return self._current_regime

    @property
    def current_regime(self) -> RegimeLabel:
        return self._current_regime

    @property
    def current_vol(self) -> float | None:
        return self._vol_history[-1] if self._vol_history else None


class StrategyMonitor:
    """Strategy-level performance and signal quality monitor.

    Tracks signal history per strategy, computes rolling quality metrics,
    detects regime changes, and generates alerts via the shared AlertEngine.
    Designed to be called after each on_data() cycle in the live runner.
    """

    def __init__(
        self,
        alert_engine: AlertEngine | None = None,
        regime_detector: RegimeDetector | None = None,
        strategy_name: str = "",
        max_signal_history: int = 1000,
    ) -> None:
        self.alert_engine = alert_engine or AlertEngine()
        self.regime_detector = regime_detector or RegimeDetector()
        self.strategy_name = strategy_name
        self._signal_history: list[SignalSnapshot] = []
        self._max_history = max_signal_history
        self._returns_history: list[float] = []
        self._consecutive_losses = 0
        self._peak_value = 100.0
        self._current_value = 100.0
        self._last_health: StrategyHealth | None = None
        self._registered_rules: set[str] = set()

    def register_default_rules(self) -> None:
        """Register standard strategy monitoring alert rules."""
        rules = build_strategy_alert_rules(self.strategy_name)
        for rule in rules:
            if rule.rule_id not in self._registered_rules:
                self.alert_engine.add_rule(rule)
                self._registered_rules.add(rule.rule_id)

    def record_signals(
        self,
        signals: list[TradeSignal],
        close_prices: NDArray | None = None,
        returns: NDArray | None = None,
    ) -> SignalSnapshot:
        """Record a batch of signals from on_data() and update health."""
        now = datetime.utcnow()

        # Build snapshot
        strengths = [abs(s.strength) for s in signals]
        directions = [s.direction for s in signals]
        long_count = sum(1 for d in directions if d > 0)

        snapshot = SignalSnapshot(
            timestamp=now,
            n_signals=len(signals),
            avg_strength=float(np.mean(strengths)) if strengths else 0.0,
            max_strength=float(np.max(strengths)) if strengths else 0.0,
            min_strength=float(np.min(strengths)) if strengths else 0.0,
            long_pct=long_count / max(len(signals), 1),
            gross_leverage=float(np.sum(strengths)),
            symbols=[s.symbol for s in signals],
            metadata={},
        )

        # Regime detection
        if returns is not None and len(returns) > 0:
            snapshot.regime = self.regime_detector.update(returns)

        self._signal_history.append(snapshot)
        if len(self._signal_history) > self._max_history:
            self._signal_history.pop(0)

        # Track portfolio value for drawdown/loss tracking
        if returns is not None:
            daily_ret = float(np.mean(returns))
            self._returns_history.append(daily_ret)
            if len(self._returns_history) > 500:
                self._returns_history.pop(0)

            self._current_value *= (1 + daily_ret)
            if daily_ret < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            if self._current_value > self._peak_value:
                self._peak_value = self._current_value

        # Evaluate alerts
        if self.alert_engine:
            state = self._build_alert_state(snapshot)
            self.alert_engine.evaluate(state)

        return snapshot

    def _build_alert_state(self, snapshot: SignalSnapshot) -> dict[str, Any]:
        """Build alert state dict from current snapshot and health."""
        drawdown_pct = (self._current_value - self._peak_value) / max(self._peak_value, 1e-10)
        return {
            "strategy": self.strategy_name,
            "n_signals": snapshot.n_signals,
            "avg_strength": snapshot.avg_strength,
            "gross_leverage": snapshot.gross_leverage,
            "long_pct": snapshot.long_pct,
            "regime": snapshot.regime.value if snapshot.regime else "unknown",
            "consecutive_losses": self._consecutive_losses,
            "drawdown_pct": drawdown_pct * 100,
            "current_value": self._current_value,
            "peak_value": self._peak_value,
        }

    @property
    def signal_count(self) -> int:
        return len(self._signal_history)

    @property
    def health(self) -> StrategyHealth:
        """Compute current strategy health assessment."""
        if not self._signal_history:
            return StrategyHealth(strategy_name=self.strategy_name)

        recent = self._signal_history[-min(20, len(self._signal_history)):]

        # Signal quality
        avg_strength = np.mean([s.avg_strength for s in recent])
        n_signals_avg = np.mean([s.n_signals for s in recent])

        # Drawdown
        drawdown_pct = (self._current_value - self._peak_value) / max(self._peak_value, 1e-10)

        # Rolling Sharpe
        recent_sharpe = None
        if len(self._returns_history) >= 20:
            r = np.array(self._returns_history[-20:])
            recent_sharpe = float(np.mean(r) / max(np.std(r, ddof=1), 1e-10) * np.sqrt(252))

        # Quality assessment
        warnings: list[str] = []
        errors: list[str] = []
        quality = SignalQuality.GOOD

        if self._consecutive_losses >= 5:
            warnings.append(f"{self._consecutive_losses} consecutive losing signals")
            quality = SignalQuality.DEGRADED
        if self._consecutive_losses >= 10:
            errors.append(f"Critical: {self._consecutive_losses} consecutive losses")
            quality = SignalQuality.POOR
        if drawdown_pct < -0.10:
            errors.append(f"Drawdown {drawdown_pct:.1%} exceeds 10% threshold")
            quality = SignalQuality.POOR
        elif drawdown_pct < -0.05:
            warnings.append(f"Drawdown {drawdown_pct:.1%} exceeds 5% threshold")
            quality = SignalQuality.DEGRADED
        if avg_strength < 0.05 and n_signals_avg > 0:
            warnings.append(f"Signal strength very low ({avg_strength:.3f})")
            quality = SignalQuality.DEGRADED
        if recent_sharpe is not None and recent_sharpe < -1.0:
            warnings.append(f"20-day Sharpe {recent_sharpe:.2f} below -1.0")
            quality = SignalQuality.DEGRADED

        regime_label = None
        recent_snapshot = self._signal_history[-1]
        if recent_snapshot.regime:
            regime_label = recent_snapshot.regime

        self._last_health = StrategyHealth(
            strategy_name=self.strategy_name,
            signal_quality=quality,
            regime=regime_label,
            regime_scale=1.0,
            consecutive_losses=self._consecutive_losses,
            max_drawdown_pct=float(drawdown_pct),
            days_since_rebalance=0,
            recent_sharpe=recent_sharpe,
            signal_count_avg=n_signals_avg,
            warnings=warnings,
            errors=errors,
        )
        return self._last_health

    @property
    def signal_history(self) -> list[SignalSnapshot]:
        return list(self._signal_history)

    def reset(self) -> None:
        """Reset all tracked state."""
        self._signal_history.clear()
        self._returns_history.clear()
        self._consecutive_losses = 0
        self._peak_value = 100.0
        self._current_value = 100.0
        self._last_health = None


def build_strategy_alert_rules(strategy_name: str = "") -> list[AlertRule]:
    """Build standard alert rules for strategy monitoring."""
    name_tag = f"[{strategy_name}] " if strategy_name else ""

    return [
        AlertRule(
            rule_id=f"{strategy_name}_consecutive_losses" if strategy_name else "consecutive_losses",
            name=f"{name_tag}Consecutive Losses",
            description="Strategy has 5+ consecutive losing signal periods",
            severity=AlertSeverity.WARNING,
            condition=lambda s: int(s.get("consecutive_losses", 0)) >= 5,
        ),
        AlertRule(
            rule_id=f"{strategy_name}_drawdown_5pct" if strategy_name else "drawdown_5pct",
            name=f"{name_tag}Drawdown > 5%",
            description="Strategy drawdown exceeds 5%",
            severity=AlertSeverity.WARNING,
            condition=lambda s: float(s.get("drawdown_pct", 0)) < -5.0,
        ),
        AlertRule(
            rule_id=f"{strategy_name}_drawdown_10pct" if strategy_name else "drawdown_10pct",
            name=f"{name_tag}Drawdown > 10%",
            description="Strategy drawdown exceeds 10%",
            severity=AlertSeverity.CRITICAL,
            condition=lambda s: float(s.get("drawdown_pct", 0)) < -10.0,
        ),
        AlertRule(
            rule_id=f"{strategy_name}_signal_starvation" if strategy_name else "signal_starvation",
            name=f"{name_tag}Signal Starvation",
            description="Strategy emitting fewer than expected signals",
            severity=AlertSeverity.WARNING,
            condition=lambda s: int(s.get("n_signals", 10)) == 0,
        ),
        AlertRule(
            rule_id=f"{strategy_name}_high_leverage" if strategy_name else "high_leverage",
            name=f"{name_tag}High Leverage",
            description="Strategy gross leverage exceeds 1.5",
            severity=AlertSeverity.WARNING,
            condition=lambda s: float(s.get("gross_leverage", 0)) > 1.5,
        ),
        AlertRule(
            rule_id=f"{strategy_name}_high_vol_regime" if strategy_name else "high_vol_regime",
            name=f"{name_tag}High Vol Regime",
            description="Strategy operating in high-vol regime",
            severity=AlertSeverity.INFO,
            condition=lambda s: s.get("regime", "") == "high_vol",
        ),
    ]
