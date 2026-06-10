"""Per-strategy drawdown kill-switch for Phase 4 validation.

The kill-switch is intentionally conservative and stateful. It tracks each
strategy's equity curve independently and halts a strategy when its drawdown
from peak exceeds the configured threshold. Once halted, the strategy remains
halted until an explicit reset; it is not re-enabled by parameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class StrategyKillSwitchConfig:
    """Configuration for a strategy-level drawdown kill-switch."""

    max_drawdown_pct: float = -0.20
    min_history_points: int = 1
    halt_on_breach: bool = True


@dataclass(slots=True)
class StrategyKillSwitchState:
    """Current state for one strategy kill-switch."""

    strategy_name: str
    current_equity: float
    peak_equity: float
    drawdown_pct: float
    is_halted: bool
    breach_count: int
    last_updated: str
    history: list[float] = field(default_factory=list)


class StrategyKillSwitch:
    """Track independent drawdown halts for multiple strategies."""

    def __init__(self, config: StrategyKillSwitchConfig | None = None) -> None:
        self.config = config or StrategyKillSwitchConfig()
        self._peaks: dict[str, float] = {}
        self._history: dict[str, list[float]] = {}
        self._halted: set[str] = set()
        self._breach_count: dict[str, int] = {}

    def update(
        self,
        strategy_name: str,
        equity: float,
        timestamp: datetime | None = None,
    ) -> StrategyKillSwitchState:
        """Update one strategy and return its kill-switch state."""
        if equity <= 0:
            raise ValueError("equity must be positive")

        if strategy_name not in self._peaks:
            self._peaks[strategy_name] = equity
            self._history[strategy_name] = []
            self._breach_count[strategy_name] = 0

        peak = self._peaks[strategy_name]
        if equity > peak:
            peak = equity
            self._peaks[strategy_name] = peak

        drawdown = (equity - peak) / peak if peak > 0 else 0.0
        history = self._history[strategy_name]
        history.append(equity)
        if len(history) > 500:
            history = history[-500:]
            self._history[strategy_name] = history

        is_halted = strategy_name in self._halted
        if (
            self.config.halt_on_breach
            and drawdown <= self.config.max_drawdown_pct
            and len(history) >= self.config.min_history_points
        ):
            self._halted.add(strategy_name)
            self._breach_count[strategy_name] += 1
            is_halted = True

        return StrategyKillSwitchState(
            strategy_name=strategy_name,
            current_equity=equity,
            peak_equity=peak,
            drawdown_pct=drawdown,
            is_halted=is_halted,
            breach_count=self._breach_count[strategy_name],
            last_updated=(timestamp or datetime.now(UTC)).isoformat(),
            history=list(history),
        )

    def is_halted(self, strategy_name: str) -> bool:
        return strategy_name in self._halted

    def halted_strategies(self) -> set[str]:
        return set(self._halted)

    def reset(self, strategy_name: str | None = None) -> None:
        if strategy_name is None:
            self._peaks.clear()
            self._history.clear()
            self._halted.clear()
            self._breach_count.clear()
            return
        self._peaks.pop(strategy_name, None)
        self._history.pop(strategy_name, None)
        self._halted.discard(strategy_name)
        self._breach_count.pop(strategy_name, None)

    def snapshot(self) -> dict[str, Any]:
        return {
            "halted": sorted(self._halted),
            "peaks": dict(self._peaks),
            "breach_count": dict(self._breach_count),
        }
