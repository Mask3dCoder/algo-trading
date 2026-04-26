"""Drawdown tracking and automatic risk reduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class DrawdownAction(str, Enum):
    NONE = "none"
    REDUCE_25 = "reduce_25"
    REDUCE_50 = "reduce_50"
    REDUCE_75 = "reduce_75"
    FLATTEN = "flatten"
    HALT = "halt"


@dataclass
class DrawdownState:
    """Current drawdown state."""
    peak: float
    current_nav: float
    drawdown: float  # Fractional drawdown (negative)
    drawdown_pct: float  # Percentage drawdown
    duration_days: int
    max_drawdown_pct: float
    max_drawdown_date: datetime | None = None
    action: DrawdownAction = DrawdownAction.NONE


@dataclass
class DrawdownConfig:
    """Configurable drawdown thresholds and actions."""
    thresholds: list[tuple[float, DrawdownAction]] = field(default_factory=lambda: [
        (-0.05, DrawdownAction.REDUCE_25),
        (-0.10, DrawdownAction.REDUCE_50),
        (-0.15, DrawdownAction.REDUCE_75),
        (-0.20, DrawdownAction.FLATTEN),
    ])
    halt_threshold: float = -0.25


class DrawdownTracker:
    """Real-time drawdown monitoring with automatic action triggers."""

    def __init__(
        self,
        config: DrawdownConfig | None = None,
        on_action: Callable[[DrawdownAction, DrawdownState], None] | None = None,
    ) -> None:
        self.config = config or DrawdownConfig()
        self.on_action = on_action
        self._peak: float | None = None
        self._current_nav: float = 0.0
        self._max_dd: float = 0.0
        self._max_dd_date: datetime | None = None
        self._peak_date: datetime | None = None
        self._dd_start: datetime | None = None
        self._last_action: DrawdownAction = DrawdownAction.NONE
        self._history: list[tuple[datetime, float]] = []

    def update(self, nav: float, ts: datetime | None = None) -> DrawdownState:
        """Update the tracker with a new NAV and return current state."""
        timestamp = ts or datetime.utcnow()
        self._current_nav = nav
        self._history.append((timestamp, nav))

        # Update peak
        if self._peak is None or nav > self._peak:
            self._peak = nav
            self._peak_date = timestamp
            self._dd_start = None
        else:
            if self._dd_start is None:
                self._dd_start = timestamp

        # Current drawdown
        dd = (nav - self._peak) / self._peak if self._peak > 0 else 0.0

        # Max drawdown
        if abs(dd) > abs(self._max_dd):
            self._max_dd = dd
            self._max_dd_date = timestamp

        # Duration
        duration = 0
        if self._dd_start is not None:
            duration = (timestamp - self._dd_start).days

        # Determine action
        action = self._determine_action(dd)

        state = DrawdownState(
            peak=self._peak,
            current_nav=nav,
            drawdown=dd,
            drawdown_pct=dd * 100.0,
            duration_days=duration,
            max_drawdown_pct=self._max_dd * 100.0,
            max_drawdown_date=self._max_dd_date,
            action=action,
        )

        if action != DrawdownAction.NONE and action != self._last_action:
            self._last_action = action
            if self.on_action:
                self.on_action(action, state)

        return state

    def _determine_action(self, dd: float) -> DrawdownAction:
        """Determine the action for a given drawdown level."""
        if dd <= self.config.halt_threshold:
            return DrawdownAction.HALT

        action = DrawdownAction.NONE
        for threshold, dd_action in self.config.thresholds:
            if dd <= threshold:
                action = dd_action
        return action

    @property
    def peak(self) -> float:
        return self._peak or 0.0

    @property
    def current_drawdown(self) -> float:
        if self._peak is None or self._peak == 0:
            return 0.0
        return (self._current_nav - self._peak) / self._peak

    @property
    def max_drawdown(self) -> float:
        return self._max_dd

    @property
    def is_in_drawdown(self) -> bool:
        return self.current_drawdown < 0.0


def drawdown_at_risk(
    returns: NDArray[np.floating],
    confidence: float = 0.95,
) -> float:
    """Probabilistic maximum drawdown estimate (Drawdown-at-Risk).

    Bootstrap the max drawdown distribution and take the (1-confidence) percentile.
    """
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)

    if n < 50:
        return float(np.abs(np.min(np.minimum.accumulate(np.cumsum(r)) - np.cumsum(r))))

    # Bootstrap
    n_boot = 1000
    max_dds = np.zeros(n_boot)
    rng = np.random.default_rng(42)

    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_returns = r[idx]
        cumulative = np.cumsum(boot_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        max_dds[i] = float(np.abs(drawdowns.min()))

    return float(np.percentile(max_dds, confidence * 100))


def conditional_dar(
    returns: NDArray[np.floating],
    confidence: float = 0.95,
) -> float:
    """Conditional Drawdown-at-Risk: average drawdown beyond DaR threshold."""
    r = np.asarray(returns, dtype=float).flatten()
    n = len(r)

    if n < 50:
        return drawdown_at_risk(r, confidence)

    cumulative = np.cumsum(r)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = np.abs(cumulative - running_max)

    dar = drawdown_at_risk(r, confidence)
    exceedances = drawdowns[drawdowns >= dar]
    if len(exceedances) == 0:
        return dar
    return float(np.mean(exceedances))


def recovery_estimate(
    current_dd: float,
    avg_return: float,
) -> float:
    """Estimate number of days to recover from current drawdown.

    Recovery_days ≈ |current_dd| / avg_daily_return
    """
    if avg_return <= 0 or current_dd >= 0:
        return float("inf")
    return float(abs(current_dd) / avg_return)


def compute_calmar_ratio(
    returns: NDArray[np.floating],
    window: int = 252,
) -> float:
    """Rolling Calmar ratio: annualized return / max drawdown."""
    r = np.asarray(returns, dtype=float).flatten()
    if len(r) < window:
        window = len(r)

    r_window = r[-window:]
    ann_return = float(np.mean(r_window)) * 252
    cumulative = np.cumsum(r_window)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = float(np.abs((cumulative - running_max).min()))

    if max_dd == 0:
        return 0.0
    return ann_return / max_dd
