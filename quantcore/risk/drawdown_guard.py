"""Drawdown circuit breaker with cooldown period.

Wraps the existing DrawdownTracker with cooldown logic: after the flatten
threshold is breached, the guard forces a position liquidation and blocks
new trades for a configurable cooldown period. A recovery gate ensures the
portfolio has healed before re-entering.

The scale_factor property is the single integration point — strategies
multiply their weights by this factor each rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantcore.risk.drawdown import DrawdownAction, DrawdownConfig, DrawdownTracker


@dataclass
class DrawdownGuardConfig:
    flatten_threshold: float = -0.15
    reduce_thresholds: list[tuple[float, float]] = field(default_factory=lambda: [
        (-0.05, 0.75),
        (-0.10, 0.50),
    ])
    cooldown_days: int = 7
    recovery_threshold: float = -0.05


@dataclass
class DrawdownGuardState:
    peak: float
    current_equity: float
    drawdown_pct: float
    scale_factor: float
    is_in_cooldown: bool
    cooldown_remaining: int
    action: str


class DrawdownGuard:
    """Drawdown circuit breaker with cooldown and recovery gate.

    Usage:
        guard = DrawdownGuard()
        state = guard.update(equity=portfolio_nav)
        weights *= guard.scale_factor  # 0.0 during flatten/cooldown
    """

    def __init__(self, config: DrawdownGuardConfig | None = None) -> None:
        self.config = config or DrawdownGuardConfig()
        dd_config = DrawdownConfig(
            thresholds=self._build_dd_thresholds(),
            halt_threshold=-1.0,  # Halt controlled by cooldown logic, not DrawdownTracker
        )
        self._tracker = DrawdownTracker(config=dd_config)
        self._current_equity: float = 0.0
        self._cooldown_counter: int = 0
        self._flattened: bool = False
        self._day_count: int = 0

    def _build_dd_thresholds(self) -> list[tuple[float, DrawdownAction]]:
        thresholds: list[tuple[float, DrawdownAction]] = []
        for dd_level, _scale in self.config.reduce_thresholds:
            if dd_level <= self.config.flatten_threshold:
                thresholds.append((dd_level, DrawdownAction.REDUCE_50))
            else:
                thresholds.append((dd_level, DrawdownAction.REDUCE_25))
        thresholds.append((self.config.flatten_threshold, DrawdownAction.FLATTEN))
        return sorted(thresholds, key=lambda t: t[0])

    def update(self, equity: float, day_count: int | None = None) -> DrawdownGuardState:
        """Update guard with current equity value.

        Args:
            equity: Current portfolio NAV.
            day_count: Optional bar counter for cooldown tracking. If None,
                increments by 1 each call (assumes daily calls).
        """
        if day_count is not None:
            self._day_count = day_count
        else:
            self._day_count += 1

        self._current_equity = equity
        dd_state = self._tracker.update(equity)
        dd_pct = dd_state.drawdown

        # Cooldown logic
        if self._flattened and self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # Check flatten trigger
        if dd_pct <= self.config.flatten_threshold and not self._flattened:
            self._flattened = True
            self._cooldown_counter = self.config.cooldown_days

        # Recovery: exit flatten if cooldown expired AND DD recovered
        if self._flattened:
            cooldown_done = self._cooldown_counter <= 0
            recovered = dd_pct > self.config.recovery_threshold
            if cooldown_done and recovered:
                self._flattened = False

        action, scale = self._resolve(dd_pct)
        cooldown_remaining = self._cooldown_counter if self._flattened else 0

        return DrawdownGuardState(
            peak=self._tracker.peak,
            current_equity=equity,
            drawdown_pct=dd_pct,
            scale_factor=scale,
            is_in_cooldown=self._flattened and self._cooldown_counter > 0,
            cooldown_remaining=cooldown_remaining,
            action=action,
        )

    def _resolve(self, dd_pct: float) -> tuple[str, float]:
        if self._flattened and self._cooldown_counter > 0:
            return ("cooldown", 0.0)
        if self._flattened:
            return ("flatten", 0.0)
        if dd_pct <= self.config.flatten_threshold:
            return ("flatten", 0.0)

        scale = 1.0
        for dd_level, scale_at_level in sorted(
            self.config.reduce_thresholds, key=lambda t: t[0]
        ):
            if dd_pct <= dd_level:
                scale = scale_at_level

        if scale < 1.0:
            return ("reduce", scale)
        return ("normal", 1.0)

    @property
    def scale_factor(self) -> float:
        if self._flattened:
            return 0.0
        dd = self._tracker.current_drawdown
        scale = 1.0
        for dd_level, scale_at_level in sorted(
            self.config.reduce_thresholds, key=lambda t: t[0]
        ):
            if dd <= dd_level:
                scale = scale_at_level
        return scale

    @property
    def is_in_cooldown(self) -> bool:
        return self._flattened and self._cooldown_counter > 0

    @property
    def is_flattened(self) -> bool:
        return self._flattened

    @property
    def current_drawdown(self) -> float:
        return self._tracker.current_drawdown

    def reset(self) -> None:
        self._tracker = DrawdownTracker(config=DrawdownConfig(
            thresholds=self._build_dd_thresholds(),
            halt_threshold=-1.0,
        ))
        self._cooldown_counter = 0
        self._flattened = False
        self._day_count = 0


# ---------------------------------------------------------------------------
# Vol-Adjusted Drawdown Guard (Task 3.2 from audit)
# ---------------------------------------------------------------------------

from dataclasses import field

import numpy as np


class VolAdjustedConfig:
    """Config: thresholds scale with vol ratio."""
    def __init__(self, base_flatten=-0.15, reduce=None, vol_window=60,
                 recovery_window=63, recovery_gap=-0.02, vol_median_window=1260):
        self.base_flatten_threshold = base_flatten
        self.base_reduce_thresholds = reduce or [(-0.05, 0.75), (-0.10, 0.50)]
        self.vol_window = vol_window
        self.recovery_window = recovery_window
        self.recovery_gap = recovery_gap
        self.vol_median_window = vol_median_window


class VolAdjustedState:
    def __init__(self, dd, eff_flatten, vol_ratio, scale, action, in_gate):
        self.drawdown_pct = dd
        self.effective_flatten_threshold = eff_flatten
        self.vol_ratio = vol_ratio
        self.scale_factor = scale
        self.action = action
        self.in_recovery_gate = in_gate


class VolAdjustedDrawdownGuard:
    """Vol-scaled thresholds + state-machine recovery gate (Task 3.2)."""

    def __init__(self, config=None):
        self.config = config or VolAdjustedConfig()
        self._equity_history = []
        self._return_history = []
        self._state = "normal"
        self._peak = 0.0

    def update(self, equity):
        if self._peak == 0.0:
            self._peak = equity
        if self._equity_history:
            prev = self._equity_history[-1]
            ret = (equity - prev) / prev if prev > 0 else 0.0
            self._return_history.append(ret)
        self._equity_history.append(equity)
        self._peak = max(self._peak, equity)
        max_hist = max(self.config.vol_median_window, self.config.recovery_window) + 10
        if len(self._equity_history) > max_hist:
            self._equity_history = self._equity_history[-max_hist:]
        if len(self._return_history) > max_hist:
            self._return_history = self._return_history[-max_hist:]
        dd_pct = (equity - self._peak) / self._peak if self._peak > 0 else 0.0
        vol_ratio = self._compute_vol_ratio()
        eff_flatten = self._scale_threshold(self.config.base_flatten_threshold, vol_ratio)
        eff_reduces = [(self._scale_threshold(t, vol_ratio), sc)
                       for t, sc in self.config.base_reduce_thresholds]
        # State transitions
        if self._state in ("normal", "reduce"):
            if dd_pct <= eff_flatten:
                self._state = "flatten"
            else:
                scale = 1.0
                for t_thr, sc in sorted(eff_reduces, key=lambda x: x[0]):
                    if dd_pct <= t_thr:
                        scale = sc
                self._state = "reduce" if scale < 1.0 else "normal"
        elif self._state == "flatten":
            if dd_pct > eff_flatten * 0.5:
                self._state = "recovery_gate"
        elif self._state == "recovery_gate":
            recent_max = self._trailing_max(self.config.recovery_window)
            gap = (equity - recent_max) / recent_max if recent_max > 0 else 0.0
            if gap >= self.config.recovery_gap:
                self._state = "normal"
        if self._state == "normal":
            sf = 1.0
        elif self._state == "reduce":
            sf = 1.0
            for t_thr, sc in sorted(eff_reduces, key=lambda x: x[0]):
                if dd_pct <= t_thr:
                    sf = sc
        else:
            sf = 0.0
        return VolAdjustedState(dd_pct, eff_flatten, vol_ratio, sf,
                                self._state, self._state == "recovery_gate")

    def _compute_vol_ratio(self):
        rets = self._return_history
        if len(rets) < 10:
            return 1.0
        w = self.config.vol_window
        cur = float(np.std(rets[-min(w, len(rets)):], ddof=1)) * 252**0.5
        med_w = self.config.vol_median_window
        if len(rets) >= med_w:
            rv = [float(np.std(rets[i:i+w], ddof=1))*252**0.5
                  for i in range(0, len(rets)-w, w)]
            med = float(np.median(rv)) if rv else cur
        else:
            med = cur
        return cur / med if med > 1e-10 else 1.0

    def _scale_threshold(self, base, vol_ratio):
        scale = (max(0.25, min(vol_ratio, 4.0))) ** 0.5
        return float(base * scale)

    def _trailing_max(self, window):
        if not self._equity_history:
            return 0.0
        return float(max(self._equity_history[-min(window, len(self._equity_history)):]))

    def reset(self):
        self._equity_history = []
        self._return_history = []
        self._state = "normal"
        self._peak = 0.0
