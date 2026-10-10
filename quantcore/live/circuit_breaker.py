"""Circuit breaker for live trading with graduated halt states.

States
------
NORMAL            : trading proceeds normally
DAILY_LOSS_HALT   : trading halted for the day (daily loss limit hit)
CRITICAL_HALT     : all trading halted (requires manual reset)

The circuit breaker is checked at pre-trade time and at end-of-day.
It can only transition forward (NORMAL → DAILY_LOSS_HALT → CRITICAL_HALT).
A manual ``reset()`` is required to move back to NORMAL.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

log = logging.getLogger(__name__)


class CircuitBreakerState(str, Enum):
    """Graduated halt states."""

    NORMAL = "normal"
    WARNING = "warning"           # elevated correlation — operator alert
    DAILY_LOSS_HALT = "daily_loss_halt"
    CRITICAL_HALT = "critical_halt"


@dataclass
class CircuitBreakerConfig:
    """Thresholds for circuit-breaker transitions."""

    daily_loss_pct: float = 0.02           # 2% equity loss → daily halt
    critical_drawdown_pct: float = 0.10    # 10% drawdown → critical halt
    consecutive_rejections: int = 5          # n rejections in a row → halt
    # High-correlation regime: when DCC-GARCH conditional corr > this,
    # daily loss limit is halved because strategies move together → portfolio
    # risk is higher than normal.
    corr_threshold: float = 0.70
    high_corr_days_persistence_alert: int = 5
    # Env var prefix for overrides
    _env_prefix: str = "QUANTCORE_RISK_"

    @classmethod
    def from_env(cls) -> CircuitBreakerConfig:
        return cls(
            daily_loss_pct=_env_float("QUANTCORE_RISK_DAILY_LOSS_PCT", "0.02"),
            critical_drawdown_pct=_env_float("QUANTCORE_RISK_CRITICAL_DRAWDOWN_PCT", "0.10"),
            consecutive_rejections=int(
                os.environ.get("QUANTCORE_RISK_CONSECUTIVE_REJECTIONS", "5")
            ),
            corr_threshold=float(
                os.environ.get("QUANTCORE_RISK_CORR_THRESHOLD", "0.70")),
            high_corr_days_persistence_alert=int(
                os.environ.get("QUANTCORE_RISK_HIGH_CORR_PERSISTENCE", "5")),
        )


def _env_float(key: str, default: str) -> float:
    try:
        return float(os.environ.get(key, default))
    except ValueError:
        log.warning("Invalid %s, using default %s", key, default)
        return float(default)


# ---------------------------------------------------------------------------
# CircuitBreaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Live-trading circuit breaker with graduated halt states.

    Parameters
    ----------
    config : CircuitBreakerConfig, optional
        Threshold configuration. Defaults to environment-driven values.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None) -> None:
        self.config = config or CircuitBreakerConfig.from_env()
        self._state = CircuitBreakerState.NORMAL
        self._reason = ""
        self._activation_time: datetime | None = None
        self._rejection_streak = 0
        self._daily_loss_halted_date: date | None = None

        # Correlation-regime tracking (updated by DCC-GARCH via check())
        self._correlation_regime: str = "NORMAL"
        self._conditional_correlation: float = 0.0
        self._high_corr_days: int = 0  # consecutive days in elevated crisis regime

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def activated(self) -> bool:
        return self._state != CircuitBreakerState.NORMAL

    def can_trade(self) -> bool:
        """Return True if trading is permitted."""
        if self._state == CircuitBreakerState.NORMAL:
            return True
        # WARNING is a non-blocking alert — trading is still permitted
        if self._state == CircuitBreakerState.WARNING:
            return True
        # DAILY_LOSS_HALT only applies until next calendar day
        if self._state == CircuitBreakerState.DAILY_LOSS_HALT:
            today = date.today()
            if self._daily_loss_halted_date != today:
                return True
        return False

    # ------------------------------------------------------------------
    # Check / transition
    # ------------------------------------------------------------------

    def check(
        self,
        equity: float,
        peak_equity: float,
        daily_pnl: float,
        rejected_last: bool = False,
        correlation_regime: str = "NORMAL",
        conditional_correlation: float = 0.0,
        correlation_regime_changed: bool = False,
    ) -> CircuitBreakerState:
        """Evaluate conditions and possibly transition state.

        Parameters
        ----------
        equity : float
            Current portfolio equity.
        peak_equity : float
            High-water equity mark.
        daily_pnl : float
            P&L since market open.
        rejected_last : bool
            Whether the last order was rejected.

        Returns
        -------
        CircuitBreakerState
            The (possibly updated) state.
        """
        # 0. Update and log correlation regime (DCC-GARCH feeds this in)
        prev_regime = self._correlation_regime
        self._correlation_regime = correlation_regime
        self._conditional_correlation = conditional_correlation

        if prev_regime != correlation_regime:
            log.info(
                "CORRELATION REGIME: %s → %s | cond_corr=%.3f | high_corr_days=%d",
                prev_regime, correlation_regime,
                conditional_correlation, self._high_corr_days,
            )
            # Alert via logging for live monitoring
            if correlation_regime == "CRISIS":
                log.warning(
                    "CRISIS correlation regime: portfolio risk elevated. "
                    "All strategies highly correlated — circuit breaker limits tightened.",
                )
            elif correlation_regime == "ELEVATED":
                log.info(
                    "ELEVATED correlation regime: daily loss limit halved until regime normalises.",
                )

        # Count consecutive high-corr days
        if correlation_regime in ("ELEVATED", "CRISIS"):
            self._high_corr_days += 1
        else:
            self._high_corr_days = 0

        # In elevated Crisis regime, persist high-corr → WARN state (not a halt, but alert)
        if (self._high_corr_days >= self.config.high_corr_days_persistence_alert
                and self._state == CircuitBreakerState.NORMAL):
            self._transition(
                CircuitBreakerState.WARNING,
                f"High correlation regime persisted {self._high_corr_days} days "
                f"(cond_corr={conditional_correlation:.3f}) — operator attention required",
            )
            self._high_corr_days = 0  # reset counter after triggering

        # Log equity snapshot BEFORE evaluation so false positives
        # can be traced immediately from the log output.
        log.info(
            "Circuit breaker check: equity=$%.2f peak=$%.2f daily_pnl=$%.2f "
            "state=%s streak=%d",
            equity, peak_equity, daily_pnl,
            self._state.value, self._rejection_streak,
        )

        # Streak tracking
        if rejected_last:
            self._rejection_streak += 1
        else:
            self._rejection_streak = 0

        # 1. Consecutive rejections → daily halt (early warning)
        if self._rejection_streak >= self.config.consecutive_rejections:
            if self._state == CircuitBreakerState.NORMAL:
                self._transition(
                    CircuitBreakerState.DAILY_LOSS_HALT,
                    f"{self._rejection_streak} consecutive order rejections",
                )
            return self._state

        # 2. Daily loss limit — tightened if correlation regime is elevated/crisis
        # When strategies are highly correlated, portfolio risk is higher per unit of loss.
        effective_daily_loss_pct = self.config.daily_loss_pct
        if self._correlation_regime in ("ELEVATED", "CRISIS"):
            effective_daily_loss_pct *= 0.5  # halve the limit in high-corr regimes
            log.info(
                "High-corr regime active: effective daily loss limit = %.2f%% (was %.2f%%)",
                effective_daily_loss_pct * 100, self.config.daily_loss_pct * 100,
            )
        if daily_pnl < -equity * effective_daily_loss_pct:
            if self._state in (CircuitBreakerState.NORMAL, CircuitBreakerState.WARNING, CircuitBreakerState.DAILY_LOSS_HALT):
                self._transition(
                    CircuitBreakerState.DAILY_LOSS_HALT,
                    f"Daily loss {daily_pnl:,.2f} exceeds {effective_daily_loss_pct:.1%} of equity "
                    f"(correlation_regime={self._correlation_regime}, cond_corr={self._conditional_correlation:.3f})",
                )
            return self._state

        # 3. Critical drawdown
        if peak_equity > 0:
            dd = (equity - peak_equity) / peak_equity
            if dd < -self.config.critical_drawdown_pct:
                self._transition(
                    CircuitBreakerState.CRITICAL_HALT,
                    f"Drawdown {dd:.2%} exceeds critical limit {self.config.critical_drawdown_pct:.1%}",
                )

        return self._state

    # ------------------------------------------------------------------
    # Manual / forced transitions
    # ------------------------------------------------------------------

    def trigger(self, state: CircuitBreakerState, reason: str = "") -> None:
        """Force the circuit breaker to a given state."""
        self._transition(state, reason)

    def reset(self) -> None:
        """Reset to NORMAL (requires operator confirmation)."""
        self._state = CircuitBreakerState.NORMAL
        self._reason = ""
        self._activation_time = None
        self._rejection_streak = 0
        self._daily_loss_halt_date = None
        log.info("Circuit breaker reset to NORMAL")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _transition(self, new_state: CircuitBreakerState, reason: str) -> None:
        if new_state == self._state:
            return
        self._state = new_state
        self._reason = reason
        self._activation_time = datetime.utcnow()
        if new_state == CircuitBreakerState.DAILY_LOSS_HALT:
            self._daily_loss_halted_date = date.today()
        log.critical(
            "CIRCUIT BREAKER → %s | %s", new_state.value.upper(), reason
        )
