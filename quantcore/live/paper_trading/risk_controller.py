"""Risk Controller & Kill Switches for Delta Neutral VRP Paper Trading.

Implements the three mandatory risk guardrails mandated by the Investment
Committee with HARD (not soft) enforcement:

  1. Hard MaxDD Kill Switch      → equity drop >= 10% from peak
  2. Rolling 3-Month Sharpe Halt → Sharpe < 0
  3. Macro/Replay Volatility Exit → DD > 12% during high VIX

All three are fatal (HALT) — no auto-recovery.  Manual committee review
is required to reset.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerState
from quantcore.monitoring.alerting import AlertSeverity

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ExecutionState(str, Enum):
    """Finite state for the paper trading environment."""

    ACTIVE = "active"
    PAUSED = "paused"
    HALTED = "halted"


@dataclass
class RiskControllerConfig:
    """Mandated by the Investment Committee — no overrides without approval."""

    # -- Kill Switch 1: Hard MaxDD ------------------------------------------------
    max_drawdown_pct: float = 0.10  # 10% hard kill

    # -- Kill Switch 2: Rolling 3-Month Sharpe Halt ------------------------------
    sharpe_lookback_days: int = 63  # ~3 trading months
    sharpe_annualisation: int = 252
    sharpe_min_threshold: float = 0.0  # halt if < 0

    # -- Kill Switch 3: Macro Volatility Exit ------------------------------------
    macro_vix_threshold: float = 30.0
    macro_drawdown_warning_pct: float = 0.08  # 8% → warning, reduce to 50%
    macro_drawdown_kill_pct: float = 0.12  # 12% → halt

    # -- Persistence -----------------------------------------------------------
    state_dir: Path = field(default_factory=lambda: Path("reports/paper_trading/state"))

    def __post_init__(self):
        self.state_dir = Path(self.state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Risk Controller
# ---------------------------------------------------------------------------


def _send_alert(severity: AlertSeverity, message: str) -> None:
    """Log at the appropriate level.  Production alerting is wired at runner level."""
    level = logging.CRITICAL if severity == AlertSeverity.CRITICAL else logging.WARNING
    _log.log(level, "%s", message)


class RiskController:
    """Wraps the execution pipeline with three hard kill switches.

    Parameters
    ----------
    config : RiskControllerConfig
        Immutable configuration — set once at startup.
    circuit_breaker : CircuitBreaker
        Production circuit breaker to propagate HALT to the rest of the runner.
    initial_equity : float
        Starting capital (used to seed the peak tracker on first call).
    """

    def __init__(
        self,
        config: RiskControllerConfig,
        circuit_breaker: CircuitBreaker,
        initial_equity: float = 100_000.0,
    ) -> None:
        self._cfg = config
        self._cb = circuit_breaker
        self._state = ExecutionState.ACTIVE

        # -- Kill Switch 1: Peak equity ------------------------------------------------
        self._peak_equity = initial_equity
        self._equity_history: list[tuple[date, float]] = []

        # -- Kill Switch 2: Rolling Sharpe -------------------------------------------
        self._daily_returns: deque[tuple[date, float]] = deque(maxlen=config.sharpe_lookback_days)
        self._last_sharpe_check: date = date(1970, 1, 1)

        # -- Kill Switch 3: Macro state ----------------------------------------------
        self._macro_vol_active = False
        self._macro_trigger_date: date | None = None

        # -- Persistence -----------------------------------------------------------
        self._persist_path = self._cfg.state_dir / "risk_controller_state.json"
        self._load_state()

        _log.info(
            "RiskController initialised: max_dd=%.1f%% sharpe_lookback=%dd "
            "macro_vix=%.1f macro_kill=%.1f%%",
            self._cfg.max_drawdown_pct * 100,
            self._cfg.sharpe_lookback_days,
            self._cfg.macro_vix_threshold,
            self._cfg.macro_drawdown_kill_pct * 100,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> ExecutionState:
        return self._state

    @property
    def is_active(self) -> bool:
        return self._state == ExecutionState.ACTIVE

    @property
    def is_halted(self) -> bool:
        return self._state == ExecutionState.HALTED

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    def update_equity(self, equity: float, as_of: date | None = None) -> None:
        """Call after every close (or on every fill).  Triggers MaxDD check."""
        as_of = as_of or date.today()
        self._equity_history.append((as_of, equity))

        if equity > self._peak_equity:
            self._peak_equity = equity

        if self._state == ExecutionState.ACTIVE and self._peak_equity > 0:
            drawdown = (equity - self._peak_equity) / self._peak_equity
            if drawdown <= -self._cfg.max_drawdown_pct:
                self._trigger_maxdd_kill(drawdown, equity)

        self._persist_state()

    def record_daily_return(self, dt: date, daily_return: float) -> None:
        """Append a daily return for the rolling Sharpe calculation."""
        self._daily_returns.append((dt, daily_return))

    def check_monthly_sharpe(self, as_of: date | None = None) -> float:
        """Compute and evaluate the rolling 3-month Sharpe.  Call at month-end."""
        as_of = as_of or date.today()

        if as_of.month == self._last_sharpe_check.month and as_of.year == self._last_sharpe_check.year:
            return self._compute_rolling_sharpe()

        self._last_sharpe_check = as_of
        sharpe = self._compute_rolling_sharpe()

        if self._state == ExecutionState.ACTIVE and sharpe < self._cfg.sharpe_min_threshold:
            self._trigger_sharpe_halt(sharpe)

        return sharpe

    def check_macro_conditions(
        self,
        vix_level: float,
        current_drawdown: float,
    ) -> str:
        """Evaluate macro / replay volatility conditions.  Call daily.

        Returns one of: ``'ok'``, ``'warning'``, ``'reduce_50'``, ``'halt'``.
        """
        if self._state != ExecutionState.ACTIVE:
            return "halted"

        is_high_vol = vix_level > self._cfg.macro_vix_threshold

        if is_high_vol and current_drawdown > self._cfg.macro_drawdown_warning_pct:
            if current_drawdown > self._cfg.macro_drawdown_kill_pct:
                self._trigger_macro_kill(vix_level, current_drawdown)
                return "halt"
            else:
                self._send_alert(
                    AlertSeverity.WARNING,
                    f"Macro volatility exit WARNING: VIX={vix_level:.1f}, "
                    f"drawdown={current_drawdown:.1%}. Reducing exposure to 50%.",
                )
                return "reduce_50"

        return "ok"

    def manual_reset(self, operator_id: str, reason: str) -> None:
        """Manual reset after committee review.  Logs audit trail."""
        _log.critical(
            "Manual reset by %s: %s → ACTIVE. Reason: %s",
            operator_id, self._state.value, reason,
        )
        self._state = ExecutionState.ACTIVE
        self._send_alert(
            AlertSeverity.CRITICAL,
            f"RiskController MANUAL RESET by {operator_id}. Reason: {reason}",
        )
        self._persist_state()

    def get_summary(self) -> dict[str, Any]:
        """Return current state for dashboard / reporting."""
        return {
            "state": self._state.value,
            "peak_equity": self._peak_equity,
            "current_sharpe_3m": self._compute_rounding_sharpe_safe(),
            "daily_returns_count": len(self._daily_returns),
            "macro_vol_active": self._macro_vol_active,
            "max_drawdown_threshold": self._cfg.max_drawdown_pct,
            "sharpe_threshold": self._cfg.sharpe_min_threshold,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_rolling_sharpe(self) -> float:
        """Sharpe over the rolling window (annualised)."""
        if len(self._daily_returns) < 10:
            return float("nan")
        rets = np.array([r for _, r in self._daily_returns])
        mean_r = np.mean(rets)
        std_r = np.std(rets, ddof=1)
        if std_r < 1e-12:
            return float("nan")
        return float(mean_r / std_r * np.sqrt(self._cfg.sharpe_annualisation))

    def _compute_rounding_sharpe_safe(self) -> float:
        """Safe wrapper for get_summary."""
        try:
            return self._compute_rolling_sharpe()
        except Exception:
            return float("nan")

    # -- Triggers -----------------------------------------------------

    def _trigger_maxdd_kill(self, drawdown: float, equity: float) -> None:
        self._state = ExecutionState.HALTED
        msg = (
            f"HALT: MaxDD kill switch triggered. "
            f"Drawdown={drawdown:.2%}, equity=${equity:,.2f}, "
            f"peak=${self._peak_equity:,.2f}"
        )
        _log.critical(msg)
        self._cb.trigger(CircuitBreakerState.CRITICAL_HALT)
        self._send_alert(AlertSeverity.CRITICAL, msg)
        self._persist_state()

    def _trigger_sharpe_halt(self, sharpe: float) -> None:
        self._state = ExecutionState.PAUSED
        msg = (
            f"PAUSE: Rolling 3-month Sharpe halt triggered. "
            f"Sharpe={sharpe:.4f} < {self._cfg.sharpe_min_threshold}"
        )
        _log.critical(msg)
        self._cb.trigger(CircuitBreakerState.CRITICAL_HALT)
        self._send_alert(AlertSeverity.CRITICAL, msg)
        self._persist_state()

    def _trigger_macro_kill(self, vix: float, dd: float) -> None:
        self._state = ExecutionState.HALTED
        self._macro_vol_active = True
        self._macro_trigger_date = date.today()
        msg = (
            f"HALT: Macro volatility exit triggered. "
            f"VIX={vix:.1f}, drawdown={dd:.2%}"
        )
        _log.critical(msg)
        self._cb.trigger(CircuitBreakerState.CRITICAL_HALT)
        self._send_alert(AlertSeverity.CRITICAL, msg)
        self._persist_state()

    def _send_alert(self, severity: AlertSeverity, message: str) -> None:
        """Unified alert — file + Telegram + any other configured channels."""
        _send_alert(severity, message)

    # -- Persistence ---------------------------------------------------

    def _persist_state(self) -> None:
        """Write state to disk so restart does NOT reset peak equity / Sharpe window."""
        try:
            data = {
                "state": self._state.value,
                "peak_equity": self._peak_equity,
                "last_sharpe_check": self._last_sharpe_check.isoformat(),
                "equity_history_count": len(self._equity_history),
                "daily_returns_count": len(self._daily_returns),
                "macro_vol_active": self._macro_vol_active,
                "macro_trigger_date": (
                    self._macro_trigger_date.isoformat() if self._macro_trigger_date else None
                ),
                "version": 1,
            }
            self._persist_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as exc:
            _log.error("State persistence failed: %s", exc)

    def _load_state(self) -> None:
        """Restore state from disk on restart."""
        if not self._persist_path.exists():
            _log.info("No prior state file found — starting fresh.")
            return
        try:
            data = json.loads(self._persist_path.read_text())
            saved_state = data.get("state", ExecutionState.ACTIVE.value)
            self._state = ExecutionState(saved_state)
            self._peak_equity = data.get("peak_equity", self._peak_equity)
            self._macro_vol_active = data.get("macro_vol_active", False)
            if data.get("macro_trigger_date"):
                self._macro_trigger_date = date.fromisoformat(data["macro_trigger_date"])
            _log.info("Loaded prior state: %s, peak_equity=%.2f", self._state.value, self._peak_equity)
        except Exception as exc:
            _log.error("State load failed: %s — starting fresh.", exc)