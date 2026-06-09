"""Vol Regime Circuit Breaker — portfolio-level risk reduction on VIX spikes.

Portfolio-level breaker (not per-strategy):
- VIX > 30: Reduce all short-vol strategies to 50% size
- VIX 1-day change > 40%: Reduce to 25% size, pause new entries 48h
- Portfolio daily loss > 3%: Flatten ALL options positions, require manual re-auth

This is the ONLY mechanism that overrides strategy-specific circuit breakers
at the portfolio level. Per-strategy breakers remain in effect as secondary guards.

Implementation: Must be checked AFTER each bar and BEFORE signal generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

import numpy as np


class VolCircuitState(str, Enum):
    """Vol regime circuit breaker states."""
    NORMAL = "normal"              # No action needed
    ELEVATED = "elevated"          # VIX > 30, reduced size
    SPIKE = "spike"                # VIX 1d change > 40%, 25% size + 48h pause
    BREACH = "breach"               # Daily loss > 3%, flatten everything
    RECOVERY_GATE = "recovery_gate"  # Post-breach, manual re-auth required


@dataclass
class VolCircuitResult:
    """Result of circuit breaker evaluation."""
    state: VolCircuitState
    current_vix: float = np.nan
    vix_1d_change_pct: float = np.nan
    portfolio_loss_pct: float = 0.0
    size_multiplier: float = 1.0    # 0.25, 0.50, or 1.0
    pause_until: datetime | None = None
    flatten_all: bool = False
    requires_manual_reauth: bool = False
    trigger_reason: str = ""


@dataclass
class VolCircuitConfig:
    """Configurable thresholds for the vol circuit breaker."""
    # VIX levels
    vix_elevated_threshold: float = 30.0
    vix_spike_threshold_pct: float = 40.0   # % change
    # Size reductions
    elevated_size_mult: float = 0.50         # 50% size when VIX > 30
    spike_size_mult: float = 0.25            # 25% size when VIX 1d change > 40%
    # Daily loss thresholds
    daily_loss_breach_pct: float = 0.03       # 3% portfolio daily loss
    # Pause duration
    spike_pause_hours: int = 48
    # Recovery gate (requires manual re-auth after breach)
    require_manual_reauth: bool = True


class VolRegimeCircuitBreaker:
    """Portfolio-level vol regime circuit breaker.

    Evaluates market conditions AFTER each bar and applies size reductions
    or position flattening before signal generation.

    Usage:
        breaker = VolRegimeCircuitBreaker()

        for bar in intraday_bars:
            result = breaker.evaluate(
                vix=current_vix,
                vix_prev=prev_vix,
                portfolio_loss_pct=todays_loss_pct,
                ts=bar.timestamp,
            )

            if result.flatten_all:
                flatten_all_options()
                send_alert("CIRCUIT BREACH — all options flattened")
            elif result.size_multiplier < 1.0:
                apply_size_reduction(result.size_multiplier)
            elif result.requires_manual_reauth:
                halt_all_signals_until_reauth()

    The circuit breaker is STATELESS across trading sessions but has
    a memory of when it last triggered (for pause tracking).
    """

    def __init__(
        self,
        config: VolCircuitConfig | None = None,
    ) -> None:
        self.config = config or VolCircuitConfig()
        self._last_vix: float = np.nan
        self._last_state: VolCircuitState = VolCircuitState.NORMAL
        self._pause_until: datetime | None = None
        self._breach_nav: float = np.nan      # NAV at breach time (for recovery tracking)
        self._breach_dt: datetime | None = None
        self._highest_vix_seen: float = np.nan
        self._trigger_log: list[dict] = []

    @property
    def state(self) -> VolCircuitState:
        return self._last_state

    def evaluate(
        self,
        vix: float,
        portfolio_value: float,
        portfolio_loss_pct: float,  # Today's loss so far (0.0 = no loss, -0.05 = 5% loss)
        prev_close_nav: float | None = None,
        ts: datetime | None = None,
        vix_history: list[float] | None = None,
    ) -> VolCircuitResult:
        """Evaluate vol circuit breaker conditions.

        Priority order (highest severity first):
        1. BREACH: portfolio daily loss > 3% → flatten all
        2. SPIKE: VIX 1-day change > 40% → 25% size, 48h pause
        3. ELEVATED: VIX > 30 → 50% size
        4. NORMAL: no action

        Args:
            vix: Current VIX level
            portfolio_value: Current portfolio NAV
            portfolio_loss_pct: Today's loss as fraction (e.g., -0.035 = -3.5%)
            prev_close_nav: Previous day close (for breach detection)
            ts: Current timestamp
            vix_history: Optional list of prior VIX values to compute 1d change

        Returns:
            VolCircuitResult with action flags
        """
        ts = ts or datetime.utcnow()
        cfg = self.config

        # Track highest VIX for spike detection
        if np.isfinite(vix):
            if not np.isfinite(self._highest_vix_seen) or vix > self._highest_vix_seen:
                self._highest_vix_seen = vix

        # Compute VIX 1-day change
        if vix_history and len(vix_history) >= 2:
            vix_prev = vix_history[-2]
        else:
            vix_prev = self._last_vix if np.isfinite(self._last_vix) else vix

        vix_1d_change_pct = 0.0
        if np.isfinite(vix_prev) and vix_prev > 0:
            vix_1d_change_pct = (vix - vix_prev) / vix_prev * 100.0

        size_mult = 1.0
        flatten_all = False
        pause_until = None
        requires_reauth = False
        state = VolCircuitState.NORMAL
        reason = ""

        # ── Priority 1: Daily loss breach > 3% ──────────────────────────────
        if portfolio_loss_pct <= -cfg.daily_loss_breach_pct:
            state = VolCircuitState.BREACH
            flatten_all = True
            requires_reauth = True
            reason = (
                f"Portfolio daily loss ({portfolio_loss_pct*100:.2f}%) exceeds "
                f"-{cfg.daily_loss_breach_pct*100:.0f}% breach threshold"
            )
            self._last_state = state
            self._breach_nav = portfolio_value
            self._breach_dt = ts
            self._pause_until = None  # Manual re-auth overrides pause

        # ── Priority 2: VIX spike > 40% 1-day change ────────────────────────
        elif abs(vix_1d_change_pct) > cfg.vix_spike_threshold_pct:
            state = VolCircuitState.SPIKE
            size_mult = cfg.spike_size_mult
            pause_until = ts + timedelta(hours=cfg.spike_pause_hours)
            reason = (
                f"VIX 1d change ({vix_1d_change_pct:.1f}%) exceeds ±{cfg.vix_spike_threshold_pct:.0f}% "
                f"→ size reduced to {size_mult*100:.0f}%, paused until {pause_until.strftime('%H:%M')}"
            )
            if self._pause_until is None or pause_until > self._pause_until:
                self._pause_until = pause_until
            self._last_state = state

        # ── Priority 3: VIX elevated > 30 ─────────────────────────────────
        elif np.isfinite(vix) and vix > cfg.vix_elevated_threshold:
            state = VolCircuitState.ELEVATED
            size_mult = cfg.elevated_size_mult
            reason = f"VIX ({vix:.1f}) > {cfg.vix_elevated_threshold:.0f} → size reduced to 50%"
            self._last_state = state

        # ── Recovery gate: if in recovery period and conditions normalizing ─────
        elif (self._pause_until is not None and ts < self._pause_until):
            state = VolCircuitState.RECOVERY_GATE
            remaining = (self._pause_until - ts).total_seconds() / 3600.0
            reason = f"Recovery gate: {remaining:.1f}h remaining in pause period"
            size_mult = cfg.spike_size_mult  # Keep reduced until full recovery

        else:
            state = VolCircuitState.NORMAL
            self._last_state = state
            # Clear pause if conditions have normalized
            if self._pause_until is not None and ts >= self._pause_until:
                self._pause_until = None

        result = VolCircuitResult(
            state=state,
            current_vix=vix,
            vix_1d_change_pct=vix_1d_change_pct,
            portfolio_loss_pct=portfolio_loss_pct,
            size_multiplier=size_mult,
            pause_until=pause_until,
            flatten_all=flatten_all,
            requires_manual_reauth=requires_reauth,
            trigger_reason=reason,
        )

        # Log trigger events
        if state != VolCircuitState.NORMAL:
            self._trigger_log.append({
                "timestamp": ts,
                "state": state.value,
                "reason": reason,
                "vix": vix,
                "vix_1d_change_pct": vix_1d_change_pct,
                "portfolio_loss_pct": portfolio_loss_pct,
                "size_mult": size_mult,
                "flatten_all": flatten_all,
            })

        self._last_vix = vix
        return result

    def can_take_signal(
        self,
        vix: float,
        vix_1d_change_pct: float,
        ts: datetime,
    ) -> tuple[bool, str]:
        """Return (can_trade, reason) pre-signal.

        Use this BEFORE generating new signals to check if the circuit
        breaker is blocking new entries.
        """
        cfg = self.config

        # Manual re-auth required
        if self._last_state == VolCircuitState.BREACH:
            return False, (
                "CIRCUIT BREACH: Manual re-auth required. "
                f"Breach triggered at {self._breach_dt}. "
                "Send authenticated POST /api/v1/circuit-reset to resume."
            )

        # In pause period
        if self._pause_until is not None and ts < self._pause_until:
            remaining = (self._pause_until - ts).total_seconds() / 3600.0
            return False, f"VIX spike pause: {remaining:.1f}h remaining"

        # VIX spike conditions still active
        if abs(vix_1d_change_pct) > cfg.vix_spike_threshold_pct:
            return False, (
                f"VIX 1d change ({vix_1d_change_pct:.1f}%) still elevated"
            )

        # VIX still elevated
        if np.isfinite(vix) and vix > cfg.vix_elevated_threshold:
            return False, f"VIX ({vix:.1f}) still elevated above {cfg.vix_elevated_threshold:.0f}"

        return True, "NORMAL — signals allowed"

    def manual_reset(self) -> None:
        """Reset after manual re-authentication post-breach."""
        self._last_state = VolCircuitState.NORMAL
        self._pause_until = None
        self._breach_dt = None
        self._breach_nav = np.nan

    def trigger_log(self) -> list[dict]:
        """Return all trigger events for reporting."""
        return list(self._trigger_log)

    def __repr__(self) -> str:
        return (
            f"VolRegimeCircuitBreaker(state={self._last_state.value}, "
            f"vix={self._last_vix:.1f}, "
            f"pause_until={self._pause_until})"
        )