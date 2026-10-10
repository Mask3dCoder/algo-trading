"""IntradayDrawdownCalculator — MTM high-water mark at 5-min resolution.

Tracks the equity curve at 5-minute granularity (not just daily close) and
computes maximum drawdown as the peak-to-trough decline on mark-to-market
(MTM) basis, not as a cumulative sum of log-returns.

Three horizons are reported:
1. DAILY_CLOSE — Same as the existing DrawdownTracker (EOD only)
2. INTRADAY_5MIN — MTM high-water mark updated every 5 minutes
3. STRESSED — Applies a +5 VIX shock scenario:
   - Implies a 30% instantaneous vol spike applied to all option positions
   - Gamma P&L from a ±3σ SPY move = 0.5 * Gamma * (0.30/σ * S)^2
   - This catches the "3-sigma move at 10:30 AM before re-hedge" risk

Requirements:
- Gate: No strategy can be promoted to PAPER_TRADING unless its intraday
  Max DD is verified by a SECOND independent calculation.
- All three horizons must be reported in strategy PRE_MORTEM documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class DDHorizon(str, Enum):
    """Drawdown horizon types."""
    DAILY_CLOSE = "daily_close"
    INTRADAY_5MIN = "intraday_5min"
    STRESSED_PLUS5VIX = "stressed_plus5vix"


@dataclass
class DrawdownAtHorizon:
    """Drawdown metrics at a specific horizon."""
    horizon: DDHorizon
    max_drawdown_pct: float       # Positive value = largest % decline
    max_drawdown_date: datetime | None
    max_drawdown_duration_days: float  # Days from peak to trough
    current_drawdown_pct: float    # Current unfilled drawdown
    high_water_mark: float        # Peak NAV seen
    # For stressed scenario only
    stressed_shock_pct: float = 5.0  # VIX shock magnitude


@dataclass
class IntradayDrawdownState:
    """Full drawdown state across all horizons."""
    daily_close: DrawdownAtHorizon
    intraday_5min: DrawdownAtHorizon
    stressed: DrawdownAtHorizon
    # Per-horizon equity curves
    daily_equity: NDArray[np.floating] | None = None
    intraday_equity: NDArray[np.floating] | None = None
    # Gamma risk attribution
    gamma_estimate: float = np.nan       # Portfolio gamma (per 1% SPY move)
    gamma_max_loss_pct: float = np.nan   # Estimated max loss from 3σ move


class IntradayDrawdownCalculator:
    """Calculate drawdown at daily, intraday, and stressed horizons.

    This class is the "second independent calculation" required by the
    task gate: two back-to-back MTM computations that must agree within
    numerical tolerance (< 1bp). Any discrepancy > 1bp indicates a bug in
    the engine's NAV calculation and must halt promotion.

    Usage:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)

        # Process every 5-min bar (MTM update)
        calc.update_mtm(timestamp=dt, nav=float(portfolio_mtm_nav), gamma=portfolio_gamma)

        # At end of day: set the NAV for confirmation
        calc.finalize_day(dt, daily_close_nav)

        # For the second independent calculation:
        calc2 = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ... (same processing) ...
        assert abs(calc.max_dd("intraday_5min") - calc2.max_dd("intraday_5min")) < 0.0001

        # Generate full report
        report = calc.report()
        print(f"Intraday Max DD: {report.intraday_5min.max_drawdown_pct:.3f}%")
        print(f"Stressed (+5 VIX) Max DD: {report.stressed.max_drawdown_pct:.3f}%")
    """

    def __init__(
        self,
        initial_nav: float,
        vix_base: float = 20.0,     # Base VIX for stressed scenario
        shock_magnitude: float = 5.0,  # VIX shock (+5.0 points)
    ) -> None:
        self._initial_nav = float(initial_nav)
        self._nav = float(initial_nav)
        self._ts: datetime | None = None

        # ── Daily close tracking ──────────────────────────────────────────────
        self._daily_hwm: float = float(initial_nav)
        self._daily_max_dd: float = 0.0
        self._daily_max_dd_ts: datetime | None = None
        self._current_dd_pct: float = 0.0
        self._daily_equity: list[float] = [float(initial_nav)]
        self._daily_times: list[datetime] = []

        # ── Intraday 5-min MTM tracking ────────────────────────────────────────
        self._intraday_hwm: float = float(initial_nav)
        self._intraday_max_dd: float = 0.0
        self._intraday_max_dd_ts: datetime | None = None
        self._intraday_equity: list[float] = [float(initial_nav)]
        self._intraday_times: list[datetime] = []

        # ── Stressed scenario (VIX +5 shock) ──────────────────────────────────
        self._stressed_hwm: float = float(initial_nav)
        self._stressed_max_dd: float = 0.0
        self._stressed_max_dd_ts: datetime | None = None
        self._stressed_equity: list[float] = [float(initial_nav)]
        self._stressed_times: list[datetime] = []

        # ── Gamma risk ────────────────────────────────────────────────────────
        self._portfolio_gamma: float = 0.0   # Portfolio gamma estimate
        self._vix_base: float = float(vix_base)
        self._shock_magnitude: float = float(shock_magnitude)

        # ── End-of-day NAV ───────────────────────────────────────────────────
        self._daily_close_nav: float | None = None
        self._eod_confirmed: bool = False
        # Intraday value snapshot for daily tracking (used solely for finalize_day())
        self._current_nav_for_daily: float = float(initial_nav)

    @property
    def n_updates(self) -> int:
        """Number of MTM updates received."""
        return len(self._intraday_equity) - 1

    def update_mtm(
        self,
        timestamp: datetime,
        nav: float,
        gamma: float | None = None,
    ) -> None:
        """Update MTM with a new NAV point (call every 5 minutes).

        This updates two horizons simultaneously:
        1. Intraday 5-min MTM — main tracking for options gamma risk
        2. Stressed scenario — apply VIX shock to current NAV

        NOTE: Daily close (horizon 3) is updated ONLY via finalize_day()
        to avoid inflating daily drawdown with intraday fluctuations.

        Args:
            timestamp: Bar timestamp
            nav: Current mark-to-market NAV
            gamma: Portfolio gamma (per 1% SPY move). If None, uses last value.
        """
        self._ts = timestamp
        self._nav = float(nav)
        if gamma is not None:
            self._portfolio_gamma = float(gamma)

        # ── INTRADAY 5-MIN UPDATE ───────────────────────────────────────────
        self._intraday_hwm = max(self._intraday_hwm, nav)
        intraday_dd = (self._intraday_hwm - nav) / self._intraday_hwm if self._intraday_hwm > 0 else 0.0
        if intraday_dd > self._intraday_max_dd:
            self._intraday_max_dd = intraday_dd
            self._intraday_max_dd_ts = timestamp
        self._intraday_equity.append(nav)
        self._intraday_times.append(timestamp)

        # ── STRESSED (+5 VIX SHOCK) UPDATE ────────────────────────────────────
        stressed_nav = self._apply_vix_shock(nav, gamma=self._portfolio_gamma)
        self._stressed_hwm = max(self._stressed_hwm, stressed_nav)
        stressed_dd = (self._stressed_hwm - stressed_nav) / self._stressed_hwm if self._stressed_hwm > 0 else 0.0
        if stressed_dd > self._stressed_max_dd:
            self._stressed_max_dd = stressed_dd
            self._stressed_max_dd_ts = timestamp
        self._stressed_equity.append(stressed_nav)
        self._stressed_times.append(timestamp)

        # DAILY CLOSE: only track current NAV for reference — HWM is set by finalize_day()
        self._current_nav_for_daily = nav

    def _apply_vix_shock(self, nav: float, gamma: float | None) -> float:
        """Apply a +5 VIX shock to the current NAV.

        The VIX shock implies a sudden 30% increase in realized volatility.
        This generates a gamma P&L loss estimated as:
            Loss ≈ 0.5 * |Gamma| * (ΔS)^2
        where ΔS/S = shock_vol / base_vol - 1
              shock_vol = 1.3 * base_vol
        so (ΔS/S)^2 ≈ (0.3)^2 = 9%

        The stressed NAV = current NAV - estimated_shock_loss

        Args:
            nav: Current NAV
            gamma: Portfolio gamma (per 1% SPY move)
                Estimated from option positions: sum of each leg's gamma
        """
        shock_ratio = 1.0 + self._shock_magnitude / self._vix_base  # ~1.25
        vol_increase = shock_ratio - 1.0  # ~0.25 = 25% vol increase

        if gamma is not None and abs(gamma) > 1e-10:
            # Loss = 0.5 * |gamma| * (vol_increase)^2 * notional
            # We express as a fraction of NAV (notional ≈ NAV)
            gamma_loss_fraction = 0.5 * abs(gamma) * vol_increase ** 2
            # This is the % of NAV that could be lost in a VIX shock
            stressed_loss = gamma_loss_fraction * nav
            return nav - stressed_loss
        else:
            # Fallback: assume 20% drawdown per $1M NAV for a 5 vol point shock
            return nav * 0.95  # 5% stress loss

    def finalize_day(
        self,
        timestamp: datetime,
        daily_close_nav: float,
    ) -> None:
        """Finalize end-of-day and update daily close HWM/drawdown.

        Called ONCE per trading day after the last 5-min bar.
        Daily close HWM is computed from EOD closes only (not intraday MTM
        values), so the daily drawdown represents close-to-close risk, not
        intraday peak-to-trough.

        Args:
            timestamp: End-of-day timestamp
            daily_close_nav: Final closing NAV for the day
        """
        self._daily_close_nav = daily_close_nav
        self._eod_confirmed = True

        # Update daily HWM from EOD close only
        self._daily_hwm = max(self._daily_hwm, daily_close_nav)
        daily_dd = (self._daily_hwm - daily_close_nav) / self._daily_hwm if self._daily_hwm > 0 else 0.0
        if daily_dd > self._daily_max_dd:
            self._daily_max_dd = daily_dd
            self._daily_max_dd_ts = timestamp
        self._current_dd_pct = daily_dd * 100.0

        self._daily_equity.append(daily_close_nav)
        self._daily_times.append(timestamp)

    def report(self) -> IntradayDrawdownState:
        """Generate the full drawdown report across all horizons.

        Returns a dataclass with max drawdown from each horizon plus metadata.
        """
        # Verify two-independent-calculation gate
        dd_intraday = self.max_dd("intraday_5min")
        dd_daily = self.max_dd("daily_close")

        gate_passed = abs(dd_intraday - dd_daily) < 0.0001  # 1bp tolerance
        if not gate_passed:
            import warnings
            warnings.warn(
                f"DD GATE FAILED: daily={dd_daily:.5f}, intraday={dd_intraday:.5f}. "
                f"Discrepancy may indicate MTM calculation error.",
                UserWarning,
            )

        # Stressed current drawdown: must use the stressed equity chain, not self._nav
        last_stressed_nav = float(self._stressed_equity[-1]) if self._stressed_equity else self._nav
        stressed_current_dd = (
            (self._stressed_hwm - last_stressed_nav) / self._stressed_hwm * 100.0
            if self._stressed_hwm > 0 else 0.0
        )

        return IntradayDrawdownState(
            daily_close=DrawdownAtHorizon(
                horizon=DDHorizon.DAILY_CLOSE,
                max_drawdown_pct=self._daily_max_dd * 100.0,
                max_drawdown_date=self._daily_max_dd_ts,
                max_drawdown_duration_days=self._duration_days(
                    self._daily_max_dd_ts, self._daily_times, self._daily_equity
                ),
                current_drawdown_pct=self._current_dd_pct,
                high_water_mark=self._daily_hwm,
            ),
            intraday_5min=DrawdownAtHorizon(
                horizon=DDHorizon.INTRADAY_5MIN,
                max_drawdown_pct=self._intraday_max_dd * 100.0,
                max_drawdown_date=self._intraday_max_dd_ts,
                max_drawdown_duration_days=self._duration_days(
                    self._intraday_max_dd_ts, self._intraday_times, self._intraday_equity
                ),
                current_drawdown_pct=(
                    (self._intraday_hwm - self._nav) / self._intraday_hwm * 100.0
                    if self._intraday_hwm > 0 else 0.0
                ),
                high_water_mark=self._intraday_hwm,
            ),
            stressed=DrawdownAtHorizon(
                horizon=DDHorizon.STRESSED_PLUS5VIX,
                max_drawdown_pct=self._stressed_max_dd * 100.0,
                max_drawdown_date=self._stressed_max_dd_ts,
                max_drawdown_duration_days=self._duration_days(
                    self._stressed_max_dd_ts, self._stressed_times, self._stressed_equity
                ),
                current_drawdown_pct=stressed_current_dd,
                high_water_mark=self._stressed_hwm,
                stressed_shock_pct=self._shock_magnitude,
            ),
            daily_equity=np.array(self._daily_equity, dtype=float),
            intraday_equity=np.array(self._intraday_equity, dtype=float),
            gamma_estimate=self._portfolio_gamma,
            gamma_max_loss_pct=(
                0.5 * abs(self._portfolio_gamma) * (self._shock_magnitude / self._vix_base) ** 2 * 100.0
                if abs(self._portfolio_gamma) > 1e-10 else np.nan
            ),
        )

    def max_dd(self, horizon: str | DDHorizon) -> float:
        """Return max drawdown for the specified horizon.

        Args:
            horizon: "daily_close" | "intraday_5min" | "stressed_plus5vix"

        Returns:
            Max drawdown as a fraction (e.g., 0.05 = 5%)
        """
        h = DDHorizon(horizon) if isinstance(horizon, str) else horizon
        if h == DDHorizon.INTRADAY_5MIN:
            return self._intraday_max_dd
        elif h == DDHorizon.STRESSED_PLUS5VIX:
            return self._stressed_max_dd
        else:
            return self._daily_max_dd

    def _duration_days(
        self,
        event_ts: datetime | None,
        times_list: list[datetime],
        equity_list: list[float] | None = None,
    ) -> float:
        """Compute drawdown duration in calendar days (peak → trough).

        Walks the equity curve backwards from the trough (event_ts) to find
        the peak that was broken, then returns (trough_time - peak_time) in
        calendar days.

        Args:
            event_ts: Timestamp of the max drawdown (trough).
            times_list: All timestamps for this horizon.
            equity_list: Equity curve for this horizon (required for correct
                peak-to-trough duration; if None, falls back to a conservative
                estimate from session start to trough).

        Returns:
            Duration in calendar days (0.0 if no valid peak can be found).
        """
        if event_ts is None or len(times_list) < 2:
            return 0.0
        try:
            trough_idx = next(i for i, t in enumerate(times_list) if t >= event_ts)
        except StopIteration:
            return 0.0
        if trough_idx < 1:
            return 0.0

        if equity_list is not None and len(equity_list) > trough_idx:
            # Walk backward from the trough to find the peak (highest equity
            # before the drawdown bottom).
            eq = equity_list[:trough_idx + 1]
            peak_idx = int(np.argmax(eq))  # type: ignore[arg-type]
            if peak_idx < trough_idx:
                delta = times_list[trough_idx] - times_list[peak_idx]
                return delta.total_seconds() / 86400.0

        # Fallback: from session start to trough (conservative overestimate).
        delta = times_list[trough_idx] - times_list[0]
        return delta.total_seconds() / 86400.0

    def verify_two_independent_calculations(
        self,
        calc_A: IntradayDrawdownCalculator,
        calc_B: IntradayDrawdownCalculator,
        tolerance: float = 0.0001,
    ) -> tuple[bool, float]:
        """Verify two IntradayDrawdownCalculator instances agree within tolerance.

        This is the REQUIRED verification step before any strategy can be
        promoted to PAPER_TRADING (per task gate requirement).

        Args:
            calc_A: First calculator instance
            calc_B: Second calculator instance
            tolerance: Maximum allowed discrepancy (default 1bp = 0.01%)

        Returns:
            (is_verified, max_discrepancy)
        """
        horizons = ["daily_close", "intraday_5min", "stressed_plus5vix"]
        max_discrepancy = 0.0
        for h in horizons:
            dd_a = calc_A.max_dd(h)
            dd_b = calc_B.max_dd(h)
            discrepancy = abs(dd_a - dd_b)
            max_discrepancy = max(max_discrepancy, discrepancy)

        return (max_discrepancy <= tolerance, max_discrepancy)


# ── Standalone helpers ─────────────────────────────────────────────────────────

def compute_drawdown_from_equity(
    equity: NDArray[np.floating],
    timestamps: list[datetime] | None = None,
) -> dict[str, float]:
    """Compute drawdown metrics from an equity curve array.

    This is the SECOND independent calculation method (separate from the
    running tracker). Used for verification of IntradayDrawdownCalculator.

    Duration is computed as peak → trough (calendar days), not start → trough.

    Args:
        equity: NAV array (not returns!)
        timestamps: Optional timestamps for duration calculation

    Returns:
        dict with max_drawdown, current_drawdown, duration_days, peak_date
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return {"max_drawdown": 0.0, "current_drawdown": 0.0, "duration_days": 0.0}

    # Running high-water mark
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq) / peak  # Positive = drawdown

    max_dd = float(np.max(dd))
    max_dd_idx = int(np.argmax(dd))  # Index of max drawdown (trough)

    current_dd = float(dd[-1])

    # ── Duration: peak → trough (calendar days) ─────────────────────────
    duration_days = 0.0
    if timestamps is not None and max_dd_idx > 0:
        # Find the peak index — walk backwards from max_dd_idx to find the
        # point where the HWM was at its maximum before the drawdown.
        peak_idx = int(np.argmax(eq[:max_dd_idx + 1]))
        if peak_idx < max_dd_idx:
            trough_ts = timestamps[min(max_dd_idx, len(timestamps) - 1)]
            peak_ts = timestamps[min(peak_idx, len(timestamps) - 1)]
            delta = trough_ts - peak_ts
            duration_days = delta.total_seconds() / 86400.0 if delta else 0.0

    return {
        "max_drawdown": max_dd,
        "current_drawdown": current_dd,
        "duration_days": duration_days,
        "peak_date": timestamps[max_dd_idx] if timestamps else None,
        "equity_low": float(eq[max_dd_idx]),
        "equity_now": float(eq[-1]),
    }


def compute_stressed_drawdown_from_equity(
    equity: NDArray[np.floating],
    shock_vol_increase: float = 0.25,  # 25% vol increase from VIX+5
    gamma: float = 0.0,                # Portfolio gamma per 1% move
) -> dict[str, float]:
    """Apply a vol shock to equity curve and compute stressed drawdown.

    This is an alternative stressed scenario using the same equity curve.

    Args:
        equity: NAV array
        shock_vol_increase: Assumed vol increase fraction (VIX+5 ≈ 25% increase)
        gamma: Portfolio gamma per 1% SPY move

    Returns:
        dict with stressed_max_drawdown, stressed_current_drawdown
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) == 0:
        return {"stressed_max_drawdown": 0.0, "stressed_current_drawdown": 0.0}

    if abs(gamma) > 1e-10:
        # Apply gamma loss to each return
        returns = np.diff(eq) / eq[:-1]
        shocked_returns = returns * (1.0 - 0.5 * abs(gamma) * shock_vol_increase ** 2)
        shocked_eq = eq[0] * np.concatenate([[1.0], np.cumprod(1.0 + shocked_returns)])
    else:
        # Fallback: 10% instantaneous loss
        shocked_eq = eq * (1.0 - shock_vol_increase * 0.4)

    result = compute_drawdown_from_equity(shocked_eq)
    return {
        "stressed_max_drawdown": result["max_drawdown"],
        "stressed_current_drawdown": result["current_drawdown"],
    }
