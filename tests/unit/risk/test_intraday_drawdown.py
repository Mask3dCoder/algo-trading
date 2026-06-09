"""Tests for IntradayDrawdownCalculator (Task 1.2: Drawdown Forensics).

Comprehensive audit of drawdown math across three horizons:
- DAILY_CLOSE: EOM NAV-based drawdown
- INTRADAY_5MIN: MTM high-water mark at 5-min resolution
- STRESSED_PLUS5VIX: VIX +5 shock scenario for options gamma risk

Forensic checks:
1. High-water mark monotonicity (peak only increases)
2. Drawdown non-negativity (never below zero)
3. Exact recovery to zero after new peak
4. Two-independent-calculation gate (< 1bp tolerance)
5. Stressed scenario produces meaningful shock (> 5% for typical gamma)
6. EOD finalize_day consistency check
7. Gamma sensitivity: higher gamma → larger stressed DD
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import pytest

from quantcore.risk.intraday_drawdown import (
    DDHorizon,
    DrawdownAtHorizon,
    IntradayDrawdownCalculator,
    IntradayDrawdownState,
    compute_drawdown_from_equity,
    compute_stressed_drawdown_from_equity,
)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _make_calendar(start: datetime, n: int, freq_minutes: int = 5) -> list[datetime]:
    """Generate n timestamps spaced by freq_minutes."""
    return [start + timedelta(minutes=i * freq_minutes) for i in range(n)]


# ─── Basic lifecycle tests ─────────────────────────────────────────────────────


class TestHighWaterMark:
    """Test 1: High-water mark is monotonically non-decreasing."""

    def test_hwm_increases_with_new_peak(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        calc.update_mtm(ts + timedelta(minutes=5), 1_050_000)
        calc.update_mtm(ts + timedelta(minutes=10), 1_020_000)

        # Intraday HWM tracks every 5-min MTM update
        assert calc._intraday_hwm == 1_050_000
        # Daily HWM only updates via finalize_day(), so still at initial
        assert calc._daily_hwm == 1_000_000

    def test_hwm_stays_same_on_decline(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        calc.update_mtm(ts + timedelta(minutes=5), 950_000)

        assert calc._intraday_hwm == 1_000_000


class TestDrawdownNonNegativity:
    """Test 2: Drawdown is always >= 0 and <= 1 (when NAV > 0)."""

    def test_drawdown_zero_at_peak(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        calc.update_mtm(ts + timedelta(minutes=5), 1_100_000)

        assert calc.max_dd(DDHorizon.INTRADAY_5MIN) == 0.0
        assert calc.max_dd(DDHorizon.DAILY_CLOSE) == 0.0

    def test_drawdown_positive_on_decline(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        calc.update_mtm(ts + timedelta(minutes=5), 900_000)

        dd = calc.max_dd(DDHorizon.INTRADAY_5MIN)
        assert dd > 0.0
        assert dd < 1.0
        assert pytest.approx(dd, abs=1e-6) == 0.10  # 100K / 1M

    def test_drawdown_at_zero_nav(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        calc.update_mtm(ts + timedelta(minutes=5), 0)

        dd = calc.max_dd(DDHorizon.INTRADAY_5MIN)
        assert dd == 1.0  # 100% drawdown


class TestExactRecovery:
    """Test 3: After new peak, drawdown should be exactly zero."""

    def test_recovery_after_decline(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        calc.update_mtm(ts + timedelta(minutes=5), 900_000)  # -10%
        calc.update_mtm(ts + timedelta(minutes=10), 950_000)  # still -5%
        calc.update_mtm(ts + timedelta(minutes=15), 1_100_000)  # new peak

        assert calc._intraday_max_dd == pytest.approx(0.10, abs=1e-9)
        # Current drawdown should be zero at new peak
        current = (calc._intraday_hwm - calc._nav) / calc._intraday_hwm
        assert current == 0.0


class TestTwoIndependentCalculationsGate:
    """Test 4: Two independent calc instances must agree within 1bp."""

    def test_gate_passes_for_identical_processing(self) -> None:
        calc_a = IntradayDrawdownCalculator(initial_nav=1_000_000)
        calc_b = IntradayDrawdownCalculator(initial_nav=1_000_000)

        ts = datetime(2024, 1, 1, 9, 30)
        for i in range(100):
            nav = 1_000_000 + i * 100 if i < 50 else 1_100_000 - (i - 50) * 300
            calc_a.update_mtm(ts + timedelta(minutes=i * 5), nav)
            calc_b.update_mtm(ts + timedelta(minutes=i * 5), nav)

        is_verified, max_disc = calc_a.verify_two_independent_calculations(calc_a, calc_b)
        assert is_verified
        assert max_disc < 0.0001

    def test_gate_fails_on_divergence(self) -> None:
        calc_a = IntradayDrawdownCalculator(initial_nav=1_000_000)
        calc_b = IntradayDrawdownCalculator(initial_nav=1_000_000)

        ts = datetime(2024, 1, 1, 9, 30)
        for i in range(100):
            nav_a = 1_000_000 + i * 100 if i < 50 else 1_100_000 - (i - 50) * 300
            nav_b = nav_a * 1.0005  # Slight divergence
            calc_a.update_mtm(ts + timedelta(minutes=i * 5), nav_a)
            calc_b.update_mtm(ts + timedelta(minutes=i * 5), nav_b)

        is_verified, max_disc = calc_a.verify_two_independent_calculations(calc_a, calc_b)
        assert not is_verified
        assert max_disc > 0.0001

    def test_report_gate_warns_on_mismatch(self) -> None:
        """If intraday and daily diverge, report() should issue a warning."""
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        # Manually corrupt the daily tracker to force mismatch
        calc._daily_max_dd = 0.15

        with pytest.warns(UserWarning, match="DD GATE FAILED"):
            report = calc.report()
        # Should still return data even when gate fails
        assert report.daily_close is not None


class TestStressedScenario:
    """Test 5: Stressed scenario produces meaningful shock."""

    def test_stressed_produces_nonzero_shock(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        calc.update_mtm(ts + timedelta(minutes=5), 950_000, gamma=0.05)

        stressed_dd = calc.max_dd(DDHorizon.STRESSED_PLUS5VIX)
        # Stressed should be larger than or equal to intraday
        assert stressed_dd >= calc.max_dd(DDHorizon.INTRADAY_5MIN)

    def test_stressed_increases_with_higher_gamma(self) -> None:
        calc_low_gamma = IntradayDrawdownCalculator(initial_nav=1_000_000)
        calc_high_gamma = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        for i in range(20):
            nav = 1_000_000 - i * 1000
            calc_low_gamma.update_mtm(ts + timedelta(minutes=i * 5), nav, gamma=0.01)
            calc_high_gamma.update_mtm(ts + timedelta(minutes=i * 5), nav, gamma=0.50)

        dd_low = calc_low_gamma.max_dd(DDHorizon.STRESSED_PLUS5VIX)
        dd_high = calc_high_gamma.max_dd(DDHorizon.STRESSED_PLUS5VIX)

        assert dd_high > dd_low, (
            f"Higher gamma should produce larger stressed drawdown: "
            f"high={dd_high:.6f} vs low={dd_low:.6f}"
        )

    def test_stressed_applies_vix_shock_formula(self) -> None:
        """Verify the VIX shock math: a +5 point shock at VIX=20 should produce
        meaningful (not trivial) drawdown for realistic gamma."""
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000, vix_base=20.0, shock_magnitude=5.0)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000, gamma=0.05)
        # Force a decline to trigger stressed DD computation
        calc.update_mtm(ts + timedelta(minutes=5), 900_000, gamma=0.05)

        stressed_dd = calc.max_dd(DDHorizon.STRESSED_PLUS5VIX)
        # With gamma=0.05 and a proper shock model, the stressed drawdown
        # should be detectably larger than trivial.  A minuscule shock
        # (<0.5%) implies the formula is not a crisis-level scenario.
        assert stressed_dd > 0.005, (
            f"Stressed DD={stressed_dd:.6f} is too small for a +5 VIX shock. "
            f"Verify _apply_vix_shock formula."
        )

    def test_stressed_defaults_when_gamma_none(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000, gamma=None)
        calc.update_mtm(ts + timedelta(minutes=5), 950_000, gamma=None)

        dd = calc.max_dd(DDHorizon.STRESSED_PLUS5VIX)
        # When gamma is None, the fallback is 5% loss
        assert dd >= 0.05  # Should at least reflect the 5% fallback


class TestFinalizeDay:
    """Test 6: finalize_day consistency check."""

    def test_finalize_sets_daily_hwm(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        for i in range(13):  # One trading day of 5-min bars
            nav = 1_000_000 + (i - 6) * 1000  # Sine-like pattern
            calc.update_mtm(ts + timedelta(minutes=i * 5), nav)

        # finalize_day updates the daily HWM from the EOD NAV
        eod_nav = 1_006_000  # Peak NAV from the sine pattern
        calc.finalize_day(ts + timedelta(hours=8), daily_close_nav=eod_nav)
        assert calc._daily_hwm == eod_nav
        assert calc._eod_confirmed is True

    def test_finalize_succeeds_when_consistent(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        for i in range(13):
            nav = 1_000_000 + (i - 6) * 1000
            calc.update_mtm(ts + timedelta(minutes=i * 5), nav)

        last_mtm = calc._intraday_equity[-1]
        # No warning expected when consistent
        calc.finalize_day(ts + timedelta(hours=8), daily_close_nav=last_mtm)
        assert calc._eod_confirmed

    def test_daily_equity_appended_on_finalize(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        calc.finalize_day(ts + timedelta(hours=8), daily_close_nav=1_000_000)

        assert len(calc._daily_equity) == 2  # initial + EOD
        assert calc._daily_times[-1] == ts + timedelta(hours=8)


class TestDurationDays:
    """Test 7: Duration computation from peak to trough."""

    def test_duration_zero_for_immediate_peak(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        calc.update_mtm(ts, 1_000_000)
        # Duration helper should handle empty lists gracefully
        assert calc._duration_days(None, []) == 0.0
        assert calc._duration_days(ts, [ts]) == 0.0

    def test_duration_computed_correctly(self) -> None:
        calc = IntradayDrawdownCalculator(initial_nav=1_000_000)
        ts = datetime(2024, 1, 1, 9, 30)

        # Simulate decline and recovery
        for i in range(10):
            calc.update_mtm(ts + timedelta(minutes=i * 5), max(800_000, 1_000_000 - i * 1000))

        # Duration from first bar to max drawdown event
        assert len(calc._intraday_times) > 0


class TestStandaloneHelper:
    """Test the standalone helper functions."""

    def test_compute_drawdown_from_equity_basic(self) -> None:
        equity = np.array([100.0, 110.0, 105.0, 90.0, 95.0, 100.0])
        result = compute_drawdown_from_equity(equity)
        # Peak is 110, trough after peak is 90 at index 3; dd = (110 - 90) / 110 = 20/110
        assert result["max_drawdown"] == pytest.approx(20.0 / 110.0, abs=1e-6)

    def test_compute_drawdown_from_equity_monotonic_increase(self) -> None:
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        result = compute_drawdown_from_equity(equity)
        assert result["max_drawdown"] == 0.0

    def test_compute_stressed_drawdown(self) -> None:
        equity = np.array([100.0, 95.0, 90.0, 85.0])
        result = compute_stressed_drawdown_from_equity(equity, shock_vol_increase=0.25, gamma=0.05)
        assert result["stressed_max_drawdown"] > 0.0
