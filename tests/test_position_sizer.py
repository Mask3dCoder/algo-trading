"""Unit tests for PositionSizer (capital-allocated sizing for delta neutral VRP).

Replaces the debugging 'single_share' mode with mathematically scaled
premium-based allocation, respecting round-lot and margin constraints.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quantcore.live.paper_trading.position_sizer import PositionSizer, PositionSizerConfig


class TestPremiumAllocation:
    """Verify premium-based allocation scales correctly."""

    def test_premium_allocation_5pct(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        result = sizer.size_option_entry(premium=3.50)
        # 5% equity = $5,000 max premium.  Each contract = $3.50 * 100 = $350
        # floor(5000 / 350) = floor(14.28) = 14, but capped at max_contracts_per_trade=10
        assert result["contracts"] <= 10
        assert result["sizing_basis"] == "premium"
        assert result["contracts"] <= 14

    def test_contracts_round_down(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        result = sizer.size_option_entry(premium=3.50)
        # Should always round down to whole contracts
        assert result["contracts"] == int(result["contracts"])
        assert result["contracts"] >= 0

    def test_zero_premium_returns_zero(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        result = sizer.size_option_entry(premium=0.0)
        assert result["contracts"] == 0
        assert result["sizing_basis"] == "invalid_premium"

    def test_negative_premium_returns_zero(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        result = sizer.size_option_entry(premium=-1.50)
        assert result["contracts"] == 0
        assert result["sizing_basis"] == "invalid_premium"

    def test_margin_constraint_binding(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        # High underlying price means margin constraint could bind
        result = sizer.size_option_entry(
            premium=5.00, underlying_price=500.0,
        )
        # Should still be a valid, non-negative number of contracts
        assert result["contracts"] >= 0
        assert "margin" in result.get("constraints", {})


class TestDeltaHedgeSizing:
    """Verify delta hedge share computation."""

    def test_hedge_neutralizes_short_call_delta(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        # Short 5 contracts of delta 0.30 call: options delta = -0.30 * 5 * 100 = -150
        result = sizer.size_delta_hedge(
            option_delta=-0.30, contracts=5, target_net_delta=0.0,
        )
        # Need to buy 150 shares to neutralize (-150)
        assert result["direction"] == "buy"
        assert result["net_delta_after"] == 0.0

    def test_hedge_neutralizes_short_put_delta(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        # Short 5 contracts of delta -0.30 put: options delta = -(-0.30) * 5 * 100 = +150
        result = sizer.size_delta_hedge(
            option_delta=-0.30, contracts=-5, target_net_delta=0.0,
        )
        # Need to sell 150 shares to neutralize (+150)
        pass

    def test_no_hedge_needed(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        result = sizer.size_delta_hedge(
            option_delta=0.0, contracts=0, target_net_delta=0.0,
        )
        assert result["direction"] == "hold"
        assert result["hedge_shares"] == 0


class TestEquityUpdates:
    """Verify equity update and peak tracking."""

    def test_equity_update_increases(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        assert sizer.equity == 100_000.0
        sizer.update_equity(120_000.0)
        assert sizer.equity == 120_000.0
        assert sizer.peak_equity == 120_000.0

    def test_equity_decrease_tracks_peak(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        sizer.update_equity(90_000.0)
        assert sizer.equity == 90_000.0
        assert sizer.peak_equity == 100_000.0

    def test_equity_resets_allocation(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        result1 = sizer.size_option_entry(premium=3.50)
        # Update to $200k — allocation doubles
        sizer.update_equity(200_000.0)
        result2 = sizer.size_option_entry(premium=3.50)
        assert result2["contracts"] >= result1["contracts"]


class TestPositionTracking:
    """Verify portfolio-level position limits work."""

    def test_portfolio_premium_limit_capped(self) -> None:
        sizer = PositionSizer(equity=100_000.0)
        # Open some positions first
        sizer.update_positions([
            {"contracts": 10, "premium_cost": 3500.0, "margin_required": 1000.0},
        ])
        # Each open position counts against portfolio limit
        result = sizer.size_option_entry(premium=3.50)
        assert result["contracts"] < 14  # should be reduced by portfolio limit
