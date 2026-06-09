"""Tests for EconomicEngineCore — the canonical economic kernel.

Validates all four avg_cost cases, edge cases (zero qty, cross-zero),
and the pure function contract (no side effects).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantcore.core.economic_kernel import FillApplication, apply_fill


def pos(quantity, avg_cost, realized_pnl=0.0):
    """Helper to create a position dict."""
    return {"symbol": "TST", "quantity": quantity, "avg_cost": avg_cost,
            "market_price": avg_cost, "realized_pnl": realized_pnl}


class TestApplyFill:
    """Core four-case avg_cost logic."""

    def test_new_buy(self):
        """Buy with no existing position creates long."""
        result = apply_fill(None, fill_price=100.0, fill_quantity=50, is_buy=True, symbol="TST")
        assert result.position is not None
        assert result.position["quantity"] == 50.0
        assert result.position["avg_cost"] == 100.0
        assert result.cash_delta == -5000.0
        assert result.realized_pnl == 0.0

    def test_new_sell(self):
        """Sell with no existing position creates short."""
        result = apply_fill(None, fill_price=50.0, fill_quantity=100, is_buy=False, symbol="TST")
        assert result.position is not None
        assert result.position["quantity"] == -100.0
        assert result.position["avg_cost"] == 50.0
        assert result.cash_delta == 5000.0

    def test_add_to_long_blended(self):
        """Buy when long → blended avg cost."""
        result = apply_fill(
            pos(100, 50.0), fill_price=60.0, fill_quantity=50, is_buy=True, symbol="TST",
        )
        expected_avg = (100 * 50.0 + 50 * 60.0) / 150
        assert result.position is not None
        assert result.position["quantity"] == 150.0
        assert abs(result.position["avg_cost"] - expected_avg) < 0.001
        assert result.realized_pnl == 0.0

    def test_cover_short_unchanged(self):
        """Buy when short → avg cost unchanged, realize PnL."""
        result = apply_fill(
            pos(-100, 50.0), fill_price=55.0, fill_quantity=30, is_buy=True, symbol="TST",
        )
        assert result.position is not None
        assert result.position["quantity"] == -70.0
        assert result.position["avg_cost"] == 50.0  # Unchanged
        assert result.realized_pnl == (50.0 - 55.0) * 30  # -150
        assert result.cash_delta == -(30 * 55.0)

    def test_reduce_long_unchanged(self):
        """Sell when long → avg cost unchanged, realize PnL."""
        result = apply_fill(
            pos(100, 50.0), fill_price=60.0, fill_quantity=40, is_buy=False, symbol="TST",
        )
        assert result.position is not None
        assert result.position["quantity"] == 60.0
        assert result.position["avg_cost"] == 50.0  # Unchanged
        assert result.realized_pnl == (60.0 - 50.0) * 40  # 400
        assert result.cash_delta == 40 * 60.0

    def test_add_to_short_blended(self):
        """Sell when short → blended avg cost."""
        result = apply_fill(
            pos(-50, 50.0), fill_price=45.0, fill_quantity=30, is_buy=False, symbol="TST",
        )
        expected_avg = (50 * 50.0 + 30 * 45.0) / 80
        assert result.position is not None
        assert result.position["quantity"] == -80.0
        assert abs(result.position["avg_cost"] - expected_avg) < 0.001
        assert result.realized_pnl == 0.0

    def test_full_close_long(self):
        """Sell enough to close a long → position is None."""
        result = apply_fill(
            pos(100, 50.0), fill_price=60.0, fill_quantity=100, is_buy=False, symbol="TST",
        )
        assert result.position is None
        assert result.realized_pnl == (60.0 - 50.0) * 100
        assert result.cash_delta == 100 * 60.0

    def test_full_close_short(self):
        """Buy enough to close a short → position is None."""
        result = apply_fill(
            pos(-100, 50.0), fill_price=55.0, fill_quantity=100, is_buy=True, symbol="TST",
        )
        assert result.position is None
        assert result.realized_pnl == (50.0 - 55.0) * 100
        assert result.cash_delta == -(100 * 55.0)

    def test_no_side_effects(self):
        """Function must not mutate inputs."""
        original = pos(100, 50.0)
        original_copy = dict(original)
        apply_fill(original, fill_price=60.0, fill_quantity=30, is_buy=False, symbol="TST")
        assert original == original_copy

    def test_deterministic(self):
        """Same inputs → same outputs."""
        args = (pos(100, 50.0),)
        kwargs = dict(fill_price=60.0, fill_quantity=30, is_buy=False, symbol="TST")
        r1 = apply_fill(*args, **kwargs)
        r2 = apply_fill(*args, **kwargs)
        assert r1.cash_delta == r2.cash_delta
        assert r1.realized_pnl == r2.realized_pnl
        if r1.position is not None and r2.position is not None:
            assert r1.position["quantity"] == r2.position["quantity"]
            assert r1.position["avg_cost"] == r2.position["avg_cost"]


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_add_to_long_crosses_zero(self):
        """Sell more than long position → transitions to short."""
        result = apply_fill(
            pos(100, 50.0), fill_price=60.0, fill_quantity=150, is_buy=False, symbol="TST",
        )
        # 100 - 150 = -50. Starts reducing long (avg unchanged for 100),
        # then adding to short (blended avg for remaining -50)
        assert result.position is not None
        assert result.position["quantity"] == -50.0
        # avg_cost stays at 50.0 for the transition case
        assert abs(result.position["avg_cost"] - 50.0) < 0.001
        # Realized: (60-50)*100 = 1000 from the reduce-long part
        assert result.realized_pnl == (60.0 - 50.0) * 100

    def test_add_to_short_crosses_zero(self):
        """Buy more than short position → transitions to long."""
        result = apply_fill(
            pos(-100, 50.0), fill_price=45.0, fill_quantity=150, is_buy=True, symbol="TST",
        )
        assert result.position is not None
        assert result.position["quantity"] == 50.0
        assert abs(result.position["avg_cost"] - 50.0) < 0.001
        assert result.realized_pnl == (50.0 - 45.0) * 100

    def test_zero_qty_position(self):
        """Position with quantity=0 should be treated as new."""
        result = apply_fill(
            pos(0, 50.0, realized_pnl=100.0),
            fill_price=60.0, fill_quantity=50, is_buy=True, symbol="TST",
        )
        assert result.position is not None
        assert result.position["quantity"] == 50.0
        assert result.position["avg_cost"] == 60.0
        # realized_pnl carries forward from the pre-existing 100
        assert result.position["realized_pnl"] == 100.0

    def test_very_small_qty_removed(self):
        """Extremely small residual quantities should resolve to None."""
        result = apply_fill(
            pos(1e-9, 50.0), fill_price=60.0, fill_quantity=1e-9, is_buy=False, symbol="TST",
        )
        assert result.position is None, "Sub-1e-10 residual qty should be treated as zero"
