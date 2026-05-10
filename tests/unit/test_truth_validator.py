"""Tests for TruthValidator — runtime economic invariant enforcement and reconciliation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from quantcore.core.truth_validator import (
    ReconciliationReport,
    PositionDelta,
    TruthValidator,
    Violation,
    reconcile,
)
from quantcore.core.types import Fill, PortfolioSnapshot, Position, Side


class TestConstruction:
    def test_default_mode_is_soft(self):
        v = TruthValidator()
        assert v.mode == "soft"

    def test_hard_mode(self):
        v = TruthValidator(mode="hard")
        assert v.mode == "hard"

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            TruthValidator(mode="invalid")

    def test_set_mode(self):
        v = TruthValidator()
        v.set_mode("hard")
        assert v.mode == "hard"


class TestPortfolioIdentity:
    def test_perfect_identity(self):
        state = PortfolioSnapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"),
                    avg_cost=Decimal("150"), market_price=Decimal("155"),
                ),
            },
            cash=Decimal("50000"),
            total_value=Decimal("50000") + Decimal("100") * Decimal("155"),
        )
        v = TruthValidator()
        res = v.check_portfolio_identity(state)
        assert res.passed, res.violations

    def test_mismatch_detected(self):
        state = PortfolioSnapshot(
            positions={
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"),
                    avg_cost=Decimal("150"), market_price=Decimal("155"),
                ),
            },
            cash=Decimal("50000"),
            total_value=Decimal("999999"),
        )
        v = TruthValidator()
        res = v.check_portfolio_identity(state)
        assert not res.passed
        assert any(v.invariant == "portfolio_identity" for v in res.violations)

    def test_empty_portfolio(self):
        state = PortfolioSnapshot(
            cash=Decimal("100000"),
            total_value=Decimal("100000"),
        )
        v = TruthValidator()
        res = v.check_portfolio_identity(state)
        assert res.passed, res.violations


class TestFillSide:
    def test_all_fills_have_side(self):
        fills = [
            Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("10"),
                 commission=Decimal("1"), side=Side.BUY),
            Fill(order_id="o2", price=Decimal("101"), quantity=Decimal("5"),
                 commission=Decimal("0.5"), side=Side.SELL),
        ]
        v = TruthValidator()
        res = v.check_fill_side(fills)
        assert res.passed, res.violations

    def test_missing_side_detected(self):
        fills = [
            Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("10"),
                 commission=Decimal("1")),
        ]
        v = TruthValidator()
        res = v.check_fill_side(fills)
        assert not res.passed
        assert any(v.invariant == "fill_side" for v in res.violations)


class TestOrderIdParsing:
    def test_clean_order_ids(self):
        fills = [
            Fill(order_id="paper-abc123def456", price=Decimal("100"),
                 quantity=Decimal("10"), commission=Decimal("1"), side=Side.BUY),
        ]
        v = TruthValidator()
        res = v.check_no_order_id_parsing(fills)
        assert res.passed, res.violations

    def test_order_id_anti_pattern_detected(self):
        fills = [
            Fill(order_id="ord_2020-01-02T14:30:00_AAPL_buy_buy",
                 price=Decimal("100"), quantity=Decimal("10"),
                 commission=Decimal("1"), side=Side.BUY),
        ]
        v = TruthValidator()
        res = v.check_no_order_id_parsing(fills)
        assert not res.passed
        assert any(v.invariant == "order_id_parsing" for v in res.violations)


class TestPositionIntegrity:
    def test_valid_positions_pass(self):
        positions = {
            "AAPL": Position(
                symbol="AAPL", quantity=Decimal("100"),
                avg_cost=Decimal("150"), market_price=Decimal("155"),
            ),
        }
        v = TruthValidator()
        res = v.check_position_integrity(positions)
        assert res.passed, res.violations

    def test_zero_avg_cost_flagged(self):
        positions = {
            "AAPL": Position.model_construct(
                symbol="AAPL", quantity=Decimal("100"),
                avg_cost=Decimal("0"), market_price=Decimal("155"),
            ),
        }
        v = TruthValidator()
        res = v.check_position_integrity(positions)
        assert not res.passed

    def test_negative_market_price_flagged(self):
        positions = {
            "AAPL": Position.model_construct(
                symbol="AAPL", quantity=Decimal("100"),
                avg_cost=Decimal("150"), market_price=Decimal("-1"),
            ),
        }
        v = TruthValidator()
        res = v.check_position_integrity(positions)
        assert not res.passed


class TestEngineStateConsistency:
    def test_kernel_agrees_with_state(self):
        """Pass when engine state after fill matches kernel from before state."""
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.total_value = 100000.0
                self.positions = {}

        before = _Mock()
        after = _Mock()

        from quantcore.core.economic_kernel import apply_fill as kernel_apply
        fa = kernel_apply(None, fill_price=100.0, fill_quantity=50.0, is_buy=True, symbol="AAPL")
        after.positions["AAPL"] = Position(
            symbol="AAPL", quantity=Decimal(str(fa.position["quantity"])),
            avg_cost=Decimal(str(fa.position["avg_cost"])),
            market_price=Decimal("100"),
            realized_pnl=Decimal(str(fa.position["realized_pnl"])),
        )
        after.cash = 100000.0 - 50 * 100.0
        after.total_value = after.cash + 50 * 100.0

        fill = Fill(order_id="t1", price=Decimal("100"), quantity=Decimal("50"),
                    commission=Decimal("0"), side=Side.BUY)

        v = TruthValidator()
        res = v.check_engine_state_consistency(before, after, fill, "AAPL")
        assert res.passed, res.violations


class TestIntegrationCheckpoints:
    def test_fill_batch(self):
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.total_value = 100000.0
                self.positions = {}

        state = _Mock()
        fill = Fill(order_id="t1", price=Decimal("100"), quantity=Decimal("50"),
                    commission=Decimal("0"), side=Side.BUY)

        v = TruthValidator()
        res = v.check_fill_batch(state, [(fill, "AAPL")])
        assert res.passed, res.violations

    def test_margin_event(self):
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.total_value = 100000.0
                self.positions = {}

        v = TruthValidator()
        res = v.check_margin_event(_Mock())
        assert res.passed, res.violations

    def test_eod_reconciliation(self):
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.total_value = 100000.0
                self.positions = {}

        v = TruthValidator()
        res = v.check_eod_reconciliation(_Mock())
        assert res.passed, res.violations


class TestHardMode:
    def test_hard_mode_raises_on_critical(self):
        v = TruthValidator(mode="hard")
        fill = Fill(order_id="test", price=Decimal("100"), quantity=Decimal("10"),
                    commission=Decimal("0"))
        with pytest.raises(AssertionError, match="CRITICAL"):
            v.check_fill_side([fill])

    def test_soft_mode_logs_no_exception(self):
        v = TruthValidator(mode="soft")
        fill = Fill(order_id="test", price=Decimal("100"), quantity=Decimal("10"),
                    commission=Decimal("0"))
        v.check_fill_side([fill])  # Should not raise


class TestEnforceMode:
    def test_enforce_mode_raises_on_critical_like_hard(self):
        """ENFORCE mode raises on CRITICAL violations (same as hard)."""
        v = TruthValidator(mode="enforce")
        fill = Fill(order_id="test", price=Decimal("100"), quantity=Decimal("10"),
                    commission=Decimal("0"))
        with pytest.raises(AssertionError, match="CRITICAL"):
            v.check_fill_side([fill])

    def test_pre_check_fill_valid(self):
        """pre_check_fill passes for a valid buy fill."""
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.positions = {}

        v = TruthValidator(mode="enforce")
        fill = Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("50"),
                    commission=Decimal("0"), side=Side.BUY)
        res = v.pre_check_fill(_Mock(), fill, "AAPL")
        assert res.passed, res.violations

    def test_pre_check_fill_valid_sell(self):
        """pre_check_fill passes for a valid sell fill."""
        from quantcore.core.types import Position

        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.positions = {
                    "AAPL": Position(
                        symbol="AAPL", quantity=Decimal("100"),
                        avg_cost=Decimal("50"), market_price=Decimal("55"),
                    ),
                }

        v = TruthValidator()
        fill = Fill(order_id="o1", price=Decimal("55"), quantity=Decimal("30"),
                    commission=Decimal("0"), side=Side.SELL)
        res = v.pre_check_fill(_Mock(), fill, "AAPL")
        assert res.passed, res.violations

    def test_pre_check_fill_missing_side_blocked(self):
        """pre_check_fill flags fills without side as CRITICAL."""
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.positions = {}

        v = TruthValidator(mode="enforce")
        fill = Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("50"),
                    commission=Decimal("0"))
        with pytest.raises(AssertionError, match="missing side"):
            v.pre_check_fill(_Mock(), fill, "AAPL")

    def test_pre_check_fill_passes_on_clean_fill(self):
        """pre_check_fill passes for valid fills regardless of mode."""
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.positions = {}

        for mode in ("soft", "hard", "enforce"):
            v = TruthValidator(mode=mode)
            fill = Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("50"),
                        commission=Decimal("0"), side=Side.BUY)
            res = v.pre_check_fill(_Mock(), fill, "AAPL")
            assert res.passed, f"{mode}: {res.violations}"

    def test_enforce_soft_is_noop(self):
        """In soft mode, pre_check_fill does not raise."""
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.positions = {}

        v = TruthValidator(mode="soft")
        fill = Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("50"),
                    commission=Decimal("0"))
        res = v.pre_check_fill(_Mock(), fill, "AAPL")
        # Should not raise; result should have violations (missing side)
        assert not res.passed


class TestReporting:
    def test_clean_summary(self):
        v = TruthValidator()
        v.check_portfolio_identity(
            PortfolioSnapshot(cash=Decimal("100"), total_value=Decimal("100")),
        )
        assert "0 violations" in v.summary()

    def test_assert_clean_raises_on_critical(self):
        v = TruthValidator()
        fill = Fill(order_id="test", price=Decimal("100"), quantity=Decimal("10"),
                    commission=Decimal("0"))
        v.check_fill_side([fill])
        with pytest.raises(AssertionError, match="CRITICAL"):
            v.assert_clean()

    def test_assert_clean_passes_with_minor_only(self):
        v = TruthValidator()
        class _Mock:
            def __init__(self):
                self.cash = 100000.0
                self.total_value = 100000.0
                self.positions = {}
        v.check_eod_reconciliation(_Mock())
        v.assert_clean()


class TestReconciliation:
    def test_identical_states(self):
        engine = {"cash": 100000.0, "total_value": 100000.0, "positions": {}}
        paper = {"cash": 100000.0, "total_value": 100000.0, "positions": {}}
        report = reconcile(engine, paper)
        assert report.is_identical
        assert len(report.divergence_trace) == 0

    def test_cash_mismatch(self):
        engine = {"cash": 100000.0, "positions": {}}
        paper = {"cash": 95000.0, "positions": {}}
        report = reconcile(engine, paper)
        assert not report.is_identical
        assert any("CASH" in t for t in report.divergence_trace)
        assert abs(report.delta_cash - 5000.0) < 0.01

    def test_position_qty_mismatch(self):
        engine = {
            "cash": 100000.0, "total_value": 100000.0,
            "positions": {
                "AAPL": {"quantity": 100.0, "avg_cost": 50.0, "market_price": 55.0, "realized_pnl": 0.0},
            },
        }
        paper = {
            "cash": 100000.0, "total_value": 100000.0,
            "positions": {
                "AAPL": {"quantity": 90.0, "avg_cost": 50.0, "market_price": 55.0, "realized_pnl": 0.0},
            },
        }
        report = reconcile(engine, paper)
        assert not report.is_identical
        assert "AAPL" in report.position_deltas
        assert report.position_deltas["AAPL"].engine_qty == 100.0
        assert report.position_deltas["AAPL"].paper_qty == 90.0

    def test_one_sided_position(self):
        engine = {
            "cash": 100000.0,
            "positions": {
                "AAPL": {"quantity": 100.0, "avg_cost": 50.0, "market_price": 55.0, "realized_pnl": 0.0},
            },
        }
        paper = {"cash": 100000.0, "positions": {}}
        report = reconcile(engine, paper)
        assert not report.is_identical
        assert any("ONLY IN ENGINE" in t for t in report.divergence_trace)

    def test_avg_cost_mismatch(self):
        engine = {
            "cash": 100000.0,
            "positions": {
                "AAPL": {"quantity": 100.0, "avg_cost": 50.0, "market_price": 55.0, "realized_pnl": 0.0},
            },
        }
        paper = {
            "cash": 100000.0,
            "positions": {
                "AAPL": {"quantity": 100.0, "avg_cost": 52.0, "market_price": 55.0, "realized_pnl": 0.0},
            },
        }
        report = reconcile(engine, paper)
        assert not report.is_identical
        assert any("avg_cost" in t.lower() for t in report.divergence_trace)


class TestReconciliationPositionModel:
    def test_with_position_objects(self):
        engine = {
            "cash": 90000.0,
            "positions": {
                "AAPL": Position(
                    symbol="AAPL", quantity=Decimal("100"),
                    avg_cost=Decimal("50"), market_price=Decimal("55"),
                    realized_pnl=Decimal("0"),
                ),
            },
        }
        paper = {
            "cash": 90000.0,
            "positions": {
                "AAPL": {"quantity": 100.0, "avg_cost": 50.0, "market_price": 55.0, "realized_pnl": 0.0},
            },
        }
        report = reconcile(engine, paper)
        assert report.is_identical, report.divergence_trace

    def test_zero_qty_skipped(self):
        engine = {
            "cash": 100000.0,
            "positions": {
                "AAPL": Position.model_construct(
                    symbol="AAPL", quantity=Decimal("0"),
                    avg_cost=Decimal("0"), market_price=Decimal("0"),
                ),
            },
        }
        paper = {"cash": 100000.0, "positions": {}}
        report = reconcile(engine, paper)
        assert report.is_identical, report.divergence_trace


class TestReplayEngine:
    def test_empty_stream_returns_initial_state(self):
        """Replaying an empty stream returns the initial state unchanged."""
        from quantcore.core.truth_validator import replay_engine
        from quantcore.backtest.engine import PortfolioState

        state = replay_engine([], initial_cash=50000.0)
        assert abs(state.cash - 50000.0) < 0.01
        assert len(state.positions) == 0

    def test_single_fill(self):
        """Single buy fill produces correct position."""
        from quantcore.core.truth_validator import replay_engine

        fills = [
            (Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("50"),
                  commission=Decimal("1"), side=Side.BUY), "AAPL"),
        ]
        state = replay_engine(fills, initial_cash=100000.0)
        assert "AAPL" in state.positions
        assert float(state.positions["AAPL"].quantity) == 50.0
        assert abs(state.cash - (100000.0 - 50 * 100.0 - 1.0)) < 0.01

    def test_determinism(self):
        """Same fill stream produces identical results every time."""
        from quantcore.core.truth_validator import replay_engine

        fills = [
            (Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("50"),
                  commission=Decimal("1"), side=Side.BUY), "AAPL"),
            (Fill(order_id="o2", price=Decimal("55"), quantity=Decimal("30"),
                  commission=Decimal("1"), side=Side.SELL), "AAPL"),
            (Fill(order_id="o3", price=Decimal("200"), quantity=Decimal("10"),
                  commission=Decimal("1"), side=Side.BUY), "GOOG"),
        ]
        state1 = replay_engine(fills, initial_cash=100000.0)
        state2 = replay_engine(fills, initial_cash=100000.0)
        assert abs(state1.cash - state2.cash) < 0.01
        assert abs(state1.total_value - state2.total_value) < 0.01
        assert set(state1.positions.keys()) == set(state2.positions.keys())
        for sym in state1.positions:
            assert float(state1.positions[sym].quantity) == float(state2.positions[sym].quantity)
            assert float(state1.positions[sym].avg_cost) == float(state2.positions[sym].avg_cost)

    def test_with_validator(self):
        """Replay with TruthValidator passes clean fills."""
        from quantcore.core.truth_validator import replay_engine, TruthValidator

        fills = [
            (Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("10"),
                  commission=Decimal("0"), side=Side.BUY), "AAPL"),
        ]
        v = TruthValidator(mode="soft")
        state = replay_engine(fills, initial_cash=100000.0, validator=v)
        assert "AAPL" in state.positions

    def test_with_enforce_validator(self):
        """Replay in enforce mode blocks fills with missing side."""
        from quantcore.core.truth_validator import replay_engine, TruthValidator

        fills = [
            (Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("10"),
                  commission=Decimal("0")), "AAPL"),  # Missing side
        ]
        v = TruthValidator(mode="enforce")
        with pytest.raises(AssertionError, match="missing side"):
            replay_engine(fills, initial_cash=100000.0, validator=v)
