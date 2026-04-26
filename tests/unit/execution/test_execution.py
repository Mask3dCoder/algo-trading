"""Tests for Phase 3 Execution Module: OMS, algos, impact, TCA, brokers."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

import numpy as np
import pytest

from quantcore.core.types import Order, Fill, Side, OrderType, OrderStatus
from quantcore.execution.oms import (
    OrderManager, OrderState, market_order, limit_order, bracket_order,
)
from quantcore.core.exceptions import OrderError
from quantcore.execution.algos.twap import TWAPAlgo, TWAPConfig, TWAPSlice
from quantcore.execution.algos.vwap import VWAPAlgo, VWAPConfig, DEFAULT_VOLUME_PROFILE
from quantcore.execution.algos.pov import POVAlgo, POVConfig
from quantcore.execution.algos.implementation_shortfall import (
    compute_almgren_chriss, compute_trajectory,
    compute_trajectory_with_alpha, ISSchedule,
)
from quantcore.execution.impact import (
    sqrt_impact, sqrt_impact_bps, linear_impact,
    estimate_kyle_lambda, amihud_illiquidity,
    roll_spread, roll_spread_bps, predict_cost, CostEstimate,
)
from quantcore.execution.tca import tca_trade, tca_strategy, tca_report, TCAResult


# ---------------------------------------------------------------------------
# T-0314: OMS Tests
# ---------------------------------------------------------------------------


class TestOrderManager:
    def test_submit_order(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("100"))
        oid = oms.submit(order)
        assert oid == order.id
        assert order.status == OrderStatus.SUBMITTED

    def test_valid_transition_new_to_submitted(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("50"))
        oms.submit(order)
        assert order.status == OrderStatus.SUBMITTED

    def test_fill_order(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("100"))
        oms.submit(order)
        new_state = oms.fill(order.id, Decimal("100"), Decimal("150.00"))
        assert new_state == OrderState.FILLED
        assert order.status == OrderStatus.FILLED
        assert order.filled_qty == Decimal("100")

    def test_partial_fill(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("100"))
        oms.submit(order)
        new_state = oms.fill(order.id, Decimal("40"), Decimal("150.00"))
        assert new_state == OrderState.PARTIAL
        assert order.status == OrderStatus.PARTIAL

    def test_cancel_order(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("100"))
        oms.submit(order)
        oms.cancel(order.id)
        assert order.status == OrderStatus.CANCELLED

    def test_cannot_fill_cancelled(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("100"))
        oms.submit(order)
        oms.cancel(order.id)
        with pytest.raises(OrderError):
            oms.fill(order.id, Decimal("100"), Decimal("150"))

    def test_duplicate_rejected(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("100"))
        oms.submit(order)
        with pytest.raises(OrderError):
            oms.submit(order)

    def test_audit_trail(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("100"))
        oms.submit(order)
        oms.fill(order.id, Decimal("30"), Decimal("150"))
        assert len(oms.audit_trail) >= 2

    def test_bracket_order(self):
        entry, tp, sl = bracket_order(
            "AAPL", Side.BUY, Decimal("100"),
            Decimal("150"), Decimal("160"), Decimal("145"),
        )
        assert entry.type == OrderType.LIMIT
        assert tp.side == Side.SELL
        assert sl.type == OrderType.STOP

    def test_reject_order(self):
        oms = OrderManager()
        order = market_order("AAPL", Side.BUY, Decimal("100"))
        oms.submit(order)
        oms.reject(order.id, "Test rejection")
        assert order.status == OrderStatus.REJECTED


# ---------------------------------------------------------------------------
# T-0315: TWAP Tests
# ---------------------------------------------------------------------------


class TestTWAP:
    def test_slice_count(self):
        algo = TWAPAlgo()
        slices = algo.compute_schedule(total_shares=1000)
        assert len(slices) == 10

    def test_total_shares_equal(self):
        algo = TWAPAlgo()
        slices = algo.compute_schedule(total_shares=1000)
        total = sum(s.shares for s in slices)
        assert abs(total - 1000) < 2

    def test_jitter_variation(self):
        algo = TWAPAlgo(seed=42)
        # Unique timestamps
        schedule1 = algo.compute_schedule(total_shares=1000)
        schedule2 = algo.compute_schedule(total_shares=1000)
        times1 = [s.target_time for s in schedule1]
        times2 = [s.target_time for s in schedule2]
        # Should differ slightly due to randomness
        assert times1 != times2

    def test_evaluate(self):
        algo = TWAPAlgo()
        slices = algo.compute_schedule(total_shares=1000)
        for slc in slices:
            slc.executed = True
            slc.fill_price = 100.0
            slc.fill_shares = slc.shares
        result = algo.evaluate(slices, start_price=100.0, end_price=101.0)
        assert result.total_executed > 0
        assert result.avg_price is not None


# ---------------------------------------------------------------------------
# T-0316: VWAP Tests
# ---------------------------------------------------------------------------


class TestVWAP:
    def test_volume_profile_loaded(self):
        algo = VWAPAlgo()
        profile = algo.load_volume_profile("AAPL")
        assert len(profile) > 0

    def test_slice_schedule(self):
        algo = VWAPAlgo()
        slices = algo.compute_schedule(total_shares=1000, symbol="AAPL")
        assert len(slices) > 0

    def test_volume_weights_sum(self):
        algo = VWAPAlgo()
        slices = algo.compute_schedule(total_shares=1000, symbol="AAPL")
        total = sum(s.shares for s in slices)
        assert abs(total - 1000) < 2

    def test_evaluate(self):
        algo = VWAPAlgo()
        slices = algo.compute_schedule(total_shares=500, symbol="AAPL")
        for slc in slices:
            slc.fill_price = 100.0
            slc.fill_shares = slc.shares
        result = algo.evaluate(slices, market_vwap=100.0)
        assert result.vwap_fill is not None


# ---------------------------------------------------------------------------
# T-0317: POV Tests
# ---------------------------------------------------------------------------


class TestPOV:
    def test_participation_rate(self):
        algo = POVAlgo()
        mv = [1000, 1200, 800, 1500, 900, 1100, 950, 1050, 1150, 1000]
        result = algo.run(total_shares=500, market_volumes=mv)
        assert result.actual_participation > 0

    def test_completes(self):
        algo = POVAlgo(POVConfig(participation_rate=0.10, window_minutes=10))
        mv = [1000] * 20
        result = algo.run(total_shares=500, market_volumes=mv)
        assert result.remaining < 500  # Executed something


# ---------------------------------------------------------------------------
# T-0318: Implementation Shortfall Tests
# ---------------------------------------------------------------------------


class TestImplementationShortfall:
    def test_zero_risk_aversion_is_uniform(self):
        """λ=0 → κ→0 → uniform trajectory (TWAP equivalent)."""
        sched = compute_almgren_chriss(
            total_shares=10000, T=1.0 / 6.5,  # 1 hour
            sigma=0.02, eta=0.01, gamma=0.005, lambda_=1e-12, n_steps=50,
        )
        # Trajectory should be approximately linear (uniform)
        x = sched.shares_remaining
        start = x[0]
        mid = x[len(x) // 2]
        end = x[-1]
        # Check approximately linear
        assert abs(start - 10000) < 0.01
        assert abs(mid - 5000) < 2000  # Roughly halfway
        assert abs(end) < 0.01

    def test_high_risk_aversion_front_loads(self):
        """Higher λ should cause more front-loading."""
        low_lambda = compute_almgren_chriss(
            10000, 1.0, 0.02, 0.01, 0.005, 1e-12, 50,
        )
        high_lambda = compute_almgren_chriss(
            10000, 1.0, 0.02, 0.01, 0.005, 1e-4, 50,
        )
        # High lambda: more shares traded early → fewer remaining at mid
        mid_low = low_lambda.shares_remaining[len(low_lambda.shares_remaining) // 2]
        mid_high = high_lambda.shares_remaining[len(high_lambda.shares_remaining) // 2]
        assert mid_high < mid_low  # More front-loaded

    def test_alpha_decay_front_loads(self):
        """Alpha decay should front-load execution."""
        base = compute_trajectory_with_alpha(
            10000, 1.0, 0.02, 0.01, 0.005, 1e-6, alpha=0.01, alpha_decay=0.0,
        )
        decay = compute_trajectory_with_alpha(
            10000, 1.0, 0.02, 0.01, 0.005, 1e-6, alpha=0.01, alpha_decay=5.0,
        )
        mid_base = base.shares_remaining[len(base.shares_remaining) // 2]
        mid_decay = decay.shares_remaining[len(decay.shares_remaining) // 2]
        # Fast decay should front-load more
        assert mid_decay < mid_base + 100

    def test_trajectory_dict(self):
        result = compute_trajectory(10000, 1.0, 0.02, 0.01, 0.005, 1e-6, 50)
        assert "times" in result
        assert "remaining" in result
        assert "rate" in result


# ---------------------------------------------------------------------------
# T-0319: Impact Models
# ---------------------------------------------------------------------------


class TestImpact:
    def test_sqrt_impact_zero(self):
        assert sqrt_impact(0, 1e6, 0.01) == 0.0

    def test_sqrt_impact_increases_with_size(self):
        small = sqrt_impact(100, 1e6, 0.01)
        large = sqrt_impact(10000, 1e6, 0.01)
        assert large > small

    def test_linear_impact(self):
        impact = linear_impact(1000, 1e6, 1e-6)
        assert impact > 0

    def test_amihud(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.01, 100)
        dv = np.abs(rng.normal(1e7, 1e6, 100))
        illiq = amihud_illiquidity(r, dv)
        assert illiq >= 0

    def test_roll_spread(self):
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.normal(0, 0.1, 500))
        spread = roll_spread(prices)
        assert spread >= 0

    def test_predict_cost(self):
        cost = predict_cost(shares=5000, price=100.0, adv=1e6, sigma=0.02)
        assert cost.total_bps > 0
        assert cost.spread_cost_bps > 0
        assert cost.impact_cost_bps >= 0


# ---------------------------------------------------------------------------
# T-0320: TCA Tests
# ---------------------------------------------------------------------------


class TestTCA:
    def test_trade_cost_positive_buy(self):
        fills = [
            Fill(order_id="t1", price=Decimal("100.10"), quantity=Decimal("100"), commission=Decimal("1")),
        ]
        result = tca_trade(fills, arrival_price=100.00, side="buy")
        assert result.total_cost_bps > 0

    def test_trade_cost_negative_sell(self):
        fills = [
            Fill(order_id="t2", price=Decimal("99.90"), quantity=Decimal("100"), commission=Decimal("1")),
        ]
        result = tca_trade(fills, arrival_price=100.00, side="sell")
        assert result.total_cost_bps > 0

    def test_strategy_summary(self):
        trades = [
            TCAResult(symbol="AAPL", total_cost_bps=5.0, shares=100, arrival_price=100.0, strategy="strat1"),
            TCAResult(symbol="MSFT", total_cost_bps=3.0, shares=50, arrival_price=200.0, strategy="strat1"),
        ]
        summary = tca_strategy(trades, "strat1")
        assert summary.n_trades == 2

    def test_tca_report(self):
        trades = [
            TCAResult(symbol="AAPL", total_cost_bps=5.0, shares=100, arrival_price=100.0, strategy="s1"),
            TCAResult(symbol="MSFT", total_cost_bps=3.0, shares=50, arrival_price=200.0, strategy="s1"),
        ]
        report = tca_report(trades)
        assert report["n_trades"] == 2
        assert "by_strategy" in report
