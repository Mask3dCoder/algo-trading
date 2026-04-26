"""Tests for Phase 3 Portfolio Module: sizing, optimization, rebalance, tax."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from quantcore.portfolio.optimization import (
    bl_optimize,
    black_litterman,
    compute_frontier,
    hrp,
    mean_variance,
    nco,
    risk_parity,
    stability_test,
)
from quantcore.portfolio.rebalance import (
    RebalanceConfig,
    RebalanceEngine,
    RebalanceTrigger,
)
from quantcore.portfolio.sizing import (
    atr_size,
    cvar_size,
    fractional_kelly,
    kelly_criterion,
    optimal_f,
    risk_parity_size,
    volatility_target,
)
from quantcore.portfolio.tax import (
    LotMethod,
    TaxLotTracker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    n, m = 500, 5
    cov = np.eye(m) * 0.0001
    cov += np.ones((m, m)) * 0.00002
    return rng.multivariate_normal(np.full(m, 0.001), cov, n)


@pytest.fixture
def two_asset_returns():
    rng = np.random.default_rng(42)
    n = 500
    return rng.multivariate_normal(
        [0.001, 0.0008],
        [[0.0001, 0.00003], [0.00003, 0.00009]],
        n,
    )


# ---------------------------------------------------------------------------
# T-0325: Position Sizing
# ---------------------------------------------------------------------------


class TestPositionSizing:
    def test_kelly_positive_edge(self):
        f = kelly_criterion(win_prob=0.55, avg_win=0.10, avg_loss=0.05)
        assert f > 0

    def test_kelly_zero_edge(self):
        f = kelly_criterion(win_prob=0.50, avg_win=0.05, avg_loss=0.05)
        assert f == 0.0

    def test_kelly_negative_edge(self):
        f = kelly_criterion(win_prob=0.30, avg_win=0.05, avg_loss=0.10)
        assert f == 0.0

    def test_fractional_kelly_less_than_full(self):
        full = kelly_criterion(0.55, 0.10, 0.05)
        frac = fractional_kelly(0.55, 0.10, 0.05, fraction=0.25)
        assert frac < full

    def test_vol_target(self):
        w = volatility_target(asset_vol=0.20, target_vol=0.10, n_assets=1)
        assert abs(w - 0.50) < 0.01  # 10%/20% = 0.5

    def test_vol_target_multiple_assets(self):
        w = volatility_target(asset_vol=0.20, target_vol=0.10, n_assets=4)
        assert 0 < w < 1

    def test_atr_size(self):
        shares = atr_size(account_value=100000, risk_pct=0.01, atr=2.0, multiplier=2.0)
        assert shares == 250  # 1000 / 4

    def test_cvar_size(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.02, 500)
        w = cvar_size(r, max_cvar=5000, account_value=100000)
        assert 0 < w <= 1

    def test_optimal_f(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.002, 0.02, 100)
        f = optimal_f(r)
        assert 0 < f <= 1

    def test_risk_parity_size(self):
        vols = np.array([0.20, 0.15, 0.25])
        corr = np.array([
            [1.0, 0.3, 0.4],
            [0.3, 1.0, 0.2],
            [0.4, 0.2, 1.0],
        ])
        w = risk_parity_size(vols, corr, target_vol=0.10)
        assert len(w) == 3
        assert all(0 < wi < 1 for wi in w)


# ---------------------------------------------------------------------------
# T-0326: MVO + Black-Litterman
# ---------------------------------------------------------------------------


class TestMVOBlackLitterman:
    def test_mvo_max_sharpe(self, sample_returns):
        result = mean_variance(sample_returns, target="max_sharpe", long_only=True)
        assert abs(result.weights.sum() - 1.0) < 1e-5
        assert all(w >= -1e-10 for w in result.weights)  # Long only
        assert result.sharpe_ratio > 0

    def test_mvo_min_variance(self, sample_returns):
        result = mean_variance(sample_returns, target="min_variance", long_only=True)
        assert abs(result.weights.sum() - 1.0) < 1e-5

    def test_mvo_single_asset(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, (100, 1))
        result = mean_variance(r, target="max_sharpe", long_only=True)
        assert result.weights[0] == 1.0

    def test_efficient_frontier(self, sample_returns):
        frontier = compute_frontier(sample_returns, n_points=20)
        assert len(frontier) == 20
        # Risk should increase along frontier
        risks = [f[0] for f in frontier]
        assert risks[0] <= risks[-1]

    def test_black_litterman_no_views(self, sample_returns):
        mc = np.ones(sample_returns.shape[1]) / sample_returns.shape[1]
        cov = np.cov(sample_returns.T) * 252
        posterior, _ = black_litterman(mc, cov, views=None)
        assert len(posterior) == sample_returns.shape[1]

    def test_black_litterman_with_views(self, sample_returns):
        mc = np.ones(sample_returns.shape[1]) / sample_returns.shape[1]
        cov = np.cov(sample_returns.T) * 252
        views = {0: 0.15, 2: 0.05}  # Asset 0 has high return, asset 2 moderate
        posterior, posterior_cov = black_litterman(mc, cov, views=views)
        assert len(posterior) == sample_returns.shape[1]
        # Asset 0 should have higher posterior return
        assert posterior[0] > posterior[2]

    def test_bl_optimize(self, sample_returns):
        mc = np.ones(sample_returns.shape[1]) / sample_returns.shape[1]
        cov = np.cov(sample_returns.T) * 252
        posterior, posterior_cov = black_litterman(mc, cov, views={0: 0.15})
        result = bl_optimize(posterior, posterior_cov, long_only=True)
        assert abs(result.weights.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# T-0327: HRP + NCO + Risk Parity
# ---------------------------------------------------------------------------


class TestHRPNCORiskParity:
    def test_hrp_weights_sum_to_one(self, sample_returns):
        result = hrp(sample_returns)
        assert abs(result.weights.sum() - 1.0) < 1e-5
        assert all(w >= -1e-10 for w in result.weights)

    def test_hrp_stability(self, sample_returns):
        results = stability_test(sample_returns, n_perturbations=20)
        # HRP should be more stable than MVO
        assert (
            results["stability_ratio"] > 1.0
            or results.get("hrp_mean_change", 0) <= results.get("mvo_mean_change", float("inf"))
        )

    def test_nco(self, sample_returns):
        result = nco(sample_returns, n_clusters=3)
        assert abs(result.weights.sum() - 1.0) < 1e-5

    def test_nco_two_assets(self, two_asset_returns):
        result = nco(two_asset_returns)
        assert abs(result.weights.sum() - 1.0) < 1e-5

    def test_risk_parity_weights(self, sample_returns):
        cov = np.cov(sample_returns.T)
        result = risk_parity(cov)
        assert abs(result.weights.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# T-0328: Rebalance Engine
# ---------------------------------------------------------------------------


class TestRebalanceEngine:
    def test_threshold_trigger(self):
        engine = RebalanceEngine(RebalanceConfig(trigger=RebalanceTrigger.THRESHOLD, drift_threshold=0.05))
        should, reason = engine.should_rebalance(
            {"AAPL": 0.60, "MSFT": 0.40},
            {"AAPL": 0.50, "MSFT": 0.50},
        )
        assert should
        assert "threshold" in reason

    def test_no_trigger_within_tolerance(self):
        engine = RebalanceEngine(RebalanceConfig(drift_threshold=0.10))
        should, _ = engine.should_rebalance(
            {"AAPL": 0.51, "MSFT": 0.49},
            {"AAPL": 0.50, "MSFT": 0.50},
        )
        assert not should

    def test_compute_trades(self):
        engine = RebalanceEngine()
        trades = engine.compute_trades(
            current_weights={"AAPL": 0.60, "MSFT": 0.40},
            target_weights={"AAPL": 0.50, "MSFT": 0.50},
            portfolio_value=100000,
            current_prices={"AAPL": 200.0, "MSFT": 400.0},
        )
        assert len(trades) >= 1

    def test_rebalance_ignores_small_drifts(self):
        engine = RebalanceEngine(RebalanceConfig(tolerance=0.02))
        trades = engine.compute_trades(
            current_weights={"AAPL": 0.505, "MSFT": 0.495},
            target_weights={"AAPL": 0.50, "MSFT": 0.50},
            portfolio_value=1000,
        )
        assert len(trades) == 0

    def test_cost_benefit_filter(self):
        engine = RebalanceEngine(
            RebalanceConfig(
                tolerance=0.001,
                cost_benefit_threshold=1000.0,  # Huge threshold
                min_trade_value=1.0,
            )
        )
        trades = engine.compute_trades(
            current_weights={"AAPL": 0.55, "MSFT": 0.45},
            target_weights={"AAPL": 0.50, "MSFT": 0.50},
            portfolio_value=10000,
        )
        # Should filter out because cost > benefit
        assert len(trades) == 0  # or very few


# ---------------------------------------------------------------------------
# T-0329: Tax-Lot Tracking
# ---------------------------------------------------------------------------


class TestTaxLotTracker:
    def test_add_lot(self):
        tracker = TaxLotTracker()
        lot = tracker.add_lot("AAPL", Decimal("100"), Decimal("150"))
        assert lot.shares == Decimal("100")
        assert lot.cost_basis == Decimal("150")

    def test_fifo_selection(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", Decimal("100"), Decimal("100"), datetime(2020, 1, 1))
        tracker.add_lot("AAPL", Decimal("100"), Decimal("150"), datetime(2021, 1, 1))

        results = tracker.realize("AAPL", Decimal("50"), Decimal("200"), LotMethod.FIFO)
        assert len(results) == 1
        assert results[0].shares == Decimal("50")
        # FIFO: oldest lot first
        assert results[0].lots[0].cost_basis == Decimal("100")

    def test_lifo_selection(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", Decimal("100"), Decimal("100"), datetime(2020, 1, 1))
        tracker.add_lot("AAPL", Decimal("100"), Decimal("150"), datetime(2021, 1, 1))

        results = tracker.realize("AAPL", Decimal("50"), Decimal("200"), LotMethod.LIFO)
        # LIFO: newest lot first
        assert results[0].lots[0].cost_basis == Decimal("150")

    def test_holding_period_classification(self):
        tracker = TaxLotTracker()
        tracker.add_lot(
            "AAPL", Decimal("100"), Decimal("100"),
            datetime.utcnow() - timedelta(days=400),  # Long term
        )
        results = tracker.realize("AAPL", Decimal("50"), Decimal("200"))
        assert results[0].is_long_term

    def test_wash_sale_detection(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", Decimal("100"), Decimal("100"), datetime.utcnow() - timedelta(days=10))
        # Sell at a loss
        results = tracker.realize("AAPL", Decimal("50"), Decimal("90"))
        assert results[0].gain < 0
        # Wash sale: sold at a loss, within 30 days of purchase
        assert results[0].is_wash_sale

    def test_harvest_candidates(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", Decimal("100"), Decimal("200"), datetime.utcnow() - timedelta(days=400))
        tracker.add_lot("MSFT", Decimal("50"), Decimal("400"), datetime.utcnow() - timedelta(days=200))

        prices = {"AAPL": 180.0, "MSFT": 350.0}
        candidates = tracker.harvest_candidates(threshold=-500, current_prices=prices)
        assert len(candidates) >= 1

    def test_unrealized_gains(self):
        tracker = TaxLotTracker()
        tracker.add_lot("AAPL", Decimal("100"), Decimal("150"))
        tracker.add_lot("MSFT", Decimal("50"), Decimal("300"))

        gains = tracker.unrealized_gains({"AAPL": 200.0, "MSFT": 350.0})
        assert gains["AAPL"] > 0
        assert gains["MSFT"] > 0

    def test_short_vs_long(self):
        tracker = TaxLotTracker()
        old_lot = tracker.add_lot("AAPL", Decimal("100"), Decimal("100"), datetime.utcnow() - timedelta(days=500))
        new_lot = tracker.add_lot("AAPL", Decimal("50"), Decimal("100"), datetime.utcnow() - timedelta(days=10))

        st, lt = tracker.short_vs_long([old_lot, new_lot], Decimal("150"))
        assert lt > 0  # Old lot has LT gain
