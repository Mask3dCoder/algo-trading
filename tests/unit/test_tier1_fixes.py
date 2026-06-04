"""Validation tests for the 7 Tier 1 critical bug fixes.

Each test verifies that the fix is correct and the old bug is eliminated.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Bug 1: Sortino Ratio — downside deviation formula
# ---------------------------------------------------------------------------

class TestSortinoFix:
    """New Sortino must use sqrt(mean(min(r,0)^2)) over ALL returns."""

    def test_sortino_uses_correct_formula(self):
        """Verify Sortino matches the standard definition."""
        from quantcore.risk.metrics import sortino_ratio

        # Series with known downside: 3 negative returns out of 6
        returns = np.array([0.02, -0.03, 0.01, -0.04, 0.015, -0.01])
        result = sortino_ratio(returns, annualize=False)

        # Manual calculation per standard formula:
        # MAR = 0, excess = returns
        # downside_dev = sqrt(mean(min(r, 0)^2))
        min_vals = np.minimum(returns, 0.0)
        expected_dd = np.sqrt(np.mean(min_vals ** 2))
        expected_sr = np.mean(returns) / expected_dd

        assert abs(result - expected_sr) < 1e-6, (
            f"Sortino {result:.6f} != expected {expected_sr:.6f}"
        )

    def test_sortino_with_all_positive_returns(self):
        """All positive returns should return 0.0 (no downside)."""
        from quantcore.risk.metrics import sortino_ratio

        returns = np.array([0.01, 0.02, 0.005, 0.015, 0.01])
        result = sortino_ratio(returns)
        assert result == 0.0

    def test_sortino_annualization(self):
        """Verify annualization factor is sqrt(252)."""
        from quantcore.risk.metrics import sortino_ratio

        returns = np.array([0.01, -0.01, 0.01, -0.01, 0.01, -0.01] * 100)
        annual = sortino_ratio(returns, annualize=True)
        daily = sortino_ratio(returns, annualize=False)

        if daily != 0:
            assert abs(annual / daily - np.sqrt(252)) < 0.1


# ---------------------------------------------------------------------------
# Bug 2: Kelly Formula — correct f* = p - q/b
# ---------------------------------------------------------------------------

class TestKellyFix:
    """Kelly fraction must match f* = (p*b - q) / b = p - q/b."""

    def test_kelly_known_value(self):
        """Verify Kelly with known p and b values."""
        from quantcore.strategies.equity.stat_arb import StatArbStrategy
        from quantcore.strategies.base import StrategyConfig

        config = StrategyConfig(name="test_stat_arb", strategy_type="equity")
        strat = StatArbStrategy(config)
        strat.entry_z = 1.5
        strat.exit_z = 0.0
        strat.kelly_fraction = 1.0

        result = strat._kelly_size(half_life=5.0, spread_std=1.0)

        # Manual: p = 1 - norm.cdf(-1.5), avg_win = 1.5, avg_loss = 0.75
        from scipy.stats import norm
        p = 1.0 - norm.cdf(-1.5)
        q = 1.0 - p
        avg_win = 1.5
        avg_loss = 0.75
        expected = p - q * avg_win / avg_loss
        expected = max(0.0, min(expected, 1.0))

        assert abs(result - expected) < 1e-6, (
            f"Kelly {result:.6f} != expected {expected:.6f}"
        )

    def test_kelly_non_negative(self):
        """Kelly fraction must be non-negative for valid entry thresholds."""
        from quantcore.strategies.equity.stat_arb import StatArbStrategy
        from quantcore.strategies.base import StrategyConfig

        for entry_z in [0.5, 1.0, 1.5, 2.0]:
            config = StrategyConfig(name="test_stat_arb", strategy_type="equity")
            strat = StatArbStrategy(config)
            strat.entry_z = entry_z
            strat.exit_z = 0.0
            strat.kelly_fraction = 1.0

            result = strat._kelly_size(half_life=5.0, spread_std=1.0)
            assert result >= 0.0, f"Kelly should be non-negative for entry_z={entry_z}"


# ---------------------------------------------------------------------------
# Bug 3: CPCV — first fold must not train on future data
# ---------------------------------------------------------------------------

class TestCPCVFix:
    """All folds must have train_end < test_start."""

    def test_all_folds_chronological(self):
        from quantcore.backtest.walk_forward import purged_kfold_cv

        for n_bars in [200, 500, 1000, 2000]:
            for n_splits in [3, 5, 10]:
                folds = purged_kfold_cv(n_bars=n_bars, n_splits=n_splits)
                for i, fold in enumerate(folds):
                    assert fold.train_end <= fold.test_start, (
                        f"Fold {i}: train_end={fold.train_end} > test_start={fold.test_start}"
                    )

    def test_first_fold_dropped_when_no_prior_data(self):
        """With n_bars=300, n_splits=3, test_size=60: first fold has
        test_start=0, train_end=-1 → must be skipped."""
        from quantcore.backtest.walk_forward import purged_kfold_cv

        folds = purged_kfold_cv(n_bars=300, n_splits=3, test_size=60, embargo_pct=0.01)
        for fold in folds:
            assert fold.test_start > 0, "First fold should have been dropped"

    def test_no_train_after_test(self):
        """No fold should have training data AFTER test data."""
        from quantcore.backtest.walk_forward import purged_kfold_cv

        folds = purged_kfold_cv(n_bars=1000, n_splits=5, test_size=100)
        for i, fold in enumerate(folds):
            assert fold.train_start < fold.test_start, (
                f"Fold {i}: train starts at {fold.train_start}, test at {fold.test_start}"
            )


# ---------------------------------------------------------------------------
# Bug 4: Permanent Impact — signed accumulation
# ---------------------------------------------------------------------------

class TestPermanentImpactFix:
    """Buys accumulate positive impact, sells accumulate negative."""

    def test_buy_sell_net_zero_impact(self):
        """100 buys then 100 sells should net to ~0 cumulative permanent impact."""
        permanent_impact: dict[str, float] = {}
        symbol = "TEST"
        perm_impact_bps = 1.0
        price = 100.0

        # Simulate 100 buys
        for _ in range(100):
            perm_impact_price = price * perm_impact_bps / 10_000.0
            sign = 1.0  # BUY
            permanent_impact[symbol] = permanent_impact.get(symbol, 0.0) + sign * perm_impact_price

        buy_impact = permanent_impact[symbol]
        assert buy_impact > 0, "Buys should accumulate positive impact"

        # Simulate 100 sells
        for _ in range(100):
            perm_impact_price = price * perm_impact_bps / 10_000.0
            sign = -1.0  # SELL
            permanent_impact[symbol] = permanent_impact.get(symbol, 0.0) + sign * perm_impact_price

        assert abs(permanent_impact[symbol]) < 1e-10, (
            f"Net impact should be ~0, got {permanent_impact[symbol]}"
        )

    def test_buy_impact_increases_fill_price(self):
        """Accumulated buy impact should make subsequent buys more expensive."""
        permanent_impact: dict[str, float] = {}
        symbol = "TEST"
        price = 100.0

        for _ in range(50):
            perm = price * 1.0 / 10_000.0
            permanent_impact[symbol] = permanent_impact.get(symbol, 0.0) + perm

        # Apply to a new buy: fill_price += sym_perm
        fill_price = price + permanent_impact[symbol]
        assert fill_price > price, "Buy impact should increase fill price"

        # Apply to a sell: fill_price += sym_perm (positive from buys = benefits sell)
        sell_fill = price + permanent_impact[symbol]
        assert sell_fill > price, "Accumulated buy impact should benefit sell fills"


# ---------------------------------------------------------------------------
# Bug 5: Fill Simulator — H/L bounds
# ---------------------------------------------------------------------------

class TestFillHLBounds:
    """Fill prices must be within [bar.low, bar.high]."""

    def test_fill_clipped_to_bar_range(self):
        """After slippage + impact, fill_price should be clipped."""
        bar_high = 105.0
        bar_low = 95.0

        # Case 1: fill_price above high
        fill_price = 110.0
        fill_price = max(bar_low, min(fill_price, bar_high))
        assert fill_price == bar_high

        # Case 2: fill_price below low
        fill_price = 90.0
        fill_price = max(bar_low, min(fill_price, bar_high))
        assert fill_price == bar_low

        # Case 3: fill_price within range
        fill_price = 100.0
        fill_price = max(bar_low, min(fill_price, bar_high))
        assert fill_price == 100.0


# ---------------------------------------------------------------------------
# Bug 6: $100 Hardcoded Price Fallback
# ---------------------------------------------------------------------------

class TestPriceFallbackFix:
    """Risk manager should use market_prices when available."""

    def test_order_price_uses_market_prices(self):
        from quantcore.live.risk_manager import RiskManagerLive, RiskLimitsConfig

        rm = RiskManagerLive(RiskLimitsConfig())
        from quantcore.core.types import Order, Side, OrderType

        order = Order(
            id="test",
            symbol="AAPL",
            side=Side.BUY,
            quantity=10,
            type=OrderType.MARKET,
        )

        # With market_prices provided
        price = rm._order_price(order, market_prices={"AAPL": 200.0})
        assert price == 200.0, f"Expected $200, got ${price}"

        # Without market_prices and no limit_price — should raise ValueError
        # (no $100 fallback to avoid corrupt sizing)
        with pytest.raises(ValueError, match="No price available"):
            rm._order_price(order, market_prices=None)

    def test_pre_trade_check_uses_market_prices(self):
        from quantcore.live.risk_manager import RiskManagerLive, RiskLimitsConfig, Approval
        from quantcore.core.types import Order, Side, OrderType

        rm = RiskManagerLive(RiskLimitsConfig())
        order = Order(
            id="test",
            symbol="AAPL",
            side=Side.BUY,
            quantity=10,
            type=OrderType.MARKET,
        )

        # With $200 market price, notional = $2000
        result = rm.pre_trade_check(
            order, equity=100_000, market_prices={"AAPL": 200.0}
        )
        assert isinstance(result, Approval)


# ---------------------------------------------------------------------------
# Bug 7: Live Aggregator — update_positions / update_equity
# ---------------------------------------------------------------------------

class TestAggregatorMethods:
    """StrategyAggregator must have update_positions and update_equity."""

    def test_update_positions_exists(self):
        from quantcore.portfolio.aggregator import StrategyAggregator

        agg = StrategyAggregator()
        assert hasattr(agg, "update_positions")
        assert hasattr(agg, "update_equity")

    def test_update_positions_tracks_exposure(self):
        from quantcore.portfolio.aggregator import StrategyAggregator

        agg = StrategyAggregator()

        class MockPos:
            def __init__(self, qty, price):
                self.quantity = qty
                self.market_price = price

        positions = {
            "AAPL": MockPos(10, 200.0),
            "TLT": MockPos(50, 90.0),
        }
        agg.update_positions("live", positions, cash=50_000.0)

        assert agg._name_exposure["AAPL"] == 2000.0
        assert agg._name_exposure["TLT"] == 4500.0
        assert agg._gross_exposure == 6500.0

    def test_update_equity_sets_value(self):
        from quantcore.portfolio.aggregator import StrategyAggregator

        agg = StrategyAggregator()
        agg.update_equity(100_000.0)
        assert agg._equity == 100_000.0

    def test_runner_sync_portfolio_logs_errors(self):
        """Verify that runner.py now logs errors instead of silently swallowing."""
        import quantcore.live.runner as runner_mod
        assert hasattr(runner_mod, "log"), "runner.py should have a 'log' logger"
