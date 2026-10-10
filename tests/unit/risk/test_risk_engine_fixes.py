"""Tests for risk engine fixes: stress tests, VaR, limits, attribution.

Validates:
  A1: apply_scenario called with correct signature
  A2: concentration_report receives reverse-stress results
  A3: check_pre_trade uses actual market price (not $100 default)
  A8: parametric_var uses sqrt(w'Σw) when weights provided
  B1: leverage check rejects orders exceeding max_leverage
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np

from quantcore.core.types import (
    Order,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
)
from quantcore.risk.engine import RiskEngine
from quantcore.risk.limits import (
    LimitStatus,
    RiskLimits,
    check_pre_trade,
)
from quantcore.risk.stress import apply_scenario, build_scenario_library
from quantcore.risk.var import parametric_var, portfolio_parametric_var

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_portfolio(
    positions: dict[str, tuple[float, float]],
    cash: float = 1_000_000.0,
) -> PortfolioSnapshot:
    """Build a PortfolioSnapshot from {symbol: (qty, price)}."""
    pos_dict = {}
    total = cash
    for sym, (qty, price) in positions.items():
        p = Position(
            symbol=sym,
            quantity=Decimal(str(qty)),
            avg_cost=Decimal(str(price)),
            market_price=Decimal(str(price)),
        )
        pos_dict[sym] = p
        total += qty * price
    return PortfolioSnapshot(
        positions=pos_dict,
        cash=Decimal(str(cash)),
        total_value=Decimal(str(total)),
    )


def _make_order(
    symbol: str = "AAPL",
    qty: float = 100,
    side: Side = Side.BUY,
    limit_price: float | None = None,
) -> Order:
    return Order(
        id="test_order",
        symbol=symbol,
        side=side,
        type=OrderType.MARKET if limit_price is None else OrderType.LIMIT,
        quantity=Decimal(str(qty)),
        limit_price=Decimal(str(limit_price)) if limit_price is not None else None,
    )


# ---------------------------------------------------------------------------
# A1: Stress tests produce results (not silent failures)
# ---------------------------------------------------------------------------

class TestStressTestsFixed:
    def test_apply_scenario_returns_result(self):
        """apply_scenario should return a StressResult, not raise."""
        portfolio = _make_portfolio({"SPY": (100, 450.0)})
        scenarios = build_scenario_library()
        # Use a real scenario name from the library
        result = apply_scenario(portfolio, scenarios["GFC_2008"])
        assert result is not None
        assert hasattr(result, "pnl")

    def test_stress_results_populated_in_assessment(self):
        """RiskEngine.assess should populate stress_results dict."""
        engine = RiskEngine(stress_scenarios=["GFC_2008", "COVID_2020"])
        returns = np.random.normal(0.0005, 0.01, 300)
        engine.update_returns(returns)
        portfolio = _make_portfolio({"SPY": (100, 450.0), "TLT": (50, 100.0)})
        assessment = engine.assess(portfolio)
        assert "GFC_2008" in assessment.stress_results
        assert "COVID_2020" in assessment.stress_results
        for name, result in assessment.stress_results.items():
            if name != "concentration":
                assert hasattr(result, "pnl")
                assert isinstance(result.pnl, (int, float))


# ---------------------------------------------------------------------------
# A2: Concentration report receives reverse-stress results
# ---------------------------------------------------------------------------

class TestConcentrationReportFixed:
    def test_concentration_report_in_stress_results(self):
        """concentration key should appear in stress_results when positions exist."""
        engine = RiskEngine()
        returns = np.random.normal(0.0005, 0.01, 300)
        engine.update_returns(returns)
        portfolio = _make_portfolio({"AAPL": (200, 150.0), "MSFT": (100, 300.0)})
        assessment = engine.assess(portfolio)
        # concentration_report returns a string, stored in stress_results
        assert "concentration" in assessment.stress_results
        assert isinstance(assessment.stress_results["concentration"], str)


# ---------------------------------------------------------------------------
# A3: Pre-trade check uses actual market price
# ---------------------------------------------------------------------------

class TestPreTradePriceFixed:
    def test_market_order_with_actual_price(self):
        """Market order at $500 stock should check $50K value, not $10K."""
        portfolio = _make_portfolio({}, cash=1_000_000.0)
        limits = RiskLimits(max_dollar_exposure_per_name=100_000.0)
        order = _make_order("AAPL", qty=100, limit_price=None)

        result = check_pre_trade(order, portfolio, limits, market_price=500.0)
        assert result.status == LimitStatus.APPROVED

    def test_market_order_without_price_rejected(self):
        """Market order with no price at all should be rejected."""
        portfolio = _make_portfolio({}, cash=1_000_000.0)
        limits = RiskLimits()
        order = _make_order("AAPL", qty=100, limit_price=None)

        result = check_pre_trade(order, portfolio, limits, market_price=None)
        assert result.status == LimitStatus.REJECTED
        assert "no limit_price and no market_price" in result.reason

    def test_concentration_uses_real_price(self):
        """Concentration check must use $500, not $100."""
        portfolio = _make_portfolio({}, cash=100_000.0)
        limits = RiskLimits(max_concentration_pct=0.25)
        order = _make_order("AAPL", qty=100, limit_price=None)

        # market_price=500 -> value=50,000 -> concentration=50,000/100,000=50% > 25%
        result = check_pre_trade(order, portfolio, limits, market_price=500.0)
        assert result.status == LimitStatus.REJECTED
        assert "Concentration" in result.reason

    def test_dollar_exposure_uses_real_price(self):
        """Dollar exposure check must use actual price."""
        portfolio = _make_portfolio({}, cash=10_000_000.0)
        limits = RiskLimits(max_dollar_exposure_per_name=50_000.0)
        order = _make_order("AAPL", qty=100, limit_price=None)

        # market_price=500 -> value=50,000 -> equals limit, APPROVED
        result = check_pre_trade(order, portfolio, limits, market_price=500.0)
        assert result.status == LimitStatus.APPROVED

        # market_price=600 -> value=60,000 > 50,000 -> REJECTED
        result2 = check_pre_trade(order, portfolio, limits, market_price=600.0)
        assert result2.status == LimitStatus.REJECTED


# ---------------------------------------------------------------------------
# A8: Parametric VaR with weights
# ---------------------------------------------------------------------------

class TestParametricVaRFixed:
    def test_scalar_sigma_unchanged(self):
        """Scalar sigma should work as before."""
        result = parametric_var(0.001, 0.02, confidence=0.95)
        assert result.method == "parametric"
        assert result.var < 0  # Loss is negative

    def test_covariance_matrix_with_weights(self):
        """With weights + cov matrix, should use sqrt(w'Σw)."""
        cov = np.array([
            [0.04, 0.01],
            [0.01, 0.09],
        ])
        w = np.array([0.6, 0.4])

        expected_sigma = np.sqrt(w @ cov @ w)
        result = parametric_var(0.0, cov, confidence=0.95, weights=w)
        from scipy import stats as sp_stats
        z = sp_stats.norm.ppf(0.05)
        expected_var = expected_sigma * z
        assert abs(result.var - expected_var) < 0.001

    def test_covariance_without_weights_fallback(self):
        """Without weights, should use simplified fallback."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = parametric_var(0.0, cov, confidence=0.95)
        assert result.var < 0

    def test_portfolio_parametric_var(self):
        """portfolio_parametric_var should compute sqrt(w'Σw) correctly."""
        cov = np.array([[0.04, 0.0], [0.0, 0.09]])
        w = np.array([1.0, 0.0])
        result = portfolio_parametric_var(w, cov, confidence=0.95)
        from scipy import stats as sp_stats
        z = sp_stats.norm.ppf(0.05)
        assert abs(result.var - 0.2 * z) < 0.001


# ---------------------------------------------------------------------------
# B1: Leverage check
# ---------------------------------------------------------------------------

class TestLeverageCheck:
    def test_leverage_within_limit(self):
        """Order that keeps leverage under limit should be approved."""
        portfolio = _make_portfolio({"SPY": (1000, 500.0)}, cash=500_000.0)
        limits = RiskLimits(max_leverage=2.0, max_dollar_exposure_per_name=1_000_000_000.0)
        order = _make_order("AAPL", qty=10, limit_price=100.0)

        # exposure = 500K + 1K = 501K, leverage = 501K/1M = 0.501 < 2.0
        result = check_pre_trade(order, portfolio, limits)
        assert result.status == LimitStatus.APPROVED

    def test_leverage_exceeds_limit(self):
        """Order that pushes leverage over limit should be rejected."""
        # Total value = 14.5M cash + 500K SPY = 15M
        # Existing exposure = 500K
        portfolio = _make_portfolio({"SPY": (1000, 500.0)}, cash=14_500_000.0)
        limits = RiskLimits(
            max_leverage=1.1,
            max_dollar_exposure_per_name=10_000_000.0,
            max_concentration_pct=2.0,
        )
        # Order = 12000 * 600 = 7,200,000
        # concentration = 7.2M / 15M = 0.48 < 2.0 -> passes
        # dollar exposure = 7.2M < 10M -> passes
        # leverage = (500K + 7.2M) / 15M = 7.7M/15M = 0.513 < 1.1 -> still passes!
        # Need: leverage > 1.1 => existing + order > 1.1 * total_value
        # 500K + order > 1.1 * 15M = 16.5M => order > 16M
        # But dollar exposure cap is 10M. So reduce total_value.
        # Use: cash=500K, SPY=1000*500=500K, total=1M, existing=500K
        # Order=12000*600=7.2M > 10M cap? No, 7.2M < 10M. But concentration=7.2M/1M=720% > 2%.
        # Use: cash=10M, SPY=10000*500=5M, total=15M, existing=5M
        # Order=20000*600=12M. concentration=12M/15M=0.8 < 2.0. dollar=12M > 10M. Rejected by dollar.
        # Use: cash=20M, SPY=20000*500=10M, total=30M, existing=10M
        # Order=20000*600=12M. concentration=12M/30M=0.4. dollar=12M > 10M. Rejected by dollar.
        # Increase dollar cap to 15M.
        limits.max_dollar_exposure_per_name = 15_000_000.0
        # Order=20000*600=12M. concentration=0.4<2.0. dollar=12M<15M.
        # leverage=(10M+12M)/30M=22M/30M=0.733<1.1. Still passes.
        # Need leverage > 1.1 => 10M+order > 33M => order > 23M => qty > 38334 @ 600
        # But max_position_notional constraint... Actually limits don't have that.
        # Let's just use smaller total_value.
        # cash=5M, SPY=500*500=250K, total=5.25M, existing=250K
        # Order=10000*600=6M. concentration=6M/5.25M=1.14<2.0. dollar=6M<15M.
        # leverage=(250K+6M)/5.25M=6.25M/5.25M=1.19>1.1. REJECTED!
        portfolio = _make_portfolio({"SPY": (500, 500.0)}, cash=5_000_000.0)
        order = _make_order("AAPL", qty=10000, limit_price=600.0)

        result = check_pre_trade(order, portfolio, limits)
        assert result.status == LimitStatus.REJECTED
        assert "Leverage" in result.reason


# ---------------------------------------------------------------------------
# Risk engine integration
# ---------------------------------------------------------------------------

class TestRiskEngineIntegration:
    def test_full_assessment_smoke(self):
        """Complete assessment should not crash and populate key fields."""
        engine = RiskEngine(limits=RiskLimits(max_leverage=3.0))
        returns = np.random.normal(0.0005, 0.01, 300)
        engine.update_returns(returns)
        portfolio = _make_portfolio({"SPY": (100, 450.0), "TLT": (50, 100.0)})
        assessment = engine.assess(portfolio)

        assert assessment.portfolio_value > 0
        assert assessment.var_historical is not None
        assert assessment.var_parametric is not None
        assert assessment.cvar_value != 0.0
        assert len(assessment.stress_results) > 0
        assert assessment.drawdown is not None
        assert assessment.risk_level in ("low", "normal", "elevated", "high", "critical")
