"""Tests for production hardening fixes — all CRITICAL and HIGH issues.

C1: apply_fill uses fill.side (not fragile order_id parsing) — avg cost correct on cover
C3: NaN guards in VaR — no silent NaN propagation
C4: Margin interest floored at cash=0 — no double-count
H1: Covariance PSD projection — no negative variance
H2: EOP force-fill — positions always closed on cleanup
H3: Defensive invariants — price>0, weight sum tolerance, NaN guards
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from pydantic import ValidationError

from quantcore.core.types import Fill, Order, OrderType, Position, Side

# ---------------------------------------------------------------------------
# C1: apply_fill uses fill.side
# ---------------------------------------------------------------------------

class TestC1_FillSideDirection:
    def test_fill_side_field_exists(self):
        """Fill model should have a side field."""
        f = Fill(order_id="test", price=Decimal("100"), quantity=Decimal("10"),
                 commission=Decimal("0"), side=Side.BUY)
        assert f.side == Side.BUY

    def test_fill_side_backtest_engine(self):
        """Backtest apply_fill should use fill.side (verified by side override)."""
        from quantcore.backtest.engine import PortfolioState

        ps = PortfolioState(1000.0)

        # Apply a BUY fill — should go LONG
        buy_fill = Fill(
            order_id="test", price=Decimal("100"), quantity=Decimal("10"),
            commission=Decimal("0"), side=Side.BUY,
        )
        ps.apply_fill(buy_fill, "AAPL")
        assert "AAPL" in ps.positions
        assert float(ps.positions["AAPL"].quantity) == 10.0
        assert ps.cash == pytest.approx(1000.0 - 0.0 - 10 * 100.0)  # cash - commission - cost

    def test_sell_fill_creates_short(self):
        """A SELL fill should create a short (negative quantity) position."""
        from quantcore.backtest.engine import PortfolioState

        ps = PortfolioState(1000.0)

        sell_fill = Fill(
            order_id="test", price=Decimal("100"), quantity=Decimal("5"),
            commission=Decimal("0"), side=Side.SELL,
        )
        ps.apply_fill(sell_fill, "MSFT")
        assert "MSFT" in ps.positions
        assert float(ps.positions["MSFT"].quantity) == -5.0

    def test_short_cover_avg_cost_unchanged(self):
        """Covering a short should NOT change avg_cost of remaining short."""
        from quantcore.backtest.engine import PortfolioState

        ps = PortfolioState(10000.0)

        # Open short: sell 100 at $50
        sell_fill = Fill(
            order_id="s1", price=Decimal("50"), quantity=Decimal("100"),
            commission=Decimal("0"), side=Side.SELL,
        )
        ps.apply_fill(sell_fill, "XYZ")
        assert float(ps.positions["XYZ"].quantity) == -100.0
        assert float(ps.positions["XYZ"].avg_cost) == 50.0

        # Cover 30 at $55 — avg_cost should stay at $50
        buy_fill = Fill(
            order_id="b1", price=Decimal("55"), quantity=Decimal("30"),
            commission=Decimal("0"), side=Side.BUY,
        )
        ps.apply_fill(buy_fill, "XYZ")
        assert float(ps.positions["XYZ"].quantity) == -70.0
        assert float(ps.positions["XYZ"].avg_cost) == 50.0  # Unchanged!
        assert float(ps.positions["XYZ"].realized_pnl) == pytest.approx((50 - 55) * 30)  # -150

    def test_long_sell_reduces_realized_pnl_correct(self):
        """Selling from a long position should correctly realize PnL."""
        from quantcore.backtest.engine import PortfolioState

        ps = PortfolioState(10000.0)

        # Buy 100 at $50
        buy = Fill(order_id="b1", price=Decimal("50"), quantity=Decimal("100"),
                   commission=Decimal("0"), side=Side.BUY)
        ps.apply_fill(buy, "XYZ")

        # Sell 40 at $60
        sell = Fill(order_id="s1", price=Decimal("60"), quantity=Decimal("40"),
                    commission=Decimal("0"), side=Side.SELL)
        ps.apply_fill(sell, "XYZ")
        assert float(ps.positions["XYZ"].realized_pnl) == pytest.approx((60 - 50) * 40)


# ---------------------------------------------------------------------------
# C3: VaR NaN propagation guards
# ---------------------------------------------------------------------------

class TestC3_VaRNaNGuards:
    def test_risk_engine_handles_zero_vol(self):
        """Zero-vol returns should not produce NaN in risk assessment."""
        from quantcore.risk.engine import RiskEngine

        engine = RiskEngine()
        returns = np.zeros(300)
        engine.update_returns(returns)
        from quantcore.core.types import PortfolioSnapshot
        portfolio = PortfolioSnapshot(total_value=Decimal("100000"))
        assessment = engine.assess(portfolio)

        # Should not crash and VaR should not be NaN
        for var_result in [assessment.var_historical,
                           assessment.var_parametric,
                           assessment.var_cornish_fisher,
                           assessment.var_monte_carlo]:
            if var_result is not None:
                assert not np.isnan(var_result.var), f"{var_result.method} produced NaN"

    def test_risk_engine_handles_nan_returns(self):
        """Risk engine should handle returns with NaN values gracefully."""
        from quantcore.risk.engine import RiskEngine

        engine = RiskEngine()
        returns = np.array([0.001, 0.002, np.nan, 0.001, -0.002])
        engine.update_returns(returns)
        from quantcore.core.types import PortfolioSnapshot
        portfolio = PortfolioSnapshot(total_value=Decimal("100000"))
        assessment = engine.assess(portfolio)

        # Should not crash — NaN in returns should be handled
        assert assessment.portfolio_value > 0

    def test_var_result_nan_doesnt_crash_classification(self):
        """NaN VaR results should not crash _classify_risk."""
        from quantcore.risk import engine as risk_engine
        from quantcore.risk.var import VaRResult

        assessment = risk_engine.RiskAssessment(
            portfolio_value=100000.0,
            var_historical=VaRResult(method="historical", confidence=0.95,
                                     horizon_days=1, var=float("nan"), cvar=float("nan")),
        )
        # _classify_risk should handle NaN VaR without errors
        level, flags = risk_engine.RiskEngine._classify_risk(None, assessment)
        assert level in ("normal", "elevated", "high", "critical")


# ---------------------------------------------------------------------------
# C4: Margin interest with negative cash
# ---------------------------------------------------------------------------

class TestC4_MarginInterestGuard:
    def test_margin_interest_cash_floored(self):
        """Margin interest should floor cash at 0 for calculation."""
        from quantcore.backtest.engine import PortfolioState
        from quantcore.core.types import Fill, Side

        ps = PortfolioState(100_000.0)

        # Create negative cash scenario via the kernel path: buy > available cash
        fill = Fill(
            order_id="test-001", price=Decimal("550"), quantity=Decimal("200"),
            commission=Decimal("1"), side=Side.BUY,
        )
        ps.apply_fill(fill, "SPY")
        ps.update_price("SPY", 550.0)
        # Cash should now be negative (~ -10,011 after slippage and commission)
        assert ps.cash < 0, f"Expected negative cash, got {ps.cash}"

        from quantcore.backtest.financing import FinancingConfig

        config = FinancingConfig(margin_rate=0.5, margin_limit=2.0)
        ps.set_financing_config(config)
        ps.daily_financing_update()

        # Should not crash, daily_interest should be based on 110K not 120K
        assert ps.total_margin_interest >= 0
        # Calculate expected: margin_used = 110K - 0 = 110K
        # daily_interest = 110K * 0.5 / 360 = 152.78
        expected = 110_000.0 * 0.5 / 360.0
        assert abs(ps.total_margin_interest - expected) < 1.0


# ---------------------------------------------------------------------------
# H1: Covariance PSD projection
# ---------------------------------------------------------------------------

class TestH1_CovariancePSD:
    def test_nearest_psd_fixes_non_psd(self):
        """_nearest_psd should project non-PSD matrix to nearest PSD."""
        from quantcore.risk.engine import _nearest_psd

        # Non-PSD matrix: negative eigenvalue
        non_psd = np.array([
            [1.0, 1.5],
            [1.5, 1.0],
        ])
        # This matrix has eigenvalues ~2.5 and -0.5
        eigvals = np.linalg.eigvalsh(non_psd)
        assert any(eigvals < 0), "Test matrix must have negative eigenvalue"

        psd = _nearest_psd(non_psd)
        psd_eigvals = np.linalg.eigvalsh(psd)
        assert all(eig >= -1e-10 for eig in psd_eigvals), "All eigenvalues should be >= 0"

        # Portfolio variance should be non-negative
        w = np.array([0.5, 0.5])
        port_var = w @ psd @ w
        assert port_var >= 0, "Portfolio variance must be non-negative"


# ---------------------------------------------------------------------------
# H2: EOP force-fill fallback
# ---------------------------------------------------------------------------

class TestH2_EOPForceFill:
    def test_force_fill_in_eop_path(self):
        """EOP cleanup code should use _force_fill fallback."""
        import inspect

        from quantcore.backtest.engine import BacktestEngine

        source = inspect.getsource(BacktestEngine.run)
        # The EOP section should reference _force_fill
        assert "_force_fill(order, bar)" in source


# ---------------------------------------------------------------------------
# H3: Defensive invariants
# ---------------------------------------------------------------------------

class TestH3_DefensiveInvariants:
    def test_pre_trade_rejects_zero_price(self):
        """check_pre_trade should reject orders with price <= 0."""
        from quantcore.core.types import PortfolioSnapshot
        from quantcore.risk.limits import LimitStatus, RiskLimits, check_pre_trade

        order = Order(
            id="test", symbol="AAPL", side=Side.BUY,
            type=OrderType.MARKET, quantity=Decimal("100"),
        )
        portfolio = PortfolioSnapshot(total_value=Decimal("100000"))
        limits = RiskLimits()

        # market_price=0 should be rejected
        result = check_pre_trade(order, portfolio, limits, market_price=0.0)
        assert result.status == LimitStatus.REJECTED

        # market_price negative should be rejected
        result2 = check_pre_trade(order, portfolio, limits, market_price=-5.0)
        assert result2.status == LimitStatus.REJECTED

    def test_price_field_gt_zero(self):
        """Pydantic should enforce price > 0 on Fill."""
        with pytest.raises(ValidationError):
            Fill(
                order_id="test", price=Decimal("0"), quantity=Decimal("10"),
                commission=Decimal("0"),
            )

    def test_weights_normalized_on_mismatch(self):
        """Risk engine should normalize weights that don't sum to 1.0."""
        # This is tested indirectly through the engine's weight normalization
        from quantcore.risk.engine import RiskEngine

        engine = RiskEngine()
        returns = np.random.normal(0.001, 0.01, 300)
        engine.update_returns(returns)
        # Per-asset returns for attribution
        per_asset = np.random.normal(0.001, 0.01, (300, 3))
        engine.update_per_asset_returns(per_asset)

        from quantcore.core.types import PortfolioSnapshot
        portfolio = PortfolioSnapshot(
            positions={
                "AAPL": Position(symbol="AAPL", quantity=Decimal("100"),
                                 avg_cost=Decimal("200"), market_price=Decimal("200")),
                "MSFT": Position(symbol="MSFT", quantity=Decimal("50"),
                                 avg_cost=Decimal("400"), market_price=Decimal("400")),
                "GOOGL": Position(symbol="GOOGL", quantity=Decimal("30"),
                                  avg_cost=Decimal("500"), market_price=Decimal("500")),
            },
            total_value=Decimal("100000"),
            cash=Decimal("30000"),
        )
        # Should not crash during attribution (normalized weights)
        assessment = engine.assess(portfolio)
        assert assessment.portfolio_value > 0


# ---------------------------------------------------------------------------
# Integration: All fixes work together
# ---------------------------------------------------------------------------

class TestIntegration_FullHardening:
    def test_fill_side_and_pnl_integration(self):
        """Full cycle: BUY → partial SELL → BUY more → SELL all, PnL correct."""
        from quantcore.backtest.engine import PortfolioState

        ps = PortfolioState(100_000.0)

        # 1. Buy 1000 shares at $100
        f1 = Fill(order_id="o1", price=Decimal("100"), quantity=Decimal("1000"),
                  commission=Decimal("25"), side=Side.BUY)
        ps.apply_fill(f1, "AAPL")
        assert float(ps.positions["AAPL"].quantity) == 1000.0
        assert float(ps.positions["AAPL"].avg_cost) == 100.0

        # 2. Sell 300 shares at $110
        f2 = Fill(order_id="o2", price=Decimal("110"), quantity=Decimal("300"),
                  commission=Decimal("10"), side=Side.SELL)
        ps.apply_fill(f2, "AAPL")
        assert float(ps.positions["AAPL"].quantity) == 700.0
        assert float(ps.positions["AAPL"].avg_cost) == 100.0  # Unchanged on partial sell
        assert float(ps.positions["AAPL"].realized_pnl) == pytest.approx((110 - 100) * 300)

        # 3. Buy 200 more at $120
        f3 = Fill(order_id="o3", price=Decimal("120"), quantity=Decimal("200"),
                  commission=Decimal("5"), side=Side.BUY)
        ps.apply_fill(f3, "AAPL")
        assert float(ps.positions["AAPL"].quantity) == 900.0
        # Blended avg cost: (700*100 + 200*120) / 900 = (70000+24000)/900 = 104.44
        expected_avg = (700 * 100.0 + 200 * 120.0) / 900.0
        assert float(ps.positions["AAPL"].avg_cost) == pytest.approx(expected_avg, abs=0.01)

        # 4. Sell 500 at $90 (loss)
        f4 = Fill(order_id="o4", price=Decimal("90"), quantity=Decimal("500"),
                  commission=Decimal("12"), side=Side.SELL)
        ps.apply_fill(f4, "AAPL")
        assert float(ps.positions["AAPL"].quantity) == 400.0
        # Realized PnL adds: (90 - 104.44) * 500 = -7222.22
        expected_loss = (90.0 - expected_avg) * 500.0
        total_realized = (110 - 100) * 300 + expected_loss
        assert float(ps.positions["AAPL"].realized_pnl) == pytest.approx(total_realized, abs=0.1)
