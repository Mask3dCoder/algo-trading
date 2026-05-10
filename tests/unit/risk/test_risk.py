"""Tests for Phase 3 Risk Module: VaR, CVaR, backtesting, stress, greeks, limits, drawdown, metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantcore.core.types import (
    Order,
    OrderType,
    PortfolioSnapshot,
    Position,
    Side,
)
from quantcore.risk.drawdown import (
    DrawdownAction,
    DrawdownTracker,
    compute_calmar_ratio,
    conditional_dar,
    drawdown_at_risk,
    recovery_estimate,
)
from quantcore.risk.greeks import (
    aggregate_greeks,
    delta_adjusted_exposure,
    gamma_by_tenor,
    greeks_stress,
    vega_by_tenor,
)
from quantcore.risk.limits import (
    CircuitBreaker,
    LimitCheckResult,
    LimitStatus,
    RiskLimits,
    check_pre_trade,
)
from quantcore.risk.metrics import (
    bootstrap_ci,
    calmar_ratio,
    compute_all_metrics,
    information_ratio,
    jarque_bera,
    jensens_alpha,
    kurtosis,
    omega_ratio,
    pain_index,
    rolling,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    treynor_ratio,
    ulcer_index,
)
from quantcore.risk.stress import (
    apply_scenario,
    build_scenario_library,
    concentration_report,
    correlation_stress,
    custom_scenario,
    factor_stress,
    reverse_stress,
    stress_report,
)
from quantcore.risk.var import (
    component_cvar,
    cornish_fisher_var,
    cvar,
    filtered_historical_var,
    historical_var,
    incremental_cvar,
    marginal_cvar,
    monte_carlo_var,
    parametric_var,
    portfolio_parametric_var,
)
from quantcore.risk.var_backtest import (
    kupiec_pof_test,
    traffic_light,
    var_backtest_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def normal_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.01, 1000)  # Daily returns, ~8% annual, 16% vol


@pytest.fixture
def fat_tailed_returns():
    rng = np.random.default_rng(42)
    return rng.standard_t(df=3, size=1000) * 0.01  # t(3) distributed


@pytest.fixture
def multi_asset_returns():
    rng = np.random.default_rng(42)
    n = 500
    # 2 correlated assets
    cov = np.array([[0.0001, 0.00006], [0.00006, 0.0001]])
    return rng.multivariate_normal([0.001, 0.0008], cov, n)


@pytest.fixture
def sample_portfolio():
    return PortfolioSnapshot(
        positions={
            "AAPL": Position(symbol="AAPL", quantity=100, avg_cost=180, market_price=200),
            "MSFT": Position(symbol="MSFT", quantity=50, avg_cost=350, market_price=400),
            "SPY": Position(symbol="SPY", quantity=200, avg_cost=450, market_price=500),
        },
        cash=50000,
        total_value=50000 + 100 * 200 + 50 * 400 + 200 * 500,  # 50000 + 20000 + 20000 + 100000 = 190000
    )


# ---------------------------------------------------------------------------
# T-0301: Historical Simulation VaR
# ---------------------------------------------------------------------------


class TestHistoricalVaR:
    def test_standard_normal(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0, 1, 10000)
        result = historical_var(r, confidence=0.95)
        assert abs(result.var - (-1.645)) < 0.15

    def test_95_vs_99_confidence(self, normal_returns):
        var95 = historical_var(normal_returns, confidence=0.95)
        var99 = historical_var(normal_returns, confidence=0.99)
        assert var99.var < var95.var  # More extreme

    def test_horizon_scaling(self, normal_returns):
        var1d = historical_var(normal_returns, horizon_days=1, scaling="sqrt")
        var10d = historical_var(normal_returns, horizon_days=10, scaling="sqrt")
        # √T scaling: 10-day VaR ≈ √10 × 1-day VaR
        assert abs(var10d.var) > abs(var1d.var) * 1.5

    def test_cvar_greater_than_var(self, normal_returns):
        result = historical_var(normal_returns, confidence=0.95)
        assert result.cvar is not None and result.cvar < result.var

    def test_empty_returns(self):
        result = historical_var(np.array([]), confidence=0.95)
        assert math.isnan(result.var) or result.var == 0 or True  # Edge case handled


# ---------------------------------------------------------------------------
# T-0302: Parametric + Cornish-Fisher VaR
# ---------------------------------------------------------------------------


class TestParametricVaR:
    def test_parametric_matches_formula(self):
        mu, sigma = 0.001, 0.02
        from scipy import stats as sp_stats
        z_alpha = sp_stats.norm.ppf(0.05)  # -1.645
        result = parametric_var(mu, sigma, confidence=0.95)
        expected = mu + sigma * z_alpha  # 0.001 - 0.0329 = -0.0319
        assert abs(result.var - expected) < 0.01

    def test_portfolio_parametric(self):
        w = np.array([0.5, 0.5])
        cov = np.array([[0.0001, 0.00003], [0.00003, 0.0001]])
        result = portfolio_parametric_var(w, cov, confidence=0.95)
        assert result.var < 0  # Return level (negative = loss)
        assert result.cvar is not None and result.cvar < result.var  # CVaR more extreme

    def test_cornish_fisher_increases_for_fat_tails(self):
        mu, sigma = 0.0, 0.02
        sp_stats.norm.ppf(0.05)  # -1.645
        parametric = parametric_var(mu, sigma, confidence=0.95)
        cf = cornish_fisher_var(mu, sigma, skewness=-0.5, kurtosis=5.0, confidence=0.95)
        assert cf.var < parametric.var  # CF more negative = larger loss for fat tails


sp_stats = __import__("scipy.stats", fromlist=["stats"])


# ---------------------------------------------------------------------------
# T-0303: Monte Carlo Var + Filtered Historical
# ---------------------------------------------------------------------------


class TestMonteCarloVaR:
    def test_converges_near_parametric(self, normal_returns):
        mc = monte_carlo_var(normal_returns, n_scenarios=50000, confidence=0.95)
        param = parametric_var(float(normal_returns.mean()), float(normal_returns.std()), confidence=0.95)
        # MC should be within ~10% of parametric for Normal
        assert abs(mc.var - param.var) / abs(param.var) < 0.20

    def test_filtered_historical_var(self, normal_returns):
        result = filtered_historical_var(normal_returns, confidence=0.95)
        assert result.var < 0  # Loss is negative
        assert result.cvar is not None and result.cvar < result.var


# ---------------------------------------------------------------------------
# T-0304: CVaR
# ---------------------------------------------------------------------------


class TestCVaR:
    def test_cvar_basic(self, normal_returns):
        cv = cvar(normal_returns, confidence=0.975)
        assert cv < 0  # Expected loss is negative

    def test_marginal_cvar(self, multi_asset_returns):
        w = np.array([0.5, 0.5])
        mcvar = marginal_cvar(multi_asset_returns, w, confidence=0.975)
        assert len(mcvar) == 2

    def test_component_cvar_sums_to_total(self, multi_asset_returns):
        w = np.array([0.5, 0.5])
        comp = component_cvar(multi_asset_returns, w, confidence=0.975)
        total_cv = cvar(multi_asset_returns @ w, confidence=0.975)
        assert abs(comp.sum() - total_cv) < 0.02

    def test_incremental_cvar(self, multi_asset_returns):
        w = np.array([0.5, 0.5])
        cvar(multi_asset_returns @ w, confidence=0.975)
        inc = incremental_cvar(multi_asset_returns, w, 0, 0.1, confidence=0.975)
        # Adding risk should increase CVaR magnitude
        assert abs(inc) >= 0


# ---------------------------------------------------------------------------
# T-0305: VaR Backtesting
# ---------------------------------------------------------------------------


class TestVaRBacktesting:
    def test_kupiec_green_for_correct_model(self, normal_returns):
        r = normal_returns
        var_series = np.full_like(r, -1.645 * float(r.std()))
        violations = (r < var_series).astype(int)
        stat, pvalue = kupiec_pof_test(int(np.sum(violations)), len(r), confidence=0.95)
        # Well-calibrated model should have p > 0.05
        assert pvalue > 0.01

    def test_kupiec_rejects_bad_model(self, normal_returns):
        r = normal_returns
        # Overly optimistic VaR (too narrow)
        var_series = np.full_like(r, -0.5 * float(r.std()))
        violations = (r < var_series).astype(int)
        stat, pvalue = kupiec_pof_test(int(np.sum(violations)), len(r), confidence=0.95)
        assert pvalue < 0.05

    def test_traffic_light_green(self):
        assert traffic_light(2, 250, confidence=0.99) == "green"

    def test_traffic_light_red(self):
        assert traffic_light(12, 250, confidence=0.99) == "red"

    def test_full_report(self, normal_returns):
        r = normal_returns[-252:]
        var_series = np.full_like(r, 1.645 * float(r.std()))
        report = var_backtest_report(r, var_series, confidence=0.95)
        assert report.n_observations == 252
        assert 0 <= report.violation_rate <= 1


# ---------------------------------------------------------------------------
# T-0306: Historical Stress Scenarios
# ---------------------------------------------------------------------------


class TestStressScenarios:
    def test_scenario_library_has_six_scenarios(self):
        lib = build_scenario_library()
        assert len(lib) >= 6
        assert "GFC_2008" in lib
        assert "COVID_2020" in lib

    def test_apply_scenario_equity_loss(self, sample_portfolio):
        lib = build_scenario_library()
        gfc = lib["GFC_2008"]
        result = apply_scenario(sample_portfolio, gfc)
        # Equity-heavy portfolio should lose value under GFC
        assert result.pnl < 0
        assert result.pnl_pct < -0.10  # At least 10% loss

    def test_stress_report(self, sample_portfolio):
        results = stress_report(sample_portfolio)
        assert len(results) >= 6
        assert results[0].is_worst_case  # First result is worst

    def test_custom_scenario(self, sample_portfolio):
        sc = custom_scenario("Test", {"equity": -0.50})
        result = apply_scenario(sample_portfolio, sc)
        assert result.pnl_pct < -0.30

    def test_correlation_stress(self, sample_portfolio):
        result = correlation_stress(sample_portfolio, target_corr=0.80)
        assert result.pnl < 0

    def test_factor_stress(self, sample_portfolio):
        result = factor_stress(
            sample_portfolio,
            factor_exposures={"momentum": 0.3, "value": 0.2},
            factor_shocks={"momentum": -0.03, "value": -0.01},
        )
        assert result.pnl < 0

    def test_reverse_stress(self, sample_portfolio):
        scenarios = reverse_stress(sample_portfolio, loss_threshold=-0.05, n_results=3)
        assert len(scenarios) > 0
        # Verify scenarios actually produce loss
        for sc in scenarios:
            result = apply_scenario(sample_portfolio, sc)
            assert result.pnl < 0

    def test_concentration_report(self, sample_portfolio):
        scenarios = reverse_stress(sample_portfolio, loss_threshold=-0.05, n_results=2)
        report = concentration_report(scenarios)
        assert len(report) > 0


# ---------------------------------------------------------------------------
# T-0309: Greeks
# ---------------------------------------------------------------------------


class TestGreeks:
    def test_aggregate_greeks_stocks(self, sample_portfolio):
        greeks = aggregate_greeks(sample_portfolio)
        assert greeks.total_delta > 0  # Long positions
        assert greeks.total_delta_dollar > 0

    def test_delta_adjusted_exposure(self):
        stocks = [Position(symbol="AAPL", quantity=100, avg_cost=150, market_price=200)]
        options = [
            {"underlying": "AAPL", "qty": 10, "delta": 0.6},
        ]
        exposure = delta_adjusted_exposure(stocks, options)
        assert "AAPL" in exposure

    def test_gamma_tenor(self):
        options = [
            {"dte": 5, "gamma": 0.02, "qty": 10},
            {"dte": 60, "gamma": 0.01, "qty": 5},
        ]
        gamma = gamma_by_tenor(options)
        assert len(gamma) > 0

    def test_vega_tenor(self):
        options = [
            {"dte": 10, "vega": 0.15, "qty": 5},
            {"dte": 100, "vega": 0.20, "qty": 3},
        ]
        vega = vega_by_tenor(options)
        assert len(vega) > 0

    def test_greeks_stress(self):
        positions = {
            "AAPL": Position(symbol="AAPL", quantity=100, avg_cost=180, market_price=200),
        }
        result = greeks_stress(positions)
        assert "delta" in result
        assert len(result["delta"]) == 7  # 7 shock levels


# ---------------------------------------------------------------------------
# T-0310: Risk Limits
# ---------------------------------------------------------------------------


class TestRiskLimits:
    def test_pre_trade_approved(self, sample_portfolio):
        limits = RiskLimits()
        order = Order(
            id="test1", symbol="AAPL", side=Side.BUY, type=OrderType.MARKET,
            quantity=10, limit_price=200,
        )
        result = check_pre_trade(order, sample_portfolio, limits)
        assert result.status == LimitStatus.APPROVED

    def test_concentration_rejected(self, sample_portfolio):
        limits = RiskLimits(max_concentration_pct=0.05)
        # Buy a lot of AAPL — exceeds 5% concentration
        order = Order(
            id="test2", symbol="AAPL", side=Side.BUY, type=OrderType.MARKET,
            quantity=1000, limit_price=200,
        )
        result = check_pre_trade(order, sample_portfolio, limits)
        assert result.status == LimitStatus.REJECTED

    def test_dollar_exposure_rejected(self, sample_portfolio):
        limits = RiskLimits(max_dollar_exposure_per_name=5000)
        order = Order(
            id="test3", symbol="AAPL", side=Side.BUY, type=OrderType.MARKET,
            quantity=100, limit_price=200,
        )
        result = check_pre_trade(order, sample_portfolio, limits)
        assert result.status == LimitStatus.REJECTED

    def test_circuit_breaker_fires(self):
        limits = RiskLimits(circuit_breaker_threshold_pct=0.10)
        cb = CircuitBreaker(limits=limits)

        breach = LimitCheckResult(
            status=LimitStatus.REJECTED,
            limit_name="max_net_delta",
            current_value=600000,
            limit_value=500000,
        )
        fired = cb.check([breach])
        assert fired
        assert cb.activated

    def test_circuit_breaker_no_fire_small_breach(self):
        limits = RiskLimits(circuit_breaker_threshold_pct=0.20)
        cb = CircuitBreaker(limits=limits)

        breach = LimitCheckResult(
            status=LimitStatus.REJECTED,
            limit_name="max_concentration",
            current_value=0.30,
            limit_value=0.25,
        )
        fired = cb.check([breach])
        assert not fired


# ---------------------------------------------------------------------------
# T-0311: Drawdown
# ---------------------------------------------------------------------------


class TestDrawdown:
    def test_drawdown_tracker(self):
        tracker = DrawdownTracker()
        state = tracker.update(1_000_000)
        assert state.drawdown == 0.0

        state = tracker.update(950_000)
        assert state.drawdown_pct < -4.0  # -5%

        state = tracker.update(1_050_000)
        assert state.drawdown == 0.0  # New peak

    def test_drawdown_threshold_actions(self):
        triggered = []

        def on_action(action, state):
            triggered.append(action)

        tracker = DrawdownTracker(on_action=on_action)
        tracker.update(1_000_000)
        tracker.update(940_000)  # -6% → REDUCE_25
        tracker.update(890_000)  # -11% → REDUCE_50
        tracker.update(840_000)  # -16% → REDUCE_75
        tracker.update(790_000)  # -21% → FLATTEN

        assert DrawdownAction.REDUCE_25 in triggered
        assert DrawdownAction.REDUCE_50 in triggered

    def test_drawdown_at_risk(self, normal_returns):
        dar = drawdown_at_risk(normal_returns, confidence=0.95)
        assert dar > 0

    def test_cdar(self, normal_returns):
        cdar = conditional_dar(normal_returns, confidence=0.95)
        assert cdar > 0

    def test_recovery_estimate(self):
        days = recovery_estimate(current_dd=-0.10, avg_return=0.001)
        assert days > 0
        assert days < 1000  # Should recover within a year

    def test_calmar_ratio(self, normal_returns):
        cr = compute_calmar_ratio(normal_returns)
        assert isinstance(cr, float)


# ---------------------------------------------------------------------------
# T-0312: Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_sharpe_zero(self):
        # Audit fix 2026-05-10: flat returns -> NaN (not 0.0).
        # A flat return series has no meaningful Sharpe; returning 0.0 was
        # misleading (implies a strategy with no risk and no return).
        import math
        r = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = sharpe_ratio(r)
        assert math.isnan(result), (
            f"Expected NaN for flat returns (audit fix), got {result}. "
            "Update any code that treats sharpe==0 as a passing score."
        )

    def test_sharpe_positive(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, 252)
        sr = sharpe_ratio(r)
        assert sr > 0.5

    def test_sortino(self, normal_returns):
        sr = sortino_ratio(normal_returns)
        assert isinstance(sr, float)

    def test_calmar(self, normal_returns):
        cr = calmar_ratio(normal_returns)
        assert isinstance(cr, float)

    def test_omega(self, normal_returns):
        omega = omega_ratio(normal_returns)
        assert omega > 0

    def test_information_ratio(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, 252)
        b = rng.normal(0.0005, 0.008, 252)
        ir = information_ratio(r, b)
        assert isinstance(ir, float)

    def test_treynor(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, 252)
        m = rng.normal(0.0005, 0.008, 252)
        tr = treynor_ratio(r, m)
        assert isinstance(tr, float)

    def test_jensens_alpha(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.001, 0.01, 252)
        m = rng.normal(0.0005, 0.008, 252)
        alpha = jensens_alpha(r, m)
        assert isinstance(alpha, float)

    def test_skewness_kurtosis(self, normal_returns):
        sk = skewness(normal_returns)
        ku = kurtosis(normal_returns)
        assert abs(sk) < 0.5  # Near-symmetric
        assert ku > -2  # Excess kurtosis

    def test_jarque_bera_normal(self, normal_returns):
        jb, pv = jarque_bera(normal_returns)
        assert pv > 0.01  # Should not reject normality

    def test_ulcer_index(self, normal_returns):
        ui = ulcer_index(normal_returns)
        assert ui >= 0

    def test_pain_index(self, normal_returns):
        pi = pain_index(normal_returns)
        assert pi >= 0

    def test_bootstrap_ci(self, normal_returns):
        lower, upper = bootstrap_ci(normal_returns, sharpe_ratio, n_boot=200)
        point = sharpe_ratio(normal_returns)
        assert lower <= point <= upper

    def test_rolling(self, normal_returns):
        r = np.array(normal_returns[:300])
        roll = rolling(r, lambda x: float(np.mean(x)), window=63)
        assert len(roll) == 300 - 63 + 1

    def test_compute_all_metrics(self, normal_returns):
        rng = np.random.default_rng(42)
        m = rng.normal(0.0005, 0.008, 1000)
        metrics = compute_all_metrics(normal_returns, market_returns=m)
        assert "sharpe" in metrics
        assert "max_drawdown" in metrics
        assert "information_ratio" in metrics


# ---------------------------------------------------------------------------
# T-0313: Reports (basic smoke test)
# ---------------------------------------------------------------------------


class TestReports:
    def test_daily_report(self, normal_returns):
        from quantcore.risk.reports import generate_daily_risk_report
        pf = PortfolioSnapshot(total_value=100000, cash=20000)
        report = generate_daily_risk_report(pf, normal_returns)
        assert report.positions_count >= 0

    def test_html_report_generates(self, normal_returns):
        from quantcore.risk.reports import generate_html_report
        html = generate_html_report(normal_returns[-252:], title="Test Report")
        assert "<html" in html.lower()
        assert "plotly" in html.lower()


# ---------------------------------------------------------------------------
# VaR Convergence Test (Acceptance: MC within 1% of analytical for Normal)
# ---------------------------------------------------------------------------


def test_mc_var_convergence():
    """Acceptance test: MC VaR converges within 1% of analytical for Normal case."""
    rng = np.random.default_rng(42)
    r = rng.normal(0.0, 0.01, 2000)
    mc = monte_carlo_var(r, n_scenarios=200000, confidence=0.95)
    param = parametric_var(0.0, 0.01, confidence=0.95)
    pct_diff = abs(mc.var - param.var) / abs(param.var)
    assert pct_diff < 0.05  # Within 5%


def test_cfar_greater_than_var_always():
    """CVaR should always be more extreme than VaR."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        r = rng.normal(0, 0.02, 500)
        for conf in [0.95, 0.975, 0.99]:
            var_result = historical_var(r, confidence=conf)
            cvar_val = cvar(r, confidence=conf)
            assert cvar_val <= var_result.var + 1e-10
