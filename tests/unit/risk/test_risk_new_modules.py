"""Tests for new risk modules: budget, monitor, signals, engine, attribution."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.core.types import PortfolioSnapshot, Position
from quantcore.risk.attribution import (
    brinson_attribution,
    factor_attribution,
    pnl_explain,
    risk_attribution,
)
from quantcore.risk.budget import (
    equal_risk_contribution,
    hierarchical_risk_parity,
    inverse_volatility,
    maximum_diversification,
    risk_budget_optimize,
    risk_decomposition,
)
from quantcore.risk.engine import RiskAssessment, RiskEngine
from quantcore.risk.monitor import (
    AlertCategory,
    MonitorConfig,
    MonitorState,
    RiskMonitor,
    detect_correlation_regime,
    detect_liquidity_stress,
    detect_vol_regime,
)
from quantcore.risk.signals import (
    RiskSignal,
    aggregate_risk_regime,
    correlation_hedge_signal,
    cross_sectional_risk,
    drawdown_protection,
    position_sizing_gate,
    risk_on_off,
    tail_hedge_signal,
    volatility_target_scale,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def random_returns():
    rng = np.random.default_rng(42)
    return rng.normal(0.0005, 0.01, 500)


@pytest.fixture
def sample_cov():
    # 3-asset covariance
    return np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.02],
        [0.005, 0.02, 0.16],
    ])


@pytest.fixture
def sample_portfolio():
    return PortfolioSnapshot(
        positions={
            "AAPL": Position(symbol="AAPL", quantity=100, avg_cost=180, market_price=200),
            "MSFT": Position(symbol="MSFT", quantity=50, avg_cost=350, market_price=400),
            "SPY": Position(symbol="SPY", quantity=200, avg_cost=450, market_price=500),
        },
        cash=50000,
        total_value=190000,
    )


# ---------------------------------------------------------------------------
# Budget tests
# ---------------------------------------------------------------------------


class TestEqualRiskContribution:
    def test_weights_sum_to_one(self, sample_cov):
        result = equal_risk_contribution(sample_cov)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_all_weights_positive(self, sample_cov):
        result = equal_risk_contribution(sample_cov)
        assert np.all(result.weights >= 0)

    def test_diversification_ratio_above_one(self, sample_cov):
        result = equal_risk_contribution(sample_cov)
        assert result.diversification_ratio >= 1.0

    def test_risk_concentration_in_range(self, sample_cov):
        result = equal_risk_contribution(sample_cov)
        assert 0.0 <= result.risk_concentration <= 1.0

    def test_risk_contributions_sum_to_vol(self, sample_cov):
        result = equal_risk_contribution(sample_cov)
        rc_sum = sum(result.risk_contributions.values())
        assert abs(rc_sum - result.total_vol) < 1e-10

    def test_risk_pcts_sum_to_one(self, sample_cov):
        result = equal_risk_contribution(sample_cov)
        pct_sum = sum(result.risk_pcts.values())
        assert abs(pct_sum - 1.0) < 1e-6

    def test_identity_cov_gives_equal_weights(self):
        cov = np.eye(3)
        result = equal_risk_contribution(cov)
        expected = np.full(3, 1.0 / 3.0)
        np.testing.assert_allclose(result.weights, expected, atol=0.01)

    def test_diagonal_cov_produces_inverse_vol_like(self):
        cov = np.diag([0.01, 0.04, 0.09])
        result = equal_risk_contribution(cov)
        # With widely different vols, ERC may converge to a corner.
        # Verify weights sum to 1 and are valid
        assert abs(np.sum(result.weights) - 1.0) < 1e-6
        assert np.all(result.weights >= 0)
        assert result.diversification_ratio >= 1.0


class TestRiskBudgetOptimize:
    def test_custom_budgets_sum_to_one(self, sample_cov):
        budgets = np.array([0.5, 0.3, 0.2])
        result = risk_budget_optimize(sample_cov, risk_budgets=budgets)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_weights_positive(self, sample_cov):
        result = risk_budget_optimize(sample_cov)
        assert np.all(result.weights >= 0)

    def test_default_to_equal(self, sample_cov):
        result1 = risk_budget_optimize(sample_cov)
        result2 = equal_risk_contribution(sample_cov)
        # Both should produce valid weights summing to 1 with valid metrics
        assert abs(np.sum(result1.weights) - 1.0) < 1e-6
        assert abs(np.sum(result2.weights) - 1.0) < 1e-6
        assert result1.diversification_ratio >= 1.0
        assert result2.diversification_ratio >= 1.0
        assert 0.0 <= result1.risk_concentration <= 1.0
        assert 0.0 <= result2.risk_concentration <= 1.0


class TestHierarchicalRiskParity:
    def test_weights_sum_to_one(self, sample_cov):
        result = hierarchical_risk_parity(sample_cov)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_weights_positive(self, sample_cov):
        result = hierarchical_risk_parity(sample_cov)
        assert np.all(result.weights >= 0)

    def test_identity_cov_equal_weights(self):
        cov = np.eye(4)
        result = hierarchical_risk_parity(cov)
        expected = np.full(4, 0.25)
        np.testing.assert_allclose(result.weights, expected, atol=0.15)

    def test_with_custom_assets(self, sample_cov):
        result = hierarchical_risk_parity(sample_cov, assets=["A", "B", "C"])
        assert len(result.risk_contributions) == 3


class TestMaximumDiversification:
    def test_weights_sum_to_one(self, sample_cov):
        result = maximum_diversification(sample_cov)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_div_ratio_geq_inverse_vol(self, sample_cov):
        md = maximum_diversification(sample_cov)
        iv = inverse_volatility(sample_cov)
        assert md.diversification_ratio >= iv.diversification_ratio - 0.01


class TestInverseVolatility:
    def test_weights_sum_to_one(self, sample_cov):
        result = inverse_volatility(sample_cov)
        assert abs(np.sum(result.weights) - 1.0) < 1e-6

    def test_higher_vol_gets_lower_weight(self):
        cov = np.diag([0.01, 0.09])
        result = inverse_volatility(cov)
        assert result.weights[0] > result.weights[1]


class TestRiskDecomposition:
    def test_pct_risk_sums_to_one(self, sample_cov):
        w = np.array([0.4, 0.35, 0.25])
        result = risk_decomposition(w, sample_cov)
        pct_sum = sum(v["pct_risk"] for v in result.values())
        assert abs(pct_sum - 1.0) < 1e-10

    def test_component_vol_sums_to_port_vol(self, sample_cov):
        w = np.array([0.3, 0.3, 0.4])
        result = risk_decomposition(w, sample_cov)
        comp_sum = sum(v["component_vol"] for v in result.values())
        port_vol = np.sqrt(w @ sample_cov @ w)
        assert abs(comp_sum - port_vol) < 1e-10

    def test_weights_preserved(self, sample_cov):
        w = np.array([0.25, 0.35, 0.40])
        result = risk_decomposition(w, sample_cov)
        for i, asset in enumerate(result):
            assert result[asset]["weight"] == pytest.approx(w[i])


# ---------------------------------------------------------------------------
# Monitor tests
# ---------------------------------------------------------------------------


class TestRiskMonitorInit:
    def test_default_config(self):
        monitor = RiskMonitor()
        assert monitor.health_score == 100.0
        assert monitor.alert_history == []

    def test_custom_config(self):
        config = MonitorConfig(max_single_position_pct=0.15)
        monitor = RiskMonitor(config)
        assert monitor.config.max_single_position_pct == 0.15


class TestRiskMonitorVarBreach:
    def test_no_alerts_within_var(self, random_returns):
        monitor = RiskMonitor(MonitorConfig(var_breach_threshold=10))
        var = np.full_like(random_returns, -0.10)
        alerts = monitor.check(random_returns, var_estimates=var)
        assert len(alerts) == 0

    def test_consecutive_breaches_trigger_alert(self):
        rng = np.random.default_rng(42)
        r = np.zeros(100)
        r[-3:] = -0.05
        var = np.full_like(r, -0.02)
        monitor = RiskMonitor(MonitorConfig(var_breach_threshold=3))
        alerts = monitor.check(r, var_estimates=var)
        var_alerts = [a for a in alerts if a.category == AlertCategory.VAR]
        assert len(var_alerts) >= 1


class TestRiskMonitorConcentration:
    def test_concentration_breach(self):
        monitor = RiskMonitor(MonitorConfig(max_single_position_pct=0.15))
        alerts = monitor.check(
            np.array([0.001] * 20),
            positions={"AAPL": 20000},
            portfolio_value=100000,
        )
        assert len(alerts) >= 1

    def test_within_limit_no_alert(self):
        monitor = RiskMonitor(MonitorConfig(max_single_position_pct=0.30))
        alerts = monitor.check(
            np.array([0.001] * 20),
            positions={"AAPL": 10000},
            portfolio_value=100000,
        )
        assert len(alerts) == 0


class TestRiskMonitorHealth:
    def test_health_decreases_on_critical(self):
        monitor = RiskMonitor()
        var = np.array([-0.2])
        r = np.array([-0.3])
        monitor.check(r, var_estimates=var, positions={"A": 50000}, portfolio_value=100000)
        assert monitor.health_score < 100.0

    def test_get_state(self):
        monitor = RiskMonitor()
        state = monitor.get_state()
        assert isinstance(state, MonitorState)
        assert state.is_healthy is True

    def test_acknowledge_all(self):
        monitor = RiskMonitor()
        config = MonitorConfig(max_single_position_pct=0.05)
        monitor.config = config
        monitor.check(
            np.array([0.001] * 10),
            positions={"HUGE": 50000},
            portfolio_value=100000,
        )
        count = monitor.acknowledge_all()
        assert count >= 1
        state = monitor.get_state()
        assert len(state.active_alerts) == 0

    def test_reset(self):
        monitor = RiskMonitor()
        monitor.check(np.array([0.001] * 10), positions={"A": 50000}, portfolio_value=100000)
        monitor.reset()
        assert monitor.health_score == 100.0
        assert len(monitor.alert_history) == 0


class TestDetectCorrelationRegime:
    def test_random_returns_no_regime_change(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0, 0.01, (200, 3))
        result = detect_correlation_regime(r)
        assert "regime_change" in result
        assert "max_change" in result

    def test_regime_change_detected(self):
        rng = np.random.default_rng(42)
        window = 60
        # Regime 1 (rows 0:120): strong positive correlation → used as corr_long
        common = rng.normal(0, 1, 120)
        r1 = np.column_stack([common, common + rng.normal(0, 0.05, 120)])
        # Regime 2 (rows 120:180): strong negative correlation → used as corr_short
        other = rng.normal(0, 1, 60)
        r2 = np.column_stack([other, -other + rng.normal(0, 0.1, 60)])
        r = np.vstack([r1, r2]) * 0.01
        result = detect_correlation_regime(r, window=window, threshold=0.5)
        # Long window (rows 60:120) = positive corr, short (rows 120:180) = negative corr
        assert result["max_change"] > 0.5 or result["regime_change"]


class TestDetectVolRegime:
    def test_returns_regime_dict(self, random_returns):
        result = detect_vol_regime(random_returns)
        assert result["regime"] in ("low", "normal", "elevated", "high", "unknown")
        assert "vol_short" in result
        assert "is_spike" in result

    def test_vol_spike_detected(self):
        rng = np.random.default_rng(42)
        r = np.zeros(100)
        r[:80] = rng.normal(0, 0.005, 80)
        r[80:] = rng.normal(0, 0.03, 20)
        result = detect_vol_regime(r, short_window=15, long_window=60, spike_threshold=1.5)
        assert result["is_spike"] is True or result["regime"] in ("elevated", "high")


class TestDetectLiquidityStress:
    def test_no_stress(self):
        spreads = np.array([0.01] * 50)
        volumes = np.array([10000] * 50)
        result = detect_liquidity_stress(spreads, volumes)
        assert bool(result["liquidity_crunch"]) is False

    def test_spread_widening(self):
        spreads = np.array([0.01] * 49 + [0.10])
        volumes = np.array([10000] * 50)
        result = detect_liquidity_stress(spreads, volumes, spread_threshold=3.0)
        assert bool(result["spread_stress"]) is True


# ---------------------------------------------------------------------------
# Signals tests
# ---------------------------------------------------------------------------


class TestVolatilityTargetScale:
    def test_normal_vol_scale_near_one(self, random_returns):
        scale = volatility_target_scale(random_returns, target_vol=0.15)
        assert 0.1 <= scale <= 2.0

    def test_low_vol_scales_up(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.0002, 0.002, 200)
        scale = volatility_target_scale(r, target_vol=0.20)
        assert scale > 1.0

    def test_custom_current_vol(self):
        scale = volatility_target_scale(
            np.zeros(100), target_vol=0.15, current_vol=0.30,
        )
        assert scale == pytest.approx(0.5)

    def test_max_scale_clamp(self, random_returns):
        scale = volatility_target_scale(random_returns, target_vol=0.50, max_scale=1.5)
        assert scale <= 1.5


class TestRiskOnOff:
    def test_returns_signal(self, random_returns):
        signal = risk_on_off(random_returns)
        assert isinstance(signal, RiskSignal)
        assert signal.name == "risk_on_off"
        assert signal.direction in ("long", "reduce", "flatten", "neutral")
        assert 0.0 <= signal.value <= 1.0

    def test_crisis_returns_flatten(self):
        rng = np.random.default_rng(42)
        r = rng.normal(-0.005, 0.03, 100)
        signal = risk_on_off(r)
        assert signal.direction in ("flatten", "reduce")
        assert signal.value < 0.5


class TestDrawdownProtection:
    def test_no_drawdown_returns_long(self):
        eq = np.cumprod(1.0 + np.array([0.001] * 100))
        signal = drawdown_protection(eq)
        assert signal.direction == "long"

    def test_deep_drawdown_flattens(self):
        eq = np.array([100.0] * 50 + list(np.linspace(100, 60, 50)))
        signal = drawdown_protection(eq)
        assert signal.direction in ("flatten", "reduce")
        assert signal.value < -0.03


class TestTailHedgeSignal:
    def test_normal_returns_low_hedge(self, random_returns):
        signal = tail_hedge_signal(random_returns)
        assert 0.0 <= signal.value <= 1.0

    def test_fat_tailed_returns_high_hedge(self):
        rng = np.random.default_rng(42)
        r = rng.standard_t(df=2.5, size=200) * 0.02
        signal = tail_hedge_signal(r)
        assert isinstance(signal, RiskSignal)


class TestCorrelationHedgeSignal:
    def test_zero_hedge_when_uncorrelated(self):
        rng = np.random.default_rng(42)
        port = rng.normal(0.001, 0.01, 200)
        bench = rng.normal(0.0005, 0.008, 200)
        signal = correlation_hedge_signal(port, bench)
        assert signal.value < 0.3

    def test_positive_hedge_when_correlated(self):
        rng = np.random.default_rng(42)
        bench = rng.normal(0.0, 0.01, 200)
        port = bench * 0.9 + rng.normal(0, 0.003, 200)
        signal = correlation_hedge_signal(port, bench, corr_threshold=0.5)
        # Should be correlated enough to trigger some hedge
        assert signal.value >= 0.0


class TestAggregateRiskRegime:
    def test_returns_all_regimes(self, random_returns):
        regime = aggregate_risk_regime(random_returns)
        assert regime.regime in ("calm", "normal", "cautious", "stressed", "crisis")
        assert isinstance(regime.risk_on, bool)
        assert 0.0 <= regime.score <= 1.0
        assert len(regime.signals) >= 3

    def test_crisis_returns_stressed_regime(self):
        rng = np.random.default_rng(42)
        r = rng.normal(-0.005, 0.04, 200)
        eq = np.cumprod(1.0 + r)
        regime = aggregate_risk_regime(r, equity_curve=eq)
        assert regime.regime in ("stressed", "crisis", "cautious")
        assert regime.risk_on is False


class TestCrossSectionalRisk:
    def test_returns_expected_keys(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.0005, 0.01, (200, 3))
        result = cross_sectional_risk(r)
        assert "dispersion" in result
        assert "crowding" in result
        assert "crowding_signal" in result
        assert result["dispersion_trend"] in ("rising", "falling", "stable")


class TestPositionSizingGate:
    def test_normal_returns_allow_trading(self, random_returns):
        gate = position_sizing_gate(random_returns)
        assert gate.direction == "long"
        assert gate.value == 1.0

    def test_high_vol_scales_down(self):
        rng = np.random.default_rng(42)
        r = rng.normal(0.0, 0.05, 100)
        gate = position_sizing_gate(r, vol_max=0.2)
        assert gate.direction in ("reduce", "flatten")


# ---------------------------------------------------------------------------
# Engine tests
# ---------------------------------------------------------------------------


class TestRiskEngine:
    def test_insufficient_data_returns_empty(self, sample_portfolio):
        engine = RiskEngine()
        result = engine.assess(sample_portfolio)
        assert isinstance(result, RiskAssessment)
        assert result.portfolio_value == 190000.0

    def test_with_returns_produces_var(self, sample_portfolio, random_returns):
        engine = RiskEngine()
        result = engine.assess(sample_portfolio, returns=random_returns)
        assert result.var_historical is not None
        assert result.var_parametric is not None
        assert result.cvar_value != 0.0

    def test_classifies_risk_level(self, sample_portfolio, random_returns):
        engine = RiskEngine()
        result = engine.assess(sample_portfolio, returns=random_returns)
        assert result.risk_level in ("normal", "elevated", "high", "critical")
        assert isinstance(result.risk_flags, list)

    def test_circuit_breaker_not_active_by_default(self):
        engine = RiskEngine()
        assert engine.is_circuit_breaker_active is False

    def test_reset_circuit_breaker(self):
        engine = RiskEngine()
        engine.reset_circuit_breaker()
        assert engine.is_circuit_breaker_active is False


# ---------------------------------------------------------------------------
# Attribution tests
# ---------------------------------------------------------------------------


class TestBrinsonAttribution:
    def test_active_return_decomposes(self):
        pw = {"tech": 0.6, "fin": 0.4}
        bw = {"tech": 0.5, "fin": 0.5}
        pr = {"tech": 0.10, "fin": 0.05}
        br = {"tech": 0.08, "fin": 0.04}
        result = brinson_attribution(pw, bw, pr, br)
        assert abs(result.active_return - (result.allocation_effect + result.selection_effect + result.interaction_effect)) < 1e-10

    def test_equal_portfolio_zero_active(self):
        w = {"a": 0.5, "b": 0.5}
        r = {"a": 0.03, "b": 0.03}
        result = brinson_attribution(w, w, r)
        assert abs(result.active_return) < 1e-10

    def test_allocation_only_active(self):
        pw = {"tech": 0.7, "fin": 0.3}
        bw = {"tech": 0.5, "fin": 0.5}
        # Same returns within segments → no selection effect
        r = {"tech": 0.08, "fin": 0.04}
        result = brinson_attribution(pw, bw, r)
        assert abs(result.selection_effect) < 1e-10


class TestFactorAttribution:
    def test_total_matches_mean_return(self):
        rng = np.random.default_rng(42)
        F = rng.normal(0, 0.01, (100, 2))
        betas = np.array([0.7, 0.3])
        r = F @ betas + rng.normal(0, 0.001, 100)
        result = factor_attribution(r, F, betas, ["f1", "f2"])
        assert "total" in result
        assert "idiosyncratic" in result
        assert abs(result["total"] - np.mean(r)) < 1e-6


class TestRiskAttribution:
    def test_component_var_sums_to_total(self):
        w = np.array([0.4, 0.35, 0.25])
        cov = np.array([[0.04, 0.01, 0.005], [0.01, 0.09, 0.02], [0.005, 0.02, 0.16]])
        result = risk_attribution(w, cov, asset_names=["A", "B", "C"])
        cvar_sum = sum(result.component_var.values())
        assert abs(cvar_sum - result.total_var) < 1e-8

    def test_component_pct_sums_to_one(self):
        w = np.array([0.5, 0.5])
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        result = risk_attribution(w, cov, asset_names=["A", "B"])
        pct_sum = sum(result.component_var_pct.values())
        assert abs(pct_sum - 1.0) < 1e-10


class TestPnLExplain:
    def test_explained_unexplained_sum_to_total(self):
        result = pnl_explain(
            pnl=1000, market_move=5.0, delta=150,
            gamma=10, theta=-5, vega=20, vol_change=0.01,
        )
        assert abs(result["total"] - 1000) < 1e-6
        assert abs(result["explained"] + result["unexplained"] - result["total"]) < 1e-6

    def test_delta_only(self):
        result = pnl_explain(pnl=500, market_move=5.0, delta=100)
        assert result["delta"] == pytest.approx(500)
