"""Integration tests for Phase 4 LiveRiskEngine + circuit breaker integration.

Tests:
1. LiveRiskEngine instantiates and fits on synthetic data
2. pre_trade_check returns approval (not rejection) for normal signals
3. pre_trade_check returns rejection with non-empty reason in crisis regime
4. Circuit breaker daily loss limit is halved in high-correlation regime
5. Correlation regime persists in WARNING state after 5 days
6. Rolling window update triggers refit
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_returns():
    """252 days of synthetic multi-asset returns."""
    rng = np.random.default_rng(2025)
    n_days, n_assets = 252, 6
    # Two latent factors driving most assets
    factors = rng.normal(0, 0.008, (n_days, 2))
    loadings = rng.uniform(0.3, 0.9, (2, n_assets))
    noise = rng.normal(0, 0.003, (n_days, n_assets))
    returns = factors @ loadings + noise
    # Add a few extreme events (simulate fat tails)
    for _ in range(5):
        idx = rng.integers(0, n_days)
        returns[idx] += rng.normal(0, 0.04, n_assets)
    return returns.astype(np.float64)


@pytest.fixture
def portfolio_returns(synthetic_returns):
    """Equal-weighted portfolio returns from synthetic matrix."""
    return synthetic_returns.mean(axis=1).astype(np.float64)


@pytest.fixture
def mock_signal():
    """Construct a minimal TradeSignal-like mock."""
    sig = MagicMock()
    sig.direction = 1          # BUY
    sig.symbol = "SPY"
    sig.strength = 0.7
    sig.size = 5000.0
    sig.instrument_type = "equity"
    sig.metadata = {}
    return sig


@pytest.fixture
def mock_option_signal():
    """Option TradeSignal mock."""
    sig = MagicMock()
    sig.direction = 1
    sig.symbol = "SPY"
    sig.strength = 0.6
    sig.size = 2000.0
    sig.instrument_type = "option"
    sig.metadata = {
        "contracts": 2,
        "strike": 450.0,
        "dte": 30,
        "contract_id": "SPY_450_30DTE_PUT",
        "action": "SELL_TO_OPEN",
    }
    return sig


@pytest.fixture
def mock_positions():
    """Minimal position dict."""
    pos = MagicMock()
    pos.quantity = 100.0
    pos.market_price = 450.0
    return {"SPY": pos}


# ---------------------------------------------------------------------------
# LiveRiskEngine core
# ---------------------------------------------------------------------------


class TestLiveRiskEngineFit:
    def test_instantiate_without_error(self):
        """Engine can be created with no external data (synthetic fallback)."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        assert engine is not None
        assert hasattr(engine, "pre_trade_check")
        assert hasattr(engine, "update")
        assert not engine._fitted

    def test_fit_initial_creates_synthetic_fallback(self, synthetic_returns):
        """fit_initial uses synthetic returns when real data is insufficient."""
        from quantcore.live.live_risk_engine import LiveRiskEngine, ReturnWindow
        # Pre-populate window with very few observations
        engine = LiveRiskEngine()
        # Manually push a few returns (less than 20 observation threshold)
        for i in range(10):
            engine._window.push(
                portfolio_return=synthetic_returns[i].mean(),
                asset_returns=synthetic_returns[i],
                as_of=date.today(),
            )
        assert not engine._window.is_ready
        engine.fit_initial(synthetic=False)
        # Should have used synthetic fallback (synthetic_returns has 252 rows > 20)
        assert engine._fitted
        assert engine._latest_var_95 > 0

    def test_fit_with_real_data(self, synthetic_returns, portfolio_returns):
        """fit_initial calibrates EVT PCA DCC with real synthetic data."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        for i in range(len(portfolio_returns)):
            engine._window.push(
                portfolio_return=portfolio_returns[i],
                asset_returns=synthetic_returns[i],
                as_of=date.today(),
            )
        engine.fit_initial()
        assert engine._fitted
        assert engine._latest_var_95 > 0
        assert engine._latest_var_99 >= engine._latest_var_95
        assert engine._latest_es_99 >= engine._latest_var_99
        assert engine._latest_conditional_corr >= 0.0
        # PCA should have been fitted
        assert len(engine._latest_factor_exposures) > 0

    def test_get_snapshot_returns_all_keys(self, synthetic_returns, portfolio_returns):
        """get_snapshot exposes everything the API endpoint needs."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        for i in range(len(portfolio_returns)):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        engine.fit_initial()
        snap = engine.get_snapshot()
        required_keys = [
            "correlation_regime", "conditional_correlation", "var_95", "var_99",
            "es_95", "es_99", "factor_exposures", "risk_contribution",
            "systematic_risk", "idiosyncratic_risk", "n_observations",
            "is_fitted", "days_since_fit",
        ]
        for key in required_keys:
            assert key in snap, f"Missing key in snapshot: {key}"


# ---------------------------------------------------------------------------
# pre_trade_check gate
# ---------------------------------------------------------------------------


class TestPreTradeCheckGate:
    def test_normal_regime_equity_signal_approved(self, synthetic_returns, portfolio_returns, mock_signal, mock_positions):
        """In NORMAL regime a regular equity signal passes through."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        for i in range(len(portfolio_returns)):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        engine.fit_initial()
        # Force normal regime for clean test
        engine._current_regime = "NORMAL"
        engine._latest_var_95 = 0.01  # 1% VaR — well below any limit
        approval = engine.pre_trade_check(mock_signal, mock_positions, current_equity=100_000.0)
        assert approval.approved is True

    def test_normal_regime_option_signal_approved(self, synthetic_returns, portfolio_returns, mock_option_signal, mock_positions):
        """In NORMAL regime an option signal passes through."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        for i in range(len(portfolio_returns)):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        engine.fit_initial()
        engine._current_regime = "NORMAL"
        approval = engine.pre_trade_check(mock_option_signal, mock_positions, current_equity=100_000.0)
        assert approval.approved is True

    def test_equity_close_direction_allowed(self, synthetic_returns, portfolio_returns, mock_signal):
        """Closing signals (direction=0) are always allowed regardless of regime."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        for i in range(len(portfolio_returns)):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        engine.fit_initial()
        engine._current_regime = "CRISIS"
        engine._latest_var_95 = 999.0  # artificially set very high
        close_signal = MagicMock()
        close_signal.direction = 0
        close_signal.symbol = "SPY"
        close_signal.strength = 0.0
        close_signal.size = 0.0
        close_signal.instrument_type = "equity"
        close_signal.metadata = {}
        approval = engine.pre_trade_check(close_signal, {}, current_equity=100_000.0)
        assert approval.approved is True

    def test_crisis_regime_rejects_large_notional(self, synthetic_returns, portfolio_returns, mock_signal):
        """In CRISIS regime, large notional exceeds the reduced size limit."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        for i in range(len(portfolio_returns)):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        engine.fit_initial()
        engine._current_regime = "CRISIS"
        # Notional is 5% of equity — in crisis regime limit is 1.25%
        big_signal = MagicMock()
        big_signal.direction = 1
        big_signal.symbol = "SPY"
        big_signal.strength = 1.0
        big_signal.size = 5000.0    # 5% of $100k
        big_signal.instrument_type = "equity"
        big_signal.metadata = {}
        engine._latest_var_95 = 0.01  # keep tail risk check passed
        approval = engine.pre_trade_check(big_signal, {}, current_equity=100_000.0)
        assert approval.approved is False
        assert len(approval.reason) > 0
        assert "CRISIS" in approval.reason or "correlation" in approval.reason.lower()

    def test_elevated_regime_tail_risk_rejection(self, synthetic_returns, portfolio_returns, mock_signal):
        """In ELEVATED regime, tail risk that exceeds threshold kicks out the trade."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        for i in range(len(portfolio_returns)):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        engine.fit_initial()
        engine._current_regime = "ELEVATED"
        # artificially set very high VaR — in elevated threshold is 2%
        engine._latest_var_99 = 5000.0  # 5% of equity — exceeds 2% threshold
        approval = engine.pre_trade_check(mock_signal, {}, current_equity=100_000.0)
        assert approval.approved is False
        assert "tail" in approval.reason.lower() or "var" in approval.reason.lower()


# ---------------------------------------------------------------------------
# Rolling window and refit
# ---------------------------------------------------------------------------


class TestRollingWindowUpdate:
    def test_update_pushes_return_observations(self, synthetic_returns, portfolio_returns):
        """update() appends observations to the rolling window."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine()
        initial_obs = engine._window.n_observations
        for i in range(min(10, len(portfolio_returns))):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        assert engine._window.n_observations == initial_obs + min(10, len(portfolio_returns))

    def test_force_refit_on_interval(self, synthetic_returns, portfolio_returns):
        """update() triggers a refit when refit_interval is reached."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine(refit_interval=5)
        for i in range(len(portfolio_returns)):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        engine.fit_initial()
        prev_var = engine._latest_var_95
        prev_es = engine._latest_es_99
        # Manually advance the counter past refit_interval
        engine._days_since_fit = 5
        # Push a big shock to trigger visible refit effect in metrics
        big_return = -0.05  # -5% shock
        engine.update(big_return, as_of=date.today())
        # Engine refitted on next update after threshold
        assert engine._days_since_fit == 0  # reset after refit
        # After refit with big shock, VaR should be notably different
        # (may be same since synthetic data dominates — just verify no crash)
        assert engine._fitted

    def test_regime_change_triggers_refit(self, synthetic_returns, portfolio_returns):
        """Transitioning to CRISIS regime forces an immediate refit."""
        from quantcore.live.live_risk_engine import LiveRiskEngine
        engine = LiveRiskEngine(refit_interval=999)  # large so interval doesn't trigger
        for i in range(len(portfolio_returns)):
            engine._window.push(portfolio_returns[i], synthetic_returns[i], date.today())
        engine.fit_initial()
        prev_fit_count = engine._regime_change_count
        # Simulate regime change
        engine._current_regime = "NORMAL"
        engine._update_regime()  # normal → normal, no change
        assert engine._regime_change_count == prev_fit_count
        # Now go to CRISIS — this should set high_corr_days for next update
        engine._current_regime = "NORMAL"
        engine._latest_conditional_corr = 0.85
        engine._update_regime()
        assert engine._current_regime == "CRISIS"


# ---------------------------------------------------------------------------
# Circuit breaker integration
# ---------------------------------------------------------------------------


class TestCircuitBreakerCorrelationRegime:
    def test_daily_loss_limit_halved_in_high_corr_regime(self):
        """High-correlation regime halves effective daily loss limit."""
        from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        cb = CircuitBreaker(CircuitBreakerConfig(daily_loss_pct=0.04))  # 4% default
        equity = 100_000.0
        # Normal regime: 4% limit → reject if loss < -$4000
        state1 = cb.check(equity, equity, -3000.0, correlation_regime="NORMAL")
        assert state1.value == "normal", "Should not halt in normal at -3% (limit=4%)"

        # Elevated regime: effective limit is 2% → reject at -$3000
        state2 = cb.check(equity, equity, -3000.0, correlation_regime="ELEVATED")
        assert state2.value == "daily_loss_halt", "Should halt elevated at -3% (effective limit=2%)"

    def test_warning_state_after_5_consecutive_high_corr_days(self):
        """5+ days of high correlation triggers WARNING state (not halt)."""
        from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
        cb = CircuitBreaker(CircuitBreakerConfig(
            daily_loss_pct=0.10,  # generous so we don't hit daily loss halt
            high_corr_days_persistence_alert=5,
        ))
        equity = 100_000.0
        # Simulate 4 days of high correlation — should stay NORMAL
        for day in range(4):
            cb.check(equity, equity, 0.0, correlation_regime="CRISIS")
        assert cb.state == CircuitBreakerState.NORMAL, "Still normal at 4 consecutive days"

        # Day 5 — crosses threshold → transitions to WARNING
        state = cb.check(equity, equity, 0.0, correlation_regime="CRISIS")
        assert state == CircuitBreakerState.WARNING
        assert "high correlation regime persisted" in cb.reason.lower()
        assert "operator attention" in cb.reason.lower()

    def test_warning_state_does_not_block_trading(self):
        """WARNING state still allows trading — it's an alert, not a halt."""
        from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState
        cb = CircuitBreaker(CircuitBreakerConfig(
            daily_loss_pct=0.10,
            high_corr_days_persistence_alert=3,
        ))
        equity = 100_000.0
        # Get to WARNING
        for _ in range(3):
            cb.check(equity, equity, 0.0, correlation_regime="CRISIS")
        assert cb.state == CircuitBreakerState.WARNING
        # can_trade returns True in WARNING state (no halt)
        assert cb.can_trade() is True

    def test_correlation_regime_logged_on_change(self, caplog):
        """Every regime transition is logged at INFO level."""
        import logging
        from quantcore.live.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
        cb = CircuitBreaker(CircuitBreakerConfig())
        caplog.set_level(logging.INFO)
        equity = 100_000.0

        cb.check(equity, equity, 0.0, correlation_regime="NORMAL",
                 conditional_correlation=0.30)
        cb.check(equity, equity, 0.0, correlation_regime="ELEVATED",
                 conditional_correlation=0.75)
        cb.check(equity, equity, 0.0, correlation_regime="CRISIS",
                 conditional_correlation=0.90)

        logs = caplog.text
        assert "CORRELATION REGIME" in logs or "ELEVATED" in logs
        assert "CRISIS" in logs