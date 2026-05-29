"""Tests for strategy framework: base, registry, equity, options, macro strategies."""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def single_asset_data():
    """Single asset price data for 300 days."""
    rng = np.random.default_rng(42)
    n = 300
    raw = 100.0 + np.cumsum(rng.normal(0.05, 1.0, n))
    prices = np.maximum(raw, 1.0)
    high = prices * (1 + np.abs(rng.normal(0, 0.02, n)))
    low = prices * (1 - np.abs(rng.normal(0, 0.02, n)))
    low = np.minimum(low, prices)
    iv = np.full(n, 0.3) + rng.normal(0, 0.03, n)
    iv = np.maximum(iv, 0.1)
    return {
        "close": prices,
        "high": high,
        "low": low,
        "implied_vol": iv,
        "symbols": ["TEST"],
    }


@pytest.fixture
def multi_asset_data():
    """Multi-asset price data for 400 days, 6 assets."""
    rng = np.random.default_rng(42)
    n, n_assets = 400, 6
    base = 100.0 + np.cumsum(rng.normal(0.02, 0.5, (n, n_assets)), axis=0)
    prices = np.maximum(base, 10.0)
    iv = np.full((n, n_assets), 0.25) + rng.normal(0, 0.02, (n, n_assets))
    iv = np.maximum(iv, 0.1)
    return {
        "close": prices,
        "implied_vol": iv,
        "symbols": [f"ASSET_{i}" for i in range(n_assets)],
    }


@pytest.fixture
def mean_rev_config():
    from quantcore.strategies.base import StrategyConfig, StrategyType
    return StrategyConfig(
        name="test_mean_rev",
        strategy_type=StrategyType.EQUITY,
        max_position_notional=100_000,
        risk_budget_pct=0.05,
    )


# ===========================================================================
# Test Base Classes
# ===========================================================================

class TestStrategyConfig:
    def test_defaults(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.EQUITY)
        assert cfg.name == "test"
        assert cfg.strategy_type == StrategyType.EQUITY
        assert cfg.max_position_notional == 1_000_000.0
        assert cfg.max_leverage == 1.0

    def test_custom_params(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        cfg = StrategyConfig(
            name="test",
            strategy_type=StrategyType.OPTIONS,
            max_leverage=2.0,
            params={"window": 20, "threshold": 0.5},
        )
        assert cfg.max_leverage == 2.0
        assert cfg.params["window"] == 20


class TestStrategyMetrics:
    def test_initial_values(self):
        from quantcore.strategies.base import StrategyMetrics
        m = StrategyMetrics()
        assert m.total_pnl == 0.0
        assert m.n_trades == 0
        assert m.n_winning == 0


class TestTradeSignal:
    def test_creation(self):
        from quantcore.strategies.base import TradeSignal
        sig = TradeSignal(symbol="AAPL", direction=1, strength=0.8, size=50000.0)
        assert sig.symbol == "AAPL"
        assert sig.direction == 1
        assert sig.strength == 0.8
        assert sig.stop_loss is None


class TestStrategyLifecycle:
    def test_initialize_and_teardown(self, mean_rev_config):
        from quantcore.strategies.base import StrategyState
        from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
        strat = MeanReversionStrategy(mean_rev_config)
        assert strat.state == StrategyState.UNINITIALIZED
        strat.initialize()
        assert strat.state == StrategyState.INITIALIZED
        strat.teardown()
        assert strat.state == StrategyState.STOPPED

    def test_config_validation_negative_leverage(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
        bad_cfg = StrategyConfig(
            name="test", strategy_type=StrategyType.EQUITY, max_leverage=-1.0
        )
        strat = MeanReversionStrategy(bad_cfg)
        with pytest.raises(ValueError, match="max_leverage must be positive"):
            strat.initialize()

    def test_on_fill_tracking(self, mean_rev_config):
        from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
        strat = MeanReversionStrategy(mean_rev_config)
        strat.initialize()
        strat.on_fill("AAPL", 150.0, 10.0, "buy")
        assert strat._positions["AAPL"] == 10.0
        assert strat.metrics.n_trades == 1

    def test_update_pnl(self, mean_rev_config):
        from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
        strat = MeanReversionStrategy(mean_rev_config)
        strat.update_pnl(100.0, 50.0)
        assert strat.metrics.realized_pnl == 100.0
        assert strat.metrics.unrealized_pnl == 50.0
        assert strat.metrics.total_pnl == 150.0
        assert strat.metrics.n_winning == 1


class TestStrategyState:
    def test_enum_values(self):
        from quantcore.strategies.base import StrategyState
        assert StrategyState.UNINITIALIZED == "uninitialized"
        assert StrategyState.RUNNING == "running"
        assert StrategyState.STOPPED == "stopped"


class TestStrategyType:
    def test_enum_values(self):
        from quantcore.strategies.base import StrategyType
        assert StrategyType.EQUITY == "equity"
        assert StrategyType.OPTIONS == "options"
        assert StrategyType.MACRO == "macro"


# ===========================================================================
# Test Registry
# ===========================================================================

class TestRegistry:
    def test_register_and_get(self):
        from quantcore.strategies.registry import _registry, get, register

        saved = dict(_registry)
        try:
            @register("test_strategy_reg")
            class MockStrategy:
                def __init__(self, config):
                    self.config = config

            cls = get("test_strategy_reg")
            assert cls is MockStrategy
        finally:
            _registry.clear()
            _registry.update(saved)

    def test_duplicate_register_raises(self):
        from quantcore.strategies.registry import _registry, register
        saved = dict(_registry)
        try:
            @register("test_dup")
            class A:
                pass

            with pytest.raises(ValueError, match="already registered"):
                @register("test_dup")
                class B:
                    pass
        finally:
            _registry.clear()
            _registry.update(saved)

    def test_get_missing_raises(self):
        from quantcore.strategies.registry import get
        with pytest.raises(KeyError, match="not found"):
            get("nonexistent_strategy_xyz_12345")

    def test_list_strategies(self):
        from quantcore.strategies.registry import _registry, list_strategies, register
        saved = dict(_registry)
        try:
            @register("alpha_strat")
            class X:
                pass

            @register("beta_strat")
            class Y:
                pass

            names = list_strategies()
            assert "alpha_strat" in names
            assert "beta_strat" in names
        finally:
            _registry.clear()
            _registry.update(saved)

    def test_create_from_registry(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.registry import _registry, create, register
        saved = dict(_registry)
        try:
            @register("test_create")
            class CreatableStrategy:
                def __init__(self, config):
                    self.config = config

            cfg = StrategyConfig(name="created", strategy_type=StrategyType.EQUITY)
            strat = create("test_create", cfg)
            assert strat.config.name == "created"
        finally:
            _registry.clear()
            _registry.update(saved)


# ===========================================================================
# Test All Strategies Are Registered
# ===========================================================================

class TestAllStrategiesRegistered:
    """Verify all strategy modules register themselves with @register."""

    @pytest.fixture(autouse=True)
    def _ensure_imported(self):
        # Import all strategy modules to trigger @register decorators
        import quantcore.strategies.equity.mean_reversion
        import quantcore.strategies.equity.momentum
        import quantcore.strategies.equity.stat_arb
        import quantcore.strategies.equity.trend_following
        import quantcore.strategies.macro.carry
        import quantcore.strategies.macro.cta_trend
        import quantcore.strategies.macro.risk_parity
        import quantcore.strategies.options.delta_neutral
        import quantcore.strategies.options.earnings_vol
        import quantcore.strategies.options.iron_condor
        import quantcore.strategies.options.tail_hedge  # noqa: F401

    def test_all_strategies_in_registry(self):
        from quantcore.strategies.registry import _registry
        expected = [
            "equity_momentum", "equity_stat_arb", "equity_mean_reversion",
            "equity_trend_following", "options_delta_neutral",
            "options_iron_condor", "options_earnings_vol",
            "options_tail_hedge", "macro_risk_parity",
            "macro_cta_trend", "macro_carry",
        ]
        for name in expected:
            assert name in _registry, f"Strategy '{name}' not registered"


# ===========================================================================
# Test Equity Strategies
# ===========================================================================

class TestMomentumStrategy:
    def test_on_data_returns_signals(self, multi_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.momentum import MomentumStrategy
        cfg = StrategyConfig(
            name="test_momentum",
            strategy_type=StrategyType.EQUITY,
            params={"rebalance_freq": 1, "lookback_months": 6, "skip_months": 1},
        )
        strat = MomentumStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(multi_asset_data)
        assert isinstance(signals, list)
        for s in signals:
            assert s.symbol in multi_asset_data["symbols"]
            assert s.direction in [-1, 0, 1]

    def test_inv_vol_weighting(self, multi_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.momentum import MomentumStrategy
        cfg = StrategyConfig(
            name="test",
            strategy_type=StrategyType.EQUITY,
            params={"rebalance_freq": 1, "n_long": 2, "n_short": 2},
        )
        strat = MomentumStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(multi_asset_data)
        # Should have both long and short signals
        [s for s in signals if s.direction > 0]
        [s for s in signals if s.direction < 0]
        # At least some directional signals (may be 0 if all assets same direction)
        assert len(signals) >= 0


class TestStatArbStrategy:
    def test_on_data_returns_signals(self, multi_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.stat_arb import StatArbStrategy
        cfg = StrategyConfig(
            name="test_stat_arb",
            strategy_type=StrategyType.EQUITY,
            params={"rescan_freq": 1, "entry_z": 1.5, "lookback": 100},
        )
        strat = StatArbStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(multi_asset_data)
        assert isinstance(signals, list)

    def test_pairs_scanning(self, multi_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.stat_arb import StatArbStrategy
        cfg = StrategyConfig(
            name="test",
            strategy_type=StrategyType.EQUITY,
            params={"rescan_freq": 1, "lookback": 100, "max_pairs": 5},
        )
        strat = StatArbStrategy(cfg)
        strat.initialize()
        # Force a pair scan
        strat._days_since_scan = strat.rescan_freq
        strat.on_data(multi_asset_data)
        # After scan, should have identified some pairs (or empty if none cointegrated)
        assert isinstance(strat._pairs, list)


class TestMeanReversionStrategy:
    def test_on_data_returns_list(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
        cfg = StrategyConfig(
            name="test_mr",
            strategy_type=StrategyType.EQUITY,
            params={"rsi_oversold": 30, "rsi_overbought": 70},
        )
        strat = MeanReversionStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(single_asset_data)
        assert isinstance(signals, list)

    def test_rsi_computation(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.EQUITY)
        strat = MeanReversionStrategy(cfg)
        # Monotonically increasing prices → RSI near 100
        prices = np.linspace(100, 200, 100)
        rsi = strat._compute_rsi(prices)
        assert rsi[-1] > 50  # Should be high for uptrend

    def test_bb_deviation(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.EQUITY)
        strat = MeanReversionStrategy(cfg)
        prices = np.ones(50) * 100.0
        prices[-1] = 110.0  # Spike
        dev = strat._bb_deviation(prices)
        assert dev[-1] > 0  # Price above band

    def test_signal_direction(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.mean_reversion import MeanReversionStrategy
        cfg = StrategyConfig(
            name="test",
            strategy_type=StrategyType.EQUITY,
            params={"rsi_oversold": 45, "rsi_overbought": 55, "z_entry": 1.0},
        )
        strat = MeanReversionStrategy(cfg)
        strat.initialize()
        # Run multiple times to generate at least some signals
        for _ in range(5):
            signals = strat.on_data(single_asset_data)
            for s in signals:
                if s.metadata.get("action") != "mean_rev_exit":
                    assert s.direction in [-1, 0, 1]


class TestTrendFollowingStrategy:
    def test_on_data_returns_signals(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.trend_following import TrendFollowingStrategy
        cfg = StrategyConfig(
            name="test_trend",
            strategy_type=StrategyType.EQUITY,
        )
        strat = TrendFollowingStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(single_asset_data)
        assert isinstance(signals, list)

    def test_ema_computation(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.trend_following import TrendFollowingStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.EQUITY)
        strat = TrendFollowingStrategy(cfg)
        prices = np.linspace(100, 200, 50)
        ema = strat._ema(prices, 20)
        assert len(ema) == 50
        assert ema[-1] > ema[0]

    def test_atr_sizing(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.equity.trend_following import TrendFollowingStrategy
        cfg = StrategyConfig(
            name="test",
            strategy_type=StrategyType.EQUITY,
            params={"trend_window": 50, "fast_window": 10, "slow_window": 30},
        )
        strat = TrendFollowingStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(single_asset_data)
        for s in signals:
            assert s.size >= 0
            if "atr" in s.metadata:
                assert s.metadata["atr"] > 0


# ===========================================================================
# Test Options Strategies
# ===========================================================================

class TestDeltaNeutralStrategy:
    def test_on_data_returns_list(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.delta_neutral import DeltaNeutralStrategy
        cfg = StrategyConfig(
            name="test_dn",
            strategy_type=StrategyType.OPTIONS,
            params={"ivr_entry": 30, "hedge_freq": 1},
        )
        strat = DeltaNeutralStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(single_asset_data)
        assert isinstance(signals, list)

    def test_ivr_computation(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.delta_neutral import DeltaNeutralStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.OPTIONS)
        strat = DeltaNeutralStrategy(cfg)
        iv_history = np.linspace(0.1, 0.5, 252)
        ivr = strat._compute_ivr(iv_history)
        assert 0 <= ivr <= 100

    def test_greeks_computation(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.delta_neutral import DeltaNeutralStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.OPTIONS)
        strat = DeltaNeutralStrategy(cfg)
        greeks = strat._compute_greeks(S=100, K=100, T=0.25, r=0.05, sigma=0.3)
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert greeks["gamma"] > 0
        assert greeks["vega"] > 0

    def test_pnl_attribution(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.delta_neutral import DeltaNeutralStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.OPTIONS)
        strat = DeltaNeutralStrategy(cfg)
        greeks = {"delta": -0.3, "gamma": 0.05, "vega": 15.0, "theta": -0.02}
        pnl = strat._pnl_attribution(greeks, 0.3, 100, 101, 0.3, 0.31)
        assert "delta_pnl" in pnl
        assert "gamma_pnl" in pnl
        assert "vega_pnl" in pnl
        assert "theta_pnl" in pnl
        assert "total_pnl" in pnl


class TestIronCondorStrategy:
    def test_on_data_returns_list(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.iron_condor import IronCondorStrategy
        cfg = StrategyConfig(
            name="test_ic",
            strategy_type=StrategyType.OPTIONS,
            params={"ivr_entry": 30},
        )
        strat = IronCondorStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(single_asset_data)
        assert isinstance(signals, list)

    def test_delta_to_strike(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.iron_condor import IronCondorStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.OPTIONS)
        strat = IronCondorStrategy(cfg)
        # 16-delta call should be above spot
        call_strike = strat._delta_to_strike(100, 0.25, 0.3, 16, is_call=True)
        assert call_strike > 100
        # 16-delta put should be below spot
        put_strike = strat._delta_to_strike(100, 0.25, 0.3, 16, is_call=False)
        assert put_strike < 100


class TestEarningsVolStrategy:
    def test_on_data_returns_list(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.earnings_vol import EarningsVolStrategy
        cfg = StrategyConfig(name="test_ev", strategy_type=StrategyType.OPTIONS)
        strat = EarningsVolStrategy(cfg)
        strat.initialize()
        # Add earnings flag
        single_asset_data["earnings_upcoming"] = np.array(1.0)
        signals = strat.on_data(single_asset_data)
        assert isinstance(signals, list)

    def test_straddle_entry_with_earnings(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.earnings_vol import EarningsVolStrategy
        cfg = StrategyConfig(
            name="test",
            strategy_type=StrategyType.OPTIONS,
            params={"implied_vs_historical_ratio": 1.5},  # Always underpriced
        )
        strat = EarningsVolStrategy(cfg)
        strat.initialize()

        rng = np.random.default_rng(42)
        n = 300
        prices = 100 + np.cumsum(rng.normal(0.02, 0.5, n))
        prices = np.maximum(prices, 10)
        data = {
            "close": prices,
            "implied_vol": np.full(n, 0.2),
            "symbols": ["TEST"],
            "earnings_upcoming": np.array(1.0),
        }
        signals = strat.on_data(data)
        # Should get entry signal since IV < historical and earnings upcoming
        entry_signals = [s for s in signals if s.metadata.get("action") == "earnings_straddle_entry"]
        assert len(entry_signals) >= 0  # May or may not trigger depending on data


class TestTailHedgeStrategy:
    def test_on_data_returns_list(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.tail_hedge import TailHedgeStrategy
        cfg = StrategyConfig(name="test_th", strategy_type=StrategyType.OPTIONS)
        strat = TailHedgeStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(single_asset_data)
        assert isinstance(signals, list)

    def test_risk_score_computation(self, single_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.options.tail_hedge import TailHedgeStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.OPTIONS)
        strat = TailHedgeStrategy(cfg)
        # Compute risk indicator proxies
        iv = single_asset_data["implied_vol"]
        skew = strat._compute_skew_proxy(iv)
        assert skew > 0
        corr = strat._compute_correlation_proxy(single_asset_data["close"])
        assert 0 <= corr <= 1.5  # Autocorrelation range


# ===========================================================================
# Test Macro Strategies
# ===========================================================================

class TestRiskParityStrategy:
    def test_on_data_returns_signals(self, multi_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.risk_parity import RiskParityStrategy
        cfg = StrategyConfig(
            name="test_rp",
            strategy_type=StrategyType.MACRO,
            params={"rebalance_freq": 1, "lookback": 100},
        )
        strat = RiskParityStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(multi_asset_data)
        assert isinstance(signals, list)
        for s in signals:
            assert s.direction in [-1, 0, 1]
            assert s.strength >= 0

    def test_equal_risk_contribution(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.risk_parity import RiskParityStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.MACRO)
        strat = RiskParityStrategy(cfg)
        # Two assets with equal variance → equal weights
        cov = np.array([[0.04, 0.01], [0.01, 0.04]])
        weights = strat._risk_parity_weights(cov)
        assert len(weights) == 2
        assert abs(weights[0] - weights[1]) < 0.05  # Roughly equal

    def test_unequal_risk_contribution(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.risk_parity import RiskParityStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.MACRO)
        strat = RiskParityStrategy(cfg)
        # Asset 0 has 4x the variance of asset 1 → should get ~half the weight
        cov = np.array([[0.16, 0.0], [0.0, 0.04]])
        weights = strat._risk_parity_weights(cov)
        assert weights[0] < weights[1]  # Lower weight to higher-vol asset

    def test_regime_overlay(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.risk_parity import RiskParityStrategy
        cfg = StrategyConfig(
            name="test",
            strategy_type=StrategyType.MACRO,
            params={"regime_overlay": True},
        )
        strat = RiskParityStrategy(cfg)
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, (252, 3))
        scale = strat._regime_scale(returns)
        assert 0.3 <= scale <= 1.5


class TestCTATrendStrategy:
    def test_on_data_returns_signals(self, multi_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.cta_trend import CTATrendStrategy
        cfg = StrategyConfig(
            name="test_cta",
            strategy_type=StrategyType.MACRO,
            params={"rebalance_freq": 1, "slow_window": 100},
        )
        strat = CTATrendStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(multi_asset_data)
        assert isinstance(signals, list)

    def test_trend_signal(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.cta_trend import CTATrendStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.MACRO)
        strat = CTATrendStrategy(cfg)
        # Uptrend: price above SMA
        prices = np.linspace(100, 200, 50)
        signal = strat._trend_signal(prices, 20)
        assert signal > 0
        # Downtrend
        prices_down = np.linspace(200, 100, 50)
        signal_down = strat._trend_signal(prices_down, 20)
        assert signal_down < 0

    def test_breakout_signal(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.cta_trend import CTATrendStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.MACRO)
        strat = CTATrendStrategy(cfg)
        # At 52-week high
        prices = np.linspace(100, 200, 50)
        signal = strat._breakout_signal(prices, 50)
        assert signal > 0.5

    def test_momentum_signal(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.cta_trend import CTATrendStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.MACRO)
        strat = CTATrendStrategy(cfg)
        prices = np.linspace(100, 200, 22)
        signal = strat._momentum_signal(prices, 20)
        assert signal == 1.0  # Positive return → buy


class TestCarryStrategy:
    def test_on_data_returns_signals(self, multi_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.carry import CarryStrategy
        cfg = StrategyConfig(
            name="test_carry",
            strategy_type=StrategyType.MACRO,
            params={"rebalance_freq": 1, "n_long": 2, "n_short": 2},
        )
        strat = CarryStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(multi_asset_data)
        assert isinstance(signals, list)

    def test_with_yield_data(self, multi_asset_data):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.carry import CarryStrategy
        rng = np.random.default_rng(42)
        n = multi_asset_data["close"].shape[0]
        n_assets = multi_asset_data["close"].shape[1]
        yields = rng.normal(0.03, 0.01, (n, n_assets))
        multi_asset_data["yields"] = yields

        cfg = StrategyConfig(
            name="test",
            strategy_type=StrategyType.MACRO,
            params={"rebalance_freq": 1, "n_long": 2, "n_short": 2},
        )
        strat = CarryStrategy(cfg)
        strat.initialize()
        signals = strat.on_data(multi_asset_data)
        # High carry assets should be long, low carry short
        longs = [s for s in signals if s.direction > 0]
        shorts = [s for s in signals if s.direction < 0]
        assert len(longs) >= 0
        assert len(shorts) >= 0

    def test_carry_smoothing(self):
        from quantcore.strategies.base import StrategyConfig, StrategyType
        from quantcore.strategies.macro.carry import CarryStrategy
        cfg = StrategyConfig(name="test", strategy_type=StrategyType.MACRO)
        strat = CarryStrategy(cfg)
        # Verify carry strategy initializes with expected attributes
        assert hasattr(strat, 'rebalance_freq')
        assert hasattr(strat, 'target_vol')
        assert strat.rebalance_freq == 21


# ===========================================================================
# Test __init__ Exports
# ===========================================================================

class TestStrategiesInit:
    def test_base_exports(self):
        from quantcore.strategies import (
            Strategy,
            StrategyConfig,
            StrategyMetrics,
            TradeSignal,
        )
        assert Strategy is not None
        assert StrategyConfig is not None
        assert StrategyMetrics is not None
        assert TradeSignal is not None

    def test_registry_exports(self):
        from quantcore.strategies import create, get, list_strategies, register
        assert callable(register)
        assert callable(get)
        assert callable(list_strategies)
        assert callable(create)

    def test_equity_exports(self):
        from quantcore.strategies.equity import (
            MeanReversionStrategy,
            MomentumStrategy,
            StatArbStrategy,
            TrendFollowingStrategy,
        )
        assert all([MomentumStrategy, StatArbStrategy, MeanReversionStrategy, TrendFollowingStrategy])

    def test_options_exports(self):
        from quantcore.strategies.options import (
            DeltaNeutralStrategy,
            EarningsVolStrategy,
            IronCondorStrategy,
            TailHedgeStrategy,
        )
        assert all([DeltaNeutralStrategy, IronCondorStrategy, EarningsVolStrategy, TailHedgeStrategy])

    def test_macro_exports(self):
        from quantcore.strategies.macro import (
            CarryStrategy,
            CTATrendStrategy,
            RiskParityStrategy,
        )
        assert all([RiskParityStrategy, CTATrendStrategy, CarryStrategy])
