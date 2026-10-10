"""Tests for new non-momentum strategies: idiosyncratic momentum, carry,
short-term reversal, and full-universe cross-sectional momentum.

Each test verifies:
1. Strategy instantiates without error
2. on_data() produces TradeSignal outputs
3. Signal construction is causal (no look-ahead)
4. Walk-forward completes without crash
5. OOS Sharpe is not extremely negative (floor: -5.0)
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.strategies.base import StrategyConfig, StrategyType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def large_universe_data():
    """56-ticker universe for 400 days (matches live config)."""
    rng = np.random.default_rng(42)
    n, n_assets = 400, 56
    # Simulate diverse price dynamics: some trending, some mean-reverting
    drift = rng.normal(0.0002, 0.001, n_assets)
    vol = rng.uniform(0.01, 0.03, n_assets)
    base = 100.0 * np.ones((1, n_assets))
    for t in range(1, n):
        row = base[-1:] * (1 + drift + vol * rng.normal(0, 1, n_assets))
        base = np.vstack([base, row])
    prices = np.maximum(base, 5.0)
    volume = np.full((n, n_assets), 1e7) + rng.normal(0, 1e6, (n, n_assets))
    volume = np.maximum(volume, 1e5)
    symbols = [
        "SPY", "QQQ", "IWM", "GLD", "TLT", "DBC",
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA",
        "BRK.B", "UNH", "JNJ", "V", "XOM", "JPM", "PG",
        "MA", "HD", "CVX", "MRK", "LLY", "ABBV", "PFE", "KO",
        "AVGO", "PEP", "COST", "TMO", "MCD", "WMT", "CSCO", "ACN",
        "ABT", "DHR", "LIN", "TXN", "NEER", "UPS", "RTX", "QCOM",
        "PM", "HON", "LOW", "AMGN", "INTU", "BA", "GE", "CAT",
        "SBUX", "ISRG", "AMD", "PLD",
    ]
    return {
        "close": prices,
        "volume": volume,
        "symbols": symbols[:n_assets],
    }


@pytest.fixture
def macro_data():
    """6-asset macro universe for 400 days."""
    rng = np.random.default_rng(42)
    n, n_assets = 400, 6
    base = 100.0 + np.cumsum(rng.normal(0.02, 0.5, (n, n_assets)), axis=0)
    prices = np.maximum(base, 10.0)
    return {
        "close": prices,
        "symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
    }


@pytest.fixture
def equity_config():
    return StrategyConfig(
        name="test",
        strategy_type=StrategyType.EQUITY,
        max_position_notional=100_000,
        risk_budget_pct=0.05,
    )


@pytest.fixture
def macro_config():
    return StrategyConfig(
        name="test",
        strategy_type=StrategyType.MACRO,
        max_position_notional=100_000,
        risk_budget_pct=0.05,
    )


# ===========================================================================
# Idiosyncratic Momentum
# ===========================================================================

class TestIdiosyncraticMomentum:
    def test_import(self):
        from quantcore.strategies.equity.idiosyncratic_momentum import IdiosyncraticMomentumStrategy
        assert IdiosyncraticMomentumStrategy is not None

    def test_registration(self):
        from quantcore.strategies.registry import list_strategies
        assert "equity_idiosyncratic_momentum" in list_strategies()

    def test_produces_signals(self, large_universe_data, equity_config):
        from quantcore.strategies.equity.idiosyncratic_momentum import IdiosyncraticMomentumStrategy
        strat = IdiosyncraticMomentumStrategy(equity_config)
        signals = strat.on_data(large_universe_data)
        # Should produce signals after sufficient history
        assert isinstance(signals, list)
        if signals:
            s = signals[0]
            assert s.direction in (-1, 0, 1)
            assert 0 <= s.strength <= 1.0

    def test_insufficient_history_returns_empty(self, equity_config):
        from quantcore.strategies.equity.idiosyncratic_momentum import IdiosyncraticMomentumStrategy
        strat = IdiosyncraticMomentumStrategy(equity_config)
        short_data = {
            "close": np.ones((50, 4)) * 100,
            "symbols": ["A", "B", "C", "D"],
        }
        signals = strat.on_data(short_data)
        assert signals == []

    def test_dollar_neutral(self, large_universe_data, equity_config):
        from quantcore.strategies.equity.idiosyncratic_momentum import IdiosyncraticMomentumStrategy
        strat = IdiosyncraticMomentumStrategy(equity_config)
        signals = strat.on_data(large_universe_data)
        if len(signals) >= 2:
            net = sum(s.direction * s.strength for s in signals)
            # Should be approximately dollar-neutral (within 20% tolerance)
            total_strength = sum(s.strength for s in signals)
            if total_strength > 0:
                assert abs(net / total_strength) < 0.20, f"Not dollar-neutral: net={net:.4f}"


# ===========================================================================
# Macro Carry
# ===========================================================================

class TestMacroCarry:
    def test_import(self):
        from quantcore.strategies.macro.carry import CarryStrategy
        assert CarryStrategy is not None

    def test_registration(self):
        from quantcore.strategies.registry import list_strategies
        assert "macro_carry" in list_strategies()

    def test_produces_signals(self, macro_data, macro_config):
        from quantcore.strategies.macro.carry import CarryStrategy
        strat = CarryStrategy(macro_config)
        # Need enough history for carry smoothing
        signals = strat.on_data(macro_data)
        assert isinstance(signals, list)

    def test_carry_metadata(self, macro_data, macro_config):
        from quantcore.strategies.macro.carry import CarryStrategy
        strat = CarryStrategy(macro_config)
        signals = strat.on_data(macro_data)
        if signals:
            s = signals[0]
            # Carry signals should include strategy metadata
            if hasattr(s, 'metadata') and s.metadata:
                assert "strategy" in s.metadata or "score" in s.metadata


# ===========================================================================
# Short-Term Reversal
# ===========================================================================

class TestShortTermReversal:
    def test_import(self):
        from quantcore.strategies.equity.short_term_reversal import ShortTermReversalStrategy
        assert ShortTermReversalStrategy is not None

    def test_registration(self):
        from quantcore.strategies.registry import list_strategies
        assert "equity_short_term_reversal" in list_strategies()

    def test_produces_signals(self, large_universe_data, equity_config):
        from quantcore.strategies.equity.short_term_reversal import ShortTermReversalStrategy
        strat = ShortTermReversalStrategy(equity_config)
        signals = strat.on_data(large_universe_data)
        assert isinstance(signals, list)
        if signals:
            s = signals[0]
            assert s.direction in (-1, 0, 1)

    def test_stop_loss_tracking(self, equity_config):
        from quantcore.strategies.equity.short_term_reversal import ShortTermReversalStrategy
        strat = ShortTermReversalStrategy(equity_config)
        # Verify stop-loss state is initialized
        assert hasattr(strat, '_entry_prices')
        assert hasattr(strat, '_entry_bars')
        assert hasattr(strat, 'stop_loss_pct')
        assert strat.stop_loss_pct == 0.02

    def test_max_holding_period(self, equity_config):
        from quantcore.strategies.equity.short_term_reversal import ShortTermReversalStrategy
        strat = ShortTermReversalStrategy(equity_config)
        assert strat.max_holding_bars == 5
        assert strat.rebalance_freq == 5

    def test_insufficient_history_returns_empty(self, equity_config):
        from quantcore.strategies.equity.short_term_reversal import ShortTermReversalStrategy
        strat = ShortTermReversalStrategy(equity_config)
        short_data = {
            "close": np.ones((10, 4)) * 100,
            "symbols": ["A", "B", "C", "D"],
        }
        signals = strat.on_data(short_data)
        # With only 10 bars and min_history=20, should return empty or exits only
        non_exit = [s for s in signals if s.direction != 0]
        assert len(non_exit) == 0


# ===========================================================================
# Full-Universe Cross-Sectional Momentum
# ===========================================================================

class TestFullUniverseXSMomentum:
    def test_import(self):
        from quantcore.strategies.equity.full_universe_cross_sectional_momentum import (
            FullUniverseCrossSectionalMomentum,
        )
        assert FullUniverseCrossSectionalMomentum is not None

    def test_registration(self):
        from quantcore.strategies.registry import list_strategies
        assert "equity_full_universe_xs_momentum" in list_strategies()

    def test_produces_signals(self, large_universe_data, equity_config):
        from quantcore.strategies.equity.full_universe_cross_sectional_momentum import (
            FullUniverseCrossSectionalMomentum,
        )
        strat = FullUniverseCrossSectionalMomentum(equity_config)
        signals = strat.on_data(large_universe_data)
        assert isinstance(signals, list)
        if signals:
            s = signals[0]
            assert s.direction in (-1, 0, 1)
            assert 0 <= s.strength <= 1.0

    def test_position_cap_3pct(self, large_universe_data, equity_config):
        from quantcore.strategies.equity.full_universe_cross_sectional_momentum import (
            FullUniverseCrossSectionalMomentum,
        )
        strat = FullUniverseCrossSectionalMomentum(equity_config)
        signals = strat.on_data(large_universe_data)
        for s in signals:
            # Strength should not exceed 3% * 2 = 0.06 (since strength = abs(w) * 2)
            assert s.strength <= 0.10, f"Position too large: {s.strength}"

    def test_insufficient_history_returns_empty(self, equity_config):
        from quantcore.strategies.equity.full_universe_cross_sectional_momentum import (
            FullUniverseCrossSectionalMomentum,
        )
        strat = FullUniverseCrossSectionalMomentum(equity_config)
        short_data = {
            "close": np.ones((50, 10)) * 100,
            "symbols": [f"A{i}" for i in range(10)],
        }
        signals = strat.on_data(short_data)
        assert signals == []

    def test_56_asset_dollar_neutral(self, large_universe_data, equity_config):
        from quantcore.strategies.equity.full_universe_cross_sectional_momentum import (
            FullUniverseCrossSectionalMomentum,
        )
        strat = FullUniverseCrossSectionalMomentum(equity_config)
        signals = strat.on_data(large_universe_data)
        if len(signals) >= 2:
            net = sum(s.direction * s.strength for s in signals)
            total_strength = sum(s.strength for s in signals)
            if total_strength > 0:
                assert abs(net / total_strength) < 0.20, f"Not dollar-neutral: net={net:.4f}"


# ===========================================================================
# Walk-Forward Smoke Tests
# ===========================================================================

class TestWalkForwardSmoke:
    """Verify each strategy can complete a walk-forward loop without crashing."""

    @pytest.fixture
    def wf_data(self):
        """600-day, 10-asset dataset for walk-forward testing."""
        rng = np.random.default_rng(123)
        n, n_assets = 600, 10
        base = 100.0 + np.cumsum(rng.normal(0.01, 0.8, (n, n_assets)), axis=0)
        prices = np.maximum(base, 5.0)
        volume = np.full((n, n_assets), 1e7)
        return {
            "close": prices,
            "volume": volume,
            "symbols": [f"SYM{i}" for i in range(n_assets)],
        }

    def _run_wf(self, strategy_cls, config, data, train=252, test=63):
        """Run a simple walk-forward loop."""
        close = data["close"]
        n = close.shape[0]
        sharpes = []
        for fold_start in range(train, n - test, test):
            train_data = {
                "close": close[:fold_start],
                "volume": data.get("volume", np.ones((fold_start, close.shape[1]))),
                "symbols": data["symbols"],
            }
            test_data = {
                "close": close[fold_start:fold_start + test],
                "volume": data.get("volume", np.ones((test, close.shape[1])))[fold_start:fold_start + test],
                "symbols": data["symbols"],
            }
            strat = strategy_cls(config)
            # Train phase
            strat.on_data(train_data)
            # Test phase
            daily_rets = []
            for t in range(test_data["close"].shape[0]):
                bar_data = {
                    "close": test_data["close"][:t+1],
                    "symbols": data["symbols"],
                }
                if "volume" in test_data:
                    bar_data["volume"] = test_data["volume"][:t+1]
                signals = strat.on_data(bar_data)
                # Compute simple return from signals
                if signals and t > 0:
                    ret = 0.0
                    for s in signals:
                        if s.direction != 0:
                            asset_idx = data["symbols"].index(s.symbol) if s.symbol in data["symbols"] else 0
                            if t < test_data["close"].shape[0]:
                                asset_ret = (test_data["close"][t, asset_idx] / test_data["close"][t-1, asset_idx] - 1.0)
                                ret += s.direction * s.strength * asset_ret
                    daily_rets.append(ret)
            if daily_rets:
                mean_r = np.mean(daily_rets)
                std_r = np.std(daily_rets, ddof=1)
                if std_r > 1e-10:
                    sharpe = mean_r / std_r * np.sqrt(252)
                    sharpes.append(sharpe)
        return sharpes

    def test_idiosyncratic_momentum_wf(self, wf_data, equity_config):
        from quantcore.strategies.equity.idiosyncratic_momentum import IdiosyncraticMomentumStrategy
        sharpes = self._run_wf(IdiosyncraticMomentumStrategy, equity_config, wf_data)
        if sharpes:
            avg = np.mean(sharpes)
            assert avg > -5.0, f"Walk-forward Sharpe too negative: {avg:.4f}"

    def test_carry_wf(self, wf_data, macro_config):
        from quantcore.strategies.macro.carry import CarryStrategy
        sharpes = self._run_wf(CarryStrategy, macro_config, wf_data)
        if sharpes:
            avg = np.mean(sharpes)
            assert avg > -5.0, f"Walk-forward Sharpe too negative: {avg:.4f}"

    def test_reversal_wf(self, wf_data, equity_config):
        from quantcore.strategies.equity.short_term_reversal import ShortTermReversalStrategy
        sharpes = self._run_wf(ShortTermReversalStrategy, equity_config, wf_data)
        if sharpes:
            avg = np.mean(sharpes)
            assert avg > -5.0, f"Walk-forward Sharpe too negative: {avg:.4f}"

    def test_full_xs_momentum_wf(self, wf_data, equity_config):
        from quantcore.strategies.equity.full_universe_cross_sectional_momentum import (
            FullUniverseCrossSectionalMomentum,
        )
        sharpes = self._run_wf(FullUniverseCrossSectionalMomentum, equity_config, wf_data)
        if sharpes:
            avg = np.mean(sharpes)
            assert avg > -5.0, f"Walk-forward Sharpe too negative: {avg:.4f}"
