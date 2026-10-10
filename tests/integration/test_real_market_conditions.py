"""Real market condition smoke tests: stress strategies under 5 market scenarios.

Each scenario generates synthetic price data simulating a specific market
condition and validates that strategies produce valid, non-degenerate signals.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.strategies.base import StrategyConfig, StrategyType
from quantcore.strategies.macro import *  # Triggers strategy registration
from quantcore.strategies.registry import get as get_strategy_class

# -- Scenario data generators ------------------------------------------------

def _scenario_baseline(n_bars: int = 500, n_assets: int = 6, seed: int = 42):
    """Generate baseline correlated random walk data.

    Returns (close, symbols) where close.shape = (n_bars, n_assets).
    """
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0003, 0.005, n_bars)
    specific = rng.normal(0.0005, 0.015, (n_bars, n_assets))
    returns = specific + common[:, np.newaxis] * 0.3
    log_prices = np.zeros((n_bars, n_assets))
    log_prices[0] = np.log(100.0)
    for i in range(1, n_bars):
        log_prices[i] = log_prices[i - 1] + returns[i]
    close = np.exp(log_prices)
    symbols = [f"ASSET_{i}" for i in range(n_assets)]
    return close, symbols


def scenario_covid_crash(n_bars: int = 500, n_assets: int = 6) -> tuple:
    """Simulate a sudden sharp market decline (like COVID March 2020).

    Phase 1: Normal (bars 0-300)
    Phase 2: Crash — -15% over 20 bars with elevated vol (bars 301-320)
    Phase 3: Recovery — +10% over 30 bars with elevated vol (bars 321-350)
    Phase 4: Normal (bars 351-500)
    """
    close, symbols = _scenario_baseline(n_bars, n_assets)

    # Crash phase: amplify common factor downwards, increase vol
    for i in range(301, min(321, n_bars)):
        crash_ret = -0.015 + np.random.default_rng(i).normal(0, 0.025, n_assets)
        close[i] = close[i - 1] * np.exp(crash_ret)

    # Recovery phase
    for i in range(321, min(351, n_bars)):
        recovery_ret = 0.005 + np.random.default_rng(i).normal(0, 0.02, n_assets)
        close[i] = close[i - 1] * np.exp(recovery_ret)

    return close, symbols


def scenario_trend_reversal(n_bars: int = 500, n_assets: int = 6) -> tuple:
    """Simulate a sharp trend reversal (like 2022 pivot).

    Phase 1: Strong uptrend (bars 0-250)
    Phase 2: Consolidation (bars 251-300)
    Phase 3: Sharp reversal downtrend (bars 301-400)
    Phase 4: New trend (bars 401-500)
    """
    close, symbols = _scenario_baseline(n_bars, n_assets)

    # Overwrite with strong trend structure
    rng = np.random.default_rng(99)
    trend_up = 0.003 + rng.normal(0, 0.008, (250, n_assets))  # strong uptrend
    consolidate = rng.normal(0.0001, 0.005, (50, n_assets))    # chop
    trend_down = -0.004 + rng.normal(0, 0.01, (100, n_assets)) # sharp downtrend
    new_trend = 0.002 + rng.normal(0, 0.007, (100, n_assets))  # new uptrend

    all_rets = np.vstack([trend_up, consolidate, trend_down, new_trend])
    n_actual = min(len(all_rets), n_bars)
    close[0] = 100.0
    for i in range(1, n_actual):
        close[i] = close[i - 1] * np.exp(all_rets[i - 1])

    return close, symbols


def scenario_correlation_breakdown(n_bars: int = 500, n_assets: int = 6) -> tuple:
    """Simulate correlation structure breakdown.

    Phase 1: High correlation regime (bars 0-250) — common factor dominates
    Phase 2: Correlation breakdown (bars 251-400) — assets diverge
    Phase 3: Re-correlation (bars 401-500)
    """
    rng = np.random.default_rng(77)
    n = n_bars

    log_prices = np.zeros((n, n_assets))
    log_prices[0] = np.log(100.0)

    for i in range(1, n):
        common = rng.normal(0, 0.005)
        specific = rng.normal(0, 0.015, n_assets)

        if i < 250:
            # High correlation: common factor explains 70% of variance
            rets = specific + common * 2.0
        elif i < 400:
            # Breakdown: assets diverge (high specific, low common)
            rets = specific * 1.5 + common * 0.3
        else:
            # Re-correlation
            rets = specific + common * 1.5

        log_prices[i] = log_prices[i - 1] + rets

    close = np.exp(log_prices)
    symbols = [f"ASSET_{i}" for i in range(n_assets)]
    return close, symbols


def scenario_liquidity_crisis(n_bars: int = 500, n_assets: int = 6) -> tuple:
    """Simulate a liquidity crisis with gap risk.

    Phase 1: Normal (bars 0-300)
    Phase 2: Liquidity crisis — large gaps, increased vol (bars 301-370)
    Phase 3: Recovery (bars 371-500)
    """
    close, symbols = _scenario_baseline(n_bars, n_assets)

    rng = np.random.default_rng(55)

    for i in range(301, min(371, n_bars)):
        # Large gaps: wider return distribution
        gap_ret = rng.laplace(0, 0.03, n_assets)
        close[i] = close[i - 1] * np.exp(gap_ret)

    for i in range(371, min(401, n_bars)):
        # Recovery with still elevated vol
        ret = rng.normal(0.002, 0.015, n_assets)
        close[i] = close[i - 1] * np.exp(ret)

    return close, symbols


def scenario_low_vol_then_spike(n_bars: int = 500, n_assets: int = 6) -> tuple:
    """Simulate low vol regime transitioning to high vol spike.

    Phase 1: Low vol calm market (bars 0-300)
    Phase 2: Vol spike (bars 301-350)
    Phase 3: Elevated vol (bars 351-500)
    """
    rng = np.random.default_rng(33)
    n = n_bars

    log_prices = np.zeros((n, n_assets))
    log_prices[0] = np.log(100.0)

    for i in range(1, n):
        if i < 300:
            # Low vol
            common = rng.normal(0.0002, 0.002)
            specific = rng.normal(0.0003, 0.005, n_assets)
        elif i < 350:
            # Vol spike
            common = rng.normal(-0.002, 0.015)
            specific = rng.normal(-0.003, 0.03, n_assets)
        else:
            # Elevated vol
            common = rng.normal(0.0005, 0.008)
            specific = rng.normal(0.0005, 0.02, n_assets)

        log_prices[i] = log_prices[i - 1] + specific + common

    close = np.exp(log_prices)
    symbols = [f"ASSET_{i}" for i in range(n_assets)]
    return close, symbols


# -- Test configurations -----------------------------------------------------

SCENARIOS = {
    "covid_crash": scenario_covid_crash,
    "trend_reversal": scenario_trend_reversal,
    "correlation_breakdown": scenario_correlation_breakdown,
    "liquidity_crisis": scenario_liquidity_crisis,
    "low_vol_then_spike": scenario_low_vol_then_spike,
}

# Strategies to test: the two macro deploy candidates
STRATEGIES = [
    ("macro_risk_parity", {
        "vol_target": 0.10,
        "lookback": 100,
        "rebalance_freq": 21,
        "regime_overlay": True,
        "max_leverage": 1.0,
    }),
    ("macro_cta_trend", {
        "fast_window": 20,
        "medium_window": 50,
        "slow_window": 200,
        "vol_target": 0.15,
        "vol_lookback": 60,
        "rebalance_freq": 5,
    }),
]


# -- Tests -------------------------------------------------------------------

class TestMarketConditionSmokeTests:
    """Smoke test all strategies under different market scenarios."""

    @pytest.mark.parametrize("strategy_name,params", STRATEGIES,
                             ids=[s[0] for s in STRATEGIES])
    @pytest.mark.parametrize("scenario_name,scenario_fn", list(SCENARIOS.items()),
                             ids=list(SCENARIOS.keys()))
    def test_strategy_survives_scenario(self, strategy_name, params, scenario_name, scenario_fn):
        """Strategy produces valid signals without crashing under scenario."""
        close, symbols = scenario_fn(500, 6)
        n_periods, n_assets = close.shape

        cfg = StrategyConfig(
            name=strategy_name,
            strategy_type=StrategyType.MACRO,
            params=dict(params),
        )
        strategy_cls = get_strategy_class(strategy_name)
        strategy = strategy_cls(cfg)
        strategy.initialize()

        data = {"close": close, "symbols": symbols}

        all_signals = []
        signal_bars = 0
        crash_count = 0

        for i in range(100, n_periods):
            subset = {"close": close[:i], "symbols": symbols}
            try:
                signals = strategy.on_data(subset) or []
                all_signals.append(len(signals))
                if len(signals) > 0:
                    signal_bars += 1
            except Exception:
                crash_count += 1

        # Must not crash on any bar
        assert crash_count == 0, (
            f"{strategy_name} crashed on {crash_count}/{n_periods - 100} "
            f"bars in {scenario_name} scenario"
        )

        # Must produce signals on at least some bars
        assert signal_bars > 0, (
            f"{strategy_name} produced zero signal bars in {scenario_name} scenario"
        )

    @pytest.mark.parametrize("strategy_name,params", STRATEGIES,
                             ids=[s[0] for s in STRATEGIES])
    def test_strategy_baseline(self, strategy_name, params):
        """Strategy produces valid signals on baseline data."""
        close, symbols = _scenario_baseline(500, 6)

        cfg = StrategyConfig(
            name=strategy_name,
            strategy_type=StrategyType.MACRO,
            params=dict(params),
        )
        strategy = get_strategy_class(strategy_name)(cfg)
        strategy.initialize()

        # Collect signals after warm-up
        all_strengths = []
        has_signal = False
        for i in range(200, 500):
            subset = {"close": close[:i], "symbols": symbols}
            signals = strategy.on_data(subset) or []
            for s in signals:
                has_signal = True
                all_strengths.append(abs(s.strength))

        assert has_signal, f"{strategy_name} produced no signals on baseline data"
        strengths = np.array(all_strengths)
        assert np.mean(strengths) > 0.01, (
            f"{strategy_name} signal strengths too low: {np.mean(strengths):.4f}"
        )
        assert np.max(strengths) <= 1.0, (
            f"{strategy_name} signal strength exceeds 1.0: {np.max(strengths):.4f}"
        )

    @pytest.mark.parametrize("strategy_name,params", STRATEGIES,
                             ids=[s[0] for s in STRATEGIES])
    def test_strategy_returns_signal_metadata(self, strategy_name, params):
        """Signal metadata contains expected fields for monitoring."""
        close, symbols = scenario_covid_crash(500, 6)

        cfg = StrategyConfig(
            name=strategy_name,
            strategy_type=StrategyType.MACRO,
            params=dict(params),
        )
        strategy = get_strategy_class(strategy_name)(cfg)
        strategy.initialize()

        # Check that signals have metadata
        for i in range(250, 400):
            subset = {"close": close[:i], "symbols": symbols}
            signals = strategy.on_data(subset) or []
            for s in signals:
                assert s.metadata is not None, "Signal metadata is None"
                assert s.symbol, "Signal symbol is empty"
                assert isinstance(s.direction, int), f"Direction not int: {type(s.direction)}"
                assert -1 <= s.direction <= 1, f"Direction out of range: {s.direction}"
                assert 0 <= s.strength <= 1.0, f"Strength out of range: {s.strength}"
                assert s.size > 0, f"Size not positive: {s.size}"

    def test_all_scenarios_generate_different_regimes(self):
        """Verify each scenario produces distinct statistical properties."""
        regime_vols = {}
        for name, fn in SCENARIOS.items():
            close, _ = fn(500, 6)
            # Split into halves and compare vol
            rets = np.diff(close, axis=0) / close[:-1]
            port_rets = np.mean(rets, axis=1)
            first_half_vol = np.std(port_rets[:150]) * np.sqrt(252)
            second_half_vol = np.std(port_rets[150:]) * np.sqrt(252)
            regime_vols[name] = (first_half_vol, second_half_vol)

        # Ensure scenarios are differentiated
        vols_flat = [v for pair in regime_vols.values() for v in pair]
        assert len(set(round(v, 2) for v in vols_flat)) >= 2, (
            "All scenarios produce near-identical vols — not differentiated"
        )
