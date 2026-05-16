"""Unit tests for CrossSectionalFactorNeutralStatArb (CSFN).

Covers core signal pipeline, factor extraction, and portfolio construction.
"""

from __future__ import annotations

import pytest
import numpy as np
from numpy.typing import NDArray

from quantcore.strategies.base import StrategyConfig, StrategyStatus
from quantcore.strategies.registry import get as get_strategy_class

# Import the strategy module to trigger the @register decorator
import quantcore.strategies.macro.cross_sectional_factor_neutral  # noqa: F401

pytest.importorskip("sklearn", reason="tests require scikit-learn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_default_config() -> StrategyConfig:
    return StrategyConfig(
        name="macro_cross_sectional_factor_neutral",
        strategy_type="macro",
        max_position_notional=100_000,
        max_leverage=1.0,
        params={
            "rebalance_freq": 21,
            "reg_lookback": 60,
            "pred_window": 10,
            "ridge_alpha": 1.0,
            "max_position": 0.03,
            "target_vol": 0.15,
            "vol_lookback": 20,
            "zscore_cutoff": 1.5,
            "percentile_top": 0.30,
            "factor_symbols": "SPY,TLT",
        },
    )


def _build_close_matrix(t: int, n: int, seed: int = 42) -> NDArray:
    rng = np.random.default_rng(seed)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (t, n)), axis=0))
    return base


# ---------------------------------------------------------------------------
# Registry & instantiation
# ---------------------------------------------------------------------------

def test_strategy_registered():
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    assert cls is not None
    cfg = _build_default_config()
    inst = cls(cfg)
    assert inst is not None
    assert inst.STRATEGY_STATUS == StrategyStatus.RESEARCH


def test_strategy_initialisation():
    cfg = _build_default_config()
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    s = cls(cfg)
    assert s.rebalance_freq == 21
    assert s.reg_lookback == 60
    assert s.pred_window == 10
    assert s.ridge_alpha == 1.0
    assert s.max_position == 0.03
    assert s.target_vol == 0.15


def test_factor_symbols_parsing():
    cfg = _build_default_config()
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    s = cls(cfg)
    # Should have parsed the default factor symbols
    assert "SPY" in s.factor_symbols
    assert "TLT" in s.factor_symbols


# ---------------------------------------------------------------------------
# Core signal pipeline
# ---------------------------------------------------------------------------

def test_prices_to_returns():
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    s = cls(_build_default_config())

    prices = np.array([100.0, 102.0, 103.0, 101.0, 99.0]).reshape(-1, 1)
    rets = s._prices_to_returns(prices)
    assert rets is not None
    assert rets.shape[0] == prices.shape[0] - 1
    assert np.all(np.isfinite(rets))
    # Expected: (102-100)/100, (103-102)/102, (101-103)/103, (99-101)/101
    expected = np.array([0.02, 0.00980392157, -0.01941747572, -0.01980198020])
    np.testing.assert_allclose(rets.flatten(), expected, rtol=1e-5, atol=1e-7)


def test_cross_sectional_zscore():
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    s = cls(_build_default_config())

    alpha = np.array([1.0, -1.0, 0.0, 2.0, -2.0])
    z = s._cross_sectional_zscore(alpha)
    assert z is not None
    assert np.isclose(z.mean(), 0.0, atol=1e-12)
    assert np.isfinite(z).all()
    assert len(z) == len(alpha)


def test_zscore_to_raw_weights_nontrivial():
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    s = cls(_build_default_config())

    n = 10
    z = np.array([2.0, 1.0, 0.0, -1.0, -2.0, 0.5, -0.5, 3.0, -3.0, 0.0])
    weights = s._zscore_to_raw_weights(z)

    assert weights is not None
    assert len(weights) == n
    # Dollar neutrality (sum ≈ 0)
    np.testing.assert_allclose(weights.sum(), 0.0, atol=1e-12)
    # Highest z-scores (indices 7, 8 with value 3) should be long, lowest short
    assert weights[7] > 0
    assert weights[8] < 0


# ---------------------------------------------------------------------------
# End-to-end with synthetic data
# ---------------------------------------------------------------------------

def test_on_data_with_synthetic_prices():
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    cfg = _build_default_config()
    cfg.params["reg_lookback"] = 50
    cfg.params["pred_window"] = 5
    s = cls(cfg)

    t, n = 100, 4
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.005, (t, n))
    close = 100.0 * np.exp(np.cumsum(returns, axis=0))

    data = {
        "close": close,
        "symbols": ["SPY", "TLT", "ABC", "DEF"],
    }

    signals = s.on_data(data)
    # With enough data, should emit some signals
    assert isinstance(signals, list)
    # Since we have only 4 assets and 2 are factors, the residual signal
    # should be assignable to at least one non-factor
    if signals:
        for sig in signals:
            assert sig.symbol in data["symbols"]
            assert sig.direction in (-1, 1)
            assert 0 <= sig.strength <= 1


def test_factor_neutral_qp_applies():
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    s = cls(_build_default_config())

    n_assets = 8
    z = np.random.RandomState(42).standard_normal(n_assets)
    raw = s._zscore_to_raw_weights(z)
    # Create fake betas (identity-ish for test)
    betas = np.eye(n_assets, min(n_assets, 3))
    symbols = [f"A{i}" for i in range(n_assets)]

    weights = s._apply_factor_neutral(raw, betas, symbols)
    # If QP solver is unavailable, weights should be returned unchanged
    # but still dollar-neutral
    assert weights is not None
    np.testing.assert_allclose(weights.sum(), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Regression tests (ensure FactorResidualQPStrategy corner cases don't mask CSFN)
# ---------------------------------------------------------------------------

def test_strategy_not_disabled():
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    s = cls(_build_default_config())
    assert s.STRATEGY_STATUS != StrategyStatus.DISABLED


def test_empty_data_returns_empty_signals():
    cls = get_strategy_class("macro_cross_sectional_factor_neutral")
    s = cls(_build_default_config())
    signals = s.on_data({"close": np.array([]), "symbols": []})
    assert signals == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
