"""Unit tests for Atlas RAMFA strategy."""

from __future__ import annotations

import numpy as np

from quantcore.strategies.base import StrategyConfig, StrategyType


def _make_data(n_periods: int = 360, n_assets: int = 12) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    trend = np.linspace(0.0, 0.18, n_periods).reshape(-1, 1)
    noise = rng.normal(0.0, 0.015, (n_periods, n_assets))
    close = 100.0 * np.exp(trend + noise)
    volume = rng.normal(1_000_000, 150_000, (n_periods, n_assets))
    return {
        "close": close,
        "volume": volume,
        "spread_bps": np.full((n_periods, n_assets), 8.0),
        "implied_vol": np.full((n_periods, n_assets), 0.25),
        "regime_score": np.full(n_periods, 80.0),
        "symbols": np.array([f"ASSET_{idx}" for idx in range(n_assets)]),
    }


def test_atlas_ramfa_generates_long_and_short_signals() -> None:
    from quantcore.strategies.equity.atlas_ramfa import AtlasRamfaStrategy

    cfg = StrategyConfig(
        name="atlas_ramfa_test",
        strategy_type=StrategyType.EQUITY,
        max_position_notional=1_000_000.0,
        max_leverage=1.5,
        params={
            "rebalance_freq": 1,
            "n_long": 3,
            "n_short": 3,
            "min_liquidity_score": 0.0,
            "single_name_max_pct": 0.03,
        },
    )
    strategy = AtlasRamfaStrategy(cfg)
    strategy.initialize()

    signals = strategy.on_data(_make_data())

    assert signals
    assert any(signal.direction > 0 for signal in signals)
    assert any(signal.direction < 0 for signal in signals)
    assert all("composite_alpha" in signal.metadata for signal in signals)
    assert sum(abs(signal.size) for signal in signals) <= 1_500_000.0 + 1e-6


def test_atlas_ramfa_returns_no_directional_signals_before_history() -> None:
    from quantcore.strategies.equity.atlas_ramfa import AtlasRamfaStrategy

    cfg = StrategyConfig(
        name="atlas_ramfa_history_test",
        strategy_type=StrategyType.EQUITY,
        max_position_notional=1_000_000.0,
        params={"rebalance_freq": 1, "min_liquidity_score": 0.0},
    )
    strategy = AtlasRamfaStrategy(cfg)
    strategy.initialize()

    short_data = _make_data(n_periods=80)
    signals = strategy.on_data(short_data)

    assert all(signal.direction == 0 for signal in signals)


def test_atlas_ramfa_regime_filter_blocks_new_entries() -> None:
    from quantcore.strategies.equity.atlas_ramfa import AtlasRamfaStrategy

    cfg = StrategyConfig(
        name="atlas_ramfa_regime_test",
        strategy_type=StrategyType.EQUITY,
        max_position_notional=1_000_000.0,
        params={"rebalance_freq": 1, "min_liquidity_score": 0.0},
    )
    strategy = AtlasRamfaStrategy(cfg)
    strategy.initialize()

    data = _make_data()
    data["regime_score"] = np.full(data["close"].shape[0], 20.0)
    signals = strategy.on_data(data)

    assert all(signal.direction == 0 for signal in signals)
