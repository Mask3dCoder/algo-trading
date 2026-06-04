"""Unit tests for PutWriteVRPStrategy — specifically the scalar-conversion fix.

The original bug: when ``implied_vol`` is a 2D array (n_periods, 1),
``float(iv_array[-1])`` raises "only 0‑dimensional arrays can be converted
to Python scalars" because ``iv_array[-1]`` is a 1‑element 1D array, not a
0‑dimensional scalar.

Fix: use ``.item()`` or explicit indexing to extract the scalar before ``float()``.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.strategies.base import StrategyConfig
from quantcore.strategies.options.put_write import PutWriteVRPStrategy


class TestPutWriteVRPStrategy:
    """Verify the strategy handles array-like implied_vol inputs correctly."""

    def test_on_data_with_2d_implied_vol(self) -> None:
        """Reproduce the original bug: 2D implied_vol (n_periods, 1) must not crash."""
        cfg = StrategyConfig(
            name="options_put_write",
            strategy_type="OPTIONS",
            max_position_notional=None,
            params={"vix_entry": 15},
        )
        strat = PutWriteVRPStrategy(cfg)

        n_periods = 252
        data: dict[str, np.ndarray] = {
            "close": np.linspace(400, 500, n_periods * 6).reshape(n_periods, 6),
            "implied_vol": np.full((n_periods, 1), 0.25),
            "symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        }

        # This must not raise "only 0-dimensional arrays can be converted to Python scalars"
        signals = strat.on_data(data)
        assert isinstance(signals, list)

    def test_on_data_with_1d_implied_vol(self) -> None:
        """1D implied_vol must also work (backward compatibility)."""
        cfg = StrategyConfig(
            name="options_put_write",
            strategy_type="OPTIONS",
            max_position_notional=None,
            params={"vix_entry": 15},
        )
        strat = PutWriteVRPStrategy(cfg)

        n_periods = 252
        data: dict[str, np.ndarray] = {
            "close": np.linspace(400, 500, n_periods * 6).reshape(n_periods, 6),
            "implied_vol": np.full(n_periods, 0.30),
            "symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        }

        signals = strat.on_data(data)
        assert isinstance(signals, list)

    def test_on_data_without_implied_vol(self) -> None:
        """Strategy must gracefully fall back to realized vol when no implied_vol."""
        cfg = StrategyConfig(
            name="options_put_write",
            strategy_type="OPTIONS",
            max_position_notional=None,
            params={"vix_entry": 15},
        )
        strat = PutWriteVRPStrategy(cfg)

        n_periods = 252
        data: dict[str, np.ndarray] = {
            "close": np.linspace(400, 500, n_periods * 6).reshape(n_periods, 6),
            "symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        }

        signals = strat.on_data(data)
        assert isinstance(signals, list)

    def test_on_data_with_empty_close(self) -> None:
        """Empty close array returns empty signals without crashing."""
        cfg = StrategyConfig(
            name="options_put_write",
            strategy_type="OPTIONS",
            max_position_notional=None,
            params={"vix_entry": 15},
        )
        strat = PutWriteVRPStrategy(cfg)

        data: dict[str, np.ndarray] = {
            "close": np.array([]),
            "symbols": [],
        }
        signals = strat.on_data(data)
        assert signals == []

    def test_on_data_with_scalar_implied_vol(self) -> None:
        """Edge case: implied_vol is a 0D (scalar) array."""
        cfg = StrategyConfig(
            name="options_put_write",
            strategy_type="OPTIONS",
            max_position_notional=None,
            params={"vix_entry": 15},
        )
        strat = PutWriteVRPStrategy(cfg)

        n_periods = 252
        data: dict[str, np.ndarray] = {
            "close": np.linspace(400, 500, n_periods * 6).reshape(n_periods, 6),
            "implied_vol": np.array(0.25),
            "symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
        }

        signals = strat.on_data(data)
        assert isinstance(signals, list)

    def test_entry_when_vix_above_threshold(self) -> None:
        """Strategy emits a sell-put signal when VIX > threshold and enough bars."""
        cfg = StrategyConfig(
            name="options_put_write",
            strategy_type="OPTIONS",
            max_position_notional=None,
            params={"vix_entry": 15},
        )
        strat = PutWriteVRPStrategy(cfg)

        n_periods = 252
        data: dict[str, np.ndarray] = {
            "close": np.linspace(400, 500, n_periods * 6).reshape(n_periods, 6),
            "implied_vol": np.full((n_periods, 1), 0.25),  # VIX = 25 > 15
            "symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
            "equity": np.array(5_000_000.0),  # large equity so position size > 0
        }

        signals = strat.on_data(data)
        # Should emit at least one signal to sell a put on SPY
        assert any(s.symbol == "SPY" and s.direction == -1 for s in signals)

    def test_signal_instrument_type_is_option(self) -> None:
        """Phase 3: option signals must carry instrument_type='option'."""
        cfg = StrategyConfig(
            name="options_put_write",
            strategy_type="OPTIONS",
            max_position_notional=None,
            params={"vix_entry": 15},
        )
        strat = PutWriteVRPStrategy(cfg)

        n_periods = 252
        data: dict[str, np.ndarray] = {
            "close": np.linspace(400, 500, n_periods * 6).reshape(n_periods, 6),
            "implied_vol": np.full((n_periods, 1), 0.25),
            "symbols": ["SPY", "QQQ", "IWM", "GLD", "TLT", "DBC"],
            "equity": np.array(5_000_000.0),
        }

        signals = strat.on_data(data)
        entry_signals = [s for s in signals if s.instrument_type == "option" and s.direction == -1]
        assert len(entry_signals) >= 1
        for sig in entry_signals:
            assert sig.metadata.get("contract_id", "")
            assert sig.metadata.get("action") == "SELL_TO_OPEN"
            assert sig.metadata.get("instrument") == "OPTION"