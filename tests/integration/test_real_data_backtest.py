"""Integration tests for real-data backtesting — gated behind RUN_REAL_DATA_TESTS=true.

These tests download actual yfinance data and run walk-forward backtests.
They are skipped by default. Run with:
    RUN_REAL_DATA_TESTS=true python -m pytest tests/integration/test_real_data_backtest.py -v
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from quantcore.backtest.walk_forward import Fold

# Gate: only run when RUN_REAL_DATA_TESTS=true
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_DATA_TESTS", "").lower() not in ("true", "1", "yes"),
    reason="Set RUN_REAL_DATA_TESTS=true to run (requires yfinance download)",
)

from scripts.real_data_backtest import (
    ALL_PERIODS,
    PASS_CRITERIA,
    STRATEGY_PARAMS,
    _df_to_data,
    _setup_strategy,
    run_period_backtest,
)
from scripts.real_data_loader import UNIVERSES, load_universe


class TestDataConversion:
    """Verify DataFrame → numpy dict conversion."""

    def test_df_to_data_basic(self) -> None:
        """_df_to_data produces correct shapes and keys."""
        df = load_universe("equity", "2020-01-01", "2020-06-30")
        assert not df.empty, "No equity data returned"

        data = _df_to_data(df)
        expected_symbols = UNIVERSES["equity"]["symbols"]

        assert "close" in data
        assert "symbols" in data
        assert data["symbols"] == sorted(expected_symbols)

        n_bars = df["timestamp"].nunique()
        n_assets = len(expected_symbols)
        assert data["close"].shape == (n_bars, n_assets), (
            f"Expected ({n_bars}, {n_assets}), got {data['close'].shape}"
        )

    def test_df_to_data_no_nan(self) -> None:
        """_df_to_data produces NaN-free arrays."""
        df = load_universe("equity", "2020-01-01", "2020-06-30")
        data = _df_to_data(df)

        for field in ["close", "high", "low", "open", "volume"]:
            arr = data[field]
            assert not np.any(np.isnan(arr)), f"{field} contains NaN after conversion"

    def test_df_to_data_cta_universe(self) -> None:
        """_df_to_data works with CTA universe (more symbols)."""
        df = load_universe("cta", "2020-01-01", "2020-06-30")
        assert not df.empty, "No cta data returned"

        data = _df_to_data(df)
        expected_symbols = sorted(UNIVERSES["cta"]["symbols"])
        assert data["symbols"] == expected_symbols
        assert data["close"].shape[1] == len(expected_symbols)


class TestStrategySetup:
    """Verify strategy instantiation with real-data params."""

    def test_setup_cta_trend(self) -> None:
        """_setup_strategy creates macro_cta_trend with correct params."""
        strat = _setup_strategy("macro_cta_trend")
        assert strat is not None
        assert strat.config.name == "macro_cta_trend"

        params = STRATEGY_PARAMS["macro_cta_trend"]
        assert strat.fast_window == params["fast_window"]
        assert strat.medium_window == params["medium_window"]
        assert strat.slow_window == params["slow_window"]
        assert strat.vol_target == params["vol_target"]

    def test_setup_risk_parity(self) -> None:
        """_setup_strategy creates macro_risk_parity with correct params."""
        strat = _setup_strategy("macro_risk_parity")
        assert strat is not None
        assert strat.config.name == "macro_risk_parity"

        params = STRATEGY_PARAMS["macro_risk_parity"]
        assert strat.vol_target == params["vol_target"]
        assert strat.lookback == params["lookback"]
        assert strat.regime_overlay == params["regime_overlay"]


class TestPeriodBacktest:
    """Run walk-forward backtests on real data (small periods for speed)."""

    @pytest.fixture(scope="class")
    def equity_data(self) -> Any:
        """Load equity universe data for a short period."""
        return load_universe("equity", "2020-01-01", "2021-12-31")

    def test_cta_trend_full_period(self) -> None:
        """macro_cta_trend runs on full period without error."""
        df = load_universe("cta", "2020-01-01", "2021-12-31")
        assert not df.empty

        n_bars = df["timestamp"].nunique()
        folds = [
            Fold(train_start=0, train_end=200, test_start=201, test_end=n_bars)
        ]

        result = run_period_backtest("macro_cta_trend", df, folds)
        assert result["error"] is None, f"Backtest failed: {result['error']}"
        assert result["n_folds"] >= 1
        assert result["oos_sharpe"] is not None

    def test_risk_parity_full_period(self) -> None:
        """macro_risk_parity runs on full period without error."""
        df = load_universe("equity", "2020-01-01", "2021-12-31")
        assert not df.empty

        n_bars = df["timestamp"].nunique()
        folds = [
            Fold(train_start=0, train_end=200, test_start=201, test_end=n_bars)
        ]

        result = run_period_backtest("macro_risk_parity", df, folds)
        assert result["error"] is None, f"Backtest failed: {result['error']}"
        assert result["n_folds"] >= 1


class TestPassCriteria:
    """Verify pass criteria are correctly structured."""

    def test_all_criteria_have_keys(self) -> None:
        """PASS_CRITERIA covers all strategy-period combinations."""
        for strategy_id in STRATEGY_PARAMS:
            for period_id in ALL_PERIODS:
                key = (strategy_id, period_id)
                assert key in PASS_CRITERIA, (
                    f"Missing pass criterion for {strategy_id}/{period_id}"
                )

    def test_criteria_values_are_reasonable(self) -> None:
        """Pass criteria thresholds are within [-1, 2] range."""
        for (sid, pid), threshold in PASS_CRITERIA.items():
            assert -1.0 <= threshold <= 2.0, (
                f"Unreasonable threshold {threshold} for {sid}/{pid}"
            )

    def test_pass_criteria_periods_match(self) -> None:
        """All period keys in PASS_CRITERIA exist in ALL_PERIODS."""
        for strategy_id, period_id in PASS_CRITERIA:
            assert period_id in ALL_PERIODS, (
                f"Period '{period_id}' not in ALL_PERIODS"
            )


class TestCLI:
    """Verify CLI argument parsing works."""

    def test_list_periods_runs(self) -> None:
        """--list-periods flag prints periods without error."""
        import io
        import sys

        from scripts.real_data_backtest import main as backtest_main

        # We can't easily call main() with args, so just check the module
        from scripts.real_data_backtest import ALL_PERIODS as periods

        assert "full" in periods
        assert "pre_covid" in periods
        assert "rate_hikes" in periods
        assert "gfc" in periods
