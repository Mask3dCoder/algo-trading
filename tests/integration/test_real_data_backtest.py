"""Integration tests for real-data backtest -- gated behind RUN_REAL_DATA_TESTS=true.

These tests download actual yfinance data and run walk-forward backtests.
They are skipped by default. Run with:
    RUN_REAL_DATA_TESTS=true python -m pytest tests/integration/test_real_data_backtest.py -v
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Gate: only run when RUN_REAL_DATA_TESTS=true
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_REAL_DATA_TESTS", "").lower() not in ("true", "1", "yes"),
    reason="Set RUN_REAL_DATA_TESTS=true to run (requires yfinance download)",
)

from scripts.real_data_backtest import (
    FULL_PERIOD,
    REGIME_PERIODS,
    SHOWSTOPPER_THRESHOLDS,
    STRATEGY_PARAMS,
    _df_to_data,
    _setup_strategy,
    check_showstoppers,
    run_full_period_backtest,
    run_regime_backtest,
)
from scripts.real_data_loader import UNIVERSES, load_universe


class TestDataConversion:
    """DataFrame to numpy dict conversion from real data."""

    @pytest.fixture(scope="class")
    def cta_data(self) -> Any:
        """Load CTA universe data."""
        return load_universe("cta", "2020-01-01", "2020-06-30")

    def test_df_to_data_basic(self, cta_data: Any) -> None:
        """_df_to_data produces correct shapes and keys."""
        df = cta_data
        assert not df.empty, "No cta data returned"

        data = _df_to_data(df)
        expected_symbols = sorted(UNIVERSES["cta"]["symbols"])

        assert "close" in data
        assert "symbols" in data
        assert data["symbols"] == expected_symbols

        n_bars = df["timestamp"].nunique()
        n_assets = len(expected_symbols)
        assert data["close"].shape == (n_bars, n_assets), (
            f"Expected ({n_bars}, {n_assets}), got {data['close'].shape}"
        )

    def test_df_to_data_no_nan(self, cta_data: Any) -> None:
        """_df_to_data produces NaN-free arrays after forward-fill."""
        data = _df_to_data(cta_data)

        for field in ["close", "high", "low", "open", "volume"]:
            arr = data[field]
            assert not np.any(np.isnan(arr)), f"{field} contains NaN after conversion"

    def test_df_to_data_rp_universe(self) -> None:
        """_df_to_data works with RP universe."""
        df = load_universe("rp", "2020-01-01", "2020-06-30")
        assert not df.empty, "No rp data returned"

        data = _df_to_data(df)
        expected_symbols = sorted(UNIVERSES["rp"]["symbols"])
        assert data["symbols"] == expected_symbols
        assert data["close"].shape[1] == len(expected_symbols)


class TestStrategySetup:
    """Strategy instantiation with real-data params."""

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
        assert strat.rebalance_freq == params["rebalance_freq"]


class TestFullPeriodBacktest:
    """Full period anchored walk-forward on real data."""

    def test_cta_trend_full_period(self) -> None:
        """macro_cta_trend runs full period without error."""
        df = load_universe(
            "cta", FULL_PERIOD["start"], FULL_PERIOD["end"]
        )
        assert not df.empty

        result = run_full_period_backtest("macro_cta_trend", df)
        assert result["error"] is None, f"Backtest failed: {result['error']}"
        assert result["n_folds"] >= 1
        assert result["oos_sharpe"] is not None
        assert result["oos_sortino"] is not None

    def test_risk_parity_full_period(self) -> None:
        """macro_risk_parity runs full period without error."""
        df = load_universe(
            "rp", FULL_PERIOD["start"], FULL_PERIOD["end"]
        )
        assert not df.empty

        result = run_full_period_backtest("macro_risk_parity", df)
        assert result["error"] is None, f"Backtest failed: {result['error']}"
        assert result["n_folds"] >= 1
        assert result["oos_sharpe"] is not None


class TestRegimeBacktest:
    """Regime-specific backtests on real data."""

    def test_covid_regime_cta(self) -> None:
        """macro_cta_trend runs COVID regime with warmup."""
        regime = REGIME_PERIODS["covid"]
        df = load_universe("cta", regime["warmup_start"], regime["end"])
        assert not df.empty

        all_dates = sorted(df["timestamp"].unique())
        regime_start = next(
            i for i, d in enumerate(all_dates) if d >= pd.Timestamp(regime["start"])
        )

        result = run_regime_backtest("macro_cta_trend", df, regime_start)
        assert result["error"] is None, f"Regime backtest failed: {result['error']}"
        assert result["oos_sharpe"] is not None
        assert result["n_oos_bars"] >= 3

    def test_gfc_regime_rp(self) -> None:
        """macro_risk_parity runs GFC regime with warmup."""
        regime = REGIME_PERIODS["gfc"]
        df = load_universe("rp", regime["warmup_start"], regime["end"])
        assert not df.empty

        all_dates = sorted(df["timestamp"].unique())
        regime_start = next(
            i for i, d in enumerate(all_dates) if d >= pd.Timestamp(regime["start"])
        )

        result = run_regime_backtest("macro_risk_parity", df, regime_start)
        assert result["error"] is None, f"Regime backtest failed: {result['error']}"
        assert result["oos_sharpe"] is not None


class TestShowstopperDetection:
    """Showstopper threshold checking."""

    def test_showstopper_thresholds_exist(self) -> None:
        """SHOWSTOPPER_THRESHOLDS covers all strategy-period combos."""
        for strategy_id in STRATEGY_PARAMS:
            for period_id in list(REGIME_PERIODS.keys()) + ["full"]:
                key = (strategy_id, period_id)
                assert key in SHOWSTOPPER_THRESHOLDS, (
                    f"Missing showstopper threshold for {strategy_id}/{period_id}"
                )

    def test_showstopper_check_works(self) -> None:
        """check_showstoppers detects violations correctly."""
        results = [
            {
                "strategy": "macro_cta_trend",
                "period": "full",
                "oos_sharpe": -0.5,
            },
            {
                "strategy": "macro_risk_parity",
                "period": "full",
                "oos_sharpe": 0.5,
            },
        ]
        violations = check_showstoppers(results)
        # macro_cta_trend/full has showstopper at 0.00, -0.5 < 0.0 => violation
        cta_violations = [
            v for v in violations if "macro_cta_trend" in v and "full" in v
        ]
        assert len(cta_violations) > 0
