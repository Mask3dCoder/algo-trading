"""Unit tests for real_data_loader — all network calls mocked."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest

from scripts.real_data_loader import (
    UNIVERSES,
    DataQualityError,
    list_universes,
    load_and_validate,
    load_universe,
    validate_data,
)


def make_clean_df(symbols: list[str] | None = None, n_days: int = 252) -> pd.DataFrame:
    """Build a clean multi-symbol OHLCV DataFrame for testing.

    Args:
        symbols: List of tickers (default SPY, QQQ).
        n_days: Number of business days.

    Returns:
        DataFrame with columns [symbol, timestamp, open, high, low, close, volume].
    """
    if symbols is None:
        symbols = ["SPY", "QQQ"]

    dates = pd.bdate_range("2020-01-01", periods=n_days, freq="B")
    rng = np.random.default_rng(42)

    rows = []
    for sym in symbols:
        price = 100.0
        for ts in dates:
            ret = rng.normal(0.0005, 0.01)
            price = max(price * (1 + ret), 10.0)
            rows.append({
                "symbol": sym,
                "timestamp": ts,
                "open": round(price * (1 + rng.uniform(-0.002, 0.002)), 2),
                "high": round(price * (1 + abs(rng.uniform(0.001, 0.005))), 2),
                "low": round(price * (1 - abs(rng.uniform(0.001, 0.005))), 2),
                "close": round(price, 2),
                "volume": int(rng.integers(1_000_000, 10_000_000)),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df


class TestUniverses:
    def test_universes_defined(self) -> None:
        """UNIVERSES dict contains expected entries."""
        assert "equity" in UNIVERSES
        assert "cta" in UNIVERSES
        assert len(UNIVERSES["equity"]["symbols"]) >= 4
        assert "SPY" in UNIVERSES["equity"]["symbols"]
        assert "SPY" in UNIVERSES["cta"]["symbols"]

    def test_universes_have_descriptions(self) -> None:
        """Each universe has description and asset_classes."""
        for name, info in UNIVERSES.items():
            assert "description" in info, f"{name} missing description"
            assert "symbols" in info, f"{name} missing symbols"
            assert "asset_classes" in info, f"{name} missing asset_classes"

    def test_list_universes_runs(self) -> None:
        """list_universes() prints without error."""
        import io
        import sys

        captured = io.StringIO()
        sys.stdout = captured
        try:
            list_universes()
        finally:
            sys.stdout = sys.__stdout__

        output = captured.getvalue()
        assert "equity" in output
        assert "cta" in output
        assert "SPY" in output


class TestLoadUniverse:
    def test_load_universe_calls_load_combined(self) -> None:
        """load_universe delegates to load_combined with correct args."""
        with mock.patch("scripts.real_data_loader.load_combined") as mock_lc:
            mock_lc.return_value = make_clean_df()

            df = load_universe("equity", "2020-01-01", "2020-12-31")

            mock_lc.assert_called_once_with(
                UNIVERSES["equity"]["symbols"],
                "2020-01-01",
                "2020-12-31",
                "1d",
                use_cache=True,
            )
            assert not df.empty

    def test_load_universe_unknown_raises(self) -> None:
        """load_universe raises ValueError for unknown universe."""
        with pytest.raises(ValueError, match="Unknown universe"):
            load_universe("unknown", "2020-01-01", "2020-12-31")

    def test_load_universe_empty_returns_empty(self) -> None:
        """load_universe returns empty DataFrame when no data."""
        with mock.patch("scripts.real_data_loader.load_combined") as mock_lc:
            mock_lc.return_value = pd.DataFrame(
                columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
            )

            df = load_universe("equity", "2020-01-01", "2020-12-31")
            assert df.empty

    def test_load_universe_passes_use_cache(self) -> None:
        """load_universe forwards use_cache parameter."""
        with mock.patch("scripts.real_data_loader.load_combined") as mock_lc:
            mock_lc.return_value = make_clean_df()

            load_universe("equity", "2020-01-01", "2020-12-31", use_cache=False)

            mock_lc.assert_called_once_with(
                UNIVERSES["equity"]["symbols"],
                "2020-01-01",
                "2020-12-31",
                "1d",
                use_cache=False,
            )


class TestValidateData:
    def test_validate_passes_clean_data(self) -> None:
        """validate_data returns passed=True for clean DataFrame."""
        df = make_clean_df(["SPY", "QQQ", "TLT"], n_days=100)
        result = validate_data(df, name="test")
        assert result["passed"] is True
        assert result["n_symbols"] == 3
        assert result["total_bars"] == 300
        assert len(result["errors"]) == 0

    def test_validate_empty_dataframe(self) -> None:
        """validate_data returns errors for empty DataFrame."""
        df = pd.DataFrame()
        result = validate_data(df, name="empty")
        assert result["passed"] is False
        assert "empty" in str(result["errors"]).lower() or "empty" in str(result["errors"])

    def test_validate_missing_columns(self) -> None:
        """validate_data rejects DataFrame with missing columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = validate_data(df, name="bad_cols")
        assert result["passed"] is False
        assert any("column" in e.lower() for e in result["errors"])

    def test_validate_detects_nan(self) -> None:
        """validate_data flags excessive NaN values."""
        df = make_clean_df(["SPY"], n_days=50)
        df.loc[df.index % 5 == 0, "close"] = float("nan")
        result = validate_data(df, name="nan_test", max_nan_pct=0.05)
        assert result["passed"] is False
        assert any("NaN" in e for e in result["errors"])

    def test_validate_reports_date_range(self) -> None:
        """validate_data reports correct date range."""
        df = make_clean_df(["SPY"], n_days=100)
        result = validate_data(df, name="range_test")
        assert result["date_range"] is not None
        start_str, end_str = result["date_range"]
        assert start_str == "2020-01-01"
        assert end_str >= "2020-04-01"

    def test_validate_produces_symbol_results(self) -> None:
        """validate_data populates per-symbol results."""
        df = make_clean_df(["SPY", "QQQ"], n_days=50)
        result = validate_data(df, name="sym_check")
        assert "SPY" in result["symbol_results"]
        assert "QQQ" in result["symbol_results"]
        assert result["symbol_results"]["SPY"]["bars"] == 50
        assert result["symbol_results"]["QQQ"]["bars"] == 50


class TestLoadAndValidate:
    def test_load_and_validate_success(self) -> None:
        """load_and_validate returns DataFrame on success."""
        with mock.patch("scripts.real_data_loader.load_universe") as mock_lu:
            mock_lu.return_value = make_clean_df(["SPY"], n_days=50)
            df = load_and_validate("equity", "2020-01-01", "2020-12-31")
            assert not df.empty

    def test_load_and_validate_raises_on_poor_quality(self) -> None:
        """load_and_validate raises DataQualityError when data is bad."""
        with mock.patch("scripts.real_data_loader.load_universe") as mock_lu:
            bad_df = make_clean_df(["SPY"], n_days=10)
            bad_df.loc[0, "close"] = float("nan")
            mock_lu.return_value = bad_df

            with pytest.raises(DataQualityError):
                load_and_validate(
                    "equity", "2020-01-01", "2020-12-31",
                    max_nan_pct=0.01,
                )

    def test_data_quality_error_is_value_error(self) -> None:
        """DataQualityError is a subclass of ValueError."""
        assert issubclass(DataQualityError, ValueError)
