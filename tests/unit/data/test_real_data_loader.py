"""Unit tests for real_data_loader -- all network calls mocked.

Note: New tests also exist at tests/unit/test_real_data_loader.py.
This file is kept for backward compatibility.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
from scripts.real_data_loader import (
    UNIVERSES,
    fetch_and_prepare,
    list_universes,
    load_universe,
    validate_data,
)


def make_clean_df(symbols: list[str] | None = None, n_days: int = 252) -> pd.DataFrame:
    """Build a clean multi-symbol OHLCV DataFrame for testing."""
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
        assert "cta" in UNIVERSES
        assert "rp" in UNIVERSES
        assert len(UNIVERSES["cta"]["symbols"]) >= 4
        assert "SPY" in UNIVERSES["cta"]["symbols"]
        assert "SPY" in UNIVERSES["rp"]["symbols"]

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
        assert "cta" in output
        assert "rp" in output
        assert "SPY" in output


class TestLoadUniverse:
    def test_load_universe_calls_load_combined(self) -> None:
        """load_universe delegates to load_bars with correct args."""
        with mock.patch("scripts.real_data_loader.load_bars") as mock_lb:
            mock_lb.return_value = make_clean_df()

            df = load_universe("cta", "2020-01-01", "2020-12-31")

            expected_symbols = UNIVERSES["cta"]["symbols"]
            assert mock_lb.call_count == len(expected_symbols)
            assert not df.empty

    def test_load_universe_unknown_raises(self) -> None:
        """load_universe raises ValueError for unknown universe."""
        with pytest.raises(ValueError, match="Unknown universe"):
            load_universe("unknown", "2020-01-01", "2020-12-31")

    def test_load_universe_empty_returns_empty(self) -> None:
        """load_universe returns empty DataFrame when no data."""
        with mock.patch("scripts.real_data_loader.load_bars") as mock_lb:
            mock_lb.return_value = pd.DataFrame(
                columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"]
            )

            df = load_universe("cta", "2020-01-01", "2020-12-31")
            assert df.empty

    def test_load_universe_retry_on_failure(self) -> None:
        """load_universe retries on failure, then continues."""
        with mock.patch("scripts.real_data_loader.load_bars") as mock_lb:
            call_count = 0
            def side_effect(sym, *a, **kw):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Transient error")
                return make_clean_df([sym], n_days=50)

            mock_lb.side_effect = side_effect
            with mock.patch("scripts.real_data_loader.time.sleep"):
                df = load_universe("cta", "2020-01-01", "2020-12-31")
            assert not df.empty


class TestValidateData:
    def test_validate_passes_clean_data(self) -> None:
        """validate_data returns empty list for clean DataFrame."""
        df = make_clean_df(["SPY", "QQQ", "TLT"], n_days=100)
        errs = validate_data(df, name="test")
        assert len(errs) == 0

    def test_validate_empty_dataframe(self) -> None:
        """validate_data returns errors for empty DataFrame."""
        df = pd.DataFrame()
        errs = validate_data(df, name="empty")
        assert len(errs) > 0
        assert "empty" in str(errs[0]).lower()

    def test_validate_missing_columns(self) -> None:
        """validate_data rejects DataFrame with missing columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        errs = validate_data(df, name="bad_cols")
        assert len(errs) > 0
        assert any("column" in e.lower() for e in errs)

    def test_validate_detects_nan(self) -> None:
        """validate_data flags excessive NaN values."""
        df = make_clean_df(["SPY"], n_days=50)
        df.loc[df.index % 5 == 0, "close"] = float("nan")
        errs = validate_data(df, name="nan_test", max_nan_pct=0.05)
        assert len(errs) > 0
        assert any("NaN" in e for e in errs)


class TestFetchAndPrepare:
    def test_fetch_and_prepare_success(self) -> None:
        """fetch_and_prepare returns DataFrame on success."""
        with mock.patch("scripts.real_data_loader.load_universe") as mock_lu:
            mock_lu.return_value = make_clean_df(
                UNIVERSES["cta"]["symbols"], n_days=50
            )
            df = fetch_and_prepare("cta", "2020-01-01", "2020-12-31")
            assert not df.empty

    def test_fetch_and_prepare_raises_on_poor_quality(self) -> None:
        """fetch_and_prepare raises ValueError when data is bad."""
        with mock.patch("scripts.real_data_loader.load_universe") as mock_lu:
            bad_df = make_clean_df(["SPY"], n_days=10)
            bad_df.loc[0, "close"] = float("nan")
            mock_lu.return_value = bad_df

            with pytest.raises(ValueError, match="validation failed"):
                fetch_and_prepare(
                    "cta", "2020-01-01", "2020-12-31",
                )
