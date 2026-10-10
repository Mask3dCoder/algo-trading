"""Unit tests for real_data_loader -- no network calls."""

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


def _clean_df(
    symbols: list[str] | None = None, n_days: int = 252
) -> pd.DataFrame:
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
    """Universe definitions are correctly structured."""

    def test_keys_and_symbols(self) -> None:
        """UNIVERSES has cta and rp keys with correct symbols."""
        assert "cta" in UNIVERSES
        assert "rp" in UNIVERSES
        assert UNIVERSES["cta"]["symbols"] == ["DBC", "GLD", "IEF", "SPY", "TLT"]
        assert UNIVERSES["rp"]["symbols"] == ["DBC", "GLD", "IEF", "SPY", "TLT"]

    def test_descriptions_and_classes(self) -> None:
        """Each universe has description and asset_classes."""
        for name, info in UNIVERSES.items():
            assert "description" in info, f"{name} missing description"
            assert "asset_classes" in info, f"{name} missing asset_classes"
            assert len(info["symbols"]) == len(info["asset_classes"]), (
                f"{name} symbol/class count mismatch"
            )

    def test_list_universes_runs(self) -> None:
        """list_universes prints without error."""
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


class TestValidateData:
    """Data quality validation (7 checks)."""

    def test_passes_clean_data(self) -> None:
        """validate_data returns empty list for clean DataFrame."""
        df = _clean_df(["SPY", "QQQ", "TLT"], n_days=100)
        errs = validate_data(df, name="test")
        assert len(errs) == 0, f"Unexpected errors: {errs}"

    def test_detects_missing_columns(self) -> None:
        """validate_data catches missing required columns."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        errs = validate_data(df)
        assert len(errs) > 0
        assert any("column" in e.lower() for e in errs)

    def test_detects_duplicates(self) -> None:
        """validate_data flags duplicate (timestamp, symbol) rows."""
        df = _clean_df(["SPY"], n_days=10)
        dup = df.iloc[[-1]].copy()
        df = pd.concat([df, dup], ignore_index=True)
        errs = validate_data(df, expected_symbols=["SPY"])
        assert any("duplicate" in e.lower() for e in errs)

    def test_detects_nan(self) -> None:
        """validate_data flags NaN above threshold."""
        df = _clean_df(["SPY"], n_days=50)
        df.loc[df.index % 5 == 0, "close"] = float("nan")
        errs = validate_data(df, max_nan_pct=0.05)
        assert len(errs) > 0
        assert any("NaN" in e for e in errs)

    def test_detects_missing_symbols(self) -> None:
        """validate_data catches missing expected symbols."""
        df = _clean_df(["SPY"], n_days=50)
        errs = validate_data(df, expected_symbols=["SPY", "QQQ"])
        assert len(errs) > 0
        assert any("QQQ" in e for e in errs)

    def test_empty_dataframe(self) -> None:
        """validate_data returns errors for empty DataFrame."""
        df = pd.DataFrame()
        errs = validate_data(df, name="empty")
        assert len(errs) > 0
        assert any("empty" in e.lower() for e in errs)


class TestLoadUniverse:
    """Data loading with retry."""

    def test_load_universe_unknown_raises(self) -> None:
        """load_universe raises ValueError for unknown universe."""
        with pytest.raises(ValueError, match="Unknown universe"):
            load_universe("unknown", "2020-01-01", "2020-12-31")

    def test_load_universe_delegates(self) -> None:
        """load_universe calls load_bars for each symbol."""
        with mock.patch("scripts.real_data_loader.load_bars") as mock_lb:
            mock_lb.return_value = _clean_df(["SPY"], n_days=50)

            df = load_universe("cta", "2020-01-01", "2020-12-31")

            expected_symbols = UNIVERSES["cta"]["symbols"]
            assert mock_lb.call_count == len(expected_symbols)
            assert not df.empty

    def test_load_universe_retry_on_failure(self) -> None:
        """load_universe retries on failure, then continues."""
        with mock.patch("scripts.real_data_loader.load_bars") as mock_lb:
            # First call raises, second succeeds
            mock_lb.side_effect = [
                Exception("First fail"),
                _clean_df(["SPY"], n_days=50),  # SPY on retry 2
            ]
            # Make subsequent symbols return data too
            def side_effect(sym, *a, **kw):
                if sym == "SPY":
                    return _clean_df(["SPY"], n_days=50)
                if sym in UNIVERSES["cta"]["symbols"]:
                    return _clean_df([sym], n_days=50)
                return pd.DataFrame()
            mock_lb.side_effect = side_effect

            # Should not raise, should return partial data
            with mock.patch("scripts.real_data_loader.time.sleep"):
                df = load_universe("cta", "2020-01-01", "2020-12-31")
            assert not df.empty


class TestFetchAndPrepare:
    """Combined fetch and validate."""

    def test_fetch_and_prepare_success(self) -> None:
        """fetch_and_prepare returns DataFrame on success."""
        with mock.patch("scripts.real_data_loader.load_universe") as mock_lu:
            mock_lu.return_value = _clean_df(
                UNIVERSES["cta"]["symbols"], n_days=50
            )
            df = fetch_and_prepare("cta", "2020-01-01", "2020-12-31")
            assert not df.empty

    def test_fetch_and_prepare_raises_on_bad_data(self) -> None:
        """fetch_and_prepare raises ValueError when data is bad."""
        with mock.patch("scripts.real_data_loader.load_universe") as mock_lu:
            bad_df = _clean_df(["SPY"], n_days=10)
            bad_df.loc[0, "close"] = float("nan")
            mock_lu.return_value = bad_df
            with pytest.raises(ValueError, match="validation failed"):
                fetch_and_prepare("cta", "2020-01-01", "2020-12-31")
