"""Unit tests for yfinance_loader — all network calls mocked."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd

from quantcore.data.yfinance_loader import (
    _cache_path,
    _normalize_columns,
    load_bars,
    load_combined,
    load_multiple,
)


class TestCachePath:
    def test_cache_path_format(self) -> None:
        path = _cache_path("SPY", "2023-01-01", "2024-01-01", "1d")
        assert "SPY" in str(path)
        assert "2023-01-01" in str(path)
        assert "2024-01-01" in str(path)
        assert "1d" in str(path)
        assert str(path).endswith(".parquet")

    def test_cache_path_replaces_slashes(self) -> None:
        path = _cache_path("BTC/USD", "2023-01-01", "2024-01-01", "1d")
        assert "/" not in path.name
        assert "_" in path.name


class TestNormalizeColumns:
    def test_normalizes_yfinance_output(self) -> None:
        raw = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "Open": [380.0, 382.0],
            "High": [385.0, 386.0],
            "Low": [379.0, 381.0],
            "Close": [384.0, 385.0],
            "Volume": [50000000, 52000000],
        })
        raw = raw.set_index("Date")

        result = _normalize_columns(raw, "SPY")
        assert list(result.columns) == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
        assert (result["symbol"] == "SPY").all()
        assert len(result) == 2
        assert result["open"].iloc[0] == 380.0

    def test_drops_adj_close_column(self) -> None:
        raw = pd.DataFrame({
            "Date": pd.to_datetime(["2023-01-03"]),
            "Open": [380.0], "High": [385.0], "Low": [379.0],
            "Close": [384.0], "Volume": [50000000], "Adj Close": [383.0],
        })
        raw = raw.set_index("Date")

        result = _normalize_columns(raw, "SPY")
        assert "adj_close" not in result.columns
        assert "adj close" not in result.columns


class TestLoadBars:
    def test_load_bars_cached(self, tmp_path, monkeypatch) -> None:
        """When cache exists, load_bars returns it without calling yfinance."""
        monkeypatch.setattr(
            "quantcore.data.yfinance_loader.CACHE_DIR",
            tmp_path / "cache",
        )

        cache_file = tmp_path / "cache" / "SPY_2023-01-01_2024-01-01_1d.parquet"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "symbol": ["SPY", "SPY"],
            "timestamp": pd.to_datetime(["2023-01-03", "2023-01-04"]),
            "open": [380.0, 382.0], "high": [385.0, 386.0],
            "low": [379.0, 381.0], "close": [384.0, 385.0],
            "volume": [50000000, 52000000],
        }).to_parquet(cache_file, index=False)

        df = load_bars("SPY", "2023-01-01", "2024-01-01", interval="1d")
        assert len(df) == 2
        assert df["symbol"].iloc[0] == "SPY"

    def test_load_bars_empty_result(self, monkeypatch) -> None:
        """When yfinance returns empty, load_bars returns empty DataFrame."""
        import yfinance as yf

        mock_ticker = mock.MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        monkeypatch.setattr(yf, "Ticker", lambda sym: mock_ticker)

        # Skip cache
        with mock.patch.object(Path, "exists", return_value=False):
            df = load_bars("INVALID", "2023-01-01", "2024-01-01", use_cache=False)
            assert df.empty
            assert list(df.columns) == ["symbol", "timestamp", "open", "high", "low", "close", "volume"]


class TestLoadMultiple:
    def test_load_multiple_skips_failures(self) -> None:
        """load_multiple should skip symbols that fail and return successes."""
        # Mock load_bars to fail for one symbol, succeed for another
        def mock_load(symbol, start, end, interval, use_cache):
            if symbol == "FAIL":
                raise RuntimeError("connection error")
            return pd.DataFrame({
                "symbol": [symbol],
                "timestamp": pd.to_datetime(["2023-01-03"]),
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.5], "volume": [1000000],
            })

        with mock.patch("quantcore.data.yfinance_loader.load_bars", side_effect=mock_load):
            result = load_multiple(["SPY", "FAIL", "QQQ"], "2023-01-01", "2024-01-01")
            assert "SPY" in result
            assert "QQQ" in result
            assert "FAIL" not in result
            assert len(result) == 2

    def test_load_multiple_empty(self) -> None:
        """load_multiple returns empty dict when all symbols fail."""
        def mock_load(symbol, start, end, interval, use_cache):
            raise RuntimeError("fail")

        with mock.patch("quantcore.data.yfinance_loader.load_bars", side_effect=mock_load):
            result = load_multiple(["X", "Y"], "2023-01-01", "2024-01-01")
            assert result == {}


class TestLoadCombined:
    def test_load_combined_concatenates(self) -> None:
        """load_combined should concatenate and sort by timestamp + symbol."""
        def mock_load(symbol, start, end, interval, use_cache):
            return pd.DataFrame({
                "symbol": [symbol],
                "timestamp": pd.to_datetime(["2023-01-03"]),
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.5], "volume": [1000000],
            })

        with mock.patch("quantcore.data.yfinance_loader.load_bars", side_effect=mock_load):
            df = load_combined(["SPY", "QQQ"], "2023-01-01", "2024-01-01")
            assert len(df) == 2
            assert set(df["symbol"].unique()) == {"SPY", "QQQ"}

    def test_load_combined_all_fail(self) -> None:
        """load_combined returns empty DataFrame when all symbols fail."""
        def mock_load(symbol, start, end, interval, use_cache):
            raise RuntimeError("fail")

        with mock.patch("quantcore.data.yfinance_loader.load_bars", side_effect=mock_load):
            df = load_combined(["X"], "2023-01-01", "2024-01-01")
            assert df.empty
