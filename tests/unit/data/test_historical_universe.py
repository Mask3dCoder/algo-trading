"""Tests for HistoricalUniverseLoader — PIT constituent lookup, CSV loading, edge cases."""

from __future__ import annotations

import datetime
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from quantcore.data.sources.historical_universe import (
    HistoricalUniverseLoader,
    _BUILTINS,
    _HISTORY_CSV,
    _loader,
    get_loader,
)


class TestHistoricalUniverseLoader:
    def test_init_loads_builtins(self):
        loader = HistoricalUniverseLoader()
        indices = loader.available_indices()
        assert "SP500" in indices
        assert "NASDAQ100" in indices

    def test_get_constituents_sp500_2016(self):
        loader = HistoricalUniverseLoader()
        result = loader.get_constituents("SP500", datetime.date(2016, 3, 1))
        expected = _BUILTINS["SP500"]["2016-01-01"]
        assert set(result) == set(expected)

    def test_get_constituents_nasdaq100_2018(self):
        loader = HistoricalUniverseLoader()
        result = loader.get_constituents("NASDAQ100", datetime.date(2018, 6, 15))
        expected = _BUILTINS["NASDAQ100"]["2018-01-01"]
        assert set(result) == set(expected)

    def test_get_constituents_exact_date_match(self):
        loader = HistoricalUniverseLoader()
        result = loader.get_constituents("SP500", datetime.date(2012, 1, 1))
        expected = _BUILTINS["SP500"]["2012-01-01"]
        assert set(result) == set(expected)

    def test_get_constituents_future_date_returns_latest(self):
        loader = HistoricalUniverseLoader()
        result = loader.get_constituents("SP500", datetime.date(2030, 1, 1))
        expected = _BUILTINS["SP500"]["2024-01-01"]
        assert set(result) == set(expected)

    def test_get_constituents_before_first_snapshot(self):
        loader = HistoricalUniverseLoader()
        result = loader.get_constituents("SP500", datetime.date(1990, 1, 1))
        assert result == []

    def test_get_constituents_unknown_index(self):
        loader = HistoricalUniverseLoader()
        result = loader.get_constituents("RUSSELL3000", datetime.date(2020, 1, 1))
        assert result == []

    def test_available_indices(self):
        loader = HistoricalUniverseLoader()
        indices = loader.available_indices()
        assert "SP500" in indices
        assert "NASDAQ100" in indices
        assert len(indices) >= 2

    def test_available_dates_sp500(self):
        loader = HistoricalUniverseLoader()
        dates = loader.available_dates("SP500")
        assert "2008-01-01" in dates
        assert "2024-01-01" in dates
        assert len(dates) == 6

    def test_available_dates_unknown_index(self):
        loader = HistoricalUniverseLoader()
        dates = loader.available_dates("NONEXISTENT")
        assert dates == []

    def test_load_from_csv_returns_count(self):
        loader = HistoricalUniverseLoader()
        assert _HISTORY_CSV.exists(), f"CSV not found at {_HISTORY_CSV}"
        n = loader.load_from_csv(_HISTORY_CSV)
        assert n > 0
        assert n == 24  # 24 constituent changes in universe_history.csv

    def test_load_from_csv_nonexistent_path(self):
        loader = HistoricalUniverseLoader()
        n = loader.load_from_csv(Path("/nonexistent/universe.csv"))
        assert n == 0

    def test_load_from_csv_empty_path_uses_default(self):
        loader = HistoricalUniverseLoader()
        n = loader.load_from_csv()
        assert n > 0

    def test_load_from_csv_adds_sp500_changes_key(self):
        loader = HistoricalUniverseLoader()
        loader.load_from_csv(_HISTORY_CSV)
        assert "SP500_CHANGES" in loader.available_indices()

    def test_csv_loaded_via_get_loader(self):
        loader = get_loader()
        indices = loader.available_indices()
        assert "SP500" in indices
        assert "NASDAQ100" in indices

    def test_sp500_2024_has_nvda(self):
        loader = HistoricalUniverseLoader()
        result = loader.get_constituents("SP500", datetime.date(2024, 6, 1))
        assert "NVDA" in result
        assert "AAPL" in result
        assert "MSFT" in result

    def test_sp500_2008_has_ge(self):
        loader = HistoricalUniverseLoader()
        result = loader.get_constituents("SP500", datetime.date(2008, 6, 1))
        assert "GE" in result
        assert "C" in result
        assert "AIG" in result


class TestGetLoaderSingleton:
    def test_get_loader_returns_same_instance(self):
        a = get_loader()
        b = get_loader()
        assert a is b

    def test_get_loader_initializes_lazily(self):
        import quantcore.data.sources.historical_universe as hu

        original = hu._loader
        try:
            hu._loader = None
            loader = hu.get_loader()
            assert loader is not None
            assert hu._loader is loader
        finally:
            hu._loader = original

    def test_get_loader_skips_missing_json(self):
        loader = get_loader()
        assert loader is not None
        assert "SP500" in loader.available_indices()


class TestCSVIntegrity:
    def test_csv_has_required_columns(self):
        import csv

        with open(_HISTORY_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
        assert fieldnames is not None
        assert "date" in fieldnames
        assert "action" in fieldnames
        assert "ticker" in fieldnames

    def test_csv_dates_are_valid(self):
        import csv

        with open(_HISTORY_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    datetime.date.fromisoformat(row["date"])
                except (ValueError, TypeError):
                    pytest.fail(f"Invalid date: {row.get('date')}")

    def test_csv_actions_are_add_or_remove(self):
        import csv

        with open(_HISTORY_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["action"] in ("add", "remove"), f"Unexpected action: {row['action']}"

    def test_fb_meta_transition_present(self):
        import csv

        with open(_HISTORY_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        fb_remove = any(r["ticker"] == "FB" and r["action"] == "remove" for r in rows)
        meta_add = any(r["ticker"] == "META" and r["action"] == "add" for r in rows)
        assert fb_remove, "FB removal (renamed to META) must be in CSV"
        assert meta_add, "META addition must be in CSV"

    def test_tsla_added_before_2021(self):
        import csv

        with open(_HISTORY_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["ticker"] == "TSLA":
                    d = datetime.date.fromisoformat(row["date"])
                    assert d <= datetime.date(2021, 1, 1)
                    assert row["action"] == "add"
                    return
        pytest.fail("TSLA addition not found in CSV")