"""Unit tests for quantcore/research/splitter/data_splitter.py."""

from __future__ import annotations

import pytest
from pandas import DataFrame, DatetimeIndex

from quantcore.research.splitter.data_splitter import (
    DataSplitter,
    SPLIT_RANGES,
    IS_START,
    IS_END,
    VAL_START,
    VAL_END,
    VAULT_START,
    VAULT_END,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def full_df() -> DataFrame:
    """Synthetic DataFrame spanning IS + Val + Vault dates."""
    idx = DatetimeIndex(
        [
            "2015-06-01",  # before IS — outside all splits
            "2016-06-15",  # IS
            "2017-03-10",  # IS
            "2018-11-30",  # IS
            "2019-07-04",  # IS
            "2020-02-29",  # IS (leap year)
            "2021-09-01",  # IS (last day)
            "2022-01-03",  # Val
            "2022-06-15",  # Val
            "2023-10-31",  # Val (last day)
            "2024-02-01",  # Vault
            "2025-07-15",  # Vault
            "2026-04-30",  # Vault (last accessible day)
            "2026-07-01",  # after vault — outside all splits
        ]
    )
    return DataFrame({"value": range(len(idx))}, index=idx)


@pytest.fixture
def is_splitter() -> DataSplitter:
    return DataSplitter(current_phase="is")


@pytest.fixture
def val_splitter() -> DataSplitter:
    return DataSplitter(current_phase="val")


@pytest.fixture
def vault_splitter() -> DataSplitter:
    return DataSplitter(current_phase="vault")


# ── Phase construction ──────────────────────────────────────────────────────────

class TestPhaseConstruction:
    def test_splits_cover_full_hypothesis_001_daterange(self):
        assert IS_START == "2016-01-01"
        assert IS_END == "2021-12-31"
        assert VAL_START == "2022-01-01"
        assert VAL_END == "2023-12-31"
        assert VAULT_START == "2024-01-01"
        assert VAULT_END == "2026-06-01"

    def test_valid_phases_are_is_val_vault(self):
        assert DataSplitter.VALID_PHASES == ["is", "val", "vault"]

    def test_unknown_phase_raises_on_init(self):
        with pytest.raises(ValueError, match="Unknown phase"):
            DataSplitter(current_phase="oot")  # type: ignore[arg-type]


# ── Slice correctness ──────────────────────────────────────────────────────────

class TestSliceCorrectness:
    def test_is_splitter_get_returns_only_is_rows(self, is_splitter, full_df):
        result = is_splitter.get(full_df, "is")
        assert len(result) == 6  # 2016-06-15 → 2021-09-01 (6 dates in IS range)
        dates = result.index.to_series().dt.strftime("%Y-%m-%d")
        assert all(dates >= "2016-01-01") and all(dates <= "2021-12-31")

    def test_val_splitter_get_returns_only_val_rows(self, val_splitter, full_df):
        result = val_splitter.get(full_df, "val")
        assert len(result) == 3
        dates = result.index.to_series().dt.strftime("%Y-%m-%d")
        assert all(dates >= "2022-01-01") and all(dates <= "2023-12-31")

    def test_vault_splitter_get_returns_only_vault_rows(self, vault_splitter, full_df):
        result = vault_splitter.get(full_df, "vault")
        assert len(result) == 3
        dates = result.index.to_series().dt.strftime("%Y-%m-%d")
        assert all(dates >= "2024-01-01") and all(dates <= "2026-06-01")

    def test_get_excludes_rows_outside_split_range(self, is_splitter, full_df):
        """Rows from 2015 and 2026 must be excluded."""
        result = is_splitter.get(full_df, "is")
        dates_str = result.index.to_series().dt.strftime("%Y-%m-%d")
        assert "2015-06-01" not in dates_str.values
        assert "2026-04-30" not in dates_str.values


# ── Quarantine enforcement ─────────────────────────────────────────────────────

class TestQuarantineEnforcement:
    def test_is_phase_cannot_access_val_raises(self, is_splitter, full_df):
        with pytest.raises(RuntimeError, match="VAULT BREACH"):
            is_splitter.get(full_df, "val")

    def test_is_phase_cannot_access_vault_raises(self, is_splitter, full_df):
        with pytest.raises(RuntimeError, match="VAULT BREACH"):
            is_splitter.get(full_df, "vault")

    def test_val_phase_cannot_access_vault_raises(self, val_splitter, full_df):
        with pytest.raises(RuntimeError, match="VAULT BREACH"):
            val_splitter.get(full_df, "vault")

    def test_vault_phase_can_access_is(self, vault_splitter, full_df):
        result = vault_splitter.get(full_df, "is")  # must not raise
        assert len(result) == 6

    def test_vault_phase_can_access_val(self, vault_splitter, full_df):
        result = vault_splitter.get(full_df, "val")  # must not raise
        assert len(result) == 3

    def test_val_phase_can_access_is(self, val_splitter, full_df):
        result = val_splitter.get(full_df, "is")  # must not raise
        assert len(result) == 6

    def test_access_future_phase_error_message_contains_attempted_phase(
        self, is_splitter, full_df
    ):
        with pytest.raises(RuntimeError, match="VAL"):
            is_splitter.get(full_df, "val")


# ── Advance ─────────────────────────────────────────────────────────────────────

class TestAdvance:
    def test_advance_returns_new_splitter(self, is_splitter):
        val_splitter = is_splitter.advance()
        assert val_splitter.phase == "val"
        assert is_splitter.phase == "is"  # original unchanged

    def test_advance_twice_reaches_vault(self, is_splitter):
        val = is_splitter.advance()
        vault = val.advance()
        assert vault.phase == "vault"

    def test_advance_from_vault_raises(self, vault_splitter):
        with pytest.raises(RuntimeError, match="one-shot"):
            vault_splitter.advance()


# ── is_accessible ──────────────────────────────────────────────────────────────

class TestIsAccessible:
    def test_is_in_is_phase(self, is_splitter):
        assert is_splitter.is_accessible("is") is True
        assert is_splitter.is_accessible("val") is False
        assert is_splitter.is_accessible("vault") is False

    def test_is_in_val_phase(self, val_splitter):
        assert val_splitter.is_accessible("is") is True
        assert val_splitter.is_accessible("val") is True
        assert val_splitter.is_accessible("vault") is False

    def test_is_in_vault_phase(self, vault_splitter):
        assert vault_splitter.is_accessible("is") is True
        assert vault_splitter.is_accessible("val") is True
        assert vault_splitter.is_accessible("vault") is True

    def test_unknown_phase_returns_false(self, is_splitter):
        assert is_splitter.is_accessible("oot") is False


# ── report ─────────────────────────────────────────────────────────────────────

class TestReport:
    def test_report_includes_all_phases(self, is_splitter):
        r = is_splitter.report()
        assert r["current_phase"] == "is"
        lines = r["phases"]
        assert len(lines) == 3
        assert any(">>> IS" in line and "accessible" in line for line in lines)
        assert any("VAL" in line and "LOCKED" in line for line in lines)

    def test_report_in_vault_shows_all_accessible(self, vault_splitter):
        r = vault_splitter.report()
        assert all("accessible" in line for line in r["phases"])