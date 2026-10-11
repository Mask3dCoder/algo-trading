"""Tests for the cross-sectional experiment preflight.

The preflight is the last thing standing between a fabricated experiment and a
real one, so its own behaviour is tested: it must FAIL on missing inputs rather
than degrade to a warning, and it must not be satisfiable by flipping a flag.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from quantcore.data.pit.market.store import ParquetPitStore  # noqa: E402
from quantcore.research.factor_preflight import (  # type: ignore[import-not-found]
    BOOTSTRAP_BLOCK_LEN,
    BOOTSTRAP_REPS,
    COST_TOTAL_BPS,
    EXPECTED_PANEL_SIZE,
    LIQUIDITY_MIN_MARKET_CAP,
    MOMENTUM_LOOKBACK,
    MOMENTUM_SKIP,
    PreflightReport,
    check_benchmark,
    check_liquidity_filter,
    check_purging,
    check_statistical_instruments,
    check_universe_panels,
    run_preflight,
)


def _bars(sym: str, n: int = 400) -> pd.DataFrame:
    flat = [100.0] * n
    return pd.DataFrame(
        {
            "symbol": sym,
            "ts": pd.bdate_range("2018-01-02", periods=n, tz="UTC"),
            "open": flat,
            "high": flat,
            "low": flat,
            "close": flat,
            "volume": flat,
            "vwap": flat,
            "transactions": flat,
        }
    )


# --- fail-closed semantics --------------------------------------------------
def test_report_with_no_checks_blocks_execution() -> None:
    """An empty report must not read as 'nothing to complain about'."""
    assert PreflightReport().can_execute is False


def test_unverified_check_blocks_execution() -> None:
    r = PreflightReport()
    r.add("thing", "UNVERIFIED", "cannot be determined")
    assert r.can_execute is False


def test_all_passing_report_allows_execution() -> None:
    r = PreflightReport()
    r.add("thing", "PASS", "verified")
    assert r.can_execute is True


# --- liquidity filter: the check that must not be soft-baked ---------------
def test_liquidity_fails_when_market_cap_unavailable(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    result = check_liquidity_filter(store)
    assert len(result.failures) == 1
    assert result.failures[0].name == "liquidity_market_cap"
    detail = result.failures[0].detail
    assert "not computable" in detail
    assert "amendment" in detail  # names the legitimate unblock routes


def test_liquidity_passes_when_market_cap_present(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    df = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "name": ["Apple"],
            "market_cap": [3.0e12],
            "shares_outstanding": [1.5e10],
            "type": ["CS"],
            "delisted_utc": [pd.NaT],
            "last_updated": [pd.Timestamp("2026-01-01", tz="UTC")],
        }
    )
    store.upsert_shard("universe", "all", df, dedupe_cols=["symbol"])
    result = check_liquidity_filter(store)
    assert not result.failures
    assert result.can_execute


def test_market_cap_threshold_is_one_billion() -> None:
    """The protocol fixes the threshold; the checker must not drift from it."""
    assert LIQUIDITY_MIN_MARKET_CAP == 1_000_000_000.0


# --- benchmark --------------------------------------------------------------
def test_benchmark_fails_when_spy_absent(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    result = check_benchmark(store)
    assert result.failures and result.failures[0].name == "benchmark_spy"


def test_benchmark_passes_with_spy_bars(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    store.upsert_shard("equity_bars", "SPY", _bars("SPY"), dedupe_cols=["symbol", "ts"])
    store.upsert_shard("equity_bars_total_return", "SPY", _bars("SPY"), dedupe_cols=["symbol", "ts"])
    result = check_benchmark(store)
    assert not result.failures, result.failures


# --- universe ---------------------------------------------------------------
def test_universe_flags_wrong_panel_size(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    result = check_universe_panels(store, ["AAPL", "MSFT"])
    names = {c.name for c in result.failures}
    assert "universe_size" in names


def test_universe_flags_missing_adjusted_series(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    syms = [f"S{i:04d}" for i in range(EXPECTED_PANEL_SIZE)]
    for s in syms:
        store.upsert_shard("equity_bars", s, _bars(s), dedupe_cols=["symbol", "ts"])

    # Raw only: the total-return leg must be reported as missing.
    result = check_universe_panels(store, syms)
    assert {c.name for c in result.failures} == {"panel_total_return"}

    # Both legs present: nothing fails.
    store2 = ParquetPitStore(tmp_path / "both")
    for s in syms:
        store2.upsert_shard("equity_bars", s, _bars(s), dedupe_cols=["symbol", "ts"])
        store2.upsert_shard("equity_bars_total_return", s, _bars(s), dedupe_cols=["symbol", "ts"])
    result2 = check_universe_panels(store2, syms)
    assert not result2.failures, [c.detail for c in result2.failures]


# --- instruments ------------------------------------------------------------
def test_statistical_instruments_match_protocol() -> None:
    result = check_statistical_instruments()
    assert not result.failures, [c.detail for c in result.failures]
    assert {c.name for c in result.checks} == {"bootstrap_contract", "dsr_contract"}


def test_bootstrap_constants_match_protocol_text() -> None:
    assert BOOTSTRAP_REPS == 2000
    assert BOOTSTRAP_BLOCK_LEN == 63


# --- purging ----------------------------------------------------------------
def test_purging_passes_with_real_horizon() -> None:
    result = check_purging(None)
    assert not result.failures, [c.detail for c in result.failures]
    assert MOMENTUM_LOOKBACK > MOMENTUM_SKIP


def test_purged_horizon_covers_the_full_12_1_lookback() -> None:
    """12-1 momentum needs T-273; purging with less than that would leak."""
    assert MOMENTUM_LOOKBACK == 273
    assert MOMENTUM_SKIP == 21


# --- cost model -------------------------------------------------------------
def test_cost_constants_match_protocol() -> None:
    assert COST_TOTAL_BPS == 20.0


# --- aggregate --------------------------------------------------------------
def test_run_preflight_blocks_on_real_empty_store(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    report = run_preflight(store, ["AAPL"])
    assert report.can_execute is False
    assert report.failures
    payload = report.as_dict()
    assert payload["can_execute"] is False
    assert payload["n_fail"] >= 1