"""PIT market-data store: as-of gating, lookahead protection, survivorship.

These tests run entirely against synthetic data written through the parquet
store — no network, no Polygon API key required.
"""

from __future__ import annotations

import pandas as pd

from quantcore.data.pit.market import ParquetPitStore
from quantcore.data.pit.market import universe as universe_mod
from quantcore.data.pit.market.asof import PitReader
from quantcore.data.pit.market.corp_actions import frame_from_rows


def _synthetic_bars(n: int = 400) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=n, freq="B", tz="UTC")
    return pd.DataFrame({
        "symbol": "SPY",
        "ts": ts,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1_000_000,
        "vwap": 100.0,
        "transactions": 1000,
    })


def test_bars_asof_never_leaks_future(tmp_path) -> None:
    store = ParquetPitStore(tmp_path)
    reader = PitReader(store)
    bars = _synthetic_bars()
    for symbol, price in (("SPY", 100.0), ("AAPL", 180.0)):
        df = bars.copy()
        df["symbol"] = symbol
        df["close"] = price
        for year, frame in df.groupby(df["ts"].dt.year):
            store.write("equity_bars", symbol, frame, str(year))

    mid = reader.equity_bars_asof(["SPY", "AAPL"], "2023-06-15")
    expected = pd.bdate_range("2023-01-02", "2023-06-15").size
    assert len(mid) == 2 * expected
    assert mid["ts"].max().normalize() <= pd.Timestamp("2023-06-15", tz="UTC")


def test_upsert_is_idempotent(tmp_path) -> None:
    store = ParquetPitStore(tmp_path)
    reader = PitReader(store)
    bars = _synthetic_bars()
    store.upsert_shard("equity_bars", "SPY", bars, dedupe_cols=["symbol", "ts"])
    store.upsert_shard("equity_bars", "SPY", bars, dedupe_cols=["symbol", "ts"])

    df = reader.equity_bars_asof(["SPY"], "2024-12-31")
    assert len(df) == len(bars)


def test_corp_action_lookahead_is_hidden(tmp_path) -> None:
    store = ParquetPitStore(tmp_path)
    reader = PitReader(store)
    rows = [
        # effective 05-01, ANNOUNCED 04-01 -> known by 05-15, must appear
        {"action_type": "split", "symbol": "SPY", "effective_date": "2023-05-01",
         "announced_at": pd.Timestamp("2023-04-01", tz="UTC"), "value": 2.0,
         "ticker_old": None, "ticker_new": None, "description": "", "source": "polygon"},
        # effective 05-01 but ANNOUNCED 06-01 -> NOT known by 05-15, must hide
        {"action_type": "dividend", "symbol": "SPY", "effective_date": "2023-05-01",
         "announced_at": pd.Timestamp("2023-06-01", tz="UTC"), "value": 1.5,
         "ticker_old": None, "ticker_new": None, "description": "", "source": "polygon"},
    ]
    store.upsert_shard("corp_actions", "all", frame_from_rows(rows),
                       dedupe_cols=["action_type", "symbol", "effective_date"])

    known = reader.corp_actions_asof(["SPY"], "2023-05-15")
    types = set(known["action_type"])
    assert "split" in types
    assert "dividend" not in types


def test_universe_is_survivorship_free(tmp_path) -> None:
    store = ParquetPitStore(tmp_path)
    reader = PitReader(store)
    universe = pd.DataFrame([
        {"symbol": "SPY", "name": "x", "market_cap": 1e12, "shares_outstanding": 1e9,
         "type": "CS", "delisted_utc": None, "last_updated": pd.Timestamp.now("UTC")},
        {"symbol": "DEAD", "name": "y", "market_cap": 5e9, "shares_outstanding": 1e8,
         "type": "CS", "delisted_utc": pd.Timestamp("2020-06-01", tz="UTC"),
         "last_updated": pd.Timestamp.now("UTC")},
    ])
    universe_mod.save_universe(store, universe, asof=pd.Timestamp("2024-01-01").date())

    syms_2019 = reader.universe_symbols_asof("2019-06-01")
    syms_2021 = reader.universe_symbols_asof("2021-06-01")
    assert "DEAD" in syms_2019 and "SPY" in syms_2019
    assert "DEAD" not in syms_2021 and "SPY" in syms_2021


def test_select_universe_keeps_delisted_tail(tmp_path) -> None:
    """Delisted names must not be evicted by the active-only market-cap floor.

    Regression for a survivorship-bias bug: the floor was applied to the merged
    frame, so a delisted ticker whose current market_cap is null/0 was dropped
    from the snapshot — excluding every company that died during the window.
    """
    class _FakeClient:
        def __init__(self, active: list[dict], delisted: list[dict]) -> None:
            self._active = active
            self._delisted = delisted

        def all_tickers(self, *, active: bool, market: str = "stocks", type_: str = "CS") -> list[dict]:
            return self._active if active else self._delisted

    client = _FakeClient(
        active=[
            {"ticker": "SPY", "name": "x", "market_cap": 1e12,
             "shares_outstanding": 1e9, "type": "CS", "delisted_utc": None,
             "updated": "2024-01-01"},
            # Active name BELOW the $2B floor — must be dropped.
            {"ticker": "TINY", "name": "t", "market_cap": 5e8,
             "shares_outstanding": 1e8, "type": "CS", "delisted_utc": None,
             "updated": "2024-01-01"},
        ],
        delisted=[
            # Dead name, null market cap today — must be retained regardless.
            {"ticker": "DEAD", "name": "y", "market_cap": None,
             "shares_outstanding": None, "type": "CS",
             "delisted_utc": "2022-06-01", "updated": "2024-01-01"},
        ],
    )
    df = universe_mod.select_universe(client, min_market_cap=2e9, top_n=500)
    syms = set(df["symbol"])
    assert "SPY" in syms
    assert "DEAD" in syms, "delisted tail must survive the market-cap floor"
    assert "TINY" not in syms, "active names below the floor are still excluded"


def test_empty_store_queries_return_empty_frames(tmp_path) -> None:
    store = ParquetPitStore(tmp_path)
    reader = PitReader(store)
    assert reader.equity_bars_asof(["SPY"], "2023-06-15").empty
    assert reader.corp_actions_asof(["SPY"], "2023-06-15").empty
    assert reader.options_quotes_asof(["SPY250116C00600000"], "2023-06-15").empty
    assert reader.universe_symbols_asof("2023-06-15") == []
