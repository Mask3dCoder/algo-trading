"""Tests for the Alpaca PIT ingestion layer.

Every test here is offline: no Alpaca call is made. The client is replaced by a
stub that replays recorded payloads, so the suite runs in CI without keys.

The cases below pin three defects that were found the hard way during the
2026-10-11 backfill:
  1. ``store.read`` silently returning nothing when given a bare string shard.
  2. Delisting inference measured against today() instead of the coverage end,
     which flagged every still-trading name as dead in a backfilled panel.
  3. Adjustment factors defaulted to 1.0 for a symbol missing one of its two
     series - asserting "no corporate action" out of nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from quantcore.data.pit.market.alpaca_ingest import (  # noqa: E402
    ACTION_SHARD,
    DELISTED_GRACE_DAYS,
    AlpacaPitIngestor,
)
from quantcore.data.pit.market.store import ParquetPitStore  # noqa: E402
from quantcore.data.sources.alpaca import AlpacaClientError, AlpacaDataClient  # noqa: E402


class _StubClient:
    """Replays a fixed {symbol: rows} map. Raises if a test hits the network."""

    def __init__(self, bars: dict[str, list[dict]] | None = None) -> None:
        self._bars = bars or {}
        self.calls: list[tuple[str, str, str]] = []

    def daily_bars(self, symbol: str, start: str, end: str, *, adjustment: str = "raw", **_: object):
        self.calls.append((symbol, start, adjustment))
        return self._bars.get(symbol, [])


def _bar(ts: str, close: float, *, symbol: str = "AAA") -> dict:
    return {
        "symbol": symbol,
        "ts": pd.Timestamp(ts, tz="UTC"),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "volume": 1000.0,
        "vwap": close,
        "transactions": 10.0,
    }


def _write_bars(store: ParquetPitStore, dataset: str, symbol: str, closes: dict[str, float]) -> None:
    df = pd.DataFrame([_bar(ts, px, symbol=symbol) for ts, px in closes.items()])
    store.upsert_shard(dataset, symbol, df, dedupe_cols=["symbol", "ts"])


# --- store contract ---------------------------------------------------------
def test_read_rejects_bare_string_shard(tmp_path: Path) -> None:
    """A string is iterable, so this used to filter on characters and return nothing."""
    store = ParquetPitStore(tmp_path)
    _write_bars(store, "equity_bars", "AAPL", {"2024-01-02": 100.0})

    with pytest.raises(TypeError, match="iterable of names"):
        store.read("equity_bars", "AAPL")

    assert len(store.read("equity_bars", ["AAPL"])) == 1


def test_set_watermark_survives_replace_failure(tmp_path: Path, monkeypatch) -> None:
    """A locked manifest (cloud sync / indexer) must not abort an ingestion run."""
    import os

    store = ParquetPitStore(tmp_path)
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src: str, dst: str) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(5, "Access is denied", dst)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    store.set_watermark("equity_bars", "AAPL", "2024-01-02")
    assert calls["n"] == 3
    assert store.get_watermark("equity_bars", "AAPL") is not None


# --- ticker normalisation ---------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("BRK-B", "BRK.B"), ("BF-B", "BF.B"), ("aapl", "AAPL"), ("BRK.B", "BRK.B")],
)
def test_share_class_ticker_normalisation(raw: str, expected: str) -> None:
    """Alpaca 400s on the dash form of share classes."""
    import importlib

    sys.path.insert(0, str(PROJECT_ROOT / "bin"))
    mod = importlib.import_module("alpaca_ingest")
    assert mod._normalise_symbol(raw) == expected


# --- delisting inference ----------------------------------------------------
def test_name_at_coverage_end_is_not_delisted(tmp_path: Path) -> None:
    """Regression: comparing the last bar to today() flagged every live name."""
    store = ParquetPitStore(tmp_path)
    ingestor = AlpacaPitIngestor(_StubClient(), store)

    _write_bars(store, "equity_bars", "LIVE", {"2024-12-30": 10.0, "2024-12-31": 10.5})
    _write_bars(store, "equity_bars", "DEAD", {"2023-10-12": 94.0, "2023-10-13": 94.42})

    # Exactly one row: DEAD's delisting. LIVE's last bar is the coverage end.
    assert ingestor.derive_corp_actions(["LIVE", "DEAD"], coverage_end="2024-12-31", verbose=False) == 1

    actions = store.read("corp_actions", [ACTION_SHARD])
    dead = actions[actions["symbol"] == "DEAD"]
    assert len(dead) == 1
    assert dead["action_type"].iloc[0] == "delisting"
    assert str(dead["effective_date"].iloc[0]) == "2023-10-13"
    assert float(dead["value"].iloc[0]) == pytest.approx(94.42)

    # The live name's last bar IS the coverage end, so it must not appear.
    assert (actions["symbol"] == "LIVE").sum() == 0


def test_delisting_grace_period_is_respected(tmp_path: Path) -> None:
    """A halt or data lag inside the grace window is not a delisting."""
    store = ParquetPitStore(tmp_path)
    ingestor = AlpacaPitIngestor(_StubClient(), store)
    recent = (pd.Timestamp("2024-12-31") - pd.Timedelta(days=DELISTED_GRACE_DAYS - 5)).strftime("%Y-%m-%d")

    _write_bars(store, "equity_bars", "HALTED", {recent: 5.0, "2024-12-31": 5.1})
    assert ingestor.derive_corp_actions(["HALTED"], coverage_end="2024-12-31", verbose=False) == 0


# --- adjustment factors -----------------------------------------------------
def test_factor_is_adjusted_over_raw(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    ingestor = AlpacaPitIngestor(_StubClient(), store)

    raw = {"2024-06-03": 2447.0, "2024-06-04": 2500.0}
    adj = {"2024-06-03": 122.35, "2024-06-04": 125.0}
    _write_bars(store, "equity_bars", "AMZN", raw)
    _write_bars(store, "equity_bars_total_return", "AMZN", adj)

    assert ingestor.derive_adjustment_factors(["AMZN"], verbose=False) == 2
    factors = store.read("adjustment_factors", ["AMZN"]).sort_values("ts")
    assert factors["factor"].iloc[0] == pytest.approx(122.35 / 2447.0, rel=1e-9)
    assert (factors["source"] == "alpaca_derived").all()


def test_missing_adjusted_series_is_skipped_not_defaulted(tmp_path: Path) -> None:
    """A factor of 1.0 would assert 'no corporate action' out of nothing."""
    store = ParquetPitStore(tmp_path)
    ingestor = AlpacaPitIngestor(_StubClient(), store)

    _write_bars(store, "equity_bars", "ORPHAN", {"2024-06-03": 100.0})

    assert ingestor.derive_adjustment_factors(["ORPHAN"], verbose=False) == 0
    assert store.read("adjustment_factors", ["ORPHAN"]).empty


def test_split_detected_from_factor_step(tmp_path: Path) -> None:
    store = ParquetPitStore(tmp_path)
    ingestor = AlpacaPitIngestor(_StubClient(), store)

    factors = pd.DataFrame(
        {
            "symbol": "AMZN",
            "ts": pd.to_datetime(["2022-06-03", "2022-06-06", "2022-06-07"], utc=True),
            "factor": [1.0, 20.0, 20.0],
            "source": "alpaca_derived",
        }
    )
    store.upsert_shard("adjustment_factors", "AMZN", factors, dedupe_cols=["symbol", "ts"])
    _write_bars(store, "equity_bars", "AMZN", {"2022-06-03": 2447.0, "2022-06-06": 124.79, "2022-06-07": 125.0})

    # coverage_end sits just after the bars, so this isolates the split: with a
    # 2024 coverage end the same fixture would ALSO (correctly) read as delisted.
    assert ingestor.derive_corp_actions(["AMZN"], coverage_end="2022-06-10", verbose=False) == 1
    actions = store.read("corp_actions", [ACTION_SHARD])
    split = actions[actions["action_type"] == "split"].iloc[0]
    assert str(split["effective_date"]) == "2022-06-06"
    assert float(split["value"]) == pytest.approx(20.0, rel=1e-3)
    assert split["source"] == "alpaca_derived"


# --- client plumbing (offline) ---------------------------------------------
def test_client_rejects_trading_host_as_data_host() -> None:
    """The repo's .env ALPACA_BASE_URL is the paper TRADING host."""
    with pytest.raises(AlpacaClientError, match="TRADING host"):
        AlpacaDataClient("k" * 20, "s" * 20, base_url="https://paper-api.alpaca.markets")


def test_client_requires_credentials() -> None:
    with pytest.raises(AlpacaClientError, match="required"):
        AlpacaDataClient("", "")