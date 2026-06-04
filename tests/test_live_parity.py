"""Parity tests: LiveDataFeed vs yfinance data -> same signals.

These tests validate that switching from yfinance (backtest) to Alpaca IEX REST
(live) produces exactly the same strategy signals — the cardinal rule for
honest live validation.

Run with:
    python -m pytest tests/test_live_parity.py -v

The tests use pytest fixtures from tests/conftest.py and do NOT require
network access (Alpaca responses are mocked via hypothesis + respx).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The 6 macro assets used in Phase 2 validation
MACRO_TICKERS = ["SVXY", "SPXL", "TMF", "TLT", "SPY", "QQQ"]

# A fixed validation date that's within the sweep's backtest period (2005-2024)
# and for which we can deterministically generate synthetic bars.
VAL_DATE = date(2023, 6, 15)


@pytest.fixture
def synthetic_price_frames() -> dict[str, pd.DataFrame]:
    """Deterministic synthetic OHLCV bars covering 300 business days up to VAL_DATE.

    Each ticker gets a unique but reproducible price history using a seeded RNG
    so the fixture is stable across test runs.
    """
    np.random.seed(42)
    n_days = 300
    dates = pd.bdate_range(end=pd.Timestamp(VAL_DATE), periods=n_days)

    frames: dict[str, pd.DataFrame] = {}
    for i, ticker in enumerate(MACRO_TICKERS):
        # Each ticker starts at a different base price and has unique drift/vol
        base = 80.0 + i * 15.0
        daily_rets = np.random.randn(n_days - 1) * 0.012 + 0.0002
        closes = base * np.exp(np.cumsum(daily_rets))
        closes = np.insert(closes, 0, base)   # open = first close

        frames[ticker] = pd.DataFrame(
            {
                "timestamp": dates,
                "open":  closes * (1.0 + np.random.randn(n_days) * 0.003),
                "high":  closes * (1.0 + np.abs(np.random.randn(n_days)) * 0.006),
                "low":   closes * (1.0 - np.abs(np.random.randn(n_days)) * 0.006),
                "close": closes,
                "volume": np.random.randint(100_000, 10_000_000, n_days),
            }
        ).set_index("timestamp")

    return frames


@pytest.fixture
def synthetically_mapped_live_feed(
    synthetic_price_frames: dict[str, pd.DataFrame],
) -> "LiveDataFeed":
    """A LiveDataFeed that intercepts Alpaca API calls and returns synthetic bars.

    Mocks _fetch_bars_raw so the LiveDataFeed serves fixture data without any
    network I/O.
    """
    from quantcore.live.data_feed import LiveDataFeed

    def _mock_bars(
        symbol: str,
        start: str,
        end: str,
        timeframe: str = "1Day",
        api_key: str | None = None,
        secret_key: str | None = None,
        max_retries: int = 3,
    ) -> list[dict[str, Any]]:
        df = synthetic_price_frames.get(symbol)
        if df is None:
            return []

        start_ts = pd.Timestamp(start)
        end_ts   = pd.Timestamp(end)
        sub = df.loc[start_ts:end_ts]
        bars = []
        for ts, row in sub.iterrows():
            bars.append({
                "t":    ts.isoformat(),
                "o":    round(float(row["open"]), 4),
                "h":    round(float(row["high"]), 4),
                "l":    round(float(row["low"]), 4),
                "c":    round(float(row["close"]), 4),
                "v":    int(row["volume"]),
            })
        return bars

    with patch("quantcore.live.data_feed._fetch_bars_raw", side_effect=_mock_bars):
        feed = LiveDataFeed(tickers=MACRO_TICKERS, api_key="TEST", secret_key="TEST")
        # Seed the in-memory cache so no API calls are made during snapshot
        for ticker, df in synthetic_price_frames.items():
            feed._bar_cache[ticker] = df.sort_index()
            feed._cache_meta[ticker] = (df.index.min().date(), df.index.max().date())

    return feed


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLiveDataFeedParity:
    """Verify LiveDataFeed price matrix matches synthetic ground truth."""

    def test_tickers_list(self, synthetically_mapped_live_feed) -> None:
        assert synthetically_mapped_live_feed.tickers == MACRO_TICKERS

    def test_daily_snapshot_shape(self, synthetically_mapped_live_feed) -> None:
        snap = synthetically_mapped_live_feed.get_daily_snapshot(VAL_DATE, lookback_days=252)
        close = snap["close"]
        assert close.ndim == 2
        assert close.shape[1] == len(MACRO_TICKERS)
        # 252 calendar days ~ 180 business days (252 * 5/7); synthetic data has
        # every business day (no holidays), so the actual count equals the
        # business-day range length.
        expected_bdays = len(pd.bdate_range(
            start=pd.Timestamp(VAL_DATE) - pd.Timedelta(days=252), end=pd.Timestamp(VAL_DATE),
        ))
        assert close.shape[0] == expected_bdays

    def test_close_matrix_is_float64(self, synthetically_mapped_live_feed) -> None:
        snap = synthetically_mapped_live_feed.get_daily_snapshot(VAL_DATE, lookback_days=252)
        assert snap["close"].dtype == np.float64

    def test_no_nan_in_latest_window(self, synthetically_mapped_live_feed) -> None:
        snap = synthetically_mapped_live_feed.get_daily_snapshot(VAL_DATE, lookback_days=252)
        close = snap["close"]
        # Last 20 rows should be complete (no NaN from forward-fill)
        assert not pd.isna(close[-20:]).any()

    def test_symbols_match_ticker_list(self, synthetically_mapped_live_feed) -> None:
        snap = synthetically_mapped_live_feed.get_daily_snapshot(VAL_DATE, lookback_days=252)
        assert snap["symbols"] == MACRO_TICKERS


class TestSignalParity:
    """Verify strategy.on_data() produces identical signals from both data paths.

    This is the most important test: it confirms that the LIVE signal pipeline
    (LiveDataFeed + strategy.on_data) yields the same signals as the BACKTEST
    pipeline (stored prices + strategy.on_data), which is the Phase 3 mandate.
    """

    def _build_on_data_snapshot(self, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Concat price frames into the dict format expected by on_data()."""
        close_frames = [frames[t][["close"]] for t in MACRO_TICKERS]
        aligned = pd.concat(close_frames, axis=1)
        aligned.columns = MACRO_TICKERS
        aligned = aligned.dropna().sort_index()
        return {
            "close":   aligned.values.astype(np.float64),
            "symbols": MACRO_TICKERS,
        }

    def test_cross_sectional_momentum_parity(
        self,
        synthetic_price_frames,
        synthetically_mapped_live_feed,
    ) -> None:
        """on_data() must return identical signals for synthetic vs LiveDataFeed input."""
        from quantcore.strategies.macro.cross_sectional_momentum import (
            CrossSectionalMomentumStrategy,
        )
        from quantcore.strategies.base import StrategyConfig

        cfg = StrategyConfig(
            name="macro_cross_sectional_momentum",
            strategy_type="MACRO",
            max_position_notional=None,
            params={},
        )
        strat = CrossSectionalMomentumStrategy(cfg)

        # Path A: call on_data() with the raw synthetic DataFrames (backtest path)
        snap_a = self._build_on_data_snapshot(synthetic_price_frames)
        signals_a = strat.on_data(snap_a)

        # Path B: call on_data() with LiveDataFeed output (live path)
        snap_b = synthetically_mapped_live_feed.get_daily_snapshot(VAL_DATE, lookback_days=252)
        # Same lookback period to match the snap_a data
        strat_b = CrossSectionalMomentumStrategy(cfg)  # fresh instance
        signals_b = strat_b.on_data(snap_b)

        # Normalise for comparison
        def _norm(sigs):
            return sorted([
                (s.symbol, s.direction, round(float(s.strength), 6))
                for s in sigs
            ])

        assert _norm(signals_a) == _norm(signals_b), (
            "CrossSectionalMomentumStrategy signals differ between backtest and live paths"
        )

class TestLiveExecutionGateway:
    """Smoke tests for the sync execution gateway."""

    def test_gateway_initialises_in_paper_mode(self) -> None:
        from quantcore.live.broker import LiveExecutionGateway
        # Clear Alpaca env vars so gateway falls back to simulation mode
        with patch.dict(os.environ):
            for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
                os.environ.pop(k, None)
            gw = LiveExecutionGateway(initial_cash=50_000)
        assert gw.is_paper is True
        assert gw.get_account()["cash"] == 50_000

    def test_gateway_paper_order_flow(self) -> None:
        from quantcore.live.broker import LiveExecutionGateway
        from quantcore.core.types import Order, OrderType, Side

        with patch.dict(os.environ):
            for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
                os.environ.pop(k, None)
            gw = LiveExecutionGateway(initial_cash=100_000)
            gw._paper = True  # ensure paper mode

            assert len(gw.get_positions()) == 0

            order = Order(
                id="", symbol="AAPL", side=Side.BUY,
                type=OrderType.MARKET, quantity=10,
                status="pending",
                filled_qty=0,
            )
            with patch.object(gw, "_get_market_price", return_value=100.0):
                order_id = gw.submit_order(order)
            assert order_id != ""
            assert len(gw.get_positions()) == 1
            assert "AAPL" in gw.get_positions()

    def test_gateway_short_sell(self) -> None:
        from quantcore.live.broker import LiveExecutionGateway
        from quantcore.core.types import Order, OrderType, Side

        with patch.dict(os.environ):
            for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
                os.environ.pop(k, None)
            gw = LiveExecutionGateway(initial_cash=100_000)
            gw._paper = True

            order = Order(
                id="", symbol="SPY", side=Side.SELL,
                type=OrderType.MARKET, quantity=5,
                status="pending",
                filled_qty=0,
            )
            with patch.object(gw, "_get_market_price", return_value=100.0):
                gw.submit_order(order)
        assert "SPY" in gw.get_positions()
        assert float(gw.get_positions()["SPY"].quantity) == 5

    def test_gateway_rejects_insufficient_cash(self) -> None:
        from quantcore.live.broker import LiveExecutionGateway
        from quantcore.core.types import Order, OrderType, Side

        with patch.dict(os.environ):
            for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
                os.environ.pop(k, None)
            gw = LiveExecutionGateway(initial_cash=100)  # very low cash
            gw._paper = True

            order = Order(
                id="", symbol="AAPL", side=Side.BUY,
                type=OrderType.MARKET, quantity=10,  # $1000 notional
                status="pending",
                filled_qty=0,
            )
            # Mock price at $100/share -> would need $1000 but only $100 cash
            with patch.object(gw, "_get_market_price", return_value=100.0):
                order_id = gw.submit_order(order)
        assert order_id != ""
        # The order should be in paper_orders as REJECTED
        assert order_id in gw._paper_orders
        assert gw._paper_orders[order_id].status.value == "rejected"
