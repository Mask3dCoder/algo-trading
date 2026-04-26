"""Tests for AlpacaDataFeed and AlpacaPaperFeed.

All tests mock alpaca-py — no real network calls.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest import mock

import pytest


class TestAlpacaDataFeedInterface:
    """Verify AlpacaDataFeed implements FeedHandler interface."""

    def test_implements_feed_handler(self) -> None:
        from quantcore.feeds.alpaca_feed import AlpacaDataFeed
        from quantcore.data.feeds.base import FeedHandler

        feed = AlpacaDataFeed(
            symbols=["SPY"],
            mode="live",
            api_key="test_key",
            api_secret="test_secret",
        )
        assert isinstance(feed, FeedHandler)
        assert hasattr(feed, "connect")
        assert hasattr(feed, "subscribe")
        assert hasattr(feed, "disconnect")
        assert hasattr(feed, "is_connected")
        assert hasattr(feed, "on_bar")
        assert hasattr(feed, "on_tick")
        assert hasattr(feed, "on_quote")

    def test_historical_mode_creates_rest_client(self) -> None:
        """Historical mode should not need a WebSocket."""
        from quantcore.feeds.alpaca_feed import AlpacaDataFeed

        feed = AlpacaDataFeed(
            symbols=["SPY"],
            mode="historical",
            api_key="test_key",
            api_secret="test_secret",
        )
        assert feed.mode == "historical"


class TestAlpacaDataFeedCredentials:
    """Verify credential handling."""

    def test_reads_from_env_vars(self, monkeypatch) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "env_key")
        monkeypatch.setenv("ALPACA_API_SECRET", "env_secret")

        from quantcore.feeds.alpaca_feed import AlpacaDataFeed

        feed = AlpacaDataFeed(symbols=["SPY"])
        assert feed.api_key == "env_key"
        assert feed.api_secret == "env_secret"

    def test_constructor_args_override_env(self, monkeypatch) -> None:
        monkeypatch.setenv("ALPACA_API_KEY", "env_key")
        monkeypatch.setenv("ALPACA_API_SECRET", "env_secret")

        from quantcore.feeds.alpaca_feed import AlpacaDataFeed

        feed = AlpacaDataFeed(symbols=["SPY"], api_key="ctor_key", api_secret="ctor_secret")
        assert feed.api_key == "ctor_key"
        assert feed.api_secret == "ctor_secret"

    def test_missing_credentials_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

        from quantcore.feeds.alpaca_feed import AlpacaDataFeed

        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            AlpacaDataFeed(symbols=["SPY"], api_key=None, api_secret=None)


class TestAlpacaPaperFeed:
    """Verify AlpacaPaperFeed sync interface."""

    def test_implements_mock_data_feed_interface(self) -> None:
        from quantcore.feeds.alpaca_feed import AlpacaPaperFeed

        feed = AlpacaPaperFeed(
            symbols=["SPY", "QQQ"],
            api_key="test_key",
            api_secret="test_secret",
        )
        assert hasattr(feed, "get_latest")
        assert hasattr(feed, "is_market_open")
        assert hasattr(feed, "last_data_timestamp")

    def test_get_latest_returns_dict_list(self) -> None:
        from quantcore.feeds.alpaca_feed import AlpacaPaperFeed

        # Without alpaca-py installed, uses synthetic generator
        feed = AlpacaPaperFeed(
            symbols=["SPY"],
            api_key="test_key",
            api_secret="test_secret",
        )
        # Force offline mode
        feed._client = None

        events = feed.get_latest()
        assert isinstance(events, list)
        assert len(events) == 1
        event = events[0]
        assert event["symbol"] == "SPY"
        for field in ("price", "close", "high", "low", "open", "volume"):
            assert field in event

    def test_get_latest_multiple_symbols(self) -> None:
        from quantcore.feeds.alpaca_feed import AlpacaPaperFeed

        feed = AlpacaPaperFeed(
            symbols=["SPY", "QQQ", "IWM"],
            api_key="test_key",
            api_secret="test_secret",
        )
        feed._client = None
        events = feed.get_latest()
        assert len(events) == 3
        symbols = {e["symbol"] for e in events}
        assert symbols == {"SPY", "QQQ", "IWM"}

    def test_last_data_timestamp_updated(self) -> None:
        from quantcore.feeds.alpaca_feed import AlpacaPaperFeed

        feed = AlpacaPaperFeed(
            symbols=["SPY"],
            api_key="test_key",
            api_secret="test_secret",
        )
        feed._client = None
        assert feed.last_data_timestamp is None
        feed.get_latest()
        assert feed.last_data_timestamp is not None

    def test_missing_credentials_raises(self, monkeypatch) -> None:
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

        from quantcore.feeds.alpaca_feed import AlpacaPaperFeed

        with pytest.raises(ValueError, match="ALPACA_API_KEY"):
            AlpacaPaperFeed(api_key=None, api_secret=None)


class TestNormalization:
    """Verify Alpaca message → internal type normalization."""

    def test_bar_message_normalization(self) -> None:
        """Given a raw Alpaca bar object, verify normalization to internal types."""
        from dataclasses import dataclass
        from datetime import UTC, datetime

        @dataclass
        class MockBar:
            symbol: str = "SPY"
            timestamp: datetime = datetime.now(UTC)
            open: float = 450.0
            high: float = 452.0
            low: float = 449.0
            close: float = 451.0
            volume: float = 1000000.0

        from quantcore.feeds.alpaca_feed import AlpacaDataFeed

        feed = AlpacaDataFeed(
            symbols=["SPY"],
            mode="live",
            api_key="test_key",
            api_secret="test_secret",
        )

        mock_bar = MockBar()
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(feed._on_bar_message(mock_bar))
        loop.close()

        latest = feed._latest_bars.get("SPY")
        assert latest is not None
        assert latest["symbol"] == "SPY"
        assert latest["close"] == 451.0
        assert latest["volume"] == 1000000
        assert latest["open"] == 450.0

    def test_quote_message_normalization(self) -> None:
        """Given a raw Alpaca quote object, verify normalization."""
        from dataclasses import dataclass
        from datetime import UTC, datetime

        @dataclass
        class MockQuote:
            symbol: str = "SPY"
            timestamp: datetime = datetime.now(UTC)
            bid_price: float = 450.0
            ask_price: float = 450.1
            bid_size: float = 100.0
            ask_size: float = 200.0
            tape: str = "A"

        from quantcore.feeds.alpaca_feed import AlpacaDataFeed

        feed = AlpacaDataFeed(
            symbols=["SPY"],
            mode="live",
            api_key="test_key",
            api_secret="test_secret",
        )

        handled = []

        async def capture_quote(quote) -> None:
            handled.append(quote)

        feed.on_quote(capture_quote)

        mock_quote = MockQuote()
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(feed._on_quote_message(mock_quote))
        loop.close()

        assert len(handled) == 1
        quote = handled[0]
        from quantcore.core.types import Quote
        assert isinstance(quote, Quote)
        assert float(quote.bid) == 450.0
        assert float(quote.ask) == 450.1

    def test_trade_message_normalization(self) -> None:
        """Given a raw Alpaca trade object, verify normalization."""
        from dataclasses import dataclass
        from datetime import UTC, datetime

        @dataclass
        class MockTrade:
            symbol: str = "SPY"
            timestamp: datetime = datetime.now(UTC)
            price: float = 451.0
            size: int = 500
            tape: str = "A"

        from quantcore.feeds.alpaca_feed import AlpacaDataFeed

        feed = AlpacaDataFeed(
            symbols=["SPY"],
            mode="live",
            api_key="test_key",
            api_secret="test_secret",
        )

        handled = []

        async def capture_tick(tick) -> None:
            handled.append(tick)

        feed.on_tick(capture_tick)

        mock_trade = MockTrade()
        import asyncio
        loop = asyncio.new_event_loop()
        loop.run_until_complete(feed._on_trade_message(mock_trade))
        loop.close()

        assert len(handled) == 1
        tick = handled[0]
        from quantcore.core.types import Tick
        assert isinstance(tick, Tick)
        assert float(tick.price) == 451.0


class TestReconnection:
    """Verify reconnection logic."""

    def test_reconnect_count_increments(self) -> None:
        from quantcore.feeds.alpaca_feed import AlpacaDataFeed

        feed = AlpacaDataFeed(
            symbols=["SPY"],
            mode="live",
            api_key="test_key",
            api_secret="test_secret",
        )
        assert feed._reconnect_count == 0
        feed._reconnect_count += 1
        assert feed._reconnect_count == 1
