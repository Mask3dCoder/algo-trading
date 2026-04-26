"""Alpaca market data feed using alpaca-py SDK.

Provides both real-time streaming (StockDataStream) and historical bars
(StockHistoricalDataClient). Implements the FeedHandler base class and
exposes a sync adapter for paper_trade.py compatibility.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import numpy as np

from quantcore.core.health_state import record_data_point
from quantcore.core.types import OHLCV, Quote, Tick
from quantcore.data.feeds.base import FeedHandler

log = logging.getLogger(__name__)


class AlpacaDataFeed(FeedHandler):
    """Alpaca market data feed using alpaca-py SDK.

    Supports two modes:
      - mode="live": connects to Alpaca's WebSocket data stream for real-time bars,
        quotes, and trades. Fires callbacks via _emit_bar, _emit_quote, _emit_tick.
      - mode="historical": fetches bars via REST API (StockHistoricalDataClient)
        for backtesting or dry-run replay. Does not connect a WebSocket.
    """

    WS_URL_IEX = "wss://stream.data.alpaca.markets/v2/iex"
    WS_URL_SIP = "wss://stream.data.alpaca.markets/v2/sip"

    def __init__(
        self,
        symbols: list[str] | None = None,
        mode: str = "live",
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
        sip: bool = False,
    ) -> None:
        super().__init__()
        self.symbols_list = symbols or []
        self.mode = mode

        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        self.base_url = base_url or os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_API_SECRET must be set as environment "
                "variables or passed to AlpacaDataFeed constructor"
            )

        self._sip = sip
        self._stream: Any = None
        self._historical_client: Any = None
        self._latest_bars: dict[str, dict[str, Any]] = {}
        self._reconnect_count: int = 0
        self._max_reconnect_count: int = 3
        self._listen_task: asyncio.Task | None = None
        self._shutdown: bool = False

    # ---- FeedHandler interface -----------------------------------------------

    async def connect(self) -> None:
        """Establish connection to Alpaca data stream."""
        if self.mode == "historical":
            try:
                from alpaca.data.historical import StockHistoricalDataClient
            except ImportError as e:
                raise ImportError(
                    "alpaca-py not installed. Run: pip install alpaca-py"
                ) from e
            self._historical_client = StockHistoricalDataClient(
                api_key=self.api_key, secret_key=self.api_secret
            )
            self._connected = True
            log.info("Alpaca historical data client initialized")
            return

        try:
            from alpaca.data.live import StockDataStream
        except ImportError as e:
            raise ImportError(
                "alpaca-py not installed. Run: pip install alpaca-py"
            ) from e

        ws_url = self.WS_URL_SIP if self._sip else self.WS_URL_IEX
        self._stream = StockDataStream(
            api_key=self.api_key,
            secret_key=self.api_secret,
            url_override=ws_url,
        )

        # Register handlers for each data type
        self._stream.subscribe_bars(self._on_bar_message, *self.symbols_list)
        self._stream.subscribe_quotes(self._on_quote_message, *self.symbols_list)
        self._stream.subscribe_trades(self._on_trade_message, *self.symbols_list)

        await self._stream.connect()
        self._connected = True
        self._shutdown = False
        log.info("Alpaca live data stream connected (%s)", "SIP" if self._sip else "IEX")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to additional symbols."""
        self.add_symbols(symbols)
        self.symbols_list.extend(symbols)
        if self._stream and self.mode == "live":
            self._stream.subscribe_bars(self._on_bar_message, *symbols)
            self._stream.subscribe_quotes(self._on_quote_message, *symbols)
            self._stream.subscribe_trades(self._on_trade_message, *symbols)
            log.info("Subscribed to %d additional symbols", len(symbols))

    async def disconnect(self) -> None:
        """Disconnect from Alpaca data stream."""
        self._shutdown = True
        if self._stream:
            try:
                self._stream.unsubscribe_bars(*self.symbols_list)
                self._stream.unsubscribe_quotes(*self.symbols_list)
                self._stream.unsubscribe_trades(*self.symbols_list)
                await self._stream.disconnect()
            except Exception:
                pass
            self._stream = None
        self._connected = False
        self._historical_client = None
        log.info("Alpaca data feed disconnected")

    # ---- WebSocket message handlers ------------------------------------------

    async def _on_bar_message(self, bar_data: Any) -> None:
        """Normalize and emit a bar event from Alpaca WebSocket."""
        record_data_point()
        bar = OHLCV(
            symbol=str(bar_data.symbol),
            timestamp=bar_data.timestamp if hasattr(bar_data, 'timestamp') else datetime.now(UTC),
            open=Decimal(str(bar_data.open)),
            high=Decimal(str(bar_data.high)),
            low=Decimal(str(bar_data.low)),
            close=Decimal(str(bar_data.close)),
            volume=int(bar_data.volume),
        )
        self._latest_bars[str(bar_data.symbol)] = {
            "symbol": str(bar_data.symbol),
            "price": float(bar_data.close),
            "timestamp": bar.timestamp,
            "close": float(bar_data.close),
            "high": float(bar_data.high),
            "low": float(bar_data.low),
            "open": float(bar_data.open),
            "volume": int(bar_data.volume),
        }
        await self._emit_bar(bar)

    async def _on_quote_message(self, quote_data: Any) -> None:
        """Normalize and emit a quote event from Alpaca WebSocket."""
        record_data_point()
        quote = Quote(
            symbol=str(quote_data.symbol),
            timestamp=quote_data.timestamp if hasattr(quote_data, 'timestamp') else datetime.now(UTC),
            bid=Decimal(str(quote_data.bid_price)),
            ask=Decimal(str(quote_data.ask_price)),
            bid_size=int(quote_data.bid_size),
            ask_size=int(quote_data.ask_size),
            exchange=str(getattr(quote_data, 'tape', '')),
        )
        await self._emit_quote(quote)

    async def _on_trade_message(self, trade_data: Any) -> None:
        """Normalize and emit a trade event from Alpaca WebSocket."""
        record_data_point()
        tick = Tick(
            symbol=str(trade_data.symbol),
            timestamp_us=int(trade_data.timestamp.timestamp() * 1e6) if hasattr(trade_data, 'timestamp') else int(time.time() * 1e6),
            price=Decimal(str(trade_data.price)),
            size=int(trade_data.size),
            exchange=str(getattr(trade_data, 'tape', '')),
        )
        await self._emit_tick(tick)

    # ---- Historical data access ----------------------------------------------

    async def get_historical_bars(
        self,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        timeframe: str = "1Min",
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch historical OHLCV bars via REST API.

        Returns dict keyed by symbol, each value is a list of bar dicts
        compatible with the internal OHLCV format.
        """
        if not self._historical_client:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
            except ImportError as e:
                raise ImportError("alpaca-py not installed") from e
            self._historical_client = StockHistoricalDataClient(
                api_key=self.api_key, secret_key=self.api_secret
            )

        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        tf_map: dict[str, Any] = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, "min"),
            "15Min": TimeFrame(15, "min"),
            "1H": TimeFrame.Hour,
            "1D": TimeFrame.Day,
        }

        syms = symbols or self.symbols_list
        request = StockBarsRequest(
            symbol_or_symbols=syms,
            timeframe=tf_map.get(timeframe, TimeFrame.Minute),
            start=start,
            end=end,
        )
        response = self._historical_client.get_stock_bars(request)

        result: dict[str, list[dict[str, Any]]] = {sym: [] for sym in syms}
        bars_by_symbol = response.data if hasattr(response, 'data') else response
        for sym in syms:
            sym_bars = bars_by_symbol.get(sym, [])
            for bar in sym_bars:
                result[sym].append({
                    "symbol": sym,
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "price": float(bar.close),
                })
        return result

    async def get_latest_bar(self, symbol: str) -> dict[str, Any] | None:
        """Get the latest bar for a single symbol via REST."""
        if not self._historical_client:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
            except ImportError as e:
                raise ImportError("alpaca-py not installed") from e
            self._historical_client = StockHistoricalDataClient(
                api_key=self.api_key, secret_key=self.api_secret
            )

        from alpaca.data.requests import StockLatestBarRequest

        request = StockLatestBarRequest(symbol_or_symbols=[symbol])
        response = self._historical_client.get_stock_latest_bar(request)
        bar_data = response[symbol] if response else None
        if bar_data is None:
            return None

        return {
            "symbol": symbol,
            "timestamp": datetime.now(UTC),
            "open": float(bar_data.open),
            "high": float(bar_data.high),
            "low": float(bar_data.low),
            "close": float(bar_data.close),
            "volume": int(bar_data.volume),
            "price": float(bar_data.close),
        }

    # ---- Reconnection logic --------------------------------------------------

    async def reconnect(self) -> None:
        """Attempt reconnection with count tracking."""
        self._reconnect_count += 1
        if self._reconnect_count > self._max_reconnect_count:
            from quantcore.core.health_state import record_data_point
            log.error("Max reconnection attempts (%d) reached", self._max_reconnect_count)
            raise ConnectionError(
                f"Failed to reconnect after {self._max_reconnect_count} attempts"
            )
        await asyncio.sleep(5.0)
        await self.connect()
        self._reconnect_count = 0

    # ---- Sync adapter for paper_trade.py -------------------------------------

    def get_latest(self) -> list[dict[str, Any]]:
        """Sync adapter: return latest bar data for all subscribed symbols.

        Compatible with paper_trade.py's MockDataFeed interface.
        """
        results = []
        for sym in self.symbols_list:
            bar = self._latest_bars.get(sym)
            if bar:
                results.append(dict(bar))
            else:
                # Generate a stub so the paper loop doesn't crash on startup
                results.append({
                    "symbol": sym,
                    "price": 100.0,
                    "timestamp": datetime.now(UTC),
                    "close": 100.0,
                    "high": 100.1,
                    "low": 99.9,
                    "open": 100.0,
                    "volume": 0,
                })
        return results

    @property
    def last_data_timestamp(self) -> datetime | None:
        """Return timestamp of the most recent bar, if any."""
        bars = list(self._latest_bars.values())
        if not bars:
            return None
        latest = max(bars, key=lambda b: b.get("timestamp", datetime.min))
        return latest.get("timestamp")

    def is_market_open(self) -> bool:
        """Best-effort check: assume open if connected and recent data."""
        if not self._connected:
            return False
        if self._latest_bars:
            ts = list(self._latest_bars.values())[0].get("timestamp")
            if ts:
                age = (datetime.now(UTC) - ts).total_seconds()
                return age < 300  # stale if no bar for 5 min
        return True


# ---------------------------------------------------------------------------
# Sync adapter for paper_trade.py direct use
# ---------------------------------------------------------------------------


class AlpacaPaperFeed:
    """Synchronous Alpaca feed adapter for paper_trade.py.

    Wraps AlpacaDataFeed so paper_trade.py can use it without async/await.
    Fetches latest bars via alpaca-py's StockHistoricalDataClient on each tick.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.symbols = symbols or ["SPY"]
        self.api_key = api_key or os.environ.get("ALPACA_API_KEY")
        self.api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        self.base_url = base_url or os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not self.api_key or not self.api_secret:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_API_SECRET environment variables required"
            )

        self._client: Any = None
        self._rng = np.random.default_rng(42)
        self._last_tick: datetime | None = None
        self._cached_bars: dict[str, dict[str, Any]] = {}
        self._use_rest: bool = True

        import importlib
        self._alpaca_available = importlib.util.find_spec("alpaca") is not None

        if self._alpaca_available:
            self._init_client()
        else:
            log.warning("alpaca-py not installed — AlpacaPaperFeed will use synthetic data")

    def _init_client(self) -> None:
        """Initialize the historical data client for REST polling."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
            )
            log.info("Alpaca paper feed initialized (REST polling)")
        except Exception as e:
            log.warning("Failed to initialize Alpaca client: %s — using synthetic data", e)
            self._client = None

    def is_market_open(self) -> bool:
        """Check if the US stock market is currently open."""
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=True,
            )
            clock = client.get_clock()
            return bool(clock.is_open)
        except Exception:
            return True

    def get_latest(self) -> list[dict[str, Any]]:
        """Fetch latest bars (via REST if available, else synthetic).

        Returns a list of dicts compatible with MockDataFeed.get_latest().
        """
        now = datetime.now(UTC)
        self._last_tick = now

        if self._client:
            return self._fetch_rest_bars(now)
        return self._generate_synthetic_bars(now)

    def _fetch_rest_bars(self, now: datetime) -> list[dict[str, Any]]:
        """Fetch latest bars via REST API."""
        try:
            from alpaca.data.requests import StockLatestBarRequest

            request = StockLatestBarRequest(symbol_or_symbols=self.symbols)
            response = self._client.get_stock_latest_bar(request)

            events = []
            for sym in self.symbols:
                bar_data = response.get(sym) if response else None
                if bar_data:
                    bar = {
                        "symbol": sym,
                        "price": float(bar_data.close),
                        "timestamp": now,
                        "close": float(bar_data.close),
                        "high": float(bar_data.high),
                        "low": float(bar_data.low),
                        "open": float(bar_data.open),
                        "volume": int(bar_data.volume),
                    }
                else:
                    bar = {
                        "symbol": sym, "price": 100.0, "timestamp": now,
                        "close": 100.0, "high": 100.1, "low": 99.9,
                        "open": 100.0, "volume": 0,
                    }
                self._cached_bars[sym] = bar
                events.append(bar)
            return events
        except Exception as e:
            log.debug("REST fetch failed, using cached/synthetic data: %s", e)
            return self._generate_synthetic_bars(now)

    def _generate_synthetic_bars(self, now: datetime) -> list[dict[str, Any]]:
        """Generate synthetic OHLCV data (used when Alpaca is unavailable)."""
        events = []
        for sym in self.symbols:
            cached = self._cached_bars.get(sym)
            price = cached["price"] if cached else 100.0
            ret = self._rng.normal(0.00005, 0.01)
            price *= (1.0 + ret)
            bar = {
                "symbol": sym, "price": price, "timestamp": now,
                "close": price, "high": price * 1.001, "low": price * 0.999,
                "open": price * 0.9995, "volume": int(abs(self._rng.normal(10000, 5000))),
            }
            self._cached_bars[sym] = bar
            events.append(bar)
        return events

    @property
    def last_data_timestamp(self) -> datetime | None:
        return self._last_tick
