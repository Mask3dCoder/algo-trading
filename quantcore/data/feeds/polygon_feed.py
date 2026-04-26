"""Polygon and Crypto WebSocket feed handlers."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from typing import Any

from quantcore.core.types import OHLCV, Tick, Quote
from quantcore.data.feeds.base import FeedHandler


class PolygonFeed(FeedHandler):
    """Polygon.io WebSocket feed for trades, quotes, and aggregates."""

    WS_URL = "wss://socket.polygon.io/stocks"

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self.api_key = api_key
        self._ws: Any = None

    async def connect(self) -> None:
        try:
            import websockets
            self._ws = await websockets.connect(self.WS_URL)
            # Authenticate
            await self._ws.send(json.dumps({"action": "auth", "params": self.api_key}))
            resp = await self._ws.recv()
            data = json.loads(resp)
            if data.get("ev") == "status" and data.get("status") == "auth_success":
                self._connected = True
            else:
                raise ConnectionError(f"Polygon auth failed: {data}")
        except ImportError:
            raise ImportError("websockets required")

    async def subscribe(self, symbols: list[str]) -> None:
        if not self._ws:
            raise RuntimeError("Not connected")
        self.add_symbols(symbols)
        # Subscribe to trades (T.), quotes (Q.), and aggregates (A.)
        params = ",".join([f"T.{s},Q.{s},A.{s}" for s in symbols])
        await self._ws.send(json.dumps({"action": "subscribe", "params": params}))

    async def listen(self) -> None:
        if not self._ws:
            return
        try:
            async for message in self._ws:
                data = json.loads(message)
                if isinstance(data, list):
                    for event in data:
                        await self._process_event(event)
        except Exception:
            self._connected = False
            await self._reconnect_with_backoff()

    async def _process_event(self, event: dict) -> None:
        ev = event.get("ev")
        if ev == "T":
            tick = Tick(
                symbol=event.get("sym", ""),
                timestamp_us=int(event.get("t", 0)),
                price=Decimal(str(event.get("p", 0))),
                size=int(event.get("s", 0)),
                exchange=event.get("x", ""),
            )
            await self._emit_tick(tick)
        elif ev == "Q":
            quote = Quote(
                symbol=event.get("sym", ""),
                timestamp=datetime.utcnow(),
                bid=Decimal(str(event.get("bp", 0))),
                ask=Decimal(str(event.get("ap", 0))),
                bid_size=int(event.get("bs", 0)),
                ask_size=int(event.get("as", 0)),
            )
            await self._emit_quote(quote)
        elif ev == "A":
            bar = OHLCV(
                symbol=event.get("sym", ""),
                timestamp=datetime.utcnow(),
                open=Decimal(str(event.get("o", 0))),
                high=Decimal(str(event.get("h", 0))),
                low=Decimal(str(event.get("l", 0))),
                close=Decimal(str(event.get("c", 0))),
                volume=int(event.get("v", 0)),
            )
            await self._emit_bar(bar)

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._connected = False


class CryptoFeed(FeedHandler):
    """CCXT-based WebSocket feed for crypto exchanges."""

    def __init__(self, exchange_id: str = "binance") -> None:
        super().__init__()
        self.exchange_id = exchange_id
        self._exchange: Any = None

    async def connect(self) -> None:
        try:
            import ccxt.async_support as ccxt_async
            exchange_class = getattr(ccxt_async, self.exchange_id)
            self._exchange = exchange_class()
            await self._exchange.load_markets()
            self._connected = True
        except ImportError:
            raise ImportError("ccxt required: pip install ccxt")

    async def subscribe(self, symbols: list[str]) -> None:
        self.add_symbols(symbols)
        # CCXT uses watch methods — actual subscription happens in listen()

    async def listen(self) -> None:
        """Watch trades using CCXT's WebSocket support."""
        if not self._exchange:
            return

        while self._connected:
            for symbol in self._subscriptions:
                try:
                    trades = await self._exchange.watch_trades(symbol)
                    for trade in trades:
                        tick = Tick(
                            symbol=symbol,
                            timestamp_us=int(trade.get("timestamp", 0)),
                            price=Decimal(str(trade.get("price", 0))),
                            size=int(trade.get("amount", 0)),
                        )
                        await self._emit_tick(tick)
                except Exception:
                    await asyncio.sleep(1)

    async def disconnect(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._connected = False
