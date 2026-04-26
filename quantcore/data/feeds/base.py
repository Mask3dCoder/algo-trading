"""Abstract feed handler and Alpaca WebSocket feed implementation."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Callable

from quantcore.core.types import OHLCV, Tick, Quote


class FeedHandler(ABC):
    """Abstract base class for real-time market data feed handlers."""

    def __init__(self) -> None:
        self._connected = False
        self._subscriptions: set[str] = set()
        self._on_tick_callbacks: list[Callable] = []
        self._on_bar_callbacks: list[Callable] = []
        self._on_quote_callbacks: list[Callable] = []
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._base_reconnect_delay = 1.0

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        ...

    @abstractmethod
    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to real-time data for the given symbols."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the data source."""
        ...

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def subscriptions(self) -> set[str]:
        return self._subscriptions.copy()

    def on_tick(self, callback: Callable) -> None:
        """Register a callback for tick events."""
        self._on_tick_callbacks.append(callback)

    def on_bar(self, callback: Callable) -> None:
        """Register a callback for bar events."""
        self._on_bar_callbacks.append(callback)

    def on_quote(self, callback: Callable) -> None:
        """Register a callback for quote events."""
        self._on_quote_callbacks.append(callback)

    async def _emit_tick(self, tick: Tick) -> None:
        for cb in self._on_tick_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(tick)
            else:
                cb(tick)

    async def _emit_bar(self, bar: OHLCV) -> None:
        for cb in self._on_bar_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(bar)
            else:
                cb(bar)

    async def _emit_quote(self, quote: Quote) -> None:
        for cb in self._on_quote_callbacks:
            if asyncio.iscoroutinefunction(cb):
                await cb(quote)
            else:
                cb(quote)

    async def _reconnect_with_backoff(self) -> None:
        """Attempt reconnection with exponential backoff."""
        for attempt in range(self._max_reconnect_attempts):
            delay = self._base_reconnect_delay * (2 ** attempt)
            await asyncio.sleep(delay)
            try:
                await self.connect()
                self._reconnect_attempts = 0
                return
            except Exception:
                self._reconnect_attempts += 1
                continue
        raise ConnectionError(f"Failed to reconnect after {self._max_reconnect_attempts} attempts")

    def add_symbols(self, symbols: list[str]) -> None:
        self._subscriptions.update(symbols)

    def remove_symbols(self, symbols: list[str]) -> None:
        self._subscriptions -= set(symbols)


class AlpacaFeed(FeedHandler):
    """Alpaca WebSocket feed for real-time trades, quotes, and bars."""

    WS_URL = "wss://stream.data.alpaca.markets/v2/iex"

    def __init__(self, api_key: str, secret_key: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self._ws: Any = None
        self._authenticated = False

    async def connect(self) -> None:
        """Connect and authenticate to Alpaca data stream."""
        try:
            import websockets
            self._ws = await websockets.connect(self.WS_URL)
            # Authenticate
            auth_msg = {"action": "auth", "key": self.api_key, "secret": self.secret_key}
            await self._ws.send(json.dumps(auth_msg))
            resp = await self._ws.recv()
            data = json.loads(resp)
            if data.get("type") == "authorization" and data.get("msg") == "connected":
                self._authenticated = True
                self._connected = True
            else:
                raise ConnectionError(f"Alpaca auth failed: {data}")
        except ImportError:
            raise ImportError("websockets package required: pip install websockets")

    async def subscribe(self, symbols: list[str]) -> None:
        """Subscribe to trades, quotes, and bars for symbols."""
        if not self._connected or not self._ws:
            raise RuntimeError("Not connected")

        self.add_symbols(symbols)
        sub_msg = {
            "action": "subscribe",
            "trades": symbols,
            "quotes": symbols,
            "bars": symbols,
        }
        await self._ws.send(json.dumps(sub_msg))

    async def listen(self) -> None:
        """Main listening loop — parse and emit events."""
        if not self._ws:
            return

        try:
            async for message in self._ws:
                data = json.loads(message)
                if isinstance(data, list):
                    for event in data:
                        await self._process_event(event)
                elif isinstance(data, dict):
                    await self._process_event(data)
        except Exception:
            self._connected = False
            await self._reconnect_with_backoff()

    async def _process_event(self, event: dict) -> None:
        """Parse an Alpaca WebSocket message and emit appropriate event."""
        event_type = event.get("T")

        if event_type == "t":
            # Trade
            tick = Tick(
                symbol=event.get("S", ""),
                timestamp_us=int(event.get("t", 0) / 1e6),
                price=Decimal(str(event.get("p", 0))),
                size=int(event.get("s", 0)),
                exchange=event.get("x", ""),
            )
            await self._emit_tick(tick)

        elif event_type == "q":
            # Quote
            quote = Quote(
                symbol=event.get("S", ""),
                timestamp=datetime.utcnow(),
                bid=Decimal(str(event.get("bp", 0))),
                ask=Decimal(str(event.get("ap", 0))),
                bid_size=int(event.get("bs", 0)),
                ask_size=int(event.get("as", 0)),
            )
            await self._emit_quote(quote)

        elif event_type == "b":
            # Bar
            bar = OHLCV(
                symbol=event.get("S", ""),
                timestamp=datetime.utcnow(),
                open=Decimal(str(event.get("o", 0))),
                high=Decimal(str(event.get("h", 0))),
                low=Decimal(str(event.get("l", 0))),
                close=Decimal(str(event.get("c", 0))),
                volume=int(event.get("v", 0)),
            )
            await self._emit_bar(bar)

    async def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        if self._ws:
            await self._ws.close()
            self._connected = False
            self._authenticated = False
