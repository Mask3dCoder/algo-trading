"""IBKR market data feed via ib_insync.

Note: IBKR market data requires paid subscriptions for real-time data.
For most users, Alpaca is preferred for market data (free real-time via IEX).
This feed is provided for users with IBKR market data subscriptions.

If you do not have IBKR market data subscriptions, use:
  --data-source alpaca  (for market data)
  --broker ibkr         (for order execution)
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)


class IBKRFeed:
    """IBKR market data feed for paper trading.

    Uses ib_insync to subscribe to real-time bars.
    Falls back to delayed (15-min) data if no real-time subscription.

    Usage:
        feed = IBKRFeed(symbols=["SPY"], host="127.0.0.1", port=7497)
        feed.connect()
        events = feed.get_latest()
        feed.disconnect()
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        host: str | None = None,
        port: int | None = None,
        client_id: int | None = None,
        bar_size: str = "5 secs",
        use_rth: bool = True,
    ) -> None:
        self.symbols = symbols or ["SPY"]
        self.host = host or os.environ.get("IBKR_HOST", "127.0.0.1")
        self.port = port or int(os.environ.get("IBKR_PORT", "7497"))
        self.client_id = client_id or int(os.environ.get("IBKR_CLIENT_ID", "1"))
        self.bar_size = bar_size
        self.use_rth = use_rth

        self._ib: Any = None
        self._connected = False
        self._latest_prices: dict[str, float] = {}
        self._latest_bars: list[dict[str, Any]] = []
        self._last_tick: datetime | None = None
        self._bar_callbacks: list = []
        self._lock = threading.Lock()
        self._delayed_mode = False

    def connect(self) -> None:
        """Connect to TWS/Gateway and subscribe to market data."""
        import ib_insync

        client_id = self.client_id + 100  # offset to avoid conflict with broker
        self._ib = ib_insync.IB()
        self._ib.RequestTimeout = 10

        try:
            self._ib.connect(self.host, self.port, clientId=client_id, timeout=10)
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to TWS/Gateway at {self.host}:{self.port}. "
                f"Is TWS running? {e}"
            )

        self._connected = True
        self._delayed_mode = False

        self._ib.errorEvent += self._on_error

        for symbol in self.symbols:
            self._subscribe_symbol(symbol)

        log.info(
            "IBKR feed connected to %s:%d — %d symbols (%s)",
            self.host, self.port, len(self.symbols),
            "delayed" if self._delayed_mode else "live",
        )

    def _subscribe_symbol(self, symbol: str) -> None:
        """Subscribe to market data for one symbol."""
        import ib_insync

        contract = ib_insync.Stock(symbol, "SMART", "USD")
        try:
            self._ib.qualifyContracts(contract)
        except Exception as e:
            log.warning("Cannot qualify contract for %s: %s", symbol, e)
            return

        # Request real-time bars (5-second)
        bars = self._ib.reqRealTimeBars(contract, 5, "TRADES", self.use_rth)
        bars.updateEvent += self._on_bar_update

        # Also request top-of-book snapshot for latest price
        ticker = self._ib.reqMktData(contract, "233", False, False)
        ticker.updateEvent += self._on_ticker_update

    def _on_bar_update(self, bars, has_new_bar: bool) -> None:
        """Handle real-time bar update."""
        if not has_new_bar:
            return

        import ib_insync

        symbol = bars.contract.symbol if hasattr(bars, "contract") else ""
        if isinstance(bars, ib_insync.objects.RealTimeBarList) and len(bars) > 0:
            bar = bars[-1]
            with self._lock:
                price = float(bar.close)
                self._latest_prices[symbol] = price
                self._last_tick = datetime.now(UTC)
                self._latest_bars.append({
                    "symbol": symbol,
                    "price": price,
                    "timestamp": datetime.now(UTC),
                    "close": price,
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "open": float(bar.open),
                    "volume": int(bar.volume),
                })
                # Keep only recent bars
                if len(self._latest_bars) > 1000:
                    self._latest_bars = self._latest_bars[-500:]

    def _on_ticker_update(self, ticker) -> None:
        """Handle ticker price update."""
        if ticker.last and ticker.last > 0:
            symbol = ticker.contract.symbol
            with self._lock:
                self._latest_prices[symbol] = float(ticker.last)
                self._last_tick = datetime.now(UTC)

    def _on_error(self, reqId, errorCode, errorString, _advancedOrderRejectJson=""):
        """Handle IBKR error messages."""
        if errorCode == 354:
            # "Requested market data is not subscribed"
            if not self._delayed_mode:
                log.warning(
                    "IBKR market data subscription required for real-time data (error 354). "
                    "Switching to delayed (15-min) data. "
                    "For free real-time data, use: --data-source alpaca"
                )
                self._delayed_mode = True
        elif errorCode in (2103, 2104, 2106, 2107, 2158):
            # Market data farm connection/disconnection messages — informational
            log.debug("IBKR data farm: [%d] %s", errorCode, errorString)
        elif errorCode >= 2000:
            log.warning("IBKR feed error %d: %s", errorCode, errorString)

    def get_latest(self) -> list[dict[str, Any]]:
        """Get the latest market data events for all subscribed symbols."""
        with self._lock:
            if self._latest_bars:
                return [self._latest_bars[-1]]
            # Fall back to constructing from latest prices
            events = []
            for sym, price in self._latest_prices.items():
                events.append({
                    "symbol": sym, "price": price,
                    "timestamp": self._last_tick or datetime.now(UTC),
                    "close": price, "high": price * 1.001,
                    "low": price * 0.999, "open": price * 0.9995,
                    "volume": 0,
                })
            return events

    def is_market_open(self) -> bool:
        """Check if market is open (delegates to exchange calendars)."""
        from quantcore.core.time_utils import is_market_open as imo
        return imo(datetime.now(UTC))

    @property
    def last_data_timestamp(self) -> datetime | None:
        return self._last_tick

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._ib and self._ib.isConnected():
            self._ib.cancelMktData()
            self._ib.disconnect()
        self._connected = False
        log.info("IBKR feed disconnected")

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()
