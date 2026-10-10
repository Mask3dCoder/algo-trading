"""Hybrid configuration: Alpaca data feed + IBKR execution.

This is the recommended production setup:
- Alpaca provides free real-time market data via IEX
- IBKR handles order execution with professional routing

Data flows: Alpaca WebSocket → signal engine → OMS → IBKR broker
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class HybridFeed:
    """Delegates market data to Alpaca, wraps IBKR for execution.

    Usage:
        hybrid = HybridFeed(
            alpaca_feed=<AlpacaPaperFeed>,
            ibkr_broker=<IBKRBroker>,
        )
        # Use hybrid.feed for data, hybrid.broker for execution
    """

    def __init__(
        self,
        feed: Any = None,
        broker: Any = None,
        symbols: list[str] | None = None,
    ) -> None:
        self.feed = feed
        self.broker = broker
        self.symbols = symbols or ["SPY"]
        self._connected = False

    def connect(self) -> None:
        """Connect both Alpaca feed and IBKR broker. Fail fast if either fails."""
        errors: list[str] = []

        if self.feed is not None:
            try:
                self.feed.connect()
                log.info("Hybrid: Alpaca feed connected")
            except Exception as e:
                errors.append(f"Alpaca feed: {e}")
        else:
            errors.append("Alpaca feed not configured")

        if self.broker is not None:
            try:
                self.broker.connect_sync()
                log.info("Hybrid: IBKR broker connected")
            except Exception as e:
                errors.append(f"IBKR broker: {e}")
        else:
            errors.append("IBKR broker not configured")

        if errors:
            raise ConnectionError(
                "Hybrid configuration failed:\n  - " + "\n  - ".join(errors)
            )

        self._connected = True
        log.info("Hybrid mode active: Alpaca data + IBKR execution — %d symbols", len(self.symbols))

    def disconnect(self) -> None:
        """Disconnect both feed and broker."""
        if self.feed:
            try:
                self.feed.disconnect()
            except Exception as e:
                log.warning("Error disconnecting Alpaca feed: %s", e)
        if self.broker:
            try:
                self.broker.disconnect_sync()
            except Exception as e:
                log.warning("Error disconnecting IBKR broker: %s", e)
        self._connected = False

    def get_latest(self) -> list[dict[str, Any]]:
        """Get latest market data from Alpaca feed."""
        if self.feed is None:
            return []
        return self.feed.get_latest()

    @property
    def last_data_timestamp(self):
        if self.feed:
            return self.feed.last_data_timestamp
        return None

    @property
    def is_connected(self) -> bool:
        return self._connected


def create_hybrid(
    symbols: list[str] | None = None,
    ibkr_host: str | None = None,
    ibkr_port: int | None = None,
    ibkr_client_id: int | None = None,
    initial_capital: float = 100_000.0,
) -> HybridFeed:
    """Factory for creating a hybrid Alpaca data + IBKR execution setup.

    Reads Alpaca credentials from environment and IBKR config from args.
    """
    from quantcore.brokers.ibkr_broker import IBKRBroker
    from quantcore.feeds.alpaca_feed import AlpacaPaperFeed

    syms = symbols or ["SPY"]

    # Alpaca feed for data
    alpaca_feed = AlpacaPaperFeed(symbols=syms)

    # IBKR broker for execution
    ibkr_broker = IBKRBroker(
        host=ibkr_host,
        port=ibkr_port,
        client_id=ibkr_client_id,
    )

    hybrid = HybridFeed(feed=alpaca_feed, broker=ibkr_broker, symbols=syms)
    return hybrid
