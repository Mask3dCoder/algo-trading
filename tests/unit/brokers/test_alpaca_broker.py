"""Tests for AlpacaBroker — no real network calls.

All external APIs are mocked via unittest.mock.patch.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any
from unittest import mock

import pytest


@pytest.fixture
def broker() -> Any:
    """Create a test AlpacaBroker with fake credentials."""
    from quantcore.brokers.alpaca_broker import AlpacaBroker

    return AlpacaBroker(
        api_key="test_key",
        secret_key="test_secret",
        mode="paper",
    )


class TestAlpacaBrokerInterface:
    """Verify AlpacaBroker implements BrokerAdapter."""

    def test_implements_broker_adapter(self, broker) -> None:
        from quantcore.execution.brokers.base import BrokerAdapter

        assert isinstance(broker, BrokerAdapter)

    def test_has_all_required_methods(self, broker) -> None:
        methods = [
            "connect", "disconnect", "submit_order", "cancel_order",
            "cancel_all_orders", "get_order", "get_orders", "get_open_orders",
            "get_position", "get_positions", "close_position",
            "close_all_positions", "get_account",
        ]
        for method in methods:
            assert hasattr(broker, method), f"Missing method: {method}"

    def test_broker_name(self, broker) -> None:
        assert broker.broker_name == "alpaca"


class TestCredentials:
    """Test credential handling."""

    def test_constructor_args_take_priority(self) -> None:
        from quantcore.brokers.alpaca_broker import AlpacaBroker

        b = AlpacaBroker(api_key="my_key", secret_key="my_secret")
        assert b.api_key == "my_key"
        assert b.secret_key == "my_secret"

    def test_missing_credentials_raises_auth_error(self, monkeypatch) -> None:
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

        from quantcore.brokers.alpaca_broker import AlpacaBroker, BrokerAuthError

        with pytest.raises(BrokerAuthError, match="ALPACA_API_KEY"):
            AlpacaBroker(api_key=None, secret_key=None)


class TestPaperVsLiveLogging:
    """Test paper vs live mode log messages."""

    def test_paper_mode_logs_safe_message(self, caplog) -> None:
        import logging

        from quantcore.brokers.alpaca_broker import AlpacaBroker

        caplog.set_level(logging.INFO)
        AlpacaBroker(api_key="k", secret_key="s", mode="paper")
        messages = [r.message for r in caplog.records]
        paper_msgs = [m for m in messages if "no real money at risk" in m.lower()]
        assert len(paper_msgs) == 1

    def test_live_mode_logs_warning(self, caplog, monkeypatch) -> None:
        from quantcore.brokers.alpaca_broker import AlpacaBroker

        # Override base URL so mode="live" is not overridden by paper URL detection
        monkeypatch.setenv("ALPACA_BASE_URL", "https://api.alpaca.markets")
        AlpacaBroker(api_key="k", secret_key="s", mode="live")
        messages = [r.message for r in caplog.records]
        live_msgs = [m for m in messages if "real money at risk" in m.lower()]
        assert len(live_msgs) == 1

    def test_paper_mode_detected_from_url(self, monkeypatch, caplog) -> None:
        monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        from quantcore.brokers.alpaca_broker import AlpacaBroker

        b = AlpacaBroker(api_key="k", secret_key="s")
        assert b.is_paper is True


class TestStatusMapping:
    """Verify Alpaca → Internal order status mapping."""

    def test_all_alpaca_statuses_mapped(self) -> None:
        from quantcore.brokers.alpaca_broker import _ALPACA_STATUS_MAP
        from quantcore.core.types import OrderStatus

        alpaca_statuses = [
            "new", "partially_filled", "filled", "done_for_day",
            "cancelled", "expired", "replaced", "pending_cancel",
            "pending_replace", "held", "accepted", "accepted_for_bidding",
            "stopped", "rejected", "suspended", "calculated",
        ]
        for status in alpaca_statuses:
            mapped = _ALPACA_STATUS_MAP.get(status)
            assert mapped is not None, f"Missing mapping for {status}"
            assert isinstance(mapped, OrderStatus), f"Invalid mapping for {status}: {mapped}"

    def test_filled_statuses_map_to_filled(self) -> None:
        from quantcore.brokers.alpaca_broker import _ALPACA_STATUS_MAP
        from quantcore.core.types import OrderStatus

        assert _ALPACA_STATUS_MAP["filled"] == OrderStatus.FILLED
        assert _ALPACA_STATUS_MAP["calculated"] == OrderStatus.FILLED

    def test_rejected_statuses_map_to_rejected(self) -> None:
        from quantcore.brokers.alpaca_broker import _ALPACA_STATUS_MAP
        from quantcore.core.types import OrderStatus

        assert _ALPACA_STATUS_MAP["rejected"] == OrderStatus.REJECTED
        assert _ALPACA_STATUS_MAP["suspended"] == OrderStatus.REJECTED

    def test_cancelled_variants_map_to_cancelled(self) -> None:
        from quantcore.brokers.alpaca_broker import _ALPACA_STATUS_MAP
        from quantcore.core.types import OrderStatus

        for status in ("done_for_day", "cancelled", "expired", "replaced"):
            assert _ALPACA_STATUS_MAP[status] == OrderStatus.CANCELLED


class TestOrderConversion:
    """Verify _alpaca_order_to_internal field mapping."""

    def test_converts_alpaca_order_to_internal(self) -> None:
        from dataclasses import dataclass
        from enum import Enum

        class MockSide(Enum):
            BUY = "buy"
            SELL = "sell"

        @dataclass
        class MockAlpacaOrder:
            id: str = "test-order-123"
            symbol: str = "SPY"
            side: MockSide = MockSide.BUY
            type: str = "market"
            qty: str = "100"
            limit_price: str | None = None
            stop_price: str | None = None
            status: str = "filled"
            filled_qty: str = "100"
            filled_avg_price: str = "450.50"

        mock_ao = MockAlpacaOrder()

        with mock.patch("alpaca.trading.enums.OrderSide", create=True) as mock_side:
            mock_side.BUY = "buy"
            mock_side.SELL = "sell"

            from quantcore.brokers.alpaca_broker import _alpaca_order_to_internal
            order = _alpaca_order_to_internal(mock_ao)

        assert order.id == "test-order-123"
        assert order.symbol == "SPY"
        assert order.quantity == Decimal("100")
        assert order.filled_qty == Decimal("100")
        assert order.avg_fill_price == Decimal("450.50")

    def test_market_order_default_type(self) -> None:
        from dataclasses import dataclass
        from enum import Enum

        class MockSide(Enum):
            BUY = "buy"
            SELL = "sell"

        @dataclass
        class MockOrder:
            id: str = "o1"
            symbol: str = "QQQ"
            side: MockSide = MockSide.SELL
            type: str = "market"
            qty: str = "50"
            limit_price: str | None = None
            stop_price: str | None = None
            status: str = "new"
            filled_qty: str = "0"
            filled_avg_price: str | None = None

        mock_ao = MockOrder()

        with mock.patch("alpaca.trading.enums.OrderSide", create=True) as mo:
            mo.BUY = "buy"
            mo.SELL = "sell"

            from quantcore.brokers.alpaca_broker import _alpaca_order_to_internal
            order = _alpaca_order_to_internal(mock_ao)

        assert order.quantity == Decimal("50")


class TestErrorHandling:
    """Verify API error handling."""

    def test_cancel_order_returns_false_on_failure(self, broker) -> None:
        """cancel_order should return False, not raise, on failure."""
        from quantcore.brokers.alpaca_broker import AlpacaBroker

        b = AlpacaBroker(api_key="k", secret_key="s", mode="paper")
        # No client connected — should return False gracefully
        assert b._client is None
        # Async method — test via sync wrapper pattern
        import asyncio

        async def run() -> bool:
            return await b.cancel_order("nonexistent")

        result = asyncio.run(run())
        assert result is False

    def test_close_position_returns_false_when_no_client(self, broker) -> None:
        import asyncio

        async def run() -> bool:
            return await broker.close_position("SPY")

        result = asyncio.run(run())
        assert result is False
