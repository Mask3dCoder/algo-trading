"""Tests for paper broker and execution fixes.

Validates:
  C1: Paper broker uses market_price instead of $100 default
  BrokerOrder.market_price field exists
"""

from __future__ import annotations

import asyncio
from decimal import Decimal

import pytest

from quantcore.execution.brokers.base import BrokerOrder
from quantcore.execution.brokers.paper import PaperBroker, PaperConfig


# ---------------------------------------------------------------------------
# C1: Paper broker fill price
# ---------------------------------------------------------------------------

class TestPaperBrokerFillPrice:
    def test_fill_at_limit_price(self):
        """Limit orders should fill at limit price (with slippage)."""
        broker = PaperBroker(PaperConfig(latency_ms=0))
        asyncio.get_event_loop().run_until_complete(broker.connect())

        order = BrokerOrder(
            symbol="AAPL",
            side="buy",
            order_type="limit",
            quantity=Decimal("100"),
            limit_price=Decimal("500.00"),
        )
        broker_id = asyncio.get_event_loop().run_until_complete(
            broker.submit_order(order)
        )
        fills = asyncio.get_event_loop().run_until_complete(broker.get_orders("filled"))
        assert len(fills) == 1
        fill_price = float(fills[0].avg_fill_price)
        # Should be near $500 + 1bps slippage, NOT near $100
        assert 499.0 < fill_price < 501.0

    def test_fill_at_market_price_when_no_limit(self):
        """Market orders should fill at market_price, not $100."""
        broker = PaperBroker(PaperConfig(latency_ms=0))
        asyncio.get_event_loop().run_until_complete(broker.connect())

        order = BrokerOrder(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=Decimal("100"),
            market_price=Decimal("750.00"),
        )
        broker_id = asyncio.get_event_loop().run_until_complete(
            broker.submit_order(order)
        )
        fills = asyncio.get_event_loop().run_until_complete(broker.get_orders("filled"))
        assert len(fills) == 1
        fill_price = float(fills[0].avg_fill_price)
        # Should be near $750, NOT $100
        assert 749.0 < fill_price < 751.0

    def test_fill_fallback_to_100_without_any_price(self):
        """Legacy: without any price, should still fill (at $100 fallback)."""
        broker = PaperBroker(PaperConfig(latency_ms=0))
        asyncio.get_event_loop().run_until_complete(broker.connect())

        order = BrokerOrder(
            symbol="AAPL",
            side="buy",
            order_type="market",
            quantity=Decimal("100"),
            # No limit_price, no market_price
        )
        broker_id = asyncio.get_event_loop().run_until_complete(
            broker.submit_order(order)
        )
        fills = asyncio.get_event_loop().run_until_complete(broker.get_orders("filled"))
        assert len(fills) == 1
        fill_price = float(fills[0].avg_fill_price)
        # Fallback to $100
        assert 99.0 < fill_price < 101.0

    def test_sell_order_uses_market_price(self):
        """Sell orders should also use market_price."""
        broker = PaperBroker(PaperConfig(latency_ms=0))
        asyncio.get_event_loop().run_until_complete(broker.connect())

        # First buy to have a position
        buy = BrokerOrder(
            symbol="AAPL", side="buy", order_type="market",
            quantity=Decimal("100"), market_price=Decimal("500.00"),
        )
        asyncio.get_event_loop().run_until_complete(broker.submit_order(buy))

        # Then sell
        sell = BrokerOrder(
            symbol="AAPL", side="sell", order_type="market",
            quantity=Decimal("100"), market_price=Decimal("500.00"),
        )
        asyncio.get_event_loop().run_until_complete(broker.submit_order(sell))

        fills = asyncio.get_event_loop().run_until_complete(broker.get_orders("filled"))
        assert len(fills) == 2
        sell_price = float(fills[1].avg_fill_price)
        # Sell should be near $500 - 1bps
        assert 499.0 < sell_price < 501.0


# ---------------------------------------------------------------------------
# BrokerOrder.market_price field
# ---------------------------------------------------------------------------

class TestBrokerOrderField:
    def test_market_price_field_exists(self):
        """BrokerOrder should accept market_price field."""
        order = BrokerOrder(
            symbol="AAPL", side="buy", order_type="market",
            quantity=Decimal("100"), market_price=Decimal("450.00"),
        )
        assert order.market_price == Decimal("450.00")

    def test_market_price_defaults_to_none(self):
        """BrokerOrder.market_price should default to None."""
        order = BrokerOrder(
            symbol="AAPL", side="buy", order_type="market",
            quantity=Decimal("100"),
        )
        assert order.market_price is None
