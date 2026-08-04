from __future__ import annotations

from decimal import Decimal

import pytest

from quantcore.brokers.ibkr_broker import _build_ibkr_contract
from quantcore.execution.brokers.base import BrokerOrder
from quantcore.live.broker import LiveExecutionGateway
from quantcore.live.orders import (
    ExecutionReport,
    InternalOrder,
    InternalOrderType,
    InternalTradeSide,
)


def test_internal_order_is_frozen_and_broker_neutral() -> None:
    order = InternalOrder(
        id="ord-1",
        symbol="AAPL",
        side=InternalTradeSide.BUY,
        order_type=InternalOrderType.MARKET,
        quantity=Decimal("10"),
    )

    assert order.id == "ord-1"
    assert order.side == InternalTradeSide.BUY
    assert order.order_type == InternalOrderType.MARKET
    assert order.quantity == Decimal("10")
    with pytest.raises(AttributeError):
        order.symbol = "MSFT"


def test_live_gateway_coerces_legacy_order_objects() -> None:
    gateway = LiveExecutionGateway(initial_cash=100_000)
    gateway._adapter = None
    gateway._paper = True

    from quantcore.core.types import Order, OrderType, Side

    legacy = Order(
        id="legacy",
        symbol="AAPL",
        side=Side.BUY,
        type=OrderType.MARKET,
        quantity=Decimal("1"),
    )

    order_id = gateway.submit_order(legacy)  # type: ignore[arg-type]

    assert order_id != ""
    assert gateway._paper_orders[order_id].status.value == "rejected"


def test_broker_order_carries_metadata_for_ibkr_translation() -> None:
    order = BrokerOrder(
        symbol="EUR.USD",
        side="buy",
        order_type="market",
        quantity=Decimal("1"),
        metadata={"sec_type": "CASH", "currency": "USD"},
    )

    assert order.metadata["sec_type"] == "CASH"
    assert order.metadata["currency"] == "USD"


def test_ibkr_contract_uses_metadata_exchange_currency_and_asset_class() -> None:
    class Ib:
        def Contract(self):
            return type("Contract", (), {})()

        def Stock(self, symbol, exchange, currency, **kwargs):
            contract = type("Contract", (), {})()
            contract.symbol = symbol
            contract.exchange = exchange
            contract.currency = currency
            contract.secType = "STK"
            contract.kwargs = kwargs
            return contract

    ib = Ib()
    order = InternalOrder(
        id="ord-2",
        symbol="BHP",
        side=InternalTradeSide.BUY,
        order_type=InternalOrderType.MARKET,
        quantity=Decimal("100"),
        metadata={"exchange": "ASX", "currency": "AUD", "sec_type": "STK"},
    )

    contract = _build_ibkr_contract(ib, order.symbol, order.metadata)

    assert contract.symbol == "BHP"
    assert contract.exchange == "ASX"
    assert contract.currency == "AUD"
    assert contract.secType == "STK"


def test_execution_report_rejected_factory() -> None:
    report = ExecutionReport.rejected("test", "ord-1", "no adapter")

    assert report.broker_name == "test"
    assert report.internal_order_id == "ord-1"
    assert report.accepted is False
    assert report.broker_order_id is None
    assert report.message == "no adapter"
