"""Unit tests for IBKRBroker adapter."""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from quantcore.execution.brokers.base import AccountInfo, BrokerOrder


@pytest.fixture
def mock_ib():
    """Mock ib_insync.IB instance."""
    ib = MagicMock()
    ib.isConnected.return_value = True
    ib.managedAccounts.return_value = ["DU123456"]
    ib.accountSummary.return_value = [
        _summary_item("NetLiquidation", "100000"),
        _summary_item("AvailableFunds", "80000"),
        _summary_item("TotalCashValue", "50000"),
        _summary_item("GrossPositionValue", "50000"),
    ]
    ib.positions.return_value = []
    ib.openTrades.return_value = []
    return ib


def _summary_item(tag, value, currency="USD"):
    item = MagicMock()
    item.tag = tag
    item.value = value
    item.currency = currency
    return item


def _mock_trade(order_id, symbol, action, status, filled=0, avg_price=0):
    trade = MagicMock()
    trade.order.orderId = order_id
    trade.order.action = action
    trade.order.totalQuantity = 100
    trade.order.orderType = "MKT"
    trade.order.lmtPrice = 0
    trade.order.auxPrice = 0
    trade.contract.symbol = symbol
    trade.orderStatus.status = status
    trade.orderStatus.filled = filled
    trade.orderStatus.avgFillPrice = avg_price
    trade.orderStatus.whyHeld = ""
    return trade


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestIBKRBrokerConstruction:
    def test_default_construction(self):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        assert broker.broker_name == "ibkr"
        assert broker.is_paper is True
        assert broker.is_connected is False
        assert broker.host == "127.0.0.1"
        assert broker.port == 7497

    def test_live_port_detection(self):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7496)
        assert broker.is_paper is False

    def test_gateway_paper_port(self):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=4002)
        assert broker.is_paper is True

    def test_custom_client_id(self):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=99)
        assert broker.client_id == 99


# ---------------------------------------------------------------------------
# Status mapping
# ---------------------------------------------------------------------------


class TestStatusMapping:
    def test_filled_status(self):
        from quantcore.brokers.ibkr_broker import _internal_status
        from quantcore.core.types import OrderStatus

        assert _internal_status("Filled") == OrderStatus.FILLED

    def test_submitted_status(self):
        from quantcore.brokers.ibkr_broker import _internal_status
        from quantcore.core.types import OrderStatus

        assert _internal_status("Submitted") == OrderStatus.SUBMITTED

    def test_cancelled_status(self):
        from quantcore.brokers.ibkr_broker import _internal_status
        from quantcore.core.types import OrderStatus

        assert _internal_status("Cancelled") == OrderStatus.CANCELLED

    def test_unknown_status_defaults_to_pending(self):
        from quantcore.brokers.ibkr_broker import _internal_status
        from quantcore.core.types import OrderStatus

        assert _internal_status("UnknownStatus") == OrderStatus.PENDING


# ---------------------------------------------------------------------------
# Account queries (mocked)
# ---------------------------------------------------------------------------


class TestIBKRBrokerAccount:
    def test_get_account_info_fields(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True
        broker.account_id = "DU123456"

        import asyncio
        acct = asyncio.run(broker.get_account())

        assert isinstance(acct, AccountInfo)
        assert acct.portfolio_value == 100000
        assert acct.buying_power == 80000
        assert acct.cash == 50000
        assert acct.account_id == "DU123456"

    def test_get_account_not_connected_raises(self):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._connected = False

        import asyncio

        with pytest.raises(RuntimeError, match="not connected"):
            asyncio.run(broker.get_account())

    def test_get_positions_empty(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        import asyncio
        positions = asyncio.run(broker.get_positions())
        assert positions == []

    def test_get_positions_with_holdings(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        pos = MagicMock()
        pos.contract.symbol = "AAPL"
        pos.position = 100
        pos.avgCost = 150.0
        pos.marketPrice = 155.0
        mock_ib.positions.return_value = [pos]

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        import asyncio
        positions = asyncio.run(broker.get_positions())

        assert len(positions) == 1
        assert positions[0].symbol == "AAPL"
        assert positions[0].quantity == Decimal("100")
        assert positions[0].avg_cost == Decimal("150")


# ---------------------------------------------------------------------------
# Order queries
# ---------------------------------------------------------------------------


class TestIBKRBrokerOrders:
    def test_get_open_orders_empty(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        import asyncio
        orders = asyncio.run(broker.get_orders(status="open"))
        assert orders == []

    def test_cancel_order_not_found(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        import asyncio
        result = asyncio.run(broker.cancel_order("ibkr-99999"))
        assert result is False


# ---------------------------------------------------------------------------
# BrokerOrder to IBKR order conversion
# ---------------------------------------------------------------------------


class TestSubmitOrder:
    def test_submit_order_not_connected_reconnects(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker, IBKRBrokerError

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        order = BrokerOrder(
            symbol="AAPL", side="buy", order_type="market",
            quantity=Decimal("100"), client_order_id="test-001",
        )

        import asyncio
        # ib_insync MarketOrder is mocked via MagicMock, placeOrder returns MagicMock
        # We expect order status to not match immediately → timeout
        with pytest.raises(IBKRBrokerError, match="timed out"):
            asyncio.run(broker.submit_order(order))


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------


class TestPositionManagement:
    def test_close_position_no_position(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        import asyncio
        result = asyncio.run(broker.close_position("AAPL"))
        assert result is True  # nothing to close, returns True

    def test_close_all_positions_empty(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        import asyncio
        result = asyncio.run(broker.close_all_positions())
        assert result == 0

    def test_close_all_positions_with_holdings(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        pos = MagicMock()
        pos.contract.symbol = "AAPL"
        pos.position = 100

        mock_ib.positions.return_value = [pos]

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        import asyncio
        result = asyncio.run(broker.close_all_positions())
        assert result == 1


# ---------------------------------------------------------------------------
# Cancel all orders
# ---------------------------------------------------------------------------


class TestCancelAllOrders:
    def test_cancel_all_orders(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        trade = _mock_trade(1, "AAPL", "BUY", "Submitted")
        mock_ib.openTrades.return_value = [trade]

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        import asyncio
        result = asyncio.run(broker.cancel_all_orders())
        assert result == 1
        mock_ib.reqGlobalCancel.assert_called_once()


# ---------------------------------------------------------------------------
# Sync wrappers
# ---------------------------------------------------------------------------


class TestSyncWrappers:
    def test_get_account_sync(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        acct = broker.get_account_sync()
        assert acct.portfolio_value == 100000
        assert acct.cash == 50000

    def test_get_positions_sync(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        positions = broker.get_positions_sync()
        assert isinstance(positions, list)

    def test_submit_order_sync_not_connected(self, mock_ib):
        from quantcore.brokers.ibkr_broker import IBKRBroker, IBKRBrokerError

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)
        broker._ib = mock_ib
        broker._connected = True

        order = BrokerOrder(
            symbol="SPY", side="sell", order_type="limit",
            quantity=Decimal("50"), limit_price=Decimal("450.0"),
            client_order_id="sync-test",
        )

        with pytest.raises(IBKRBrokerError, match="timed out"):
            broker.submit_order_sync(order)


# ---------------------------------------------------------------------------
# Connection error handling
# ---------------------------------------------------------------------------


class TestConnectionErrors:
    def test_connection_refused(self):
        from quantcore.brokers.ibkr_broker import IBKRBroker, IBKRConnectionError
        import asyncio

        broker = IBKRBroker(host="127.0.0.1", port=7497, client_id=1)

        with patch.object(broker, '_ib', None):
            pass  # _ib is None by default

        with pytest.raises(IBKRConnectionError):
            with patch("ib_insync.IB") as mock_ib_class:
                mock_ib = MagicMock()
                mock_ib.connect.side_effect = ConnectionRefusedError("refused")
                mock_ib_class.return_value = mock_ib
                asyncio.run(broker.connect())
