"""Test that cancel_order/cancel_all_orders have no NameError (bug fix).

Original bug (broker.py:188): the log line referenced an undefined variable `symbol`
instead of `record.symbol`, causing NameError crashes in live trading.

These tests verify:
1. cancel_order() completes without NameError for a valid paper order
2. cancel_all_orders() completes without NameError for multiple paper orders
3. Unknown order IDs are handled gracefully (no crash)
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from quantcore.core.types import Order, OrderStatus, OrderType, Side
from quantcore.live.broker import LiveExecutionGateway


class TestCancelOrderNameError:
    """Verify cancel_order() has no undefined variable NameError."""

    def _create_pending_order(
        self, gw: LiveExecutionGateway, symbol: str = "AAPL"
    ) -> str:
        """Helper: inject a pending paper order directly into the gateway.

        Skips submit_order() to avoid immediate paper filling.  This lets
        tests verify cancellation logic on truly PENDING orders.
        """
        from quantcore.live.broker import _PaperOrderRecord

        record = _PaperOrderRecord(
            symbol=symbol,
            side=Side.BUY,
            quantity=Decimal("1"),
            order_type="market",
        )
        record.status = OrderStatus.PENDING
        gw._paper_orders[record.id] = record
        gw._orders[record.id] = gw._order_to_typed(record)
        return record.id

    def test_cancel_order_no_name_error(self) -> None:
        """cancel_order() on a valid pending paper order must NOT raise NameError."""
        with patch.dict("os.environ"):
            for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
                pass
            gw = LiveExecutionGateway(initial_cash=100_000)
            gw._paper = True

            # Create a pending order directly (submit_order fills immediately)
            order_id = self._create_pending_order(gw)

            # Cancel it — must NOT raise NameError
            result = gw.cancel_order(order_id)
            assert result is True

    def test_cancel_order_unknown_id_graceful(self) -> None:
        """cancel_order() with a non-existent order ID must return False, not crash."""
        with patch.dict("os.environ"):
            for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
                pass
            gw = LiveExecutionGateway(initial_cash=100_000)
            gw._paper = True

            # Cancel something that was never submitted
            result = gw.cancel_order("NOT_A_REAL_ID_12345")
            assert result is False

    def test_cancel_all_orders_no_name_error(self) -> None:
        """cancel_all_orders() with multiple pending orders must NOT raise NameError."""
        with patch.dict("os.environ"):
            for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
                pass
            gw = LiveExecutionGateway(initial_cash=100_000)
            gw._paper = True

            # Create several pending orders directly
            order_ids = []
            for symbol in ["AAPL", "SPY", "QQQ"]:
                oid = self._create_pending_order(gw, symbol)
                order_ids.append(oid)

            # Cancel all — must NOT raise NameError
            result = gw.cancel_all_orders()
            assert result is True

            # Verify all are cancelled
            for oid in order_ids:
                rec = gw._paper_orders.get(oid)
                assert rec is not None
                assert rec.status == OrderStatus.CANCELLED

    def test_cancel_all_orders_no_orders_present(self) -> None:
        """cancel_all_orders() when no orders exist must return True, not crash."""
        with patch.dict("os.environ"):
            for k in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL"):
                pass
            gw = LiveExecutionGateway(initial_cash=100_000)
            gw._paper = True

            result = gw.cancel_all_orders()
            assert result is True