from __future__ import annotations

from decimal import Decimal

import pytest

from quantcore.brokers.alpaca_broker import AlpacaBroker
from quantcore.brokers.ibkr_broker import IBKRBroker
from quantcore.core.types import Side

try:
    from quantcore.execution.brokers.base import InstrumentSpec

    _HAS_INSTRUMENT_SPEC = True
except ImportError:  # InstrumentSpec lives on broker-consolidation-wip branch
    _HAS_INSTRUMENT_SPEC = False

from quantcore.live.execution_state import (
    ExecutionEvent,
    ExecutionEventState,
    ExecutionStateMonitor,
)

requires_instrument_spec = pytest.mark.skipif(
    not _HAS_INSTRUMENT_SPEC,
    reason="InstrumentSpec + broker classmethod conversions moved to broker-consolidation-wip; "
    "re-enable after that branch merges",
)


@requires_instrument_spec
def test_instrument_spec_normalizes_and_validates_signature() -> None:
    instrument = InstrumentSpec(" bhp ", "stk", "asx", "aud")

    assert instrument.symbol == "BHP"
    assert instrument.asset_class == "STK"
    assert instrument.exchange == "ASX"
    assert instrument.currency == "AUD"
    assert instrument.multiplier == Decimal("1")

    with pytest.raises(ValueError):
        InstrumentSpec("BHP", "STOCK", "ASX", "AUD")

    with pytest.raises(ValueError):
        InstrumentSpec("BHP", "STK", "ASX", "AUD", Decimal("0"))


@requires_instrument_spec
def test_ibkr_adapter_exposes_typed_contract_conversion() -> None:
    class Ib:
        def Stock(self, symbol: str, exchange: str, currency: str):
            return type("Contract", (), {
                "symbol": symbol,
                "exchange": exchange,
                "currency": currency,
                "secType": "STK",
            })()

    instrument = InstrumentSpec("BHP", "STK", "ASX", "AUD")
    contract = IBKRBroker.to_ibkr_contract(instrument, Ib())

    assert contract.symbol == "BHP"
    assert contract.exchange == "ASX"
    assert contract.currency == "AUD"
    assert contract.secType == "STK"


@requires_instrument_spec
def test_alpaca_adapter_exposes_typed_symbol_conversion() -> None:
    assert AlpacaBroker.to_alpaca_symbol(InstrumentSpec("AAPL", "STK", "NASDAQ", "USD")) == "AAPL"

    assert AlpacaBroker.to_alpaca_symbol(InstrumentSpec("BTC/USD", "CRYPTO", "CRYPTO", "USD")) == "BTC/USD"

    with pytest.raises(ValueError, match="does not support"):
        AlpacaBroker.to_alpaca_symbol(InstrumentSpec("ES", "OPT", "GLOBEX", "USD"))


def _event(
    state: ExecutionEventState,
    *,
    execution_hash: str = "exec-1",
    quantity: Decimal = Decimal("10"),
    price: Decimal = Decimal("100"),
    filled_quantity: Decimal = Decimal("0"),
) -> ExecutionEvent:
    return ExecutionEvent(
        broker_name="test",
        internal_order_id="ord-1",
        broker_order_id="ord-1",
        state=state,
        symbol="AAPL",
        side=Side.BUY,
        quantity=quantity,
        filled_quantity=filled_quantity,
        remaining_quantity=quantity - filled_quantity,
        price=price,
        estimated_price=price,
        execution_hash=execution_hash,
    )


def test_execution_state_monitor_tracks_reservations_and_inventory() -> None:
    monitor = ExecutionStateMonitor()

    assert monitor.process_event(_event(ExecutionEventState.SUBMITTED))
    reservation = monitor.in_flight_reservations["ord-1"]
    assert reservation.reserved_cash == Decimal("1000")
    assert reservation.reserved_margin == Decimal("0")

    assert monitor.process_event(
        _event(
            ExecutionEventState.PARTIALLY_FILLED,
            execution_hash="exec-2",
            quantity=Decimal("4"),
            filled_quantity=Decimal("4"),
        )
    )
    position = monitor.positions["AAPL"]
    assert position.quantity == Decimal("4")
    assert position.avg_cost == Decimal("100")
    assert monitor.in_flight_reservations["ord-1"].reserved_cash == Decimal("600")

    duplicate = _event(
        ExecutionEventState.PARTIALLY_FILLED,
        execution_hash="exec-2",
        quantity=Decimal("4"),
        filled_quantity=Decimal("4"),
    )
    assert monitor.process_event(duplicate) is False
    assert monitor.positions["AAPL"].quantity == Decimal("4")

    assert monitor.process_event(
        _event(
            ExecutionEventState.FILLED,
            execution_hash="exec-3",
            quantity=Decimal("6"),
            filled_quantity=Decimal("10"),
        )
    )
    assert monitor.positions["AAPL"].quantity == Decimal("10")
    assert monitor.cash == Decimal("-1000")
    assert "ord-1" not in monitor.in_flight_reservations


def test_execution_state_monitor_purges_reservations_on_terminal_events() -> None:
    monitor = ExecutionStateMonitor()
    monitor.process_event(_event(ExecutionEventState.SUBMITTED))
    assert "ord-1" in monitor.in_flight_reservations

    monitor.process_event(
        ExecutionEvent(
            broker_name="test",
            internal_order_id="ord-1",
            broker_order_id="ord-1",
            state=ExecutionEventState.REJECTED,
            symbol="AAPL",
            side=Side.BUY,
            execution_hash="terminal-1",
        )
    )

    assert "ord-1" not in monitor.in_flight_reservations
    assert monitor.orders["ord-1"].state == ExecutionEventState.REJECTED


@pytest.mark.asyncio
async def test_execution_state_monitor_worker_processes_queued_events() -> None:
    monitor = ExecutionStateMonitor()
    monitor.start()

    monitor.enqueue(_event(ExecutionEventState.SUBMITTED))
    await monitor.drain()

    assert "ord-1" in monitor.in_flight_reservations
    await monitor.stop()


def test_reconcile_skips_symbols_with_in_flight_order_collision() -> None:
    monitor = ExecutionStateMonitor()
    monitor.process_event(_event(ExecutionEventState.SUBMITTED))

    result = monitor.reconcile_with_broker_snapshot(
        broker_orders={"ord-1": {"symbol": "AAPL"}},
        broker_positions={"AAPL": {"quantity": "0"}, "MSFT": {"quantity": "1"}},
    )

    assert result.skipped_order_ids == {"ord-1"}
    assert result.skipped_symbols == {"AAPL"}
    assert result.reconciled_symbols == {"MSFT"}
    assert result.should_sync_account is False
