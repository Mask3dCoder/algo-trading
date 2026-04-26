"""Unit tests for the event bus."""

from __future__ import annotations

import pytest

from quantcore.core.events import Event, EventBus, EventType


@pytest.mark.asyncio
async def test_subscribe_and_publish() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.TICK, handler)
    event = Event(type=EventType.TICK, data={"symbol": "SPY", "price": 450.0})
    await bus.publish(event)

    assert len(received) == 1
    assert received[0].data["symbol"] == "SPY"


@pytest.mark.asyncio
async def test_multiple_subscribers() -> None:
    bus = EventBus()
    results: list[str] = []

    async def handler_a(event: Event) -> None:
        results.append("a")

    async def handler_b(event: Event) -> None:
        results.append("b")

    bus.subscribe(EventType.BAR, handler_a)
    bus.subscribe(EventType.BAR, handler_b)
    await bus.publish(Event(type=EventType.BAR))

    assert set(results) == {"a", "b"}


@pytest.mark.asyncio
async def test_sync_handler() -> None:
    bus = EventBus()
    received: list[Event] = []

    def sync_handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.SIGNAL, sync_handler)
    await bus.publish(Event(type=EventType.SIGNAL))

    assert len(received) == 1


@pytest.mark.asyncio
async def test_unsubscribe() -> None:
    bus = EventBus()

    async def handler(event: Event) -> None:
        pass

    bus.subscribe(EventType.ORDER, handler)
    assert bus.subscriber_count == 1
    bus.unsubscribe(EventType.ORDER, handler)
    assert bus.subscriber_count == 0


@pytest.mark.asyncio
async def test_no_subscribers_no_error() -> None:
    bus = EventBus()
    await bus.publish(Event(type=EventType.FILL))


@pytest.mark.asyncio
async def test_clear() -> None:
    bus = EventBus()

    async def handler(event: Event) -> None:
        pass

    bus.subscribe(EventType.RISK_ALERT, handler)
    bus.clear()
    assert bus.subscriber_count == 0
