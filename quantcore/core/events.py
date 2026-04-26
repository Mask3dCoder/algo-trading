"""Async-compatible event bus for inter-component communication."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class EventType(Enum):
    TICK = auto()
    BAR = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()
    RISK_ALERT = auto()
    REGIME_CHANGE = auto()


@dataclass
class Event:
    """Base event carried on the bus."""
    type: EventType
    data: Any = None
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0)


Handler = Callable[[Event], Coroutine[Any, Any, None] | None]


class EventBus:
    """Lightweight async pub/sub event bus.

    Handlers may be sync or async. All handlers are invoked concurrently
    per event via asyncio.gather.
    """

    def __init__(self) -> None:
        self._subscribers: defaultdict[EventType, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        """Register a handler for an event type."""
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        """Remove a handler for an event type."""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        """Publish an event to all registered handlers concurrently."""
        handlers = self._subscribers.get(event.type, [])
        if not handlers:
            return
        coros = []
        for h in handlers:
            result = h(event)
            if asyncio.iscoroutine(result):
                coros.append(result)
        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

    def publish_sync(self, event: Event) -> None:
        """Publish from a sync context (creates a new event loop if needed)."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.publish(event))
        except RuntimeError:
            asyncio.run(self.publish(event))

    def clear(self) -> None:
        """Remove all subscribers."""
        self._subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        return sum(len(v) for v in self._subscribers.values())
