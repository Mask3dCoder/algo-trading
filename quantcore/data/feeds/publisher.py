"""Data publisher — publish bars/ticks to Redis pub/sub and Kafka topics."""

from __future__ import annotations

from collections import deque
from typing import Any

import msgpack

from quantcore.core.types import OHLCV, Tick


class RingBuffer:
    """Fixed-size ring buffer with drop-oldest policy for backpressure."""

    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._dropped: int = 0

    def push(self, item: Any) -> None:
        if len(self._buffer) >= self.capacity:
            self._dropped += 1
        self._buffer.append(item)

    def pop(self) -> Any:
        if self._buffer:
            return self._buffer.popleft()
        raise IndexError("Ring buffer is empty")

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def is_empty(self) -> bool:
        return len(self._buffer) == 0


class DataPublisher:
    """Publish normalized ticks/bars to Redis pub/sub and Kafka topics.

    Uses a ring buffer for backpressure (drop-oldest policy).
    """

    def __init__(
        self,
        redis_client: Any = None,
        kafka_producer: Any = None,
        buffer_capacity: int = 10_000,
        redis_channel_prefix: str = "quantcore:bars",
        kafka_topic: str = "quantcore-bars",
    ) -> None:
        self._redis = redis_client
        self._kafka = kafka_producer
        self._buffer = RingBuffer(capacity=buffer_capacity)
        self._redis_channel = redis_channel_prefix
        self._kafka_topic = kafka_topic
        self._published_count = 0

    async def publish_bar(self, bar: OHLCV) -> None:
        """Publish a bar to Redis and Kafka."""
        message = {
            "symbol": bar.symbol,
            "timestamp": bar.timestamp.isoformat(),
            "open": str(bar.open),
            "high": str(bar.high),
            "low": str(bar.low),
            "close": str(bar.close),
            "volume": bar.volume,
        }
        serialized = msgpack.packb(message, use_bin_type=True)
        self._buffer.push(serialized)

        # Publish to Redis
        if self._redis:
            try:
                channel = f"{self._redis_channel}:{bar.symbol}"
                await self._redis.publish(channel, serialized)
            except Exception:
                pass

        # Publish to Kafka
        if self._kafka:
            try:
                self._kafka.produce(
                    self._kafka_topic,
                    key=bar.symbol.encode(),
                    value=serialized,
                )
            except Exception:
                pass

        self._published_count += 1

    async def publish_tick(self, tick: Tick) -> None:
        """Publish a tick to Redis and Kafka."""
        message = {
            "symbol": tick.symbol,
            "timestamp_us": tick.timestamp_us,
            "price": str(tick.price),
            "size": tick.size,
        }
        serialized = msgpack.packb(message, use_bin_type=True)
        self._buffer.push(serialized)

        if self._redis:
            try:
                await self._redis.publish(f"quantcore:ticks:{tick.symbol}", serialized)
            except Exception:
                pass

        if self._kafka:
            try:
                self._kafka.produce("quantcore-ticks", key=tick.symbol.encode(), value=serialized)
            except Exception:
                pass

    @property
    def stats(self) -> dict[str, int]:
        return {
            "published": self._published_count,
            "buffer_size": self._buffer.size,
            "buffer_dropped": self._buffer.dropped,
        }
