"""Redis cache layer with msgpack serialization, pub/sub, and TTL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import msgpack


@dataclass
class CacheConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str = ""
    ttl_ohlcv: int = 3600       # 1 hour
    ttl_latest_price: int = 5   # 5 seconds


class CacheManager:
    """Async Redis cache manager with msgpack serialization."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._redis: Any = None

    async def connect(self) -> None:
        """Create Redis connection."""
        try:
            import redis.asyncio as aioredis
            url = f"redis://{self.config.host}:{self.config.port}/{self.config.db}"
            if self.config.password:
                url = f"redis://:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.db}"
            self._redis = aioredis.from_url(url, decode_responses=False)
            await self._redis.ping()
        except ImportError:
            self._redis = None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()

    @property
    def is_connected(self) -> bool:
        return self._redis is not None

    def _serialize(self, data: Any) -> bytes:
        """Serialize data with msgpack."""
        return msgpack.packb(data, use_bin_type=True)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize data with msgpack."""
        return msgpack.unpackb(data, raw=False)

    async def set_ohlcv(self, symbol: str, date: str, data: dict, ttl: int | None = None) -> None:
        """Cache OHLCV data with TTL."""
        if not self._redis:
            return
        key = f"ohlcv:{symbol}:{date}"
        ttl = ttl or self.config.ttl_ohlcv
        await self._redis.set(key, self._serialize(data), ex=ttl)

    async def get_ohlcv(self, symbol: str, date: str) -> dict | None:
        """Get cached OHLCV data."""
        if not self._redis:
            return None
        key = f"ohlcv:{symbol}:{date}"
        data = await self._redis.get(key)
        return self._deserialize(data) if data else None

    async def set_latest_price(self, symbol: str, price: float, ttl: int | None = None) -> None:
        """Cache latest price with short TTL."""
        if not self._redis:
            return
        key = f"price:{symbol}"
        ttl = ttl or self.config.ttl_latest_price
        await self._redis.set(key, self._serialize({"price": price}), ex=ttl)

    async def get_latest_price(self, symbol: str) -> float | None:
        """Get latest cached price."""
        if not self._redis:
            return None
        data = await self._redis.get(f"price:{symbol}")
        if data:
            return float(self._deserialize(data)["price"])
        return None

    async def publish(self, channel: str, message: dict) -> None:
        """Publish a message to a Redis channel."""
        if not self._redis:
            return
        await self._redis.publish(channel, self._serialize(message))

    async def subscribe(self, channel: str, callback) -> None:
        """Subscribe to a Redis channel with a callback."""
        if not self._redis:
            return
        pubsub = self._redis.pubsub()
        await pubsub.subscribe(channel)
        async for message in pubsub.listen():
            if message["type"] == "message":
                data = self._deserialize(message["data"])
                await callback(data)

    async def invalidate(self, pattern: str) -> int:
        """Delete keys matching a pattern (cache busting)."""
        if not self._redis:
            return 0
        keys = []
        async for key in self._redis.scan_iter(match=pattern):
            keys.append(key)
        if keys:
            return await self._redis.delete(*keys)
        return 0
