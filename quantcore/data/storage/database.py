"""Async database access layer for TimescaleDB with connection pooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    name: str = "quantcore"
    user: str = "quantcore"
    password: str = "quantcore"
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class DatabaseManager:
    """Async database manager with connection pooling (asyncpg backend)."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self.config = config or DatabaseConfig()
        self._pool: Any = None

    async def connect(self) -> None:
        """Create connection pool."""
        try:
            import asyncpg
            self._pool = await asyncpg.create_pool(
                self.config.dsn,
                min_size=2,
                max_size=self.config.pool_size,
            )
        except ImportError:
            # Fallback: store config for sync operations
            self._pool = None

    async def close(self) -> None:
        """Close connection pool."""
        if self._pool:
            await self._pool.close()

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    async def execute(self, query: str, *args: Any) -> str:
        """Execute a SQL statement."""
        if not self._pool:
            raise RuntimeError("Database not connected")
        async with self._pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list[dict]:
        """Fetch rows as list of dicts."""
        if not self._pool:
            raise RuntimeError("Database not connected")
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(r) for r in rows]

    async def fetchval(self, query: str, *args: Any) -> Any:
        """Fetch a single value."""
        if not self._pool:
            raise RuntimeError("Database not connected")
        async with self._pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def insert_ohlcv(self, data: pd.DataFrame) -> int:
        """Insert OHLCV data using bulk COPY protocol. Returns number of rows."""
        if not self._pool:
            raise RuntimeError("Database not connected")
        if data.empty:
            return 0
        columns = ["symbol", "timestamp", "open", "high", "low", "close", "volume", "adj_close"]
        records = [
            (
                str(row.get("symbol", "")),
                row.get("timestamp"),
                float(row.get("open", 0)),
                float(row.get("high", 0)),
                float(row.get("low", 0)),
                float(row.get("close", 0)),
                int(row.get("volume", 0)),
                float(row.get("adj_close", row.get("close", 0))),
            )
            for _, row in data.iterrows()
        ]
        return await self.bulk_insert("ohlcv_daily", columns, records)

    async def query_ohlcv(
        self,
        symbol: str,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Query OHLCV data for a symbol and date range."""
        if not self._pool:
            raise RuntimeError("Database not connected")

        query = "SELECT * FROM ohlcv_daily WHERE symbol = $1"
        args: list[Any] = [symbol]
        idx = 2

        if start:
            query += f" AND timestamp >= ${idx}"
            args.append(start)
            idx += 1
        if end:
            query += f" AND timestamp <= ${idx}"
            args.append(end)
            idx += 1

        query += " ORDER BY timestamp"
        rows = await self.fetch(query, *args)
        return pd.DataFrame(rows)

    async def bulk_insert(self, table: str, columns: list[str], data: list[tuple]) -> int:
        """Bulk insert using COPY protocol for maximum throughput."""
        if not self._pool:
            raise RuntimeError("Database not connected")

        async with self._pool.acquire() as conn, conn.transaction():
            await conn.copy_records_to_table(table, records=data, columns=columns)
        return len(data)


# SQL for initial schema migration
SCHEMA_SQL = """
-- OHLCV daily (hypertable)
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    adj_close DOUBLE PRECISION,
    dividend DOUBLE PRECISION DEFAULT 0,
    split_factor DOUBLE PRECISION DEFAULT 1,
    PRIMARY KEY (symbol, timestamp)
);

-- OHLCV intraday (hypertable)
CREATE TABLE IF NOT EXISTS ohlcv_intraday (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (symbol, timestamp)
);

-- Ticks (hypertable)
CREATE TABLE IF NOT EXISTS ticks (
    symbol TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    size INTEGER NOT NULL,
    exchange TEXT DEFAULT '',
    conditions TEXT DEFAULT '',
    PRIMARY KEY (symbol, timestamp)
);

-- Options chains
CREATE TABLE IF NOT EXISTS options_chains (
    underlying TEXT NOT NULL,
    expiry DATE NOT NULL,
    strike DOUBLE PRECISION NOT NULL,
    right TEXT NOT NULL CHECK (right IN ('C', 'P')),
    timestamp TIMESTAMPTZ NOT NULL,
    bid DOUBLE PRECISION,
    ask DOUBLE PRECISION,
    last DOUBLE PRECISION,
    volume BIGINT DEFAULT 0,
    open_interest BIGINT DEFAULT 0,
    iv DOUBLE PRECISION,
    delta DOUBLE PRECISION,
    gamma DOUBLE PRECISION,
    theta DOUBLE PRECISION,
    vega DOUBLE PRECISION
);

-- Corporate actions
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    action_type TEXT NOT NULL CHECK (action_type IN ('split', 'dividend', 'merger', 'spinoff', 'delisting')),
    value DOUBLE PRECISION NOT NULL,
    description TEXT DEFAULT '',
    PRIMARY KEY (symbol, date, action_type)
);

-- Orders
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    type TEXT NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    limit_price DOUBLE PRECISION,
    status TEXT NOT NULL DEFAULT 'pending',
    strategy_id TEXT DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

-- Fills
CREATE TABLE IF NOT EXISTS fills (
    id SERIAL PRIMARY KEY,
    order_id TEXT NOT NULL REFERENCES orders(id),
    price DOUBLE PRECISION NOT NULL,
    quantity DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL
);

-- Positions
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT NOT NULL,
    quantity DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol)
);

-- Audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    hash TEXT NOT NULL DEFAULT ''
);

-- Convert to hypertables (requires TimescaleDB extension)
-- These are run separately after the extension is enabled
"""
