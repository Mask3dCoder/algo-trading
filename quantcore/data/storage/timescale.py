"""TimescaleDB-specific features: continuous aggregates, compression, retention."""

from __future__ import annotations

from quantcore.data.storage.database import DatabaseManager

TIMESCALE_SETUP_SQL = """
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Convert tables to hypertables
SELECT create_hypertable('ohlcv_daily', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('ohlcv_intraday', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('ticks', 'timestamp', if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day');

-- Continuous aggregates: tick → 1min bars
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_1min
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 minute', timestamp) AS bucket,
    FIRST(price, timestamp) AS open,
    MAX(price) AS high,
    MIN(price) AS low,
    LAST(price, timestamp) AS close,
    SUM(size) AS volume
FROM ticks
GROUP BY symbol, bucket;

-- 1min → 5min
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_5min
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('5 minutes', bucket) AS bucket,
    FIRST(open, bucket) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, bucket) AS close,
    SUM(volume) AS volume
FROM ohlcv_1min
GROUP BY symbol, bucket;

-- 5min → hourly
CREATE MATERIALIZED VIEW IF NOT EXISTS ohlcv_hourly
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', bucket) AS bucket,
    FIRST(open, bucket) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, bucket) AS close,
    SUM(volume) AS volume
FROM ohlcv_5min
GROUP BY symbol, bucket;

-- Refresh policies (every 5 minutes)
SELECT add_continuous_aggregate_policy('ohlcv_1min',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);

SELECT add_continuous_aggregate_policy('ohlcv_5min',
    start_offset => INTERVAL '12 hours',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);

-- Compression policy: compress chunks older than 7 days
ALTER TABLE ohlcv_intraday SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);
SELECT add_compression_policy('ohlcv_intraday', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy
SELECT add_retention_policy('ticks', INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('ohlcv_intraday', INTERVAL '2 years', if_not_exists => TRUE);
"""


class TimescaleManager:
    """Manager for TimescaleDB-specific operations."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    async def setup(self) -> None:
        """Set up TimescaleDB extension, hypertables, and policies."""
        for statement in TIMESCALE_SETUP_SQL.split(";"):
            stmt = statement.strip()
            if stmt:
                try:
                    await self.db.execute(stmt)
                except Exception:
                    pass  # Some statements may fail if already set up

    async def refresh_aggregates(self) -> None:
        """Manually refresh all continuous aggregates."""
        for view in ["ohlcv_1min", "ohlcv_5min", "ohlcv_hourly"]:
            try:
                await self.db.execute(f"CALL refresh_continuous_aggregate('{view}', NULL, NULL)")
            except Exception:
                pass

    async def get_compression_stats(self, table: str = "ohlcv_intraday") -> dict:
        """Get compression statistics for a table."""
        try:
            rows = await self.db.fetch(
                "SELECT * FROM timescaledb_information.compression_stats WHERE hypertable_name = $1",
                table,
            )
            return rows[0] if rows else {}
        except Exception:
            return {}

    async def get_chunk_info(self, table: str = "ohlcv_daily") -> list[dict]:
        """Get information about chunks for a hypertable."""
        try:
            return await self.db.fetch(
                "SELECT * FROM timescaledb_information.chunks WHERE hypertable_name = $1 ORDER BY range_start",
                table,
            )
        except Exception:
            return []
