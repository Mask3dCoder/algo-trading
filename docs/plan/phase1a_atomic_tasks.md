# Phase 1A — Scaffold, Data Infra, Feeds

---

### T-0101 · Project metadata & dependencies
**Files:** `pyproject.toml`
**Deps:** none
**Steps:**
1. Create `pyproject.toml` with `[project]` metadata (name=quantcore, python≥3.12)
2. Add all dependency groups from master plan (core, ml, stats, viz, infra, dev)
3. Configure `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest]` sections
**Accept:** `pip install -e ".[dev]"` succeeds. `ruff check .` and `mypy --version` run.

---

### T-0102 · Dev tooling
**Files:** `Makefile`, `.env.example`
**Deps:** T-0101
**Steps:**
1. `Makefile` targets: `install`, `test`, `lint`, `typecheck`, `format`, `docker-up`, `docker-down`
2. `.env.example` with placeholders: `DATABASE_URL`, `REDIS_URL`, `KAFKA_BROKER`, `ALPACA_KEY`, `POLYGON_KEY`
**Accept:** `make lint` and `make typecheck` run (may show zero-file warnings). `.env.example` documents all required vars.

---

### T-0103 · Package skeleton
**Files:** ~60 `__init__.py` files across all directories
**Deps:** T-0101
**Steps:**
1. Create every directory from master plan structure
2. Place `__init__.py` in each Python package directory
3. Each `__init__.py` is empty or has a module-level docstring only
**Accept:** `python -c "import quantcore; import quantcore.core; import quantcore.data; import quantcore.math_models"` works.

---

### T-0104 · Config loader
**Files:** `quantcore/core/config.py`, `config/default.toml`, `config/development.toml`
**Deps:** T-0101, T-0103
**Steps:**
1. `config/default.toml`: sections `[database]`, `[redis]`, `[kafka]`, `[logging]`, `[brokers.alpaca]`, `[brokers.ibkr]`, `[risk_limits]`, `[execution]` with sensible defaults
2. `config/development.toml`: override DB to localhost, logging to pretty-print
3. `core/config.py`: `get_config() → Settings` using dynaconf, env layering (default → env-specific → env vars), type-safe access
4. Unit test: load config, verify defaults, verify env override
**Accept:** `get_config().database.host` returns correct value. Env var `QUANTCORE_DATABASE__HOST=x` overrides.

---

### T-0105 · Structured logging
**Files:** `quantcore/core/logging.py`
**Deps:** T-0104
**Steps:**
1. Configure structlog: JSON renderer for prod, ConsoleRenderer for dev (auto-detect from config)
2. `get_logger(name) → BoundLogger` factory
3. Add context: timestamp (ISO), level, logger name, correlation_id
4. Log level from config
5. Unit test: capture log output, verify JSON structure
**Accept:** `get_logger("test").info("hello", x=1)` produces `{"event":"hello","x":1,"timestamp":...}` in prod mode.

---

### T-0106 · Core data types
**Files:** `quantcore/core/types.py`
**Deps:** T-0103
**Steps:**
1. Pydantic models: `OHLCV` (symbol, timestamp, o, h, l, c, volume, adj_close), `Tick` (symbol, timestamp_us, price, size, exchange), `Quote` (symbol, bid, ask, bid_size, ask_size)
2. `Order` (id, symbol, side, type, quantity, limit_price, status, strategy_id, timestamps), `Fill` (order_id, price, quantity, commission, timestamp)
3. `Position` (symbol, quantity, avg_cost, market_price, unrealized_pnl, realized_pnl)
4. `Signal` (name, symbol, direction, strength, confidence, timestamp, metadata)
5. `PortfolioSnapshot` (timestamp, positions, cash, total_value, returns)
6. All with proper validators, serialization, `__eq__`, `__hash__`
7. Unit test: create, serialize/deserialize, validate constraints
**Accept:** All models validate correctly. Invalid data (negative quantity, future timestamp) raises `ValidationError`.

---

### T-0107 · Constants & time utilities
**Files:** `quantcore/core/constants.py`, `quantcore/core/time_utils.py`
**Deps:** T-0103
**Steps:**
1. `constants.py`: `TRADING_DAYS_PER_YEAR=252`, `ANNUALIZATION_FACTOR=√252`, `MARKET_OPEN=9:30 ET`, `MARKET_CLOSE=16:00 ET`, `RISK_FREE_RATE=0.05`
2. `time_utils.py`: `is_market_open(dt)`, `next_market_open(dt)`, `trading_days_between(start, end)`, `to_utc(dt)`, `to_eastern(dt)`
3. Use `exchange_calendars` or `pandas_market_calendars` for NYSE calendar
4. Unit test: verify known holidays, market hours, day counts
**Accept:** `trading_days_between("2024-01-01","2024-12-31")` returns 252±1. `is_market_open` correct for NYSE.

---

### T-0108 · Exceptions & event bus
**Files:** `quantcore/core/exceptions.py`, `quantcore/core/events.py`
**Deps:** T-0103
**Steps:**
1. `exceptions.py`: `QuantCoreError` (base), `DataError`, `ExecutionError`, `RiskLimitBreached`, `ConfigError`, `BrokerError`, `ValidationError`
2. `events.py`: `EventBus` class — `subscribe(event_type, handler)`, `publish(event)`, async-compatible (`asyncio.Queue`-backed)
3. Event types enum: `TICK`, `BAR`, `SIGNAL`, `ORDER`, `FILL`, `RISK_ALERT`, `REGIME_CHANGE`
4. Unit test: subscribe handler, publish event, verify handler called
**Accept:** `EventBus` delivers events to all subscribers. Async handlers work with `asyncio.run`.

---

### T-0109 · Docker infrastructure
**Files:** `docker-compose.yml`, `Dockerfile`
**Deps:** T-0101
**Steps:**
1. `Dockerfile`: multi-stage (builder installs deps, runtime copies venv), Python 3.12-slim base
2. `docker-compose.yml` services: `timescaledb` (timescale/timescaledb:latest-pg16, port 5432), `redis` (redis:7, port 6379), `kafka` + `zookeeper` (confluentinc images, ports 9092/2181)
3. Volumes for DB persistence
4. Health checks on all services
5. Network: single bridge network `quantcore-net`
**Accept:** `docker-compose up -d` starts all services. `psql`, `redis-cli ping`, Kafka topic creation all succeed.

---

### T-0110 · Database schema & access layer
**Files:** `quantcore/data/storage/database.py`, `alembic/versions/001_initial.py`
**Deps:** T-0109, T-0106
**Steps:**
1. Set up Alembic for migrations
2. SQL schema: `ohlcv_daily` (hypertable), `ohlcv_intraday` (hypertable), `ticks` (hypertable, microsecond ts), `options_chains`, `corporate_actions`, `orders`, `fills`, `positions`, `audit_log`
3. `database.py`: `DatabaseManager` class — async connection pool (asyncpg), `insert_ohlcv()`, `query_ohlcv(symbol, start, end)`, `insert_ticks()`, `bulk_insert()` (COPY protocol)
4. Integration test: insert 10K rows, query back, verify correctness and timing
**Accept:** Bulk insert of 100K OHLCV rows completes in <5s. Queries by symbol+date range use index. Hypertable chunking verified.

---

### T-0111 · TimescaleDB features
**Files:** `quantcore/data/storage/timescale.py`
**Deps:** T-0110
**Steps:**
1. Continuous aggregates: 1min ticks → 5min bars → hourly → daily (define refresh policies)
2. Compression policy: compress chunks older than 7 days
3. Retention policy: configurable (default: ticks=90 days, bars=10 years)
4. Helper: `refresh_aggregates()`, `get_compression_stats()`, `get_chunk_info()`
5. Integration test: insert minute data, verify 5min/hourly aggregates auto-populate
**Accept:** Continuous aggregates refresh within configured interval. Compression reduces storage by >5×.

---

### T-0112 · Redis cache layer
**Files:** `quantcore/data/storage/cache.py`
**Deps:** T-0109, T-0106
**Steps:**
1. `CacheManager`: async Redis connection (aioredis)
2. `get/set_ohlcv(symbol, date)` with TTL (configurable, default 1 hour)
3. `get/set_latest_price(symbol)` with 5s TTL
4. `publish/subscribe(channel, message)` for real-time data distribution
5. `invalidate(pattern)` for cache busting
6. Serialization: msgpack for performance
7. Unit test (mock Redis) + integration test (real Redis)
**Accept:** Cache hit returns data in <1ms. Cache miss falls through to DB. Pub/sub delivers messages to subscribers.

---

### T-0113 · DataSource base + Yahoo adapter
**Files:** `quantcore/data/sources/base.py`, `quantcore/data/sources/yahoo.py`
**Deps:** T-0106, T-0110
**Steps:**
1. `base.py`: abstract `DataSource` — `async fetch_ohlcv(symbol, start, end) → list[OHLCV]`, `async fetch_options(symbol) → list[OptionsChain]`, `supported_assets() → list[str]`
2. `yahoo.py`: `YahooDataSource(DataSource)` using yfinance — batch download, rate limiting, retry with backoff
3. Handle splits/dividends (yfinance auto-adjusts, but store raw + adjusted)
4. Unit test: mock yfinance response, verify OHLCV parsing. Integration test: download 1 year SPY.
**Accept:** Downloads 20 years of SPY daily data in <30s. OHLCV models validate. Handles network errors gracefully.

---

### T-0114 · Polygon adapter
**Files:** `quantcore/data/sources/polygon.py`
**Deps:** T-0113
**Steps:**
1. `PolygonDataSource(DataSource)`: REST client for aggregates, trades, quotes, options, reference data
2. Rate limiting (5 req/min free tier, configurable)
3. Pagination handling for large result sets
4. Options chain fetching: all expirations and strikes for a symbol
5. Reference data: ticker details, market status
6. Unit test: mock HTTP responses, verify parsing
**Accept:** Fetches daily bars, tick data, and options chains. Rate limiter prevents 429 errors.

---

### T-0115 · Alpha Vantage + FRED adapters
**Files:** `quantcore/data/sources/alpha_vantage.py`, `quantcore/data/sources/fred.py`
**Deps:** T-0113
**Steps:**
1. `AlphaVantageDataSource`: fundamentals (income statement, balance sheet, cash flow), earnings calendar, economic indicators
2. `FREDDataSource`: yield curves (2Y, 5Y, 10Y, 30Y), GDP, CPI, unemployment, Fed funds rate
3. Rate limiting for both (AV: 5/min free, FRED: 120/min)
4. Store fetched data in TimescaleDB `fundamentals` / `economic_indicators` tables
5. Unit test: mock HTTP, verify parsing and storage
**Accept:** Can fetch and store 5 years of fundamental data for any S&P500 company. FRED yields match published values.

---

### T-0116 · Corporate actions handler
**Files:** `quantcore/data/corporate_actions.py`
**Deps:** T-0110, T-0113
**Steps:**
1. `adjust_for_split(prices, split_ratio, split_date)`: adjust OHLCV and volume
2. `adjust_for_dividend(prices, div_amount, ex_date)`: proportional price adjustment
3. `handle_merger(old_symbol, new_symbol, ratio, effective_date)`: remap + adjust
4. `handle_delisting(symbol, last_date)`: mark as delisted, retain historical data
5. `get_adjustment_factors(symbol, date_range) → list[AdjustmentEvent]`
6. Unit test: known AAPL 4:1 split (2020-08-31) produces correct adjusted prices
**Accept:** Adjusted prices match Yahoo Finance adjusted close for AAPL, TSLA (post-split). Delisted stocks retained with flag.

---

### T-0117 · Universe manager
**Files:** `quantcore/data/sources/universe.py`
**Deps:** T-0113, T-0116
**Steps:**
1. Static universes: `SP500`, `SP100`, `RUSSELL2000`, `NASDAQ100`, sector ETFs
2. Universe loader: fetch current constituents from Wikipedia/data source
3. Point-in-time universe reconstruction: `get_universe("SP500", date) → list[str]` using historical membership data
4. ADV filter: `filter_by_adv(symbols, min_adv=1_000_000)`
5. Price filter: `filter_by_price(symbols, min_price=5.0)`
6. Unit test: current SP500 has ~500 symbols. Historical SP500 on 2008-01-01 differs from today.
**Accept:** Universe returns correct constituent count. Point-in-time avoids survivorship bias.

---

### T-0118 · Outlier detection
**Files:** `quantcore/data/quality/outlier_detection.py`
**Deps:** T-0106
**Steps:**
1. `zscore_outliers(series, threshold=4.0) → mask`: flag points >threshold σ from mean
2. `grubbs_test(series, alpha=0.05) → list[int]`: iterative single-outlier test
3. `iqr_outliers(series, factor=1.5) → mask`: flag outside Q1-factor*IQR, Q3+factor*IQR
4. `return_outliers(prices, max_sigma=10) → mask`: flag daily returns >max_sigma σ
5. `detect_all(series) → OutlierReport`: run all methods, return union/intersection
6. Unit test: inject 3 outliers into Normal(0,1) series of 1000, verify detection
**Accept:** >95% recall on injected outliers. <5% false positive rate on clean data.

---

### T-0119 · Missing data handling
**Files:** `quantcore/data/quality/missing_data.py`
**Deps:** T-0106
**Steps:**
1. `detect_gaps(series, freq="B") → list[DateRange]`: identify missing business days
2. `forward_fill(series, max_gap=5) → series`: ffill with max gap limit
3. `interpolate(series, method="cubic") → series`: scipy interpolation
4. `em_impute(matrix) → matrix`: EM algorithm for multivariate missing data
5. `impute(series, method="ffill") → tuple[series, ImputationReport]`: apply method + log what was filled
6. Unit test: remove 5 known points, impute, verify error < threshold
**Accept:** Forward-fill handles gaps ≤ max_gap. Cubic interpolation error < 1% on smooth series. EM converges in <100 iterations.

---

### T-0120 · Stale quotes & cross-validation
**Files:** `quantcore/data/quality/stale_quotes.py`, `quantcore/data/quality/cross_validation.py`
**Deps:** T-0106, T-0107
**Steps:**
1. `detect_stale(prices, max_unchanged=5) → mask`: flag consecutive identical prices (excluding market closed)
2. `cross_validate(source_a, source_b, threshold=0.01) → DiscrepancyReport`: compare two price series, flag divergence > threshold
3. `majority_vote(sources: dict[str, series]) → series`: take median of 3+ sources as "truth"
4. Unit test: inject stale quotes, verify detection. Create mismatched sources, verify flagging.
**Accept:** Stale detection catches 5+ unchanged prices during market hours. Cross-validation flags >1% discrepancies.

---

### T-0121 · Feed handler base + Alpaca
**Files:** `quantcore/data/feeds/base.py`, `quantcore/data/feeds/alpaca_feed.py`
**Deps:** T-0106, T-0108
**Steps:**
1. `base.py`: abstract `FeedHandler` — `async connect()`, `async subscribe(symbols)`, `async on_tick(callback)`, `async on_bar(callback)`, `disconnect()`, reconnect logic
2. `alpaca_feed.py`: `AlpacaFeed(FeedHandler)` — WebSocket to Alpaca data stream, parse trades/quotes/bars, normalize to `Tick`/`Quote`/`OHLCV`
3. Reconnect on disconnect with exponential backoff
4. Subscription management (add/remove symbols while connected)
5. Unit test: mock WebSocket, verify message parsing. Integration test: connect to Alpaca paper, receive SPY ticks.
**Accept:** Connects to Alpaca paper data stream. Receives and parses ticks. Auto-reconnects on disconnect.

---

### T-0122 · Polygon + crypto feeds
**Files:** `quantcore/data/feeds/polygon_feed.py`, `quantcore/data/feeds/crypto_feed.py`
**Deps:** T-0121
**Steps:**
1. `PolygonFeed(FeedHandler)`: WebSocket for trades, quotes, aggregates. Handle auth, subscription messages.
2. `CryptoFeed(FeedHandler)`: CCXT WebSocket, support Binance/Coinbase. Normalize to internal types.
3. Both: reconnect logic, message rate monitoring
4. Unit test: mock WebSocket messages, verify parsing
**Accept:** Both feeds parse their respective message formats correctly. Normalized output matches `Tick`/`OHLCV` models.

---

### T-0123 · Bar aggregation + publishing
**Files:** `quantcore/data/feeds/aggregator.py`, `quantcore/data/feeds/publisher.py`
**Deps:** T-0121, T-0112
**Steps:**
1. `aggregator.py`: `BarAggregator` — accumulate ticks into bars (1s, 1min, 5min configurable), emit `OHLCV` on bar close. Also compute running VWAP, TWAP.
2. `publisher.py`: `DataPublisher` — publish bars/ticks to Redis pub/sub channel and Kafka topic (configurable). Serialize with msgpack.
3. Ring buffer for backpressure (configurable size, drop-oldest)
4. Unit test: feed 100 ticks, verify correct 1-min bar. Publisher delivers to mock Redis.
**Accept:** Bar aggregation produces correct OHLCV from tick stream. Publisher throughput >10K msgs/sec to Redis.
