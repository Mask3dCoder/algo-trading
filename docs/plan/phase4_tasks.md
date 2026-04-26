# Phase 4 — Live Ops & Production Tasks

**Depends on:** Phase 3 complete (risk, execution, portfolio, regime)

---

## AGENT-P4-01: Alternative Data — Sentiment NLP

**Goal:** NLP pipeline for financial sentiment extraction from news, filings, and social media.

### Tasks
1. Implement `quantcore/alt_data/sentiment/news.py`:
   - News ingestion: RSS feeds (Reuters, Bloomberg, SeekingAlpha, MarketWatch)
   - Article text extraction and cleaning (trafilatura or newspaper3k)
   - Rate-limited scraping with polite crawling
   - Store articles in PostgreSQL with metadata (source, timestamp, tickers mentioned)
2. Implement `quantcore/alt_data/sentiment/finbert.py`:
   - Load FinBERT model (ProsusAI/finbert from HuggingFace)
   - Sentence-level sentiment scoring: positive/negative/neutral probabilities
   - Document-level aggregation (attention-weighted sentence scores)
   - Entity extraction: map sentiment to specific tickers
   - Batch inference pipeline (GPU if available, CPU fallback)
3. Implement `quantcore/alt_data/sentiment/filings.py`:
   - SEC EDGAR integration: download 10-K, 10-Q, 8-K filings
   - Section extraction (MD&A, Risk Factors)
   - Sentiment scoring per section
   - Loughran-McDonald financial sentiment dictionary (as baseline)
   - Change-in-sentiment between consecutive filings
4. Implement `quantcore/alt_data/sentiment/social.py`:
   - Reddit API: r/wallstreetbets, r/stocks, r/investing
   - StockTwits API: message stream by ticker
   - Mention volume tracking (unusual activity detection)
   - Sentiment aggregation per ticker per time period
   - Bot/spam filtering heuristics
5. Implement `quantcore/alt_data/sentiment/earnings_calls.py`:
   - Transcript ingestion (from free sources or APIs)
   - NLP analysis: tone, uncertainty language, forward-looking statements
   - Topic modeling (LDA/NMF) on management commentary
   - Q&A section sentiment vs prepared remarks comparison
6. Implement `quantcore/alt_data/sentiment/signals.py`:
   - Sentiment momentum: rolling change in aggregate sentiment
   - Sentiment divergence: price trend vs sentiment trend
   - Analyst revision breadth (from sentiment on analyst reports)
   - Short interest sentiment proxy
   - Composite sentiment score per ticker
7. Write tests: FinBERT correctly classifies known positive/negative headlines. Sentiment signals have nonzero IC.

**Acceptance:** FinBERT accuracy >85% on financial sentiment benchmark. Sentiment signal produces measurable IC on 1-week forward returns.

---

## AGENT-P4-02: Options Flow Intelligence

**Goal:** Real-time options flow analysis for alpha and risk signals.

### Tasks
1. Implement `quantcore/alt_data/options_flow/unusual_activity.py`:
   - Define UOA criteria: volume > 2× average OI, premium > $100K, etc.
   - Classify: sweep (aggressive, multi-exchange) vs block (single fill)
   - Directional inference: opening buy vs closing sell (OI change confirmation)
   - Filter noise: market maker hedging, dividend-related, index rebalance
2. Implement `quantcore/alt_data/options_flow/put_call.py`:
   - Put/Call ratio: by volume, by OI, by dollar premium
   - Equity-only vs index P/C (different signals)
   - Rolling percentile rank of P/C ratio
   - Extreme reading detection
3. Implement `quantcore/alt_data/options_flow/gex.py`:
   - GEX calculation: Σ(OI × contract_multiplier × gamma × spot²) per strike
   - Separate customer vs dealer gamma (sign convention)
   - Net GEX profile across all strikes
   - Gamma flip level: where GEX crosses zero
   - Charm flow: daily delta decay → expected hedging flow
   - Vanna exposure: dDelta/dVol → vol-move hedging flow
4. Implement `quantcore/alt_data/options_flow/max_pain.py`:
   - Max pain calculation: strike that minimizes total option holder P&L
   - Pin risk probability estimation
   - Expiration-week pinning signal
5. Write tests: GEX calculation matches manual computation. Max pain identifies correct strike.

**Acceptance:** GEX correctly identifies support/resistance zones validated against historical price action. UOA detection catches known large trades.

---

## AGENT-P4-03: Paper Trading System

**Goal:** Full simulation of live trading with real-time data but simulated execution.

### Tasks
1. Implement `quantcore/live/paper.py`:
   - Real-time data feed consumption (from AGENT-P1-05 feeds)
   - Route orders through `PaperBroker` (simulated fills)
   - Realistic fill simulation: slippage, partial fills, latency
   - Commission model matching target live broker
   - Simulated margin and buying power
2. Implement paper trading loop:
   - Market open: initialize positions, load strategy state
   - Intraday: process bars/ticks → generate signals → submit orders → track fills
   - Market close: compute daily P&L, risk metrics, generate daily report
   - Handle pre/post-market data
3. Position reconciliation: compare internal state vs paper broker state
4. Performance tracking:
   - Real-time P&L vs benchmark
   - Strategy-level attribution
   - Risk metric compliance (VaR utilization, drawdown)
   - Comparison to backtest expectations (strategy degradation detection)
5. Paper trading graduation criteria (configurable):
   - Minimum 2 weeks of paper trading
   - Sharpe > minimum threshold
   - Max drawdown < limit
   - No system errors or unhandled exceptions
   - Risk limits never breached
6. Write tests: paper trading loop processes 1 full trading day of recorded data correctly

**Acceptance:** Paper trading runs for 5 consecutive days without error. P&L tracking matches manual verification. Fill simulation produces realistic costs.

---

## AGENT-P4-04: Live Trading Infrastructure

**Goal:** Production-ready live trading with safety controls.

### Tasks
1. Implement `quantcore/live/trader.py`:
   - Main trading loop: event-driven, async
   - Strategy orchestrator: run multiple strategies concurrently
   - Signal → Order → Risk Check → Execute → Fill → Update pipeline
   - Graceful startup: load state, reconcile positions, verify data feeds
   - Graceful shutdown: cancel open orders, save state, generate EOD report
2. Implement `quantcore/live/watchdog.py`:
   - Heartbeat monitoring: trader must send heartbeat every N seconds
   - Dead man's switch: if no heartbeat for 2 minutes, emergency flatten
   - Process health: CPU, memory, disk monitoring
   - Data feed health: alert if no ticks for >30 seconds during market hours
   - Automatic restart on crash (with max restart count)
3. Implement `quantcore/live/reconciliation.py`:
   - Position reconciliation: compare internal positions vs broker positions every 5 minutes
   - P&L reconciliation: internal vs broker reported
   - Cash reconciliation
   - Alert on any discrepancy > threshold
   - Auto-correction for small discrepancies, manual intervention for large
4. Implement emergency procedures:
   - Emergency flatten: cancel all orders, close all positions at market
   - Kill switch: API endpoint and keyboard shortcut to halt all trading
   - Configurable auto-halt triggers: drawdown, VaR breach, system error count
   - Post-halt: require manual restart with confirmation
5. Implement state persistence:
   - Save strategy state to database every N minutes
   - Save on every fill event
   - Crash recovery: reload state from last checkpoint
   - Replay missed events from Kafka after recovery
6. Cash and margin management:
   - Track available cash, margin used, buying power
   - Reject orders that would exceed margin
   - Margin call detection and automatic position reduction
7. Write tests: simulate crash/recovery, verify state consistency. Kill switch halts all activity within 1 second.

**Acceptance:** System survives simulated crashes and recovers without position discrepancies. Kill switch works under all conditions. Watchdog restarts crashed processes.

---

## AGENT-P4-05: Real-Time Monitoring Dashboard

**Goal:** Plotly Dash web application for comprehensive system monitoring.

### Tasks
1. Implement `quantcore/dashboard/app.py`:
   - Dash application with multi-page layout
   - Dark theme, professional aesthetic
   - Auto-refresh via WebSocket / Dash intervals (1-second for P&L, 5-second for risk)
   - Authentication (basic auth or API key)
2. Implement `quantcore/dashboard/panels/portfolio.py`:
   - Real-time P&L: intraday chart, MTD, YTD, inception
   - Equity curve with benchmark overlay (SPY, 60/40)
   - Position table: symbol, quantity, market value, P&L, weight, Greeks
   - Risk utilization bars (VaR used / limit, drawdown / limit)
   - Asset allocation pie chart
3. Implement `quantcore/dashboard/panels/strategy.py`:
   - P&L by strategy (stacked area chart)
   - Active signals table: signal name, direction, strength, confidence
   - Recent trades table: time, symbol, side, quantity, price, slippage
   - Order book: pending orders with status
   - Strategy state: running / paused / halted per strategy
4. Implement `quantcore/dashboard/panels/risk.py`:
   - VaR and CVaR gauges (current vs limit)
   - Greeks exposure: delta, gamma, vega heatmaps by underlying
   - Drawdown meter with threshold lines
   - Live correlation matrix (heatmap, updated every minute)
   - Factor exposure bar chart (market, size, value, momentum)
   - Stress test results table (scenario → P&L impact)
5. Implement `quantcore/dashboard/panels/market.py`:
   - Regime probability bars (HMM state probabilities)
   - Volatility surface (3D plotly surface, interactive)
   - Term structure chart (VIX futures curve)
   - Sector rotation heatmap (returns by sector, multiple horizons)
   - Market breadth indicators (A/D line, new highs/lows)
6. Implement `quantcore/dashboard/panels/execution.py`:
   - VWAP performance: actual vs benchmark VWAP
   - Slippage distribution histogram
   - Fill quality: % filled at or better than limit
   - TCA summary by strategy and time period
   - Latency metrics: order-to-fill time distribution
7. Implement `quantcore/dashboard/panels/research.py`:
   - Signal IC tracking over time (line chart per signal)
   - Alpha decay curves
   - Backtest comparison table (multiple strategies side-by-side)
   - Walk-forward result timeline
8. Implement `quantcore/dashboard/alerts.py`:
   - Alert rules engine: condition → action (email, Slack webhook, SMS via Twilio)
   - Alert types: risk limit breach, drawdown threshold, system error, data feed down, unusual P&L
   - Alert history log
   - Configurable alert suppression (no repeat for N minutes)
9. Write tests: dashboard renders without error. Data flows from Redis to dashboard correctly.

**Acceptance:** Dashboard loads in <3 seconds. All panels display correct data. Alerts fire within 5 seconds of condition.

---

## AGENT-P4-06: Infrastructure & Performance

**Goal:** Production infrastructure, containerization, and performance optimization.

### Tasks
1. Implement Docker setup:
   - `Dockerfile`: multi-stage build (builder + runtime), minimal image
   - `docker-compose.yml`: full stack (app, TimescaleDB, Redis, Kafka, Zookeeper, dashboard)
   - Docker health checks for all services
   - Volume mounts for data persistence
   - Environment variable configuration
2. Implement Kubernetes manifests (in `k8s/` directory):
   - Deployments: trader, dashboard, data-feeds, each as separate deployment
   - Services: internal (ClusterIP) and external (LoadBalancer for dashboard)
   - ConfigMaps and Secrets for configuration
   - PersistentVolumeClaims for database storage
   - HorizontalPodAutoscaler for data feed handlers
   - Liveness and readiness probes
3. Performance optimization:
   - Profile all hot paths with `cProfile` + `line_profiler`
   - Replace Python loops with vectorized NumPy/Polars operations
   - Apply `numba.jit` to tight numerical loops (Monte Carlo, indicator computation)
   - Connection pooling for database (asyncpg pool)
   - Redis pipeline for batch operations
   - Kafka batch producer for high-throughput
4. Implement `quantcore/api/main.py`:
   - FastAPI REST API for external interaction
   - Endpoints: `/health`, `/positions`, `/orders`, `/risk`, `/strategies`, `/config`
   - POST endpoints: `/orders/submit`, `/strategies/{id}/pause`, `/emergency/flatten`
   - WebSocket endpoint for real-time P&L streaming
   - API key authentication
   - Rate limiting
5. Implement CI/CD pipeline (GitHub Actions or similar):
   - Lint (ruff) → Type check (mypy) → Unit tests → Integration tests → Build Docker → Push to registry
   - Separate staging and production deployments
6. Monitoring infrastructure:
   - Prometheus metrics export (via `prometheus_client`)
   - Key metrics: order latency, fill rate, data feed lag, memory usage, CPU
   - Grafana dashboards (docker-compose includes Grafana + Prometheus)
7. Write load tests: simulate 1M+ ticks/sec through feed handler, measure throughput and latency

**Acceptance:** Docker stack starts with single `docker-compose up`. API responds in <50ms. Feed handler sustains 100K+ msgs/sec.

---

## AGENT-P4-07: Compliance & Audit

**Goal:** Immutable audit trail, pre-trade compliance, and regulatory reporting.

### Tasks
1. Implement `quantcore/compliance/audit_log.py`:
   - Append-only audit log for every order, fill, cancellation, modification
   - Fields: timestamp, user/strategy, action, details, pre-trade risk check result
   - Store in dedicated TimescaleDB table with no UPDATE/DELETE permissions
   - Log rotation and archival (compress after 90 days)
   - Cryptographic hash chain (each entry includes hash of previous) for tamper detection
2. Implement `quantcore/compliance/rules.py`:
   - Pre-trade compliance rules engine:
     - Restricted list check (no trading in blacklisted symbols)
     - Position limit check (regulatory: >5% ownership reporting threshold)
     - Short-sale locate verification
     - Wash sale prevention (30-day lookback)
     - Pattern day trader rule compliance (if applicable)
     - Market manipulation prevention (no layering, spoofing patterns)
   - Rules configurable via TOML
   - Every order must pass compliance before submission
3. Implement `quantcore/compliance/reporting.py`:
   - Daily trade blotter export (CSV/Excel)
   - Monthly performance report
   - Tax reporting: realized gains/losses, wash sale adjustments
   - Regulatory filing data preparation (13F format if applicable)
   - Risk report for compliance review
4. Implement data retention policies:
   - Trade data: 7 years (regulatory minimum)
   - Market data: configurable (default 5 years)
   - Audit logs: 7 years, immutable
   - Automated archival to cold storage
5. Write tests: audit log rejects UPDATE/DELETE operations. Compliance rules correctly block restricted trades. Hash chain detects tampering.

**Acceptance:** Audit trail is provably immutable. Compliance rules block all configured violations. Reports generate correctly for sample data.

---

## AGENT-P4-08: Research & Experimentation Framework

**Goal:** Tools for systematic alpha research, hypothesis testing, and experiment tracking.

### Tasks
1. Implement `quantcore/research/alpha_research.py`:
   - Alpha hypothesis template: define signal, universe, holding period, expected IC
   - Automated single-signal backtest: IC, turnover, decay, capacity
   - Multiple testing correction: Bonferroni, Holm-Bonferroni, BH (False Discovery Rate)
   - Deflated Sharpe Ratio (Bailey & Lopez de Prado): adjust for number of trials
   - Minimum backtest length calculation (for statistical significance)
2. Implement `quantcore/research/experiment_tracker.py`:
   - Experiment logging: parameters, results, artifacts (plots, reports)
   - Version control of strategy configs
   - A/B test framework: compare strategy variants on same data
   - Parameter sensitivity analysis: grid sweep with heatmap output
   - Store experiments in database with searchable metadata
3. Implement research notebooks structure:
   - `notebooks/01_data_exploration.ipynb` — template
   - `notebooks/02_signal_research.ipynb` — template
   - `notebooks/03_strategy_backtest.ipynb` — template
   - `notebooks/04_factor_analysis.ipynb` — template
   - Pre-configured with imports, data loading, standard analysis functions
4. Implement `quantcore/research/capacity.py`:
   - Strategy capacity estimation: at what AUM does alpha decay to zero?
   - Market impact scaling analysis
   - Liquidity-constrained backtest (impact grows with AUM)
5. Write tests: deflated Sharpe correctly penalizes multiple trials. Experiment tracker persists and retrieves results.

**Acceptance:** Research framework correctly identifies overfitted strategies (deflated Sharpe < threshold). Experiment tracker stores and retrieves full experiment history.
