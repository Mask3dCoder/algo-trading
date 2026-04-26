# Phase 4 — Live Ops & Production (Atomic Tasks)

---

### T-0401 · News Ingestion
**Files:** `quantcore/alt_data/sentiment/news.py`
**Deps:** T-0110
**Steps:**
1. RSS feed parser for Reuters, SeekingAlpha, MarketWatch (configurable feed URLs)
2. `ArticleFetcher`: download article text, extract body via trafilatura
3. Ticker extraction: regex + NER to map articles to symbols
4. Store in `articles` table (source, url, title, body, tickers, timestamp)
5. Rate limiting, deduplication (by URL hash)
6. Test: fetch 10 articles, verify text extraction and ticker mapping.
**Accept:** Extracts clean article text. Ticker extraction precision >80%. Deduplication works.

---

### T-0402 · FinBERT Sentiment Scoring
**Files:** `quantcore/alt_data/sentiment/finbert.py`
**Deps:** T-0401
**Steps:**
1. Load `ProsusAI/finbert` from HuggingFace transformers
2. `score_text(text) → SentimentScore(positive, negative, neutral)`
3. Sentence-level scoring with document-level aggregation (attention-weighted)
4. Batch inference pipeline (batch size configurable, GPU if available)
5. `score_articles(articles) → list[(article_id, ticker, sentiment)]`
6. Store scores in `sentiment_scores` table
7. Test: "Revenue exceeded expectations" → positive. "Bankruptcy filed" → negative.
**Accept:** FinBERT accuracy >85% on financial headlines. Batch processes 100 articles in <60s (CPU).

---

### T-0403 · SEC Filing NLP
**Files:** `quantcore/alt_data/sentiment/filings.py`
**Deps:** T-0402
**Steps:**
1. SEC EDGAR API: download 10-K, 10-Q, 8-K by CIK/ticker
2. Section extraction: MD&A, Risk Factors (regex-based section detection)
3. Loughran-McDonald dictionary: count positive/negative/uncertain words
4. FinBERT scoring on extracted sections
5. `filing_sentiment_change(ticker) → float`: delta between consecutive filings
6. Test: parse known 10-K, verify section extraction.
**Accept:** Section extraction works for >90% of EDGAR filings. Sentiment change signal has nonzero IC.

---

### T-0404 · Social Media Sentiment
**Files:** `quantcore/alt_data/sentiment/social.py`
**Deps:** T-0402
**Steps:**
1. Reddit API (PRAW): fetch posts from r/wallstreetbets, r/stocks (configurable)
2. StockTwits API: message stream by ticker
3. Mention volume tracker: count per ticker per hour
4. Aggregate sentiment per ticker per time bucket (FinBERT on post text)
5. Bot/spam heuristics: account age, karma threshold, repeated text
6. Test: fetch 100 posts from r/wallstreetbets, score sentiment.
**Accept:** Mention volume spikes correlate with known meme-stock events. Spam filter removes >50% of noise.

---

### T-0405 · Composite Sentiment Signals
**Files:** `quantcore/alt_data/sentiment/signals.py`
**Deps:** T-0402, T-0403, T-0404
**Steps:**
1. `sentiment_momentum(ticker, lookback=5) → signal`: rolling Δ in aggregate sentiment
2. `sentiment_divergence(ticker, price_trend, sentiment_trend) → signal`: disagreement
3. `aggregate_sentiment(news, filings, social, weights) → composite_score`
4. Normalize all signals to [-1, 1]
5. `signal_evaluation(composite, forward_returns) → IC`: measure predictive power
6. Test: composite signal has higher IC than any single source.
**Accept:** Composite IC > individual source ICs. Signal updates at configured frequency.

---

### T-0406 · Unusual Options Activity Detection
**Files:** `quantcore/alt_data/options_flow/unusual_activity.py`
**Deps:** T-0106
**Steps:**
1. `detect_uoa(trades, avg_oi, thresholds) → list[UOATrade]`: volume > 2× OI, premium > $100K
2. `classify_trade(trade) → "sweep"|"block"|"normal"`: multi-exchange = sweep
3. `infer_direction(trade, oi_change) → "opening_buy"|"closing_sell"|...`
4. Filter: exclude market-maker hedges (delta-neutral pairs), dividend plays
5. `uoa_summary(trades, period) → DataFrame`: top unusual activity ranked by premium
6. Test: inject known large trade, verify detection and classification.
**Accept:** UOA detection catches trades meeting criteria. Classification >80% accurate on labeled data.

---

### T-0407 · GEX Calculation
**Files:** `quantcore/alt_data/options_flow/gex.py`
**Deps:** T-0130
**Steps:**
1. `calculate_gex(options_chain, spot) → dict[strike, gex]`: Σ(OI × gamma × 100 × spot²) per strike
2. Sign convention: calls have positive dealer gamma (customers buy), puts negative
3. `net_gex_profile(gex_by_strike) → GEXProfile`: net across all strikes
4. `gamma_flip_level(profile) → float`: where cumulative GEX crosses zero
5. `charm_flow_estimate(options_chain, spot) → float`: daily delta decay → hedging pressure
6. Test: GEX positive near current spot for typical option chain. Gamma flip level reasonable.
**Accept:** GEX calculation matches manual computation for simple chain. Gamma flip level within 2% of spot.

---

### T-0408 · Max Pain + Put/Call Ratio
**Files:** `quantcore/alt_data/options_flow/max_pain.py`, `quantcore/alt_data/options_flow/put_call.py`
**Deps:** T-0106
**Steps:**
1. `max_pain(options_chain) → float`: strike minimizing total option holder P&L at expiry
2. `pin_probability(spot, max_pain, vol, dte) → float`: probability of pinning
3. `put_call_ratio(data, method="volume"|"oi"|"dollar") → float`
4. `pcr_signal(ratio, lookback=20) → zscore`: extreme readings as contrarian signal
5. Test: max pain near ATM for typical chain. P/C > 1 during fearful markets.
**Accept:** Max pain correct for hand-computed example. P/C ratio matches CBOE published values.

---

### T-0409 · Paper Trading System
**Files:** `quantcore/live/paper.py`
**Deps:** T-0121, T-0321, T-0221
**Steps:**
1. `PaperTrader`: main loop — connect feeds → process bars → run strategies → route to PaperBroker
2. Market open/close handling: init at open, EOD report at close
3. Real-time P&L tracking, risk metric computation
4. Performance vs backtest comparison: detect strategy degradation
5. Graduation criteria check: 2 weeks, Sharpe > threshold, max DD < limit, zero errors
6. State save/restore: persist at EOD, resume next day
7. Test: run paper trader on 1 day of recorded data, verify P&L matches expected.
**Accept:** Paper trades for full day without crash. P&L tracking correct. State persists across restarts.

---

### T-0410 · Live Trading Loop
**Files:** `quantcore/live/trader.py`
**Deps:** T-0409, T-0310, T-0322
**Steps:**
1. `LiveTrader`: async event loop — feed → signal → risk check → order → fill → update
2. Strategy orchestrator: run N strategies concurrently, aggregate orders
3. Startup: load config, connect broker, reconcile positions, verify feeds
4. Shutdown: cancel open orders, save state, generate EOD report
5. Error handling: catch all exceptions, log, continue (never crash silently)
6. Test: simulate full trading day with mock broker and recorded feed.
**Accept:** Processes 1000+ signals per day. No unhandled exceptions. State consistent after shutdown/restart.

---

### T-0411 · Watchdog & Health Monitoring
**Files:** `quantcore/live/watchdog.py`
**Deps:** T-0410
**Steps:**
1. `Watchdog`: separate process monitoring trader heartbeat
2. Heartbeat: trader sends ping every 10s, watchdog expects within 30s
3. Dead man's switch: no heartbeat for 2 min → emergency flatten via broker API
4. Health checks: CPU <90%, memory <85%, disk <90%, feed latency <5s
5. Auto-restart: if trader crashes, restart up to 3 times, then halt + alert
6. Test: kill trader process, verify watchdog detects and restarts.
**Accept:** Detects trader death within 30s. Emergency flatten executes. Auto-restart works.

---

### T-0412 · Position Reconciliation
**Files:** `quantcore/live/reconciliation.py`
**Deps:** T-0410
**Steps:**
1. `reconcile_positions(internal, broker) → list[Discrepancy]`: compare every 5 minutes
2. `reconcile_cash(internal, broker) → Discrepancy`
3. `reconcile_pnl(internal, broker) → Discrepancy`
4. Auto-correct: small discrepancies (< $100) → adjust internal, log warning
5. Manual escalation: large discrepancies → alert, halt trading
6. Test: inject position mismatch, verify detection and handling.
**Accept:** Detects 1-share discrepancy. Auto-corrects small mismatches. Escalates large ones.

---

### T-0413 · Emergency Procedures
**Files:** `quantcore/live/trader.py` (append emergency methods)
**Deps:** T-0410
**Steps:**
1. `emergency_flatten()`: cancel all open orders, close all positions at market
2. `kill_switch()`: exposed via API endpoint AND keyboard interrupt handler
3. Auto-halt triggers: drawdown > limit, VaR breach, >5 errors in 1 minute, feed down >1 min
4. Post-halt: require manual `resume()` with confirmation flag
5. Audit log: every emergency action logged with reason and timestamp
6. Test: trigger each auto-halt condition, verify trading stops within 1 second.
**Accept:** Kill switch halts all activity in <1s. Auto-halt fires correctly. Manual resume required.

---

### T-0414 · Dashboard Skeleton
**Files:** `quantcore/dashboard/app.py`
**Deps:** T-0112
**Steps:**
1. Plotly Dash app with multi-tab layout (Portfolio, Strategy, Risk, Market, Execution, Research)
2. Dark theme (plotly dark template + custom CSS)
3. Auto-refresh: 1s interval for P&L, 5s for risk metrics
4. Basic auth (configurable username/password)
5. Responsive layout (works on 1080p and 1440p)
6. Test: app starts, all tabs render without error.
**Accept:** Dashboard loads in <3s. Dark theme applied. Auth gate works.

---

### T-0415 · Portfolio Dashboard Panel
**Files:** `quantcore/dashboard/panels/portfolio.py`
**Deps:** T-0414
**Steps:**
1. Real-time P&L card: intraday, MTD, YTD, inception
2. Equity curve chart with SPY benchmark overlay
3. Positions table: symbol, qty, mkt_value, P&L, weight, delta, gamma
4. Risk utilization bars: VaR used/limit, DD current/limit
5. Asset allocation donut chart
6. Test: panel renders with sample data.
**Accept:** All components display correct data. Charts update on refresh.

---

### T-0416 · Strategy Dashboard Panel
**Files:** `quantcore/dashboard/panels/strategy.py`
**Deps:** T-0414
**Steps:**
1. P&L by strategy stacked area chart
2. Active signals table: name, symbol, direction, strength, timestamp
3. Recent trades table: time, symbol, side, qty, price, slippage
4. Strategy status indicator: running/paused/halted per strategy
5. Test: panel renders with sample multi-strategy data.
**Accept:** Strategy-level P&L sums to portfolio P&L. Status indicators update correctly.

---

### T-0417 · Risk Dashboard Panel
**Files:** `quantcore/dashboard/panels/risk.py`
**Deps:** T-0414
**Steps:**
1. VaR/CVaR gauge charts (current value vs limit)
2. Greeks heatmap: delta and gamma by underlying (color-coded)
3. Drawdown waterfall chart with threshold lines
4. Correlation matrix heatmap (top 20 positions)
5. Factor exposure bar chart (market, size, value, momentum, quality)
6. Test: panel renders with sample risk data.
**Accept:** Gauges correctly show utilization %. Heatmaps are readable and color-scaled.

---

### T-0418 · Market Intelligence Panel
**Files:** `quantcore/dashboard/panels/market.py`
**Deps:** T-0414, T-0330
**Steps:**
1. Regime probability bars (HMM state probabilities, stacked bar)
2. Volatility surface 3D chart (strike × expiry × IV, interactive rotation)
3. VIX term structure line chart (spot + futures)
4. Sector rotation heatmap (returns: 1D, 1W, 1M, 3M by sector)
5. Market breadth: A/D line, new highs vs lows bar chart
6. Test: panel renders with sample market data.
**Accept:** 3D vol surface is interactive. Regime bars sum to 100%. Sector heatmap readable.

---

### T-0419 · Execution Analytics Panel
**Files:** `quantcore/dashboard/panels/execution.py`
**Deps:** T-0414, T-0320
**Steps:**
1. VWAP performance chart: actual vs benchmark (scatter by order)
2. Slippage histogram (distribution of slippage in bps)
3. Fill quality: % filled at or better than limit
4. Latency distribution: order-to-fill time histogram
5. TCA summary table by strategy
6. Test: panel renders with sample TCA data.
**Accept:** Charts correctly display execution quality metrics. Slippage distribution has correct shape.

---

### T-0420 · Alert Engine
**Files:** `quantcore/dashboard/alerts.py`
**Deps:** T-0414
**Steps:**
1. `AlertRule`: condition (callable) + action (email/Slack/SMS) + cooldown (min re-alert interval)
2. Built-in rules: risk_limit_breach, drawdown_threshold, system_error, feed_down, unusual_pnl
3. Actions: `send_email(smtp)`, `send_slack(webhook_url)`, `send_sms(twilio)`
4. Alert history: log all alerts with timestamp, condition, action taken
5. Suppression: no repeat alert for same condition within cooldown window
6. Test: trigger alert condition, verify action called. Verify suppression during cooldown.
**Accept:** Alerts fire within 5s of condition. Suppression prevents spam. History logged.

---

### T-0421 · Docker Production Setup
**Files:** `Dockerfile`, `docker-compose.yml` (update from T-0109)
**Deps:** T-0109, T-0414, T-0410
**Steps:**
1. Multi-stage Dockerfile: builder (install all deps) → runtime (copy venv, minimal image)
2. Compose services: `trader`, `dashboard`, `data-feeds`, `timescaledb`, `redis`, `kafka`, `zookeeper`
3. Health checks on all services (HTTP /health for app, pg_isready for DB, etc.)
4. Volumes: DB data, config files, logs
5. Resource limits: memory and CPU per container
6. Test: `docker-compose up` starts full stack. All health checks pass.
**Accept:** Full stack starts in <2 minutes. App containers restart on failure. Logs accessible via `docker logs`.

---

### T-0422 · Kubernetes Manifests
**Files:** `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/configmap.yaml`, `k8s/hpa.yaml`
**Deps:** T-0421
**Steps:**
1. Deployments: trader (1 replica), dashboard (2 replicas), data-feeds (1 per source)
2. Services: ClusterIP for internal, LoadBalancer for dashboard
3. ConfigMap: from `config/production.toml`. Secrets: API keys, DB credentials
4. PVC for TimescaleDB storage
5. HPA: scale data-feeds based on CPU (target 70%)
6. Liveness/readiness probes: HTTP /health endpoints
7. Test: `kubectl apply` in local minikube, verify pods running.
**Accept:** All pods start and pass readiness probes. HPA scales feed handlers under load.

---

### T-0423 · FastAPI REST Layer
**Files:** `quantcore/api/main.py`, `quantcore/api/routes/`
**Deps:** T-0410
**Steps:**
1. FastAPI app with routers: `/health`, `/positions`, `/orders`, `/risk`, `/strategies`, `/config`
2. POST: `/orders/submit`, `/orders/cancel/{id}`, `/strategies/{id}/pause`, `/strategies/{id}/resume`, `/emergency/flatten`
3. WebSocket: `/ws/pnl` for real-time P&L streaming
4. Auth: API key in header (`X-API-Key`)
5. Rate limiting: 100 req/min per key
6. OpenAPI docs auto-generated at `/docs`
7. Test: hit each endpoint, verify response schema.
**Accept:** All endpoints respond correctly. Auth rejects invalid keys. WebSocket streams P&L.

---

### T-0424 · CI/CD Pipeline
**Files:** `.github/workflows/ci.yml`
**Deps:** T-0101
**Steps:**
1. Trigger: push to main, pull requests
2. Steps: checkout → setup Python → install deps → ruff check → mypy → pytest (unit) → pytest (integration with docker-compose) → build Docker image
3. Cache: pip cache, Docker layer cache
4. Staging deploy: on main branch merge
5. Test: push commit, verify pipeline runs all steps.
**Accept:** Pipeline completes in <10 minutes. Failures block merge. Docker image pushed to registry.

---

### T-0425 · Performance Optimization
**Files:** various hot-path modules
**Deps:** T-0147, all Phase 1-3 modules
**Steps:**
1. Profile with `cProfile`: identify top-10 slowest functions
2. Apply `@numba.jit(nopython=True)` to tight numerical loops (MC simulation, indicator computation)
3. Replace any Python loops over DataFrames with vectorized operations
4. Connection pooling: verify asyncpg pool configured correctly
5. Redis pipelining: batch reads/writes
6. Benchmark: before/after timing for top-10 functions
7. Test: no correctness regression after optimization.
**Accept:** Top-5 hot paths at least 2× faster. No numerical accuracy loss. Memory usage stable.

---

### T-0426 · Immutable Audit Trail
**Files:** `quantcore/compliance/audit_log.py`
**Deps:** T-0110
**Steps:**
1. `AuditLog`: append-only table (no UPDATE/DELETE permissions for app user)
2. Fields: id, timestamp, actor (strategy/user), action, entity_type, entity_id, details (JSON), prev_hash
3. Hash chain: each entry's hash includes previous entry's hash (tamper detection)
4. `log_event(actor, action, details)`: write to DB
5. `verify_chain() → bool`: recompute hashes, detect any tampering
6. Retention: 7 years, compressed after 90 days
7. Test: insert 100 entries, verify chain. Tamper one entry, verify detection.
**Accept:** Chain verification passes on clean data. Detects single-entry tampering. No UPDATE/DELETE possible.

---

### T-0427 · Pre-Trade Compliance Rules
**Files:** `quantcore/compliance/rules.py`
**Deps:** T-0314
**Steps:**
1. `ComplianceEngine`: rule chain — every order must pass all rules
2. Rules: restricted_list (no blacklisted symbols), position_limit (5% ownership threshold), short_locate, wash_sale_prevention (30-day lookback), pattern_day_trader
3. `check(order, portfolio, rules) → Approved | Rejected(rule, reason)`
4. Rules configurable in TOML: `[compliance.restricted_list]`, `[compliance.position_limits]`
5. Test: order for restricted symbol rejected. Wash sale detected and blocked.
**Accept:** All rules enforced. Every rejection logged with reason. Rules configurable without code change.

---

### T-0428 · Compliance Reporting
**Files:** `quantcore/compliance/reporting.py`
**Deps:** T-0426
**Steps:**
1. `daily_trade_blotter(date) → CSV/Excel`: all trades with timestamps, prices, commissions
2. `monthly_performance(month) → Report`: returns, AUM, fees
3. `tax_report(year) → Report`: realized gains/losses, wash sale adjustments, short/long term
4. `regulatory_data(period) → dict`: data formatted for potential regulatory filings
5. Test: generate blotter for sample trading day. Tax report matches manual calculation.
**Accept:** Blotter matches audit log. Tax calculations correct. Reports export to Excel/CSV.

---

### T-0429 · Alpha Research Framework
**Files:** `quantcore/research/alpha_research.py`
**Deps:** T-0216
**Steps:**
1. `AlphaHypothesis`: template — signal definition, universe, holding period, expected IC
2. `single_signal_backtest(signal, returns) → AlphaReport`: IC, turnover, decay, capacity estimate
3. `deflated_sharpe(sharpe, n_trials, lookback) → deflated`: Bailey & Lopez de Prado
4. `minimum_backtest_length(sharpe, n_trials) → years`: required years for significance
5. `multiple_testing_correction(p_values, method="BH") → adjusted_pvalues`: Bonferroni, BH (FDR)
6. Test: deflated Sharpe < raw Sharpe. Correction increases p-values.
**Accept:** Deflated Sharpe correctly penalizes data mining. BH controls FDR at specified level.

---

### T-0430 · Experiment Tracker
**Files:** `quantcore/research/experiment_tracker.py`
**Deps:** T-0110
**Steps:**
1. `ExperimentTracker`: log experiments with params, results, artifacts
2. `log_experiment(name, params, metrics, artifacts) → experiment_id`
3. `query_experiments(filters) → list[Experiment]`: search by name, date, metric range
4. `compare(exp_ids) → ComparisonTable`: side-by-side metrics
5. `parameter_sensitivity(base_params, param_name, values) → heatmap_data`: grid sweep
6. Store in `experiments` table with JSON params/metrics
7. Test: log 10 experiments, query by metric, compare top 3.
**Accept:** Experiments persist across sessions. Query filters work correctly. Comparison table renders.
