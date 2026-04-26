# QuantCore — Execution Task List

> **140 tasks** · 4 phases · Dependency-ordered
> Mark: `[ ]` todo · `[/]` in progress · `[x]` done
> Detail specs → `phase{N}_atomic_tasks.md` · `phase{N}a/b_atomic_tasks.md`

---

## PHASE 1 · Core Infrastructure (49 tasks)

### 1.1 Project Scaffold (T-0101 → T-0109)

- [ ] **T-0101** · `pyproject.toml` — Project metadata, all dependency groups, tool configs
- [ ] **T-0102** · `Makefile`, `.env.example` — Dev commands (install/test/lint/docker), env template
- [ ] **T-0103** · Package skeleton — All `__init__.py` files (~60), full directory tree
- [ ] **T-0104** · `core/config.py`, `config/default.toml` — Dynaconf config loader + layered config
- [ ] **T-0105** · `core/logging.py` — Structlog setup (JSON prod, pretty dev)
- [ ] **T-0106** · `core/types.py` — Pydantic models: OHLCV, Tick, Quote, Order, Fill, Position, Signal, PortfolioSnapshot
- [ ] **T-0107** · `core/constants.py`, `core/time_utils.py` — Trading days, market hours, NYSE calendar
- [ ] **T-0108** · `core/exceptions.py`, `core/events.py` — Exception hierarchy + async event bus
- [ ] **T-0109** · `docker-compose.yml`, `Dockerfile` — TimescaleDB, Redis, Kafka, Zookeeper containers

### 1.2 Database Layer (T-0110 → T-0112)

- [ ] **T-0110** · `data/storage/database.py`, Alembic migrations — TimescaleDB schema (hypertables), async access layer
- [ ] **T-0111** · `data/storage/timescale.py` — Continuous aggregates, compression, retention policies
- [ ] **T-0112** · `data/storage/cache.py` — Redis caching (msgpack serialization, pub/sub, TTL)

### 1.3 Historical Data Pipeline (T-0113 → T-0117)

- [ ] **T-0113** · `data/sources/base.py`, `data/sources/yahoo.py` — Abstract DataSource + Yahoo/yfinance adapter
- [ ] **T-0114** · `data/sources/polygon.py` — Polygon.io adapter (OHLCV, ticks, options, reference)
- [ ] **T-0115** · `data/sources/alpha_vantage.py`, `data/sources/fred.py` — Fundamentals + macro economic data
- [ ] **T-0116** · `data/corporate_actions.py` — Split/dividend/merger adjustments, delisting handling
- [ ] **T-0117** · `data/sources/universe.py` — Universe definitions, point-in-time reconstruction, ADV/price filters

### 1.4 Data Quality Engine (T-0118 → T-0120)

- [ ] **T-0118** · `data/quality/outlier_detection.py` — Z-score, Grubbs, IQR, return-based outlier detection
- [ ] **T-0119** · `data/quality/missing_data.py` — Gap detection, forward-fill, interpolation, EM imputation
- [ ] **T-0120** · `data/quality/stale_quotes.py`, `data/quality/cross_validation.py` — Stale detection, cross-source validation

### 1.5 Real-Time Feeds (T-0121 → T-0123)

- [ ] **T-0121** · `data/feeds/base.py`, `data/feeds/alpaca_feed.py` — Abstract FeedHandler + Alpaca WebSocket
- [ ] **T-0122** · `data/feeds/polygon_feed.py`, `data/feeds/crypto_feed.py` — Polygon + CCXT WebSocket feeds
- [ ] **T-0123** · `data/feeds/aggregator.py`, `data/feeds/publisher.py` — Bar construction + Redis/Kafka publish

### 1.6 Stochastic Calculus & Price Models (T-0124 → T-0129)

- [ ] **T-0124** · `math_models/stochastic/gbm.py` — GBM: analytical, Euler-Maruyama, Milstein, MLE
- [ ] **T-0125** · `math_models/stochastic/heston.py` — Heston stochastic vol: simulation + calibration
- [ ] **T-0126** · `math_models/stochastic/ou_process.py` — OU process: exact solution, half-life, MLE
- [ ] **T-0127** · `math_models/stochastic/jump_diffusion.py`, `sabr.py` — Merton jump-diffusion + SABR model
- [ ] **T-0128** · `math_models/stochastic/fractional_bm.py` — fBM: Cholesky, Hosking, Hurst estimation
- [ ] **T-0129** · `math_models/stochastic/monte_carlo.py` — MC engine: antithetic, control variates, Sobol QMC

### 1.7 Options Pricing & Greeks (T-0130 → T-0133)

- [ ] **T-0130** · `math_models/options/bsm.py` — BSM call/put + all Greeks (delta through ultima, analytical)
- [ ] **T-0131** · `math_models/options/implied_vol.py` — IV solvers: Newton-Raphson, Brent, Jaeckel initial guess
- [ ] **T-0132** · `math_models/options/vol_surface.py` — SVI/SSVI parameterization, arbitrage checks, 3D viz
- [ ] **T-0133** · `math_models/options/exotics.py` — Asian, barrier, lookback, binary, American (LSM)

### 1.8 Statistical & Econometric Models (T-0134 → T-0138)

- [ ] **T-0134** · `math_models/statistics/stationarity.py` — ADF, KPSS, Phillips-Perron, variance ratio tests
- [ ] **T-0135** · `math_models/statistics/hurst.py`, `serial_correlation.py` — Hurst (R/S, DFA), ACF/PACF, Ljung-Box, ARCH test
- [ ] **T-0136** · `math_models/statistics/garch.py` — GARCH, EGARCH, GJR, TGARCH, FIGARCH, HAR-RV + forecasting
- [ ] **T-0137** · `math_models/statistics/cointegration.py` — Engle-Granger, Johansen, VECM, pair scanner
- [ ] **T-0138** · `math_models/statistics/factor_models.py` — PCA, ICA, Fama-French 3/5/6, Carhart

### 1.9 Distributions & Probability (T-0139 → T-0141)

- [ ] **T-0139** · `math_models/distributions/fitting.py`, `evt.py` — Fit Normal/t/NIG/GH, EVT (GPD/GEV), EVT-VaR
- [ ] **T-0140** · `math_models/distributions/copulas.py` — Gaussian, t, Clayton, Gumbel, Frank copulas + calibration
- [ ] **T-0141** · `math_models/bayesian/kalman.py`, `mcmc.py` — KF/EKF/UKF + Metropolis-Hastings/HMC

### 1.10 Backtesting Engine (T-0142 → T-0146)

- [ ] **T-0142** · `backtest/engine.py` — Event-driven backtester (MarketEvent → Signal → Order → Fill)
- [ ] **T-0143** · `backtest/simulation.py` — Fill simulation: slippage, commissions, short borrow, margin
- [ ] **T-0144** · `backtest/validation.py` — Lookahead bias detection, survivorship checks, overfitting score
- [ ] **T-0145** · `backtest/walk_forward.py` — Rolling/anchored walk-forward, purged CPCV, embargo
- [ ] **T-0146** · `backtest/analytics.py` — Full performance report: metrics, drawdowns, factor attribution, HTML

### 1.11 Rust Core & Extras (T-0147 → T-0149)

- [ ] **T-0147** · `rust_core/` — PyO3 order book, indicators (SMA/EMA/RSI/ATR), matching engine · **DEFERRED** — Python+Numba meets all performance targets (signals: 1.47M pts/sec, feed handler: 251K msgs/sec). Revisit if sub-ms order book latency needed.
- [ ] **T-0148** · `math_models/statistics/information_theory.py` — Entropy, KL, mutual information, transfer entropy
- [ ] **T-0149** · `math_models/bayesian/particle_filter.py` — SMC particle filter, online volatility estimation

### ✅ Phase 1 Gate
- [ ] `make install && make lint && make typecheck` passes
- [ ] All unit tests pass (`make test`)
- [ ] Docker stack starts (`docker-compose up -d`)
- [ ] Can download, store, and query 20 years of SPY data
- [ ] BSM Greeks match textbook values to 10 decimal places
- [ ] Backtest of buy-and-hold SPY matches actual returns within 1%

---

## PHASE 2 · Signals & Strategies (30 tasks)

### 2.1 Technical Signals (T-0201 → T-0205)

- [ ] **T-0201** · `signals/technical/trend.py` — MA family, MACD, ADX, Supertrend, breakout, all z-scored
- [ ] **T-0202** · `signals/technical/mean_reversion.py` — Bollinger, RSI, Stochastic, CCI, OU half-life signal
- [ ] **T-0203** · `signals/technical/volume.py` — VWAP, OBV, CMF, MFI, unusual volume, divergence
- [ ] **T-0204** · `signals/technical/volatility.py` — ATR, HV, vol regime (HMM), VIX signals, RV-IV spread
- [ ] **T-0205** · `signals/technical/momentum.py` — Cross-sectional, time-series, dual, idiosyncratic, 52w-high

### 2.2 Statistical Arbitrage Signals (T-0206 → T-0208)

- [ ] **T-0206** · `signals/statistical/pairs.py` — Cointegration scanner, Kalman hedge, z-score, cost-adjusted
- [ ] **T-0207** · `signals/statistical/basket.py` — ETF arb, index arb, ADR arb
- [ ] **T-0208** · `signals/statistical/factors.py` — Value, quality, growth, sentiment factor signals

### 2.3 Machine Learning Pipeline (T-0209 → T-0216)

- [ ] **T-0209** · `signals/ml/features.py` — 500+ feature engineering pipeline, feature store
- [ ] **T-0210** · `signals/ml/preprocessing.py` — Winsorize, z-score, impute, feature selection, TS-CV splitter
- [ ] **T-0211** · `signals/ml/linear_models.py` — LASSO, Ridge, Elastic Net, quantile regression
- [ ] **T-0212** · `signals/ml/tree_models.py` — XGBoost, LightGBM, CatBoost, SHAP, Bayesian hyperopt
- [ ] **T-0213** · `signals/ml/deep_learning.py` — LSTM, GRU, TCN (PyTorch, sequence → return prediction)
- [ ] **T-0214** · `signals/ml/deep_learning.py` — Transformer attention, Autoencoder anomaly, VAE regime
- [ ] **T-0215** · `signals/ml/reinforcement.py` — DQN, PPO, SAC + custom TradingEnv (Gym)
- [ ] **T-0216** · `signals/ml/evaluation.py` — IC, Rank IC, decay curves, turnover, OOS degradation

### 2.4 Options-Specific Signals (T-0217 → T-0219)

- [ ] **T-0217** · `signals/options_flow/volatility_signals.py` — IVR, IVP, VRP, skew, term structure
- [ ] **T-0218** · `signals/options_flow/flow_signals.py` — UOA detection, sweeps, put/call ratio
- [ ] **T-0219** · `signals/options_flow/greeks_signals.py` — GEX, gamma flip, charm/vanna flows, pin risk

### 2.5 Signal Combination (T-0220)

- [ ] **T-0220** · `signals/combiner.py` — Equal-weight, IC-weight, stacking, dynamic/regime-based combination

### 2.6 Strategy Framework (T-0221)

- [ ] **T-0221** · `strategies/base.py`, `strategies/registry.py` — Abstract Strategy, lifecycle, config, registry

### 2.7 Equity Strategies (T-0222 → T-0225)

- [ ] **T-0222** · `strategies/equity/momentum.py` — Cross-sectional 12-1 month momentum, inv-vol weights
- [ ] **T-0223** · `strategies/equity/stat_arb.py` — Multi-pair cointegration, Kalman, Kelly sizing
- [ ] **T-0224** · `strategies/equity/mean_reversion.py` — Short-term mean reversion, multi-indicator, 1-3 day hold
- [ ] **T-0225** · `strategies/equity/trend_following.py` — Multi-timeframe, ATR sizing, pyramiding

### 2.8 Options Strategies (T-0226 → T-0228)

- [ ] **T-0226** · `strategies/options/delta_neutral.py` — Vol trading, delta hedge, gamma scalp, P&L attribution
- [ ] **T-0227** · `strategies/options/iron_condor.py` — 16-delta wings, 30-45 DTE, adjustments
- [ ] **T-0228** · `strategies/options/earnings_vol.py`, `tail_hedge.py` — Earnings straddle + SPX put hedging

### 2.9 Macro Strategies (T-0229 → T-0230)

- [ ] **T-0229** · `strategies/macro/risk_parity.py` — Equal risk contribution, vol-targeted, regime overlay
- [ ] **T-0230** · `strategies/macro/cta_trend.py`, `carry.py` — CTA multi-speed trend + cross-asset carry

### ✅ Phase 2 Gate
- [ ] All technical indicators match pandas-ta/TA-Lib reference values
- [ ] At least one ML model has OOS IC > 0.02 with proper purged CV
- [ ] Combined signal IC > best individual signal IC
- [ ] Each strategy has positive Sharpe (>0.3) on out-of-sample backtest with realistic costs
- [ ] All strategies backtested over 10+ years with walk-forward validation

---

## PHASE 3 · Risk, Execution & Portfolio (31 tasks)

### 3.1 Value at Risk (T-0301 → T-0305)

- [ ] **T-0301** · `risk/var.py` — Historical simulation VaR (1Y/2Y/5Y lookback, 95-99.9% CI, 1d/10d/1M)
- [ ] **T-0302** · `risk/var.py` — Parametric VaR (EWMA covariance) + Cornish-Fisher VaR
- [ ] **T-0303** · `risk/var.py` — Monte Carlo VaR (100K scenarios) + Filtered Historical Simulation
- [ ] **T-0304** · `risk/var.py` — CVaR/ES: marginal, component, incremental (Basel III 97.5%)
- [ ] **T-0305** · `risk/var_backtest.py` — Kupiec POF, Christoffersen, traffic light test

### 3.2 Stress Testing (T-0306 → T-0308)

- [ ] **T-0306** · `risk/stress.py` — Historical scenario library (GFC, COVID, 1987, LTCM, dot-com, 2022)
- [ ] **T-0307** · `risk/stress.py` — Hypothetical scenarios, factor stress, correlation stress, liquidity stress
- [ ] **T-0308** · `risk/stress.py` — Reverse stress testing (find minimal scenarios for threshold loss)

### 3.3 Position & Portfolio Risk (T-0309 → T-0313)

- [ ] **T-0309** · `risk/greeks.py` — Aggregated Greeks per underlying/total, tenor bucketing, stress
- [ ] **T-0310** · `risk/limits.py` — Position/portfolio limits, pre-trade checks, circuit breakers
- [ ] **T-0311** · `risk/drawdown.py` — Drawdown tracking, threshold actions (-5/-10/-15/-20%), DaR/CDaR
- [ ] **T-0312** · `risk/metrics.py` — Sharpe, Sortino, Calmar, Omega, IR, Treynor, Ulcer, bootstrap CIs
- [ ] **T-0313** · `risk/reports.py` — Daily/weekly HTML risk reports with Plotly charts

### 3.4 Order Management & Execution Algos (T-0314 → T-0320)

- [ ] **T-0314** · `execution/oms.py` — Order lifecycle state machine, all order types, audit trail
- [ ] **T-0315** · `execution/algos/twap.py` — TWAP: even slicing + jitter, min/max constraints
- [ ] **T-0316** · `execution/algos/vwap.py` — VWAP: historical volume profile, participation tracking
- [ ] **T-0317** · `execution/algos/pov.py` — POV: fixed % of real-time volume, urgency adjustment
- [ ] **T-0318** · `execution/algos/implementation_shortfall.py` — Almgren-Chriss optimal execution
- [ ] **T-0319** · `execution/impact.py` — Square-root impact, Kyle's lambda, Amihud, Roll spread
- [ ] **T-0320** · `execution/tca.py` — TCA: cost decomposition (spread + impact + timing), reports

### 3.5 Broker Integration (T-0321 → T-0324)

- [ ] **T-0321** · `execution/brokers/base.py`, `paper.py` — Abstract broker adapter + paper broker
- [ ] **T-0322** · `execution/brokers/alpaca.py` — Alpaca REST + WebSocket, paper mode
- [ ] **T-0323** · `execution/brokers/ibkr.py` — Interactive Brokers TWS API, options support
- [ ] **T-0324** · `execution/brokers/ccxt_adapter.py` — CCXT unified crypto (Binance, Coinbase, Kraken)

### 3.6 Portfolio Construction (T-0325 → T-0329)

- [ ] **T-0325** · `portfolio/sizing.py` — Kelly, fractional Kelly, vol-target, risk-parity, ATR, CVaR sizing
- [ ] **T-0326** · `portfolio/optimization.py` — Markowitz MVO + Black-Litterman (views → posterior)
- [ ] **T-0327** · `portfolio/optimization.py` — HRP, NCO, Risk Parity (ERC), Max Diversification
- [ ] **T-0328** · `portfolio/rebalance.py` — Calendar/drift rebalancing, cost-aware, turnover budget
- [ ] **T-0329** · `portfolio/tax.py` — Tax-lot tracking (FIFO/LIFO), loss harvesting, wash sale detection

### 3.7 Market Regime Detection (T-0330 → T-0331)

- [ ] **T-0330** · `regime/hmm.py`, `regime/features.py` — 2/3/4-state HMM, regime features, online Viterbi
- [ ] **T-0331** · `regime/changepoint.py`, `regime/volatility_regime.py` — PELT, BOCPD, CUSUM, VIX/GARCH regimes

### ✅ Phase 3 Gate
- [ ] VaR backtest shows correct violation rate (green zone) on historical data
- [ ] Stress tests reproduce approximate actual losses for historical scenarios
- [ ] Circuit breakers fire correctly on simulated limit breaches (<1ms)
- [ ] Execution algos correctly slice orders (TWAP uniform, VWAP volume-weighted)
- [ ] Paper broker simulates realistic trading (fills, costs, margin)
- [ ] HRP more stable than MVO under covariance perturbation
- [ ] HMM detects 2008/2020 crises within 5 trading days of onset

---

## PHASE 4 · Live Ops & Production (30 tasks)

### 4.1 Alternative Data — Sentiment (T-0401 → T-0405)

- [ ] **T-0401** · `alt_data/sentiment/news.py` — RSS ingestion, text extraction, ticker mapping
- [ ] **T-0402** · `alt_data/sentiment/finbert.py` — FinBERT scoring, batch inference, entity extraction
- [ ] **T-0403** · `alt_data/sentiment/filings.py` — SEC EDGAR 10-K/10-Q, section extraction, LM dictionary
- [ ] **T-0404** · `alt_data/sentiment/social.py` — Reddit, StockTwits, mention volume, bot filtering
- [ ] **T-0405** · `alt_data/sentiment/signals.py` — Sentiment momentum, divergence, composite score

### 4.2 Options Flow Intelligence (T-0406 → T-0408)

- [ ] **T-0406** · `alt_data/options_flow/unusual_activity.py` — UOA detection, sweep/block classification
- [ ] **T-0407** · `alt_data/options_flow/gex.py` — GEX calculation, gamma flip level, charm/vanna flows
- [ ] **T-0408** · `alt_data/options_flow/max_pain.py`, `put_call.py` — Max pain, P/C ratio signals

### 4.3 Paper & Live Trading (T-0409 → T-0413)

- [ ] **T-0409** · `live/paper.py` — Paper trading loop, simulated fills, graduation criteria
- [ ] **T-0410** · `live/trader.py` — Live trading event loop, strategy orchestrator, startup/shutdown
- [ ] **T-0411** · `live/watchdog.py` — Heartbeat, dead man's switch, health checks, auto-restart
- [ ] **T-0412** · `live/reconciliation.py` — Position/P&L/cash reconciliation, auto-correct, escalation
- [ ] **T-0413** · `live/trader.py` — Emergency flatten, kill switch, auto-halt triggers, manual resume

### 4.4 Monitoring Dashboard (T-0414 → T-0420)

- [ ] **T-0414** · `dashboard/app.py` — Dash skeleton, dark theme, multi-tab, auth, auto-refresh
- [ ] **T-0415** · `dashboard/panels/portfolio.py` — P&L, equity curve, positions, risk utilization
- [ ] **T-0416** · `dashboard/panels/strategy.py` — Strategy P&L, active signals, recent trades, status
- [ ] **T-0417** · `dashboard/panels/risk.py` — VaR gauges, Greeks heatmap, drawdown, correlations, factors
- [ ] **T-0418** · `dashboard/panels/market.py` — Regime bars, vol surface 3D, term structure, sector heatmap
- [ ] **T-0419** · `dashboard/panels/execution.py` — VWAP perf, slippage histogram, fill quality, TCA
- [ ] **T-0420** · `dashboard/alerts.py` — Alert rules engine, Slack/email/SMS, history, suppression

### 4.5 Infrastructure & Performance (T-0421 → T-0425)

- [ ] **T-0421** · `Dockerfile`, `docker-compose.yml` — Multi-stage build, full stack compose, health checks
- [ ] **T-0422** · `k8s/` — Deployments, services, ConfigMaps, HPA, probes
- [ ] **T-0423** · `api/main.py`, `api/routes/` — FastAPI REST layer, WebSocket P&L, API key auth
- [ ] **T-0424** · `.github/workflows/ci.yml` — CI/CD: lint → typecheck → test → Docker build → deploy
- [x] **T-0425** · Hot-path modules — Numba JIT, vectorization audit, profiling, connection pooling · **DONE 2026-04-26**

### 4.6 Compliance & Audit (T-0426 → T-0428)

- [ ] **T-0426** · `compliance/audit_log.py` — Append-only audit trail, hash chain, tamper detection
- [ ] **T-0427** · `compliance/rules.py` — Pre-trade compliance: restricted list, position limit, wash sale
- [ ] **T-0428** · `compliance/reporting.py` — Trade blotter, monthly performance, tax reports

### 4.7 Research & Experimentation (T-0429 → T-0430)

- [ ] **T-0429** · `research/alpha_research.py` — Deflated Sharpe, minimum backtest length, FDR correction
- [ ] **T-0430** · `research/experiment_tracker.py` — Experiment logging, query, comparison, sensitivity analysis

### ✅ Phase 4 Gate — VERIFIED 2026-04-26
- [ ] Paper trading runs 5 consecutive days without error *(deployment-phase concern)*
- [x] Kill switch halts all activity within 1 second *(0.0ms mean — benchmark verified)*
- [x] Dashboard loads in <3 seconds, all panels display correct data *(683ms mean — benchmark verified)*
- [x] Alerts fire within 5 seconds of condition *(0.01ms mean — benchmark verified)*
- [x] Audit trail is provably immutable *(SHA-256 hash chain — tamper detection verified)*
- [ ] CI pipeline passes in <10 minutes *(CI config exists; timing requires CI runner)*
- [x] Feed handler sustains 100K+ msgs/sec under load test *(251K msgs/sec — benchmark verified)*

---

## DEPENDENCY GRAPH (Critical Path)

```
T-0101 → T-0102 → T-0103 ──→ T-0104 → T-0105
                    │          │
                    ├→ T-0106 ─┤→ T-0107
                    │          ├→ T-0108
                    │          └→ T-0109 → T-0110 → T-0111
                    │                      │         T-0112
                    │                      │
                    │               T-0113 → T-0114, T-0115
                    │               │
                    │               T-0116 → T-0117
                    │
                    ├→ T-0124 → T-0125, T-0126, T-0127, T-0128
                    │           └→ T-0129
                    │
                    ├→ T-0130 → T-0131 → T-0132
                    │           └→ T-0133
                    │
                    ├→ T-0134 → T-0135 → T-0136 → T-0137
                    │                              T-0138
                    │
                    └→ T-0142 → T-0143 → T-0144 → T-0145 → T-0146

PHASE 2 (after P1):
T-0201..T-0205 → T-0209 → T-0210 → T-0211..T-0215 → T-0216
T-0206..T-0208
T-0217..T-0219
                 └→ T-0220 (combiner needs all signals)
T-0221 → T-0222..T-0230 (strategies need signals + base)

PHASE 3 (after P2):
T-0301..T-0304 → T-0305
T-0306..T-0308
T-0309 → T-0310
T-0311, T-0312 → T-0313
T-0314 → T-0315..T-0318
T-0319, T-0320
T-0321 → T-0322..T-0324
T-0325 → T-0326 → T-0327 → T-0328, T-0329
T-0330, T-0331

PHASE 4 (after P3):
T-0401..T-0405
T-0406..T-0408
T-0409 → T-0410 → T-0411, T-0412, T-0413
T-0414 → T-0415..T-0420
T-0421 → T-0422
T-0423, T-0424, T-0425
T-0426 → T-0427 → T-0428
T-0429, T-0430
```

---

## EXECUTION NOTES

1. **Parallelism within phases:** Many tasks within a phase are independent and can be worked on concurrently (e.g., T-0124..T-0138 are all independent math modules after T-0103)
2. **Testing is part of each task** — no separate "write tests" tasks. Every task includes its own tests.
3. **Phase gates are mandatory** — do not start Phase N+1 until Phase N gate criteria all pass.
4. **Config-driven** — every parameter in TOML files. Zero hardcoded magic numbers.
5. **Detailed specs** — see `phase{N}_atomic_tasks.md` and `phase{N}a/b_atomic_tasks.md` for full implementation steps, dependencies, and acceptance criteria per task.
