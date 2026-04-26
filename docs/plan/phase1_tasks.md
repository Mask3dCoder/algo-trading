# Phase 1 — Core Infrastructure Tasks

## AGENT-P1-01: Project Scaffold

**Goal:** Create full project skeleton with packaging, config, and dev tooling.

### Tasks
1. Init `pyproject.toml` with all dependency groups from master plan
2. Create full directory tree (all `__init__.py`, empty modules per master plan structure)
3. Create `config/default.toml` with sections: `[database]`, `[redis]`, `[kafka]`, `[logging]`, `[brokers]`, `[risk_limits]`, `[execution]`
4. Implement `quantcore/core/config.py` — dynaconf-based loader with env layering
5. Implement `quantcore/core/logging.py` — structlog setup (JSON prod, pretty dev)
6. Implement `quantcore/core/types.py` — shared types: `OHLCV`, `Tick`, `Order`, `Fill`, `Position`, `Signal`, `PortfolioSnapshot` as Pydantic models
7. Implement `quantcore/core/constants.py` — trading days/year (252), annualization factors, market hours
8. Implement `quantcore/core/exceptions.py` — `DataError`, `ExecutionError`, `RiskLimitBreached`, `ConfigError`
9. Implement `quantcore/core/events.py` — lightweight pub/sub event bus (async-compatible)
10. Implement `quantcore/core/time_utils.py` — market calendar, timezone conversions, trading session detection
11. Create `docker-compose.yml` with PostgreSQL+TimescaleDB, Redis, Kafka+Zookeeper
12. Create `Dockerfile` for the app
13. Create `Makefile` with targets: `install`, `test`, `lint`, `typecheck`, `format`, `docker-up`, `docker-down`
14. Create `.env.example`
15. Configure `ruff`, `mypy`, `pytest` in `pyproject.toml`

**Acceptance:** `make install && make lint && make typecheck` passes. Docker stack starts.

---

## AGENT-P1-02: Database Layer

**Goal:** TimescaleDB schema and access layer for all market data.

### Tasks
1. Design schema: `ohlcv_daily`, `ohlcv_intraday`, `ticks`, `options_chains`, `corporate_actions`, `fundamentals`, `signals`, `orders`, `fills`, `positions`, `portfolio_snapshots`
2. Create SQL migration files (use Alembic)
3. `ohlcv_daily`: hypertable partitioned by time, columns: `symbol`, `timestamp`, `open`, `high`, `low`, `close`, `volume`, `adj_close`, `dividend`, `split_factor`
4. `ticks`: hypertable, columns: `symbol`, `timestamp` (microsecond), `price`, `size`, `exchange`, `conditions`
5. `options_chains`: `underlying`, `expiry`, `strike`, `right`, `timestamp`, `bid`, `ask`, `last`, `volume`, `oi`, `iv`, `delta`, `gamma`, `theta`, `vega`
6. Implement `quantcore/data/storage/database.py` — async connection pool (asyncpg), CRUD operations
7. Implement `quantcore/data/storage/timescale.py` — TimescaleDB-specific: continuous aggregates, compression policies, retention policies
8. Implement `quantcore/data/storage/cache.py` — Redis caching layer with TTL, invalidation
9. Write integration tests against Docker TimescaleDB

**Acceptance:** Can insert and query 1M+ rows efficiently. Continuous aggregates working for 1min→5min→hourly→daily rollups.

---

## AGENT-P1-03: Historical Data Manager

**Goal:** Multi-source historical data download, normalization, and storage.

### Tasks
1. Define abstract `DataSource` interface: `fetch_ohlcv()`, `fetch_options()`, `fetch_fundamentals()`, `supported_assets()`
2. Implement adapters in `quantcore/data/sources/`:
   - `yahoo.py` — yfinance: OHLCV for all US equities, indices, ETFs
   - `polygon.py` — Polygon.io: tick data, options, reference data
   - `alpha_vantage.py` — fundamentals, earnings, economic indicators
   - `nasdaq_datalink.py` — Quandl replacement: macro data, alternative data
   - `fred.py` — Federal Reserve data: rates, yield curves, economic indicators
3. Implement `quantcore/data/corporate_actions.py`:
   - Split adjustment (forward/reverse)
   - Dividend adjustment
   - Merger/acquisition handling
   - Spinoff handling
   - Delisting tracking (survivorship bias correction)
4. Implement `quantcore/data/sources/universe.py`:
   - Universe definitions: S&P500, Russell 2000, full US equity, sector-based
   - Point-in-time universe reconstruction (who was in S&P500 on date X?)
   - ADV (average daily volume) filtering
5. Implement bulk download orchestrator: parallel downloads with rate limiting, checkpoint/resume, progress tracking
6. Data normalization pipeline: timezone standardization (UTC), price adjustment, volume normalization
7. Write unit tests for each adapter (mock HTTP responses)
8. Write integration test: download 1 year of SPY data, verify adjustments

**Acceptance:** Can download full S&P500 daily history (20+ years), store adjusted, query efficiently.

---

## AGENT-P1-04: Data Quality Engine

**Goal:** Automated detection and correction of data anomalies.

### Tasks
1. Implement `quantcore/data/quality/outlier_detection.py`:
   - Z-score method (configurable threshold, default 4σ)
   - Grubbs test for single outliers
   - IQR method (1.5× and 3× IQR)
   - Return-based outlier detection (flag >10σ daily moves)
2. Implement `quantcore/data/quality/missing_data.py`:
   - Detection: gap identification in time series
   - Forward-fill (configurable max gap)
   - Linear / cubic spline interpolation
   - EM algorithm for multivariate missing data
   - Log warnings for any imputed data
3. Implement `quantcore/data/quality/stale_quotes.py`:
   - Detect unchanged prices over configurable window
   - Cross-reference with exchange trading hours
4. Implement `quantcore/data/quality/cross_validation.py`:
   - Compare prices across 2+ sources
   - Flag discrepancies > configurable threshold
   - Automated source selection (majority vote)
5. Implement `quantcore/data/quality/report.py`:
   - Generate data quality report: completeness %, outlier count, adjustments applied
6. Write tests with synthetic data containing known anomalies

**Acceptance:** Pipeline catches injected outliers, gaps, and stale quotes with >95% recall.

---

## AGENT-P1-05: Real-Time Data Feed Handler

**Goal:** WebSocket-based real-time market data with order book reconstruction.

### Tasks
1. Implement `quantcore/data/feeds/base.py` — abstract `FeedHandler`: `connect()`, `subscribe()`, `on_tick()`, `on_quote()`, `on_bar()`
2. Implement `quantcore/data/feeds/alpaca_feed.py` — Alpaca WebSocket: trades, quotes, bars
3. Implement `quantcore/data/feeds/polygon_feed.py` — Polygon WebSocket: trades, quotes, aggregates
4. Implement `quantcore/data/feeds/crypto_feed.py` — CCXT WebSocket: multiple crypto exchanges
5. Implement `quantcore/data/feeds/orderbook.py`:
   - Level 2 order book reconstruction from tick stream
   - Maintain sorted bid/ask arrays
   - Calculate spread, mid, microprice, VWAP from book
   - Book imbalance signal
   - **Rust implementation** in `rust_core/src/orderbook.rs` with PyO3 binding
6. Implement `quantcore/data/feeds/aggregator.py`:
   - Real-time bar construction (1s, 1min, 5min) from tick stream
   - Real-time VWAP, TWAP calculation
   - Tick-to-trade ratio
7. Implement `quantcore/data/feeds/publisher.py` — publish normalized ticks/bars to Redis pub/sub and Kafka topic
8. Backpressure handling: ring buffer, drop-oldest policy when consumer is slow
9. Write tests with recorded WebSocket sessions (replay)

**Acceptance:** Handles 100K+ msgs/sec per feed. Order book reconstruction matches exchange snapshots.

---

## AGENT-P1-06: Stochastic Calculus & Price Models

**Goal:** Full SDE solver library with Monte Carlo engine.

### Tasks
1. Implement `quantcore/math_models/stochastic/gbm.py`:
   - Geometric Brownian Motion: `dS = μS dt + σS dW`
   - Analytical solution: `S(t) = S₀ exp((μ-σ²/2)t + σW(t))`
   - Euler-Maruyama discretization
   - Milstein scheme (higher-order)
   - Parameter estimation from historical data (MLE)
2. Implement `quantcore/math_models/stochastic/heston.py`:
   - Heston stochastic volatility: price + variance SDEs with correlation
   - Full truncation scheme for variance process (no negative variance)
   - Characteristic function (semi-analytical pricing)
   - Calibration to market options prices (Levenberg-Marquardt)
3. Implement `quantcore/math_models/stochastic/ou_process.py`:
   - Ornstein-Uhlenbeck: `dX = θ(μ - X)dt + σdW`
   - Analytical solution and transition density
   - Half-life calculation: `t½ = ln(2)/θ`
   - MLE parameter estimation
   - Use for mean-reversion trading signal generation
4. Implement `quantcore/math_models/stochastic/jump_diffusion.py`:
   - Merton jump-diffusion: GBM + Poisson jumps
   - Simulation with compound Poisson process
   - Characteristic function for pricing
5. Implement `quantcore/math_models/stochastic/sabr.py`:
   - SABR model for vol smile: forward + vol dynamics
   - Hagan approximation for implied vol
   - Calibration to market smile
6. Implement `quantcore/math_models/stochastic/fractional_bm.py`:
   - Fractional Brownian Motion with Hurst parameter H
   - Cholesky method and Hosking method for simulation
   - Hurst exponent estimation (R/S, DFA)
7. Implement `quantcore/math_models/stochastic/monte_carlo.py`:
   - Vectorized path generation (NumPy, 100K+ paths)
   - Variance reduction: antithetic variates, control variates, importance sampling
   - Quasi-Monte Carlo with Sobol sequences (`scipy.stats.qmc`)
   - GPU acceleration stub (CuPy optional)
   - Convergence diagnostics
8. Write tests: compare MC prices to analytical solutions (GBM call = BSM), verify convergence rates

**Acceptance:** GBM MC call price within 0.1% of BSM analytical. Heston calibration fits market smile within 0.5 vol points.

---

## AGENT-P1-07: Options Pricing & Greeks Engine

**Goal:** Complete options mathematics library.

### Tasks
1. Implement `quantcore/math_models/options/bsm.py`:
   - Black-Scholes-Merton: `call()`, `put()` with full formulas
   - All first-order Greeks: delta, gamma, theta, vega, rho (analytical)
   - All higher-order Greeks: vanna, volga, charm, speed, color, ultima (analytical)
   - Numerical Greeks via finite differences (verify against analytical)
2. Implement `quantcore/math_models/options/implied_vol.py`:
   - Newton-Raphson IV solver (primary, using vega)
   - Brent's method fallback (for edge cases)
   - Rational approximation initial guess (Jaeckel/Let's Be Rational)
   - Handle edge cases: deep ITM/OTM, near-expiry, zero vega
3. Implement `quantcore/math_models/options/vol_surface.py`:
   - SVI parameterization: `w(k) = a + b(ρ(k-m) + √((k-m)² + σ²))`
   - SSVI (Surface SVI) for full surface
   - Arbitrage-free interpolation (calendar spread, butterfly arbitrage checks)
   - Term structure modeling (flat, linear, Nelson-Siegel)
   - Surface visualization (3D plotly)
4. Implement `quantcore/math_models/options/exotics.py`:
   - Asian options: geometric (analytical), arithmetic (MC)
   - Barrier options: knock-in/out, up/down (analytical + MC)
   - Lookback options (MC)
   - Binary/Digital options (analytical)
   - American options: Longstaff-Schwartz LSM
   - Bermudan options: LSM with exercise boundary
5. Write exhaustive tests: BSM vs known values, put-call parity, Greek sign/magnitude checks, IV round-trip

**Acceptance:** BSM matches textbook values to 10 decimal places. IV solver converges in <10 iterations for 99.9% of cases. Put-call parity holds to machine precision.

---

## AGENT-P1-08: Statistical Analysis Engine

**Goal:** Time-series econometrics toolkit.

### Tasks
1. Implement `quantcore/math_models/statistics/stationarity.py`:
   - ADF test (wrap statsmodels, add interpretation)
   - KPSS test
   - Phillips-Perron test
   - Variance Ratio test (Lo-MacKinlay)
   - Unified API: `test_stationarity(series) → StationarityResult`
2. Implement `quantcore/math_models/statistics/hurst.py`:
   - R/S analysis
   - Detrended Fluctuation Analysis (DFA)
   - Wavelet-based estimation
   - Interpretation: H<0.5 mean-reverting, H=0.5 random walk, H>0.5 trending
3. Implement `quantcore/math_models/statistics/serial_correlation.py`:
   - ACF/PACF computation and plotting
   - Ljung-Box test
   - ARCH effects test (Engle's LM test)
4. Implement `quantcore/math_models/statistics/garch.py`:
   - GARCH(p,q) — wrap `arch` library with clean API
   - EGARCH (leverage effect)
   - GJR-GARCH (threshold asymmetry)
   - TGARCH
   - FIGARCH (long memory)
   - Model selection: AIC/BIC comparison
   - Forecast: 1-step and multi-step ahead volatility
   - HAR-RV model (realized volatility with 1d, 5d, 22d components)
5. Implement `quantcore/math_models/statistics/cointegration.py`:
   - Engle-Granger two-step test
   - Johansen test (trace + max eigenvalue)
   - VECM estimation
   - Rolling cointegration test
   - Pair scanner: test all N*(N-1)/2 pairs in universe
6. Implement `quantcore/math_models/statistics/factor_models.py`:
   - PCA on returns (scree plot, explained variance)
   - ICA
   - Fama-French 3/5/6-factor regression
   - Carhart 4-factor
   - Factor exposure estimation (OLS, WLS, robust)
   - Factor data download (Kenneth French data library)
7. Implement `quantcore/math_models/statistics/information_theory.py`:
   - Shannon entropy
   - KL divergence
   - Mutual information (continuous variables via KDE)
   - Transfer entropy (directional info flow)
8. Write tests against known results (e.g., ADF on random walk should fail to reject)

**Acceptance:** All tests match statsmodels/arch outputs. GARCH forecasts verified against published results. Pair scanner processes 500-symbol universe in <60s.

---

## AGENT-P1-09: Distribution Fitting & Probability

**Goal:** Fat-tailed distribution fitting, EVT, copulas, Bayesian inference.

### Tasks
1. Implement `quantcore/math_models/distributions/fitting.py`:
   - Fit and compare: Normal, Student-t, Skewed-t, NIG, Variance Gamma, GH
   - Goodness-of-fit: KS test, Anderson-Darling, QQ plots
   - AIC/BIC model selection
   - Alpha-Stable distribution (Lévy) fitting via characteristic function
2. Implement `quantcore/math_models/distributions/evt.py`:
   - GEV distribution fitting (block maxima)
   - GPD fitting (peaks over threshold)
   - Threshold selection (mean excess plot)
   - EVT-based VaR and CVaR
3. Implement `quantcore/math_models/distributions/copulas.py`:
   - Gaussian copula
   - Student-t copula
   - Clayton, Gumbel, Frank copulas
   - Vine copulas (C-vine, D-vine) — use `pyvinecopulib` if available
   - MLE calibration
   - Simulation from fitted copula
   - Tail dependence coefficient calculation
4. Implement `quantcore/math_models/bayesian/kalman.py`:
   - Standard Kalman Filter
   - Extended Kalman Filter (EKF)
   - Unscented Kalman Filter (UKF)
   - Application: dynamic hedge ratio estimation
5. Implement `quantcore/math_models/bayesian/mcmc.py`:
   - Metropolis-Hastings sampler
   - Hamiltonian Monte Carlo (HMC) — or wrap `pymc`
   - Convergence diagnostics (Gelman-Rubin, ESS, trace plots)
   - Application: Bayesian updating of return distributions
6. Implement `quantcore/math_models/bayesian/particle_filter.py`:
   - Sequential Monte Carlo for non-linear state-space models
   - Application: online volatility estimation
7. Write tests: fitted distributions recover known parameters from synthetic data

**Acceptance:** Student-t fit on S&P500 returns significantly beats Normal by AIC. Copula simulations preserve tail dependence.

---

## AGENT-P1-10: Optimization Methods

**Goal:** Numerical optimization library for portfolio and parameter tuning.

### Tasks
1. Implement `quantcore/math_models/optimization/classical.py`:
   - Newton-Raphson (1D and multi-D)
   - Gradient descent, SGD, Adam
   - L-BFGS-B (wrap scipy)
   - SQP (wrap scipy)
   - Interior point (wrap scipy/cvxpy)
2. Implement `quantcore/math_models/optimization/metaheuristic.py`:
   - Genetic algorithm
   - Simulated annealing
   - Particle swarm optimization
   - Bayesian optimization (Gaussian process surrogate — wrap `scikit-optimize`)
3. Implement `quantcore/math_models/optimization/portfolio.py`:
   - Markowitz MVO (min variance, max Sharpe, target return)
   - Black-Litterman (equilibrium returns + views → posterior)
   - Risk Parity / Equal Risk Contribution (scipy root finding)
   - Max Diversification portfolio
   - HRP (hierarchical clustering → quasi-diag → recursive bisection)
   - NCO (Nested Cluster Optimization)
   - Kelly Criterion (full and fractional)
   - CVaR optimization (linear programming via cvxpy)
   - Robust optimization with uncertainty sets
4. Constraints handling: long-only, box constraints, sector limits, turnover limits, cardinality
5. Write tests: MVO on 2-asset case matches analytical solution. HRP more stable than MVO under perturbation.

**Acceptance:** All optimizers converge on standard test functions. Portfolio optimizers produce valid weights (sum=1, within constraints).

---

## AGENT-P1-11: Basic Backtesting Engine

**Goal:** Event-driven backtester with realistic simulation.

### Tasks
1. Implement `quantcore/backtest/engine.py`:
   - Event-driven architecture: `MarketEvent` → `SignalEvent` → `OrderEvent` → `FillEvent`
   - Event queue with proper time ordering
   - Support bar-by-bar (daily, intraday) simulation
   - Strict no-lookahead enforcement (data only available at bar close)
   - Strategy interface: `on_bar()`, `on_signal()`, `generate_orders()`
2. Implement `quantcore/backtest/simulation.py`:
   - Fill simulation: market orders fill at next open (or configurable)
   - Limit order simulation against high/low
   - Slippage models: fixed, percentage, volume-based, volatility-based
   - Commission models: per-share, per-trade, tiered
   - Spread simulation from historical bid-ask
   - Short borrow cost
   - Margin and financing cost
3. Implement `quantcore/backtest/validation.py`:
   - Lookahead bias detector (scan for future data access)
   - Survivorship bias check (verify universe includes delisted)
   - Data snooping score (in-sample vs OOS degradation ratio)
4. Implement `quantcore/backtest/walk_forward.py`:
   - Rolling window walk-forward
   - Anchored walk-forward
   - Combinatorial Purged Cross-Validation (CPCV)
   - Embargo period between train/test
5. Implement `quantcore/backtest/analytics.py`:
   - Return metrics: total, annualized, monthly distribution, rolling
   - Risk metrics: Sharpe, Sortino, Calmar, Omega, IR, max drawdown
   - Bootstrap confidence intervals on all metrics
   - Trade analytics: win rate, profit factor, expectancy, avg hold time
   - Drawdown analysis: top-10 episodes with depth/duration/recovery
   - Factor attribution (Fama-French regression on strategy returns)
   - Monte Carlo equity curve simulation (bootstrap trades)
   - Generate full HTML report with plotly charts
6. Write tests: backtest a buy-and-hold SPY strategy, verify returns match actual

**Acceptance:** Backtest of known strategy reproduces published results within 1% (after costs). Walk-forward produces proper train/test splits with no leakage.

---

## AGENT-P1-12: Rust Core (Hot Paths)

**Goal:** Rust implementations for latency-critical components via PyO3.

### Tasks
1. Set up `rust_core/` Cargo project with PyO3 and maturin build
2. Implement `rust_core/src/orderbook.rs`:
   - `OrderBook` struct with sorted bid/ask levels (BTreeMap)
   - `apply_update()`, `best_bid()`, `best_ask()`, `spread()`, `midprice()`, `microprice()`
   - `book_imbalance()`, `vwap(depth)`
3. Implement `rust_core/src/indicators.rs`:
   - Vectorized SMA, EMA with streaming API
   - ATR, Bollinger Bands, RSI
   - All operating on `&[f64]` slices for zero-copy
4. Implement `rust_core/src/matching.rs`:
   - Simple simulated matching engine for backtesting
   - Price-time priority
   - Market/limit order types
5. Create Python bindings via PyO3 in `rust_core/src/lib.rs`
6. Add `maturin develop` to Makefile
7. Benchmark: Rust vs Python implementations, document speedup

**Acceptance:** Rust order book 10x+ faster than Python equivalent. Indicators match NumPy results to machine precision.
