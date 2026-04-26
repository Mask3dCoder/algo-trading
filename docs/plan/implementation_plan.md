# Algorithmic Trading System — Master Implementation Plan

## 1. Project Identity

| Field | Value |
|---|---|
| **Codename** | `quantcore` |
| **Root package** | `quantcore/` |
| **Language** | Python 3.12+ (primary), Rust (latency-critical paths) |
| **Workspace** | `c:\Users\rochv\Downloads\projetcs\algo sw` |

---

## 2. Delivery Phases

| Phase | Name | Modules | Est. Effort |
|---|---|---|---|
| **P1** | Core Infrastructure | 1 (Data), 2 (Math Models), 7 (Backtesting basics), 15 (Project scaffold) | Foundation |
| **P2** | Signals & Strategies | 3 (Signals), 4 (Strategies), 2.2 (Options Pricing), 5 (ML models) | Alpha layer |
| **P3** | Risk, Execution & Portfolio | 5 (Risk Mgmt), 6 (Execution/OMS), 8 (Portfolio Construction), 9 (Regime Detection) | Production safety |
| **P4** | Live Ops & Monitoring | 10 (Alt Data), 11 (Live Trading), 12 (Paper Trading), 13 (Monitoring/Dashboard), 14 (Infra/Perf), 16 (Compliance) | Go-live |

> [!IMPORTANT]
> Each phase MUST be fully tested and reviewed before the next phase begins. No phase skipping.

---

## 3. Project Directory Structure

```
quantcore/
├── pyproject.toml                   # Project metadata, dependencies, build config
├── Makefile                         # Common dev commands
├── docker-compose.yml               # Local dev stack (Postgres, Redis, Kafka)
├── Dockerfile                       # App container
├── .env.example                     # Environment variable template
├── config/
│   ├── default.toml                 # Default configuration
│   ├── development.toml             # Dev overrides
│   ├── production.toml              # Prod overrides
│   └── strategies/                  # Per-strategy TOML configs
│       ├── momentum.toml
│       ├── stat_arb.toml
│       └── ...
├── quantcore/                       # Main Python package
│   ├── __init__.py
│   ├── core/                        # Cross-cutting utilities
│   │   ├── config.py                # TOML config loader (dynaconf)
│   │   ├── logging.py               # Structured logging (structlog)
│   │   ├── types.py                 # Shared type definitions
│   │   ├── constants.py             # Financial constants
│   │   ├── exceptions.py            # Custom exception hierarchy
│   │   ├── events.py                # Event bus (pub/sub)
│   │   └── time_utils.py            # Timezone-aware helpers
│   ├── data/                        # MODULE 1 — Data Infrastructure
│   │   ├── sources/                 # Data source adapters
│   │   ├── storage/                 # DB access layer (TimescaleDB)
│   │   ├── feeds/                   # Real-time feed handlers
│   │   ├── quality/                 # Data quality engine
│   │   └── corporate_actions.py
│   ├── math_models/                 # MODULE 2 — Mathematical Models
│   │   ├── stochastic/              # SDE solvers, Monte Carlo
│   │   ├── options/                 # BSM, Greeks, exotics, vol surface
│   │   ├── statistics/              # Time-series tests, GARCH family
│   │   ├── distributions/           # Distribution fitting, EVT, copulas
│   │   ├── optimization/            # Numerical optimizers
│   │   └── bayesian/                # MCMC, Kalman filters
│   ├── signals/                     # MODULE 3 — Signal Generation
│   │   ├── technical/               # Price/volume indicators
│   │   ├── statistical/             # Stat-arb, pairs, factor signals
│   │   ├── ml/                      # ML signal pipeline
│   │   ├── options_flow/            # Options-specific signals
│   │   └── combiner.py              # Signal combination engine
│   ├── strategies/                  # MODULE 4 — Strategy Library
│   │   ├── base.py                  # Abstract strategy interface
│   │   ├── equity/                  # Equity strategies
│   │   ├── options/                 # Options strategies
│   │   ├── macro/                   # Multi-asset / macro
│   │   └── registry.py              # Strategy registry
│   ├── portfolio/                   # MODULE 8 — Portfolio Construction
│   │   ├── sizing.py                # Position sizing methods
│   │   ├── optimization.py          # MVO, BL, HRP, NCO
│   │   ├── rebalance.py             # Rebalancing engine
│   │   └── tax.py                   # Tax-aware optimization
│   ├── risk/                        # MODULE 5 — Risk Management
│   │   ├── var.py                   # VaR / CVaR engines
│   │   ├── stress.py                # Stress testing
│   │   ├── greeks.py                # Greeks aggregation
│   │   ├── limits.py                # Limit monitoring & breakers
│   │   ├── drawdown.py              # Drawdown management
│   │   ├── metrics.py               # Risk-adjusted metrics
│   │   └── reports.py               # Risk report generation
│   ├── execution/                   # MODULE 6 — Execution & OMS
│   │   ├── oms.py                   # Order management system
│   │   ├── algos/                   # TWAP, VWAP, IS, POV
│   │   ├── brokers/                 # Broker adapters (IB, Alpaca, CCXT)
│   │   ├── impact.py                # Market impact models
│   │   └── tca.py                   # Transaction cost analysis
│   ├── backtest/                    # MODULE 7 — Backtesting
│   │   ├── engine.py                # Event-driven simulator
│   │   ├── simulation.py            # Fill simulation, slippage
│   │   ├── analytics.py             # Performance analytics
│   │   ├── validation.py            # Bias detection
│   │   └── walk_forward.py          # Walk-forward / CPCV
│   ├── regime/                      # MODULE 9 — Regime Detection
│   │   ├── hmm.py                   # Hidden Markov Models
│   │   ├── changepoint.py           # Changepoint detection
│   │   ├── features.py              # Regime feature engineering
│   │   └── volatility_regime.py     # GARCH / VIX regimes
│   ├── alt_data/                    # MODULE 10 — Alternative Data
│   │   ├── sentiment/               # NLP / FinBERT pipeline
│   │   └── options_flow/            # GEX, UOA, flow intel
│   ├── live/                        # MODULE 11 — Live Trading
│   │   ├── trader.py                # Live trading loop
│   │   ├── reconciliation.py        # Position reconciliation
│   │   ├── watchdog.py              # Health / dead-man switch
│   │   └── paper.py                 # Paper trading shim
│   ├── dashboard/                   # MODULE 13 — Monitoring & Dashboard
│   │   ├── app.py                   # Plotly Dash app
│   │   ├── panels/                  # Dashboard panels
│   │   └── alerts.py                # Alert engine
│   ├── compliance/                  # MODULE 16 — Compliance & Audit
│   │   ├── audit_log.py             # Immutable audit trail
│   │   ├── rules.py                 # Pre-trade compliance rules
│   │   └── reporting.py             # Regulatory reporting
│   └── api/                         # FastAPI REST layer
│       ├── main.py
│       └── routes/
├── rust_core/                       # Rust crate for hot paths
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── orderbook.rs             # Order book reconstruction
│       ├── matching.rs              # Simulated matching engine
│       └── indicators.rs            # Vectorized indicator calc
├── tests/                           # Mirrors quantcore/ structure
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── notebooks/                       # Research notebooks
├── scripts/                         # CLI scripts
└── docs/                            # Sphinx / MkDocs
```

---

## 4. Cross-Cutting Concerns

### 4.1 Configuration
- All parameters in TOML files under `config/` — **zero hardcoded values**
- Use `dynaconf` for layered config (default → env-specific → env vars)
- Per-strategy config files in `config/strategies/`

### 4.2 Logging
- `structlog` with JSON output for production, pretty-print for dev
- Levels: `DEBUG` (backtest internals), `INFO` (live trading), `WARNING` (risk near-breaches), `ERROR` (risk breaches, order failures), `CRITICAL` (system failures)

### 4.3 Testing
- `pytest` with fixtures for market data, config, and mock brokers
- Every math function tested against known analytical solutions
- Property-based tests (`hypothesis`) for numerical stability
- Integration tests with Docker-composed services

### 4.4 Performance
- `numba` JIT for tight numerical loops
- Vectorized NumPy/Polars — **no Python loops on hot paths**
- `cProfile` + `line_profiler` for profiling
- Rust via PyO3 for order book, matching engine, indicator hot paths

### 4.5 Type Safety
- Full type hints on every function signature
- `pydantic` models for all data contracts
- `mypy --strict` in CI

---

## 5. Dependency Groups

```toml
[project.dependencies]
# Core
numpy = ">=1.26"
scipy = ">=1.12"
pandas = ">=2.2"
polars = ">=0.20"

# Econometrics & Stats
arch = ">=7.0"
statsmodels = ">=0.14"
hmmlearn = ">=0.3"
ruptures = ">=1.1"

# ML
scikit-learn = ">=1.4"
xgboost = ">=2.0"
lightgbm = ">=4.3"
torch = ">=2.2"
shap = ">=0.45"

# Optimization
cvxpy = ">=1.4"
pypfopt = ">=1.5"

# Backtesting
vectorbt = ">=0.26"

# Execution / Brokers
alpaca-trade-api = ">=3.0"
ccxt = ">=4.2"

# Infrastructure
fastapi = ">=0.110"
uvicorn = ">=0.29"
redis = ">=5.0"
psycopg = ">=3.1"
sqlalchemy = ">=2.0"
kafka-python = ">=2.0"

# Visualization
plotly = ">=5.20"
dash = ">=2.16"

# Config / Logging
dynaconf = ">=3.2"
structlog = ">=24.1"

# Performance
numba = ">=0.59"

# NLP
transformers = ">=4.39"

# Dev
pytest = ">=8.1"
mypy = ">=1.9"
ruff = ">=0.3"
hypothesis = ">=6.99"
```

---

## 6. Companion Task Breakdowns

The detailed, execution-ready task lists are in companion artifacts:

- [Phase 1 Tasks](file:///C:/Users/rochv/.gemini/antigravity/brain/820c1653-4b98-4599-a441-ed3019872ea9/phase1_tasks.md) — Data Infra, Math Models, Backtest Basics, Scaffold
- [Phase 2 Tasks](file:///C:/Users/rochv/.gemini/antigravity/brain/820c1653-4b98-4599-a441-ed3019872ea9/phase2_tasks.md) — Signals, Strategies, Options Pricing, ML
- [Phase 3 Tasks](file:///C:/Users/rochv/.gemini/antigravity/brain/820c1653-4b98-4599-a441-ed3019872ea9/phase3_tasks.md) — Risk, Execution, Portfolio, Regime
- [Phase 4 Tasks](file:///C:/Users/rochv/.gemini/antigravity/brain/820c1653-4b98-4599-a441-ed3019872ea9/phase4_tasks.md) — Live Trading, Dashboard, Alt Data, Compliance, Infra

---

## 7. Verification Plan

### Automated
- `pytest` suite with >90% coverage on math modules
- CI pipeline: lint → type-check → unit tests → integration tests
- Backtest reproducibility checks (deterministic seeded runs)
- Numerical accuracy tests against closed-form solutions (BSM, Greeks)

### Manual
- Paper trade each strategy for ≥2 weeks before live
- Risk dashboard visual inspection under stress scenarios
- Broker integration smoke tests on sandbox/paper accounts
- Performance profiling under simulated load (1M+ msgs/sec feed)
