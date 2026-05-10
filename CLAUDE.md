# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantCore — Institutional-grade algorithmic trading system with event-driven backtesting, live trading, risk management, and real-time monitoring. Python ≥3.12 required.

## Build & Development Commands

```bash
# Install (minimal with dev, stats, and data dependencies)
pip install -e ".[dev,stats,data]"
# Or use make
make install

# Optional dependency groups: stats, ml, optim, exec, infra, viz, nlp, perf, data, dev
# Install specific group: pip install -e ".[ml]"
# Install all groups: pip install -e ".[all]"

# Run all tests (734 tests)
make test
# or directly:
python -m pytest tests/ -v --tb=short

# Run a single test file
python -m pytest tests/unit/backtest/test_engine.py -v

# Run a specific test
python -m pytest tests/unit/backtest/test_engine.py::test_backtest_basic -v

# Skip slow/integration tests
python -m pytest tests/ -v -m "not slow and not integration"

# Lint
make lint
# or: python -m ruff check quantcore/ tests/

# Format code
make format
# or: python -m ruff check --fix quantcore/ tests/ && python -m ruff format quantcore/ tests/

# Type checking (mypy strict mode)
make typecheck
# or: python -m mypy quantcore/

# Test with coverage
make test-cov

# Docker services (TimescaleDB, Redis, Kafka)
make docker-up
make docker-down

# CLI Commands (main.py)
python main.py backtest --strategy momentum --data data.csv
python main.py live --broker paper
python main.py api --port 8000
python main.py report --input results.csv
python main.py risk --input returns.csv

# Real data backtest (Phase A validation)
python scripts/real_data_backtest.py
python scripts/real_data_backtest.py --strategy macro_factor_residual

# Paper trading
python scripts/paper_trade.py

# Strategy diagnostics and validation scripts
python scripts/strategy_audit.py
python scripts/diagnostic_summary.py
```

## Architecture

Event-driven architecture with these core modules:

```
quantcore/
├── core/          Pydantic types (OHLCV, Order, Fill), events, config, logging (structlog)
├── data/          Data sources, feeds (WebSocket), quality checks, universe definition
├── strategies/     Base class with lifecycle: initialize → on_data → on_fill → teardown
│   ├── equity/    Momentum, stat-arb, mean-reversion, trend
│   ├── options/   Delta-neutral, iron condor, earnings plays
│   └── macro/     CTA trend, risk parity, carry, factor residual
├── backtest/      Event loop: MarketEvent → SignalEvent → OrderEvent → FillEvent
├── risk/          VaR (historical/parametric/MC), CVaR, drawdown, limits, attribution
├── execution/     OMS, TWAP/VWAP/POV algos, TCA, broker adapters (Alpaca/IBKR/CCXT)
├── portfolio/     Position sizing (Kelly/vol-target), optimization (MVO/BL/HRP), rebalancing
├── live/          Live runner, watchdog, kill switch (<1ms), reconciliation
├── math_models/   Stochastic (GBM/Heston/SABR), options pricing, statistics, Bayesian
├── signals/       Technical, statistical (pairs/factors), ML (XGB/LSTM/RL), options flow
├── regime/        HMM, changepoint detection, volatility regimes
├── api/           FastAPI REST + WebSocket (uvicorn)
├── dashboard/     Dash app with portfolio, risk, execution panels
├── compliance/    Hash-chained immutable audit trail, pre-trade rules
└── alt_data/      SEC EDGAR sentiment, social media, options flow (GEX/UOA)

monitoring/        Strategy monitoring, signal quality, regime detection, performance drift
strategies/        Deployment manifests (deployment_manifest.yaml)
scripts/           Validation, diagnostics, paper trading, and data loading utilities
```

### Key Patterns

**Strategy Interface** (`quantcore/strategies/base.py`):
- Inherits `Strategy` ABC with `initialize()`, `on_data()`, `on_fill()`, `teardown()`
- Emits `TradeSignal` objects (symbol, direction -1/0/+1, strength, size)
- Configured via `StrategyConfig` dataclass
- Register strategies with `@register` decorator for discovery via `quantcore.strategies.registry`

**Backtest Engine** (`quantcore/backtest/engine.py`):
- Event-driven: `Event` objects with `MarketData`, `SignalData`, `OrderData` wrapped
- Heap-ordered event queue by timestamp
- Integrates risk limits, position sizing, regime filtering, drawdown tracking

**Walk-Forward Validation** (`quantcore/backtest/walk_forward.py`):
- `anchored_walk_forward()` for out-of-sample testing
- `Fold` dataclass with train/validation/test splits

**Types** (`quantcore/core/types.py`):
- Frozen Pydantic models: `OHLCV`, `Tick`, `Quote`, `Order`, `Fill`, `Position`
- Enums: `Side`, `OrderType`, `OrderStatus`, `Direction`

## Testing

- **Unit tests**: `tests/unit/` — mirrors `quantcore/` structure
- **Integration tests**: `tests/integration/` — requires Docker services
- **Adversarial tests**: `tests/adversarial/` — chaos testing, broker emulation
- **pytest markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.gpu`, `@pytest.mark.optional_dep`
- **Fixtures**: `tests/conftest.py`, `tests/fixtures/`

## Strategy Deployment Workflow

Strategies go through Phase A (real-data validation) before deployment:

| Status | Meaning | Criteria |
|--------|---------|----------|
| DEPLOY | Ready for live trading | OOS Sharpe > 1.0, positive across all regimes, capacity > $500K, DD < 10% |
| MONITOR | Live validation needed | OOS Sharpe > 0.5 |
| PAPER | Paper trading | OOS Sharpe > 0.0, Phase A passed |
| REJECT | Failed validation | OOS Sharpe ≤ 0.0 or structural flaw |

**Current Strategy Status** (from README):
- `macro_factor_residual` — **PAPER** (OOS SR +0.911, 6/6 regimes PASS, max DD -3.5%)
- `macro_risk_parity` — **MONITOR** (OOS SR +0.128, noise-limited)
- `macro_cta_trend` — **REJECTED** (OOS SR -0.147, structural failure on real data)

**Phase A Validation**:
```bash
python scripts/real_data_backtest.py --strategy macro_factor_residual
```
Evaluates regime-specific performance (GFC, COVID, Rate Hikes, Bull, etc.) using yfinance data.

## Performance Notes

- Hot paths use **Numba JIT** (`@njit`) with pure Python fallback via `try/except ImportError`
- Signal generation: 1.47M pts/sec, Feed handler: 251K msgs/sec
- Kill switch latency: <1ms
- HTTP clients use connection pooling (httpx.AsyncClient, requests.Session)
- Rust core (`rust_core/`) deferred — Python + Numba meets all performance targets

## Key Decisions

- **Rust core deferred.** Python + Numba exceeds all performance targets (signals at 1.47M pts/sec, feed handler at 251K msgs/sec). The `rust_core/` directory skeleton remains as a future extension point if sub-millisecond order book matching latency is required.

- **Numba over Cython/C extensions.** JIT decorators on pure NumPy functions give 30x speedup with zero build complexity. The `try/except ImportError` fallback pattern preserves portability — the system runs everywhere Python does, with or without an LLVM toolchain.

- **Connection pooling everywhere.** All HTTP data sources share httpx.AsyncClient sessions; synchronous callers use requests.Session. This eliminates the dominant latency source in production data pipelines.

- **TimescaleDB + Redis hybrid storage.** Hypertables with continuous aggregates for historical data; Redis with msgpack serialization for real-time state, pub/sub, and caching.

## Phase Gate Status

| Gate | Status |
|---|---|
| Phase 1 — Core Infrastructure | 734 tests, all passing |
| Phase 2 — Signals & Strategies | Technical/matching validated |
| Phase 3 — Risk, Execution & Portfolio | VaR backtests green |
| Phase 4 — Live Ops & Production | All 5 gates pass |

Phase 4 gate results:
- Kill switch <1s: **0.0ms**
- Dashboard load <3s: **683ms**
- Alert fire <5s: **0.01ms**
- Audit trail immutable: **SHA-256 chain verified**
- Feed handler >100K msgs/sec: **251K**

## Code Style

- **Ruff** for linting/formatting (target Python 3.12, line-length 100)
- **mypy** strict mode for type checking
- **Pydantic v2** for data models (frozen models for value objects)
- Math variables may use single-letter names (l, I, O) — ruff E741 ignored
- Lambda assignments allowed in strategy code — ruff E731 ignored

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push/PR to main
- Python 3.12, installs `[dev,stats,viz]` dependencies
- Lint with ruff (warnings only, never fails build)
- Unit tests (excluding integration) with coverage reporting
- Test results and coverage uploaded as artifacts

## Known Risks

1. **Factor residual is the only strategy to pass Phase A** (SR +0.911, 6/6 regimes PASS). Risk parity (SR +0.128) avoids rejection but is noise-limited to MONITOR. All other 6 strategies REJECTED.

2. **Factor residual's PnL is dominated by side-effect factor exposures** (84.5% of absolute PnL). The residual signal quality is extreme (Sharpe +11.98) but small in magnitude. Improved factor-neutral construction could raise total Sharpe toward DEPLOY territory.

3. **2022 Rate Hikes is the weakest regime** (SR +0.232). While not a showstopper, the strategy's residual signal struggled during the rapid hiking cycle.

4. **Factor-component assets (SPY, DBC, GLD, TLT) have near-zero residual variance** since they are in the factor definitions. Their residual autocorrelation is structurally near 1.0 (not a model flaw). Only IEF carries meaningful residual variation for signal extraction.

5. **Cross-sectional ranking is statistically significant but economically too weak** for this universe. Score vs forward return r = 0.055 (p < 0.00001). With 6 macro assets, cross-sectional degrees of freedom are insufficient for a diversified market-neutral signal.

6. **Both CTA and RP strategies degrade above $500K-$1M AUM** (market impact + margin costs). Factor residual capacity is estimated at ~$2M due to long-short structure.

7. **Real-data backtest uses yfinance (OHLCV only)** — no tick-level or order book validation.

## Graphify Integration

This project has a graphify knowledge graph at graphify-out/. A PreToolUse hook reminds Claude to check the graph before searching raw files.

- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- For cross-module "how does X relate to Y" questions, prefer `graphify query "<question>"`, `graphify path "<A>" "<B>"`, or `graphify explain "<concept>"` over grep — these traverse the graph's EXTRACTED + INFERRED edges instead of scanning files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)
