# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QuantCore — Institutional-grade algorithmic trading system with event-driven backtesting, live trading, risk management, and real-time monitoring. Python ≥3.12 required.

**⚠️ System under active reconstruction.** A forensic audit found zero deployable alpha. Current rebuild status: Phase 4 — Production Risk Controls. No strategy has been validated for live trading without going through the phase gate process.

**⚠️ REAL MONEY AT RISK** — The `--live` flag routes orders to Alpaca's live broker. Only `--scale single_share` or `--scale fractional` are permitted with `--live`.

## Build & Development Commands

```bash
# Install (minimal with dev, stats, and data dependencies)
pip install -e ".[dev,stats,data]"
# or: make install

# Optional dependency groups: stats, ml, optim, exec, infra, viz, nlp, perf, data, dev
# Install specific group: pip install -e ".[ml]"
# Install all groups: pip install -e ".[all]"

# Run all tests (734+ tests)
make test
# or: python -m pytest tests/ -v --tb=short

# Run a single test file
python -m pytest tests/unit/backtest/test_engine.py -v

# Run a specific test
python -m pytest tests/unit/backtest/test_engine.py::test_backtest_basic -v

# Skip slow/integration tests
python -m pytest tests/ -v -m "not slow and not integration"

# Run only integration tests (requires Docker services)
make test-integration
# or: python -m pytest tests/ -v -m integration

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

# Paper trading
python run_paper_trading.py --check                      # verify Alpaca connectivity
python run_paper_trading.py                               # paper trading (default)
python run_paper_trading.py --live --scale single_share   # LIVE: single share only

# Verification
python scripts/verify_honest_foundation.py   # outputs HONEST_FOUNDATION_REPORT.json
```

## Architecture

Event-driven architecture with these core modules:

```
quantcore/
├── core/               Pydantic types (OHLCV, Order, Fill), EventBus (async pub/sub),
│                        config (dynaconf), logging (structlog), economic_kernel
├── data/               Sources, feeds (WebSocket), quality checks, universe, vol_surface,
│                        yfinance/FRED loaders, options loader, intraday loader
├── math_models/        Stochastic (GBM/Heston/SABR/OU/Jump), options pricing (BSM/IV/
│                        exotics/vol_surface), statistics (GARCH/cointegration/factors),
│                        Bayesian (KF/MCMC), distributions
├── signals/            Technical (trend/mr/vol/volume), statistical (pairs/basket/factors/
│                        kalman_hedge), ML (XGB/LGB/CatBoost, LSTM/Transformer, RL),
│                        options_flow (GEX/UOA/volatility)
├── strategies/         Strategy ABC with lifecycle: initialize → on_data → on_fill → teardown
│   ├── equity/         Momentum, stat-arb, mean-reversion, trend, value, quality, etc.
│   ├── options/        Delta-neutral, iron condor, earnings, put-write, tail-hedge
│   └── macro/          CTA trend, risk parity, carry, factor residual, regime-adaptive
├── alpha/              Alpha signal generators (delta_neutral, iron_condor, pairs_finder,
│                        residual_momentum, stat_arb_mean_reversion, fred_signals)
├── backtest/           Event-driven engine (MarketEvent→SignalEvent→OrderEvent→FillEvent),
│                        walk-forward, analytics, HTML reports, market impact, financing
├── risk/               VaR (historical/parametric/MC/CVaR), stress tests, drawdown,
│                        limits, circuit breaker, greeks, intraday risk, PSR
├── execution/          OMS, TWAP/VWAP/POV/IS/iceberg algos, TCA, smart router,
│                        broker adapters (Alpaca/IBKR/CCXT/paper)
├── portfolio/          Sizing (Kelly/vol-target), optimization (MVO/BL/HRP),
│                        aggregator, rebalancing, beta-neutral, tax-aware
├── regime/             HMM, changepoint detection, volatility regimes, regime filter
├── alt_data/           SEC EDGAR sentiment (LM dictionary), options flow (GEX/UOA)
├── dashboard/          Dash app: portfolio, strategy, risk, execution panels + alert engine
├── compliance/         Hash-chained immutable audit trail, pre-trade rules, reporting
├── api/                FastAPI REST + WebSocket, API key auth
├── live/               Paper/live trader, risk manager, circuit breaker, kill switch,
│                        reconciliation, watchdog, orchestrator, production runner
└── research/           Deflated Sharpe, FDR correction, experiment tracking, bootstrap
```

### Key Patterns

**Strategy Interface** (`quantcore/strategies/base.py`):
- Inherits `Strategy` ABC with `initialize()`, `on_data()`, `on_fill()`, `teardown()`
- `on_data()` receives `dict[str, NDArray]` and returns `list[TradeSignal]`
- `TradeSignal`: symbol, direction (-1/0/+1), strength [0,1], size (notional), stop_loss, take_profit
- Register strategies with `@register("name")` decorator for discovery via `quantcore.strategies.registry`
- Strategy lifecycle states: `UNINITIALIZED → INITIALIZED → RUNNING → STOPPED/ERROR`
- Strategy status enum: `DISABLED`, `RESEARCH`, `DEVELOPMENT`, `PAPER_TRADING`, `LIVE`

**Backtest Engine** (`quantcore/backtest/engine.py`):
- Event-driven: `Event` objects with `EventType.MARKET/SIGNAL/ORDER/FILL`
- Heap-ordered event queue by timestamp
- Integrates risk limits, position sizing, drawdown tracking, market impact, financing costs

**Event Bus** (`quantcore/core/events.py`):
- Async pub/sub: `EventBus` with sync/async handlers invoked concurrently via `asyncio.gather`
- Event types: `TICK`, `BAR`, `SIGNAL`, `ORDER`, `FILL`, `RISK_ALERT`, `REGIME_CHANGE`

**Core Types** (`quantcore/core/types.py`):
- Frozen Pydantic models: `OHLCV`, `Tick`, `Quote`, `Order`, `Fill`, `Position`
- Enums: `Side`, `OrderType`, `OrderStatus`, `Direction`
- `OHLCV` has model validators for bar integrity (high ≥ low, etc.)

**Walk-Forward Validation** (`quantcore/backtest/walk_forward.py`):
- `anchored_walk_forward()` for out-of-sample testing
- `Fold` dataclass with train/validation/test splits

**Live Runner** (`quantcore/live/runner.py`):
- Flow: MarketData → Regime → Strategy → Sizing → RiskCheck → OMS → Execution → Monitor
- Kill switch: API endpoint (`/kill-switch/activate`) or file-based (`KILL_SWITCH` file)
- Circuit breaker: daily loss and drawdown limits per strategy

**Configuration**:
- `config/live.yaml` — multi-strategy live config (broker, strategies, risk, execution, schedule)
- `config/default.toml` / `config/development.toml` — dynaconf settings
- Environment variables override YAML with `QUANTCORE_` prefix
- API keys must come from environment variables, never hardcoded

## Testing

- **Unit tests**: `tests/unit/` — mirrors `quantcore/` structure
- **Integration tests**: `tests/integration/` — requires Docker services
- **Adversarial tests**: `tests/adversarial/` — chaos testing, broker emulation
- **Alpha tests**: `tests/alpha/` — alpha signal tests
- **Top-level tests**: `test_backtest_no_lookahead.py`, `test_live_parity.py`, `test_live_risk.py`
- **pytest markers**: `@pytest.mark.slow`, `@pytest.mark.integration`, `@pytest.mark.gpu`, `@pytest.mark.optional_dep`
- **asyncio mode**: `auto` (no need for `@pytest.mark.asyncio`)
- **Fixtures**: `tests/conftest.py`, `tests/fixtures/`
- **Deselect markers**: `-m "not slow and not integration"`

## Strategy Deployment Workflow

Strategies go through phase gates before deployment. Status is tracked in `StrategyStatus` enum.

| Status | Meaning | Criteria |
|--------|---------|----------|
| `LIVE` | Full compliance sign-off | OOS paper Sharpe ≥ 0.3 over live period |
| `PAPER_TRADING` | OOS validated, paper only | OOS Sharpe CI_lower(90%) ≥ 0.3, walk-forward passed |
| `DEVELOPMENT` | Walk-forward in progress | OOS Sharpe CI < 0.3 |
| `RESEARCH` | Unit-tested only | Not yet walk-forward validated |
| `DISABLED` | No deployable edge | Failed validation or structural flaw |

Strategies are registered with `@register("name")` and configured in `config/live.yaml` under `strategies:`. The registry (`quantcore.strategies.registry`) provides `get()`, `list_strategies()`, `create()`, and `clear()`.

**Production Status (2026-05-29)**: 30-day paper trading validation in progress. See `config/live.yaml` for active strategy weights and risk-parity allocations.

## Performance Notes

- Hot paths use **Numba JIT** (`@njit`) with pure Python fallback via `try/except ImportError`
- Signal generation: 1.47M pts/sec, Feed handler: 251K msgs/sec, Kill switch: <1ms
- HTTP clients use connection pooling (httpx.AsyncClient, requests.Session)
- TimescaleDB + Redis hybrid storage for historical vs. real-time data
- Rust core (`rust_core/`) deferred — Python + Numba meets all performance targets

## Code Style

- **Ruff** for linting/formatting (target Python 3.12, line-length 100)
- **mypy** strict mode for type checking
- **Pydantic v2** for data models (frozen models for value objects)
- Math variables may use single-letter names (l, I, O) — ruff E741 ignored
- Lambda assignments allowed in strategy code — ruff E731 ignored
- Blanket `type: ignore` and `noqa` are allowed — ruff PGH003/PGH004 ignored

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`):
- Runs on push/PR to main
- Python 3.12, installs `[dev,stats,viz]` dependencies
- Lint with ruff (warnings only, never fails build)
- Unit tests (excluding integration) with coverage reporting

## Key Decisions

- **Numba over Cython/C extensions.** JIT decorators on pure NumPy functions give 30x speedup with zero build complexity. The `try/except ImportError` fallback pattern preserves portability.

- **Connection pooling everywhere.** All HTTP data sources share httpx.AsyncClient sessions; synchronous callers use requests.Session.

- **TimescaleDB + Redis hybrid storage.** Hypertables with continuous aggregates for historical data; Redis with msgpack serialization for real-time state, pub/sub, and caching.

- **Rust core deferred.** Python + Numba exceeds all performance targets. The `rust_core/` directory skeleton remains as a future extension point.

## Scripts Directory

`scripts/` contains validation, diagnostics, paper trading, and data loading utilities. Key scripts:
- `scripts/real_data_backtest.py` — Phase A real-data validation with regime-specific performance
- `scripts/run_paper_trading.py` — Paper/live trading launcher
- `scripts/verify_honest_foundation.py` — Foundation verification
- `scripts/validate_all_strategies.py` — Batch strategy validation
- `scripts/strategy_audit.py` — Strategy diagnostics
- `scripts/deploy_improvements.py` — Deployment improvement checks
- `scripts/pre_deploy_check.py` — Pre-deployment validation
