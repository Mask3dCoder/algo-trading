# QuantCore

> **⚠️ DISCLAIMER — SYSTEM UNDER ACTIVE RECONSTRUCTION**
>
> A forensic audit conducted on 2026-05-09 found this system to have **zero
> deployable alpha**. The backtest engine contained look-ahead bias; the
> validation pipeline bypassed all realistic costs; Sharpe ratios were
> computed incorrectly; and purged cross-validation crashed at runtime.
>
> **No strategy has been validated for live trading.**
> All previously reported Sharpe ratios and performance figures are unreliable.
> See [`FORENSIC_AUDIT_REPORT.md`](FORENSIC_AUDIT_REPORT.md) for the full
> analysis and [`INTERNAL_NOTES.md`](INTERNAL_NOTES.md) for the recovery plan.
>
> Current rebuild status: **Phase 4 — Production Risk Controls** in progress.
>
> **⚠️ REAL MONEY AT RISK** — The `--live` flag routes orders to Alpaca's live
> broker. Only `--scale single_share` or `--scale fractional` are permitted
> with `--live`. See [`PHASE4_LIVE_DEPLOYMENT.md`](PHASE4_LIVE_DEPLOYMENT.md)
> for graduated deployment procedures and risk controls.

---

Institutional-grade algorithmic trading system — historical backtesting, live trading, risk management, and real-time monitoring.

## Quick Start

```bash
pip install -e ".[dev,stats,data]"    # minimal install
make test                              # 734 tests
python main.py backtest --data data.csv
python main.py live --broker paper
python main.py api --port 8000
python main.py report --input results.csv
python main.py risk --input returns.csv
python run_paper_trading.py --check                      # verify Alpaca connectivity
python run_paper_trading.py                               # paper trading (default)
python run_paper_trading.py --live --scale single_share   # LIVE: single share only
```

## Architecture

```
quantcore/
├── core/               Config, types (Pydantic), logging (structlog), event bus
├── data/               Sources, feeds (WebSocket), storage (TimescaleDB/Redis), quality
├── math_models/        Stochastic (GBM/Heston/SABR), options pricing (BSM/IV/exotics),
│                       statistics (GARCH/cointegration/factors), Bayesian (KF/MCMC)
├── signals/            Technical (trend/mr/vol/volume), statistical (pairs/basket/factors),
│                       ML (XGB/LGB/CatBoost, LSTM/Transformer, RL), options flow
├── strategies/         Equity (momentum/stat-arb/mr/trend), options (delta-neutral/IC/earnings),
│                       macro (risk-parity/CTA/carry)
├── backtest/           Event-driven engine, walk-forward, analytics, HTML reports
├── risk/               VaR (historical/parametric/MC/CVaR), stress tests, drawdown, limits
├── execution/          OMS, TWAP/VWAP/POV/IS algos, TCA, broker adapters (Alpaca/IBKR/CCXT)
├── portfolio/          Sizing (Kelly/vol-target), optimization (MVO/BL/HRP), rebalancing
├── regime/             HMM, changepoint detection, volatility regimes
├── alt_data/           SEC EDGAR sentiment (LM dictionary), social media, options flow (GEX/UOA)
├── dashboard/          Dash app: portfolio, strategy, risk, market, execution panels + alert engine
├── compliance/         Hash-chained immutable audit trail, pre-trade rules, reporting
├── api/                FastAPI REST + WebSocket, API key auth
├── live/               Paper/live trader, risk manager, circuit breaker, kill switch, reconciliation
├── research/           Deflated Sharpe, FDR correction, experiment tracking
└── main.py             CLI entry point
```

## Performance

| Component | Throughput | Notes |
|---|---|---|
| Signal generation | 1.47M pts/sec | Numba JIT + pandas vectorized |
| Feed handler | 251K msgs/sec | WebSocket event processing |
| OMS | 60K orders/sec | Order lifecycle state machine |
| Backtest engine | 5.6K bars/sec | Full event loop with fills |
| Kill switch | <1ms | Emergency flatten latency |
| Alert evaluation | <0.1ms | Rule condition evaluation |
| Dashboard load | ~680ms | Dash app creation + first render |

All hot-path loops compiled with Numba JIT; fallback to pure Python when Numba is unavailable. HTTP clients use connection pooling (httpx.AsyncClient, requests.Session).

## Strategy Status (Post-Audit Validation — Phase 3)

| Strategy | Status | OOS Sharpe | CI_lower (90%) | Walk-Forward | Notes |
|---|---|---|---|---|---|
| `macro_cross_sectional_momentum` | **DEPLOY** | +0.85 | +0.33 | 10-yr expanding | Phase 3 live ready |
| `macro_state_selective_momentum` | **DEPLOY** | +0.81 | +0.40 | 10-yr expanding | Phase 3 live ready |
| `macro_factor_residual` | **PAPER** | +0.911 | Unreported | 6/6 regimes | Side-effect dominated, residual thin |
| `macro_risk_parity` | **MONITOR** | +0.128 | — | — | Noise-limited; pure beta |
| `macro_cta_trend` | **DISABLED** | -0.147 | — | — | Structural negative alpha |
| `equity_momentum` | **PAPER** | Unvalidated | — | — | 12-1 mo edge decayed post-2000 |

Both DEPLOY strategies passed the 10-year walk-forward with CI_lower(90%) ≥ 0.3.
See [`PHASE3_LAUNCH.md`](PHASE3_LAUNCH.md) for live paper trading setup.

## Production Status

**30-day paper trading validation started: 2026-05-27**

| Strategy | Registry Name | Data Source | Weight | Status |
|---|---|---|---|---|
| `macro_cross_sectional_momentum` | `macro_cross_sectional_momentum` | Alpaca daily | 0.35 | RUNNING |
| `macro_state_selective_momentum` | `macro_state_selective_momentum` | Alpaca daily | 0.35 | RUNNING |
| `pairs_trading_intraday` | `equity_stat_arb` | Alpaca intraday (5min) | 0.30 | RUNNING |

- **Health API**: `http://localhost:8000/health` — liveness probe
- **Status API**: `http://localhost:8000/status` — positions, equity, events
- **Kill switch**: `curl -X POST http://localhost:8000/kill-switch/activate`
- **Daily checklist**: [`operations/daily_checklist.md`](operations/daily_checklist.md)
- **Graduation criteria**: Paper Sharpe >= 0.3, max DD < 10%, >= 20/30 days without crash

## Phase Gate Status

| Gate | Criterion | Status |
|---|---|---|
| Phase 1 — Core Infrastructure | 734 tests passing | **PASS** |
| Phase A OOS Validation | CI_lower(90%) ≥ 0.3, 10-yr walk-forward | **PASS** (2 strategies) |
| Phase B Code Quality | Same `on_data()` pipeline, no separate live signals | **PASS** |
| Phase C Parity Test | LiveDataFeed vs backtest data → identical signals | `pytest tests/test_live_parity.py` |
| Phase D Live Bootstrap | Paper trading ≥ 20 trading days without crash | **IN PROGRESS** (started 2026-05-27) |
| Phase E Live Sign-off | OOS paper Sharpe ≥ 0.3 over live period | **PENDING** |
| **Phase 4 — Risk Controls** | Pre-trade checks, circuit breaker, kill switch, graduated scale | **PASS** (tests green) |

For Phase 3 live start, see [`PHASE3_LAUNCH.md`](PHASE3_LAUNCH.md).
For Phase 4 risk controls and graduated live deployment, see [`PHASE4_LIVE_DEPLOYMENT.md`](PHASE4_LIVE_DEPLOYMENT.md).

## Verification

```bash
# Run the honest foundation verification script
python scripts/verify_honest_foundation.py
# Outputs: HONEST_FOUNDATION_REPORT.json
```

## Development

```bash
make install        # pip install -e ".[dev,stats,data]"
make test           # pytest with 734 tests
make test-cov       # with coverage
make lint           # ruff check
make format         # ruff check --fix + format
make typecheck      # mypy strict mode
make docker-up      # TimescaleDB, Redis, Kafka
```

Python ≥3.12 required. Optional dependency groups: `stats`, `ml`, `optim`, `exec`, `infra`, `viz`, `nlp`, `perf`, `data`, `dev` — install with `pip install -e ".[group]"` or `pip install -e ".[all]"`.

## Production Deployment (Docker)

The system runs as a containerized service with zero terminal commands during normal operation.

### Quick Start

```bash
# 1. Configure (edit config/live.yaml with your broker credentials and strategies)
cp config/live.yaml config/live.yaml.local   # optional override

# 2. Start
docker compose up -d

# 3. Verify
curl http://localhost:8000/health    # {"status": "ok", "uptime_seconds": ...}
curl http://localhost:8000/status    # positions, equity, recent events

# 4. Stop trading (kill switch)
curl -X POST http://localhost:8000/kill-switch/activate

# 5. Resume trading
curl -X POST http://localhost:8000/kill-switch/deactivate

# 6. View logs
docker compose logs -f quantcore
```

### File-Based Kill Switch

Create a `KILL_SWITCH` file at the project root to halt trading immediately (checked every 10 seconds):

```bash
touch KILL_SWITCH    # halt trading
rm KILL_SWITCH       # resume trading
```

### Configuration

All settings live in `config/live.yaml`:

| Section | Key Settings |
|---|---|
| `broker` | `type` (paper/alpaca/ibkr), credentials, base URL |
| `strategies` | List of `{name, enabled, weight, params}` — names must match `@register()` |
| `universe` | `tickers` and `macro_assets` lists |
| `risk` | Position limits, daily loss, drawdown, circuit breaker thresholds |
| `execution` | `scale_mode` (single_share/fractional/full), initial cash |
| `schedule` | `trading_time`, `reconciliation_time` (ET), `timezone` |
| `alerting` | File log, Slack webhook, email SMTP |
| `kill_switch` | `file_path`, `check_interval_sec` |
| `api` | `host`, `port`, `enabled` |

Environment variables override YAML values with `QUANTCORE_` prefix (e.g. `QUANTCORE_BROKER__ALPACA__API_KEY`).

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe — uptime, memory, data freshness |
| `/status` | GET | Full trading status — positions, equity, events, circuit breaker |
| `/kill-switch/activate` | POST | Emergency halt — stops all trading |
| `/kill-switch/deactivate` | POST | Resume trading after operator review |
| `/docs` | GET | OpenAPI interactive documentation |

### Docker Compose Volumes

| Mount | Purpose |
|---|---|
| `./config:/app/config:ro` | Strategy and broker configuration |
| `./reports:/app/reports` | Equity CSV, event log, alert log |
| `./data:/app/data` | Cached market data |
| `./logs:/app/logs` | Application logs |
| `./KILL_SWITCH:/app/KILL_SWITCH:ro` | File-based kill switch |

### Monitoring

- **Health check**: Docker HEALTHCHECK polls `/health` every 60s; auto-restarts on failure
- **Alerts**: Configured in `config/live.yaml` under `alerting` — file log always on, Slack/email optional
- **Equity curve**: `reports/live_equity.csv` — date, cash, positions value, total equity, daily return
- **Event log**: `reports/live_events.jsonl` — every order submission as newline-delimited JSON

## Key Decisions

- **T-0147 `rust_core/` — Deferred.** Python + Numba exceeds all performance targets: signals at 1.47M pts/sec and feed handler at 251K msgs/sec leave no latency gap that justifies maintaining a Rust/PyO3 binding layer today. The `rust_core/` directory skeleton remains as a future extension point if sub-millisecond order book matching latency is required.

- **Numba over Cython/C extensions.** JIT decorators on pure NumPy functions give 30x speedup with zero build complexity. The `try/except ImportError` fallback pattern preserves portability — the system runs everywhere Python does, with or without an LLVM toolchain.

- **Connection pooling everywhere.** All HTTP data sources share httpx.AsyncClient sessions; synchronous callers (SEC EDGAR, social media) use requests.Session. This eliminates the dominant latency source in production data pipelines.

- **TimescaleDB + Redis hybrid storage.** Hypertables with continuous aggregates for historical data; Redis with msgpack serialization for real-time state, pub/sub, and caching.
