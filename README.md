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
> Current rebuild status: **Q1 FIX FOUNDATION** in progress.

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
├── live/               Paper/live trader, watchdog (dead man's switch), kill switch, reconciliation
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

## Strategy Status (Honest — Post-Audit 2026-05-09)

| Strategy | Status | OOS Sharpe (Audit) | Reason |
|---|---|---|---|
| `macro_cta_trend` | **DISABLED** | -0.147 | Structural negative alpha post-2008 |
| `macro_risk_parity` | **DISABLED** | +0.128 | Pure beta, not alpha |
| `macro_factor_residual` | **DISABLED** | 0.000 | Broken — degenerates to 1 asset |
| `equity_momentum` | **DISABLED** | Unvalidated | 12-1 mo edge decayed post-2000 |
| `equity_earnings_momentum` | **DISABLED** | Unvalidated | Circular logic |
| `macro_cross_sectional_momentum_v2` | **DEVELOPMENT** | +0.55 (but 88% DD) | Real signal; leverage cap fix applied |
| `equity/low_volatility` | **RESEARCH** | Unvalidated | Tier-1 alpha candidate (Q2) |
| `options/delta_neutral` | **RESEARCH** | Unvalidated | Tier-1 alpha candidate (Q2) |

No strategy has OOS Sharpe 90% CI lower bound ≥ 0.3 with the corrected engine.
Re-validation in progress.

## Verification

```bash
# Run the honest foundation verification script
python scripts/verify_honest_foundation.py
# Outputs: HONEST_FOUNDATION_REPORT.json
```

## Phase Gate Status (Historical — Pre-Audit)

> **These results are INVALID.** They were produced with a look-ahead-biased
> backtest engine and a validation pipeline that bypassed all execution costs.

| Gate | Old Status | Post-Audit Status |
|---|---|---|
| Phase 1 — Core Infrastructure | 734 tests passing | Infrastructure sound; metrics buggy |
| Phase 2 — Signals & Strategies | "Validated" | NOT validated (all use broken validator) |
| Phase 3 — Risk, Execution & Portfolio | "VaR backtests green" | Risk engine decorative |
| Phase 4 — Live Ops & Production | "All 5 gates pass" | Paper trading should be halted |

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

## Key Decisions

- **T-0147 `rust_core/` — Deferred.** Python + Numba exceeds all performance targets: signals at 1.47M pts/sec and feed handler at 251K msgs/sec leave no latency gap that justifies maintaining a Rust/PyO3 binding layer today. The `rust_core/` directory skeleton remains as a future extension point if sub-millisecond order book matching latency is required.

- **Numba over Cython/C extensions.** JIT decorators on pure NumPy functions give 30x speedup with zero build complexity. The `try/except ImportError` fallback pattern preserves portability — the system runs everywhere Python does, with or without an LLVM toolchain.

- **Connection pooling everywhere.** All HTTP data sources share httpx.AsyncClient sessions; synchronous callers (SEC EDGAR, social media) use requests.Session. This eliminates the dominant latency source in production data pipelines.

- **TimescaleDB + Redis hybrid storage.** Hypertables with continuous aggregates for historical data; Redis with msgpack serialization for real-time state, pub/sub, and caching.
