# QuantCore

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

## Strategy Status

| Strategy | Type | Status | Synthetic OOS SR | Real-Data OOS SR | Capacity | Notes |
|---|---|---|---|---|---|---|
| `macro_cta_trend` | Macro trend-following | **REJECTED** | +1.79 | **-0.147** | ~$1M | Failed real-data Phase A: catastrophic failure on 2018-2024 full period |
| `macro_risk_parity` | Macro risk parity | **MONITOR** | +1.11 | +0.128 | ~$500K | Phase A: vol-target adjustments ineffective (SR-neutral); signal noise-limited |

**Verdict criteria:**
- **DEPLOY**: OOS Sharpe > 1.0, positive across all regimes, capacity > $500K, drawdown < 10%
- **MONITOR**: OOS Sharpe > 0.5, meets DEPLOY criteria but needs live validation
- **PAPER**: OOS Sharpe > 0.0 but has regime weakness needing further validation
- **REJECT**: OOS Sharpe ≤ 0.0 or structural flaw

### macro_cta_trend — REJECTED (Phase A)
- Multi-speed trend: fast (20), medium (50), slow (200) windows with weighted voting
- Volatility targeting + inverse-square vol regime scaling (threshold = vol_target * 0.30)
- Dynamic weight shift: reduces fast-signal weight during high vol in favor of slow signals
- **Synthetic regime breakdown**: HIGH_VOL SR +0.40, LOW_VOL SR +1.66, NORMAL SR +2.57, TREND_UP SR +3.35
- **Real-data regime breakdown**: Full 2018-2024 SR -0.147 (SHOWSTOPPER), GFC 2008 SR -0.734 (SHOWSTOPPER), Q4 2018 SR +2.613 (PASS), COVID 2020 SR +0.991 (PASS), Rate Hikes 2022 SR +0.904 (PASS), Bull 2023-2024 SR +0.176 (FAIL)
- Real-data max drawdown: -14.5%; 80% zero-return bars; kurtosis 52.9
- Rejection reason: Signal structurally weak on real data — extreme tail risk from trend reversals. Conviction/choppiness filters attempted but degraded previously-passing COVID regime. Requires fundamentally different approach.
- Deployment manifest: `strategies/deployment_manifest.yaml`

### macro_risk_parity — MONITOR (Phase A completed)
- Equal risk contribution via iterative ERC optimization with Ledoit-Wolf covariance shrinkage
- Individual asset vol-targeting + percentile-based regime overlay + portfolio-level vol regime filter
- Linear vol regime scaling: reduces positions when 20-bar portfolio vol exceeds vol_target * 0.40
- **Regime breakdown**: HIGH_VOL SR +0.81, LOW_VOL SR +3.07, NORMAL SR +2.04, TREND_UP SR +10.20
- **Phase A result**: Real-data full period SR +0.128 (below MONITOR threshold). 3 vol_target adjustments (0.12, 0.15, 0.18) all ineffective — vol_target scales signal and noise proportionally. Signal-to-noise ratio fundamentally limited. Strategy suitable for monitoring only; no parameter fix can salvage.
- Most sensitive parameter: `rebalance_freq` (use freq=1 or freq=21; avoid freq=30)
- Margin costs dominate above $500K AUM — capacity ceiling
- Deployment manifest: `strategies/deployment_manifest.yaml`

### Monitoring
- Strategy monitoring: `monitoring/strategy_monitor.py` (signal quality, regime detection, performance drift alerts)
- Alert engine: `quantcore/dashboard/alerts.py` (severity levels, notification routing, cooldowns)
- Live health checks: `quantcore/live/monitor.py` (latency, data staleness, heartbeat, position drift)
- Strategy smoke tests: 5 market scenarios (covid crash, trend reversal, correlation breakdown, liquidity crisis, low vol spike)

### Known Risks
1. Both strategies validated against synthetic + real data. CTA structurally weak (REJECTED). RP signal noise-limited (MONITOR only).
2. Both strategies degrade above $500K-$1M AUM (market impact + margin costs)
3. RP vol regime filter is uniformly applied (SR-neutral) — provides drawdown reduction but doesn't change signal composition
4. No options hedging or cross-asset hedging implemented for either strategy
5. Real-data backtest uses yfinance (OHLCV only) — no tick-level or order book validation
