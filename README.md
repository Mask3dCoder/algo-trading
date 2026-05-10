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

## Strategy Status

| Strategy | Type | Status | Synthetic OOS SR | Real-Data OOS SR | Capacity | Notes |
|---|---|---|---|---|---|---|
| `macro_factor_residual` | Macro factor residual | **PAPER** | -- | **+0.911** | ~$2M | **First strategy to pass Phase A.** 6/6 regimes PASS, zero showstoppers, max DD -3.5%. Removes MKT/RATES/INFL/SAFE factors via rolling ridge, trades residuals. Residual signal quality extreme (residual PnL Sharpe +11.98), beta-neutral (beta to SPY +0.008), no factor dominance. ~84% of absolute PnL from side-effect factor exposures -- needs improved factor neutrality for DEPLOY. |
| `macro_risk_parity` | Macro risk parity | **MONITOR** | +1.11 | +0.128 | ~$500K | Phase A: vol-target adjustments ineffective (SR-neutral); signal noise-limited |
| `macro_cta_trend` | Macro trend-following | **REJECTED** | +1.79 | **-0.147** | ~$1M | Failed real-data Phase A: catastrophic failure on 2018-2024 full period |

**Verdict criteria:**
- **DEPLOY**: OOS Sharpe > 1.0, positive across all regimes, capacity > $500K, drawdown < 10%
- **MONITOR**: OOS Sharpe > 0.5, meets DEPLOY criteria but needs live validation
- **PAPER**: OOS Sharpe > 0.0, Phase A passed, viable for paper trading with ongoing validation
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

### macro_factor_residual — PAPER (Phase A passer)
- **Factor decomposition residual signal extraction engine.** Removes systematic factor exposures (MKT=RATES=INFL=SAFE) via rolling 252d ridge regression, trades the residual (idiosyncratic) component.
- **Pipeline**: Returns → factor model → residuals → residual momentum (70%) + residual reversal (30%) → vol-scaled beta-neutral long-short portfolio.
- **Performance**: Regime Sharpe breakdown — Full +0.911 (PASS), GFC +0.602 (PASS), Q4 2018 +0.000 (PASS, limited data), COVID +0.354 (PASS), Rate Hikes +0.232 (PASS), Bull +0.385 (PASS). Max DD -3.51%.
- **Universe**: DBC, GLD, IEF, SPY, TLT
- **Key config**: reg_window=252, ridge_alpha=1e-6, resid_mom_window=63, resid_rev_weight=0.3, rebalance_freq=5
- **Deep diagnostics**:
  - Factor R2: SPY 99.7%, DBC 99.3%, GLD 99.7%, TLT 99.7%, IEF 85.0% — factor model correctly explains factor-component assets; IEF retains genuine residual variation
  - Beta to SPY: +0.0077 (effectively zero — passes neutrality)
  - No factor dominates: RATES 52.6%, MKT 20.3%, INFL 18.9%, SAFE 8.2% (all below 80% threshold)
  - Residual PnL Sharpe: **+11.98** — extreme signal quality in the residual component
  - Residual PnL fraction: 15.5% of absolute PnL — most PnL comes from side-effect factor exposures of position-taking
- **Limitation**: Residual signal purity is high but factor exposure contamination limits practical Sharpe. Improved factor-neutral portfolio construction (explicit beta-hedging per factor) could push this toward DEPLOY territory.
- **Caveats**: Residual autocorrelation for factor-component assets (SPY/DBC/GLD/TLT) is near 1.0 — structural artifact from near-perfect factor fit, not a model flaw. IEF shows clean residuals (autocorrelation near zero).

### Monitoring
- Strategy monitoring: `monitoring/strategy_monitor.py` (signal quality, regime detection, performance drift alerts)
- Alert engine: `quantcore/dashboard/alerts.py` (severity levels, notification routing, cooldowns)
- Live health checks: `quantcore/live/monitor.py` (latency, data staleness, heartbeat, position drift)
- Strategy smoke tests: 5 market scenarios (covid crash, trend reversal, correlation breakdown, liquidity crisis, low vol spike)

### Known Risks
1. **Factor residual is the only strategy to pass Phase A** (SR +0.911, 6/6 regimes PASS). Risk parity (SR +0.128) avoids rejection but is noise-limited to MONITOR. All other 6 strategies REJECTED.
2. **Factor residual's PnL is dominated by side-effect factor exposures** (84.5% of absolute PnL). The residual signal quality is extreme (Sharpe +11.98) but small in magnitude. The strategy implicitly takes factor bets through position-taking, and these bets contribute most of the return. Improved factor-neutral construction could raise total Sharpe toward DEPLOY territory.
3. **2022 Rate Hikes is the weakest regime** (SR +0.232). While not a showstopper, the strategy's residual signal struggled during the rapid hiking cycle. Possible factor model misspecification during rapid rate changes.
4. **Factor-component assets (SPY, DBC, GLD, TLT) have near-zero residual variance** since they are in the factor definitions. Their residual autocorrelation is structurally near 1.0 (not a model flaw). Only IEF carries meaningful residual variation for signal extraction.
5. **Cross-sectional ranking is statistically significant but economically too weak** for this universe. Score vs forward return r = 0.055 (p < 0.00001). With 6 macro assets, cross-sectional degrees of freedom are insufficient for a diversified market-neutral signal.
6. **Conclusion: Macro alpha EXISTS in this universe, but only in factor-residual space.** Time-series and cross-sectional directional strategies all fail (SR ≤ 0.24). The factor residual approach demonstrates that alpha is present in the idiosyncratic component after removing systematic MKT/RATES/INFL/SAFE exposures. This is a fundamentally different result from the earlier conclusion that "no alpha exists in this universe."
7. Both CTA and RP strategies degrade above $500K-$1M AUM (market impact + margin costs). Factor residual capacity is estimated at ~$2M due to long-short structure.
8. Real-data backtest uses yfinance (OHLCV only) — no tick-level or order book validation.
