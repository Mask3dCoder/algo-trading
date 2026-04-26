# QuantCore — Master Task Index

> Each task ID format: `T-PPNN` where PP = phase, NN = sequential number.
> Each task affects ≤3 files, is independently testable, and has explicit acceptance criteria.
> Detailed task specs are in the per-phase files linked below.

## Phase 1 — Core Infrastructure (41 tasks)
→ [phase1_atomic_tasks.md](file:///C:/Users/rochv/.gemini/antigravity/brain/820c1653-4b98-4599-a441-ed3019872ea9/phase1_atomic_tasks.md)

| ID | Module | Files | Description |
|---|---|---|---|
| T-0101 | Scaffold | `pyproject.toml` | Project metadata, all dependency groups |
| T-0102 | Scaffold | `Makefile`, `.env.example` | Dev commands and env template |
| T-0103 | Scaffold | full tree `__init__.py` files | Create empty package skeleton |
| T-0104 | Core | `core/config.py`, `config/default.toml` | Dynaconf config loader + default config |
| T-0105 | Core | `core/logging.py` | Structlog setup |
| T-0106 | Core | `core/types.py` | Pydantic data models (OHLCV, Tick, Order, Fill, Position, Signal) |
| T-0107 | Core | `core/constants.py`, `core/time_utils.py` | Financial constants + market calendar |
| T-0108 | Core | `core/exceptions.py`, `core/events.py` | Exception hierarchy + event bus |
| T-0109 | Infra | `docker-compose.yml`, `Dockerfile` | Containerization |
| T-0110 | DB | `data/storage/database.py`, SQL migrations | TimescaleDB schema + async access layer |
| T-0111 | DB | `data/storage/timescale.py` | Continuous aggregates, compression, retention |
| T-0112 | DB | `data/storage/cache.py` | Redis caching layer |
| T-0113 | Data | `data/sources/base.py`, `data/sources/yahoo.py` | Abstract DataSource + Yahoo adapter |
| T-0114 | Data | `data/sources/polygon.py` | Polygon.io adapter |
| T-0115 | Data | `data/sources/alpha_vantage.py`, `data/sources/fred.py` | AV + FRED adapters |
| T-0116 | Data | `data/corporate_actions.py` | Split/dividend/merger adjustments |
| T-0117 | Data | `data/sources/universe.py` | Universe definitions + point-in-time reconstruction |
| T-0118 | Data | `data/quality/outlier_detection.py` | Z-score, Grubbs, IQR outlier detection |
| T-0119 | Data | `data/quality/missing_data.py` | Gap detection, forward-fill, interpolation, EM |
| T-0120 | Data | `data/quality/stale_quotes.py`, `data/quality/cross_validation.py` | Stale + cross-source checks |
| T-0121 | Feeds | `data/feeds/base.py`, `data/feeds/alpaca_feed.py` | Abstract feed + Alpaca WebSocket |
| T-0122 | Feeds | `data/feeds/polygon_feed.py`, `data/feeds/crypto_feed.py` | Polygon + CCXT feeds |
| T-0123 | Feeds | `data/feeds/aggregator.py`, `data/feeds/publisher.py` | Bar construction + Redis/Kafka publish |
| T-0124 | Math | `math_models/stochastic/gbm.py` | GBM: SDE, analytical, Euler, Milstein, MLE |
| T-0125 | Math | `math_models/stochastic/heston.py` | Heston model: simulation + calibration |
| T-0126 | Math | `math_models/stochastic/ou_process.py` | OU process: solution, half-life, MLE |
| T-0127 | Math | `math_models/stochastic/jump_diffusion.py`, `math_models/stochastic/sabr.py` | Merton JD + SABR |
| T-0128 | Math | `math_models/stochastic/fractional_bm.py` | fBM: Cholesky, Hosking, Hurst estimation |
| T-0129 | Math | `math_models/stochastic/monte_carlo.py` | MC engine: vectorized, variance reduction, Sobol |
| T-0130 | Math | `math_models/options/bsm.py` | BSM pricing + all Greeks (analytical) |
| T-0131 | Math | `math_models/options/implied_vol.py` | IV solvers: Newton-Raphson, Brent, Jaeckel |
| T-0132 | Math | `math_models/options/vol_surface.py` | SVI/SSVI parameterization + arbitrage checks |
| T-0133 | Math | `math_models/options/exotics.py` | Asian, barrier, lookback, American (LSM) |
| T-0134 | Stats | `math_models/statistics/stationarity.py` | ADF, KPSS, PP, variance ratio tests |
| T-0135 | Stats | `math_models/statistics/hurst.py`, `math_models/statistics/serial_correlation.py` | Hurst + ACF/PACF/Ljung-Box |
| T-0136 | Stats | `math_models/statistics/garch.py` | GARCH family (G, EG, GJR, T, FI) + HAR-RV |
| T-0137 | Stats | `math_models/statistics/cointegration.py` | Engle-Granger, Johansen, VECM, pair scanner |
| T-0138 | Stats | `math_models/statistics/factor_models.py` | PCA, ICA, Fama-French, Carhart |
| T-0139 | Prob | `math_models/distributions/fitting.py`, `math_models/distributions/evt.py` | Distribution fitting + EVT |
| T-0140 | Prob | `math_models/distributions/copulas.py` | Gaussian, t, Clayton, Gumbel, Frank copulas |
| T-0141 | Prob | `math_models/bayesian/kalman.py`, `math_models/bayesian/mcmc.py` | KF/EKF/UKF + MCMC |

## Phase 2 — Signals & Strategies (30 tasks)
→ [phase2_atomic_tasks.md](file:///C:/Users/rochv/.gemini/antigravity/brain/820c1653-4b98-4599-a441-ed3019872ea9/phase2_atomic_tasks.md)

| ID | Module | Files | Description |
|---|---|---|---|
| T-0201 | Signals | `signals/technical/trend.py` | MA family, MACD, ADX, Supertrend, breakout |
| T-0202 | Signals | `signals/technical/mean_reversion.py` | Bollinger, RSI, Stochastic, CCI, OU half-life |
| T-0203 | Signals | `signals/technical/volume.py` | VWAP, OBV, CMF, MFI, unusual volume |
| T-0204 | Signals | `signals/technical/volatility.py` | ATR, HV, vol regime, VIX signals |
| T-0205 | Signals | `signals/technical/momentum.py` | Cross-sectional, time-series, dual momentum |
| T-0206 | Signals | `signals/statistical/pairs.py` | Coint scanner, Kalman hedge, z-score signals |
| T-0207 | Signals | `signals/statistical/basket.py` | ETF arb, index arb, ADR arb |
| T-0208 | Signals | `signals/statistical/factors.py` | Value, quality, growth, sentiment factors |
| T-0209 | ML | `signals/ml/features.py` | 500+ feature engineering pipeline |
| T-0210 | ML | `signals/ml/preprocessing.py` | Winsorize, z-score, impute, feature selection |
| T-0211 | ML | `signals/ml/linear_models.py` | LASSO, Ridge, Elastic Net, quantile regression |
| T-0212 | ML | `signals/ml/tree_models.py` | XGBoost, LightGBM, CatBoost + SHAP |
| T-0213 | ML | `signals/ml/deep_learning.py` | LSTM, GRU, TCN (PyTorch) |
| T-0214 | ML | `signals/ml/deep_learning.py` | Transformer, Autoencoder, VAE |
| T-0215 | ML | `signals/ml/reinforcement.py` | DQN, PPO, SAC + Gym environment |
| T-0216 | ML | `signals/ml/evaluation.py` | IC, Rank IC, decay curves, OOS degradation |
| T-0217 | Signals | `signals/options_flow/volatility_signals.py` | IVR, IVP, IV-HV spread, term structure |
| T-0218 | Signals | `signals/options_flow/flow_signals.py` | UOA, sweeps, P/C ratio |
| T-0219 | Signals | `signals/options_flow/greeks_signals.py` | GEX, charm flows, vanna, pin risk |
| T-0220 | Signals | `signals/combiner.py` | Equal-weight, IC-weight, stacking, dynamic |
| T-0221 | Strategy | `strategies/base.py`, `strategies/registry.py` | Abstract strategy + registry |
| T-0222 | Strategy | `strategies/equity/momentum.py` | Cross-sectional momentum strategy |
| T-0223 | Strategy | `strategies/equity/stat_arb.py` | Multi-pair cointegration strategy |
| T-0224 | Strategy | `strategies/equity/mean_reversion.py` | Short-term mean reversion |
| T-0225 | Strategy | `strategies/equity/trend_following.py` | Multi-timeframe trend following |
| T-0226 | Strategy | `strategies/options/delta_neutral.py` | Delta-neutral vol trading + gamma scalp |
| T-0227 | Strategy | `strategies/options/iron_condor.py` | Iron condor with adjustments |
| T-0228 | Strategy | `strategies/options/earnings_vol.py`, `strategies/options/tail_hedge.py` | Earnings vol + tail hedge |
| T-0229 | Strategy | `strategies/macro/risk_parity.py` | Risk parity allocation |
| T-0230 | Strategy | `strategies/macro/cta_trend.py`, `strategies/macro/carry.py` | CTA trend + carry |

## Phase 3 — Risk, Execution & Portfolio (30 tasks)
→ [phase3_atomic_tasks.md](file:///C:/Users/rochv/.gemini/antigravity/brain/820c1653-4b98-4599-a441-ed3019872ea9/phase3_atomic_tasks.md)

| ID | Module | Files | Description |
|---|---|---|---|
| T-0301 | Risk | `risk/var.py` | Historical simulation VaR |
| T-0302 | Risk | `risk/var.py` | Parametric + Cornish-Fisher VaR |
| T-0303 | Risk | `risk/var.py` | Monte Carlo VaR + filtered historical sim |
| T-0304 | Risk | `risk/var.py` | CVaR: marginal, component, incremental |
| T-0305 | Risk | `risk/var_backtest.py` | Kupiec, Christoffersen, traffic light tests |
| T-0306 | Risk | `risk/stress.py` | Historical scenario library + application engine |
| T-0307 | Risk | `risk/stress.py` | Hypothetical + factor stress tests |
| T-0308 | Risk | `risk/stress.py` | Reverse stress testing |
| T-0309 | Risk | `risk/greeks.py` | Aggregated Greeks + tenor bucketing |
| T-0310 | Risk | `risk/limits.py` | Position/portfolio limits + circuit breakers |
| T-0311 | Risk | `risk/drawdown.py` | Drawdown tracking + automatic risk reduction |
| T-0312 | Risk | `risk/metrics.py` | Full risk-adjusted metrics suite |
| T-0313 | Risk | `risk/reports.py` | Daily/weekly HTML risk reports |
| T-0314 | Exec | `execution/oms.py` | Order lifecycle state machine + audit |
| T-0315 | Exec | `execution/algos/twap.py` | TWAP execution algorithm |
| T-0316 | Exec | `execution/algos/vwap.py` | VWAP execution algorithm |
| T-0317 | Exec | `execution/algos/pov.py` | POV execution algorithm |
| T-0318 | Exec | `execution/algos/implementation_shortfall.py` | Almgren-Chriss IS |
| T-0319 | Exec | `execution/impact.py` | Market impact models |
| T-0320 | Exec | `execution/tca.py` | Transaction cost analysis |
| T-0321 | Exec | `execution/brokers/base.py`, `execution/brokers/paper.py` | Abstract broker + paper broker |
| T-0322 | Exec | `execution/brokers/alpaca.py` | Alpaca broker adapter |
| T-0323 | Exec | `execution/brokers/ibkr.py` | Interactive Brokers adapter |
| T-0324 | Exec | `execution/brokers/ccxt_adapter.py` | CCXT crypto adapter |
| T-0325 | Portfolio | `portfolio/sizing.py` | Kelly, vol-target, risk-parity, ATR sizing |
| T-0326 | Portfolio | `portfolio/optimization.py` | MVO + Black-Litterman |
| T-0327 | Portfolio | `portfolio/optimization.py` | HRP + NCO + Risk Parity |
| T-0328 | Portfolio | `portfolio/rebalance.py` | Rebalance engine + turnover management |
| T-0329 | Portfolio | `portfolio/tax.py` | Tax-lot tracking + loss harvesting |
| T-0330 | Regime | `regime/hmm.py`, `regime/features.py` | HMM regime detection + features |
| T-0331 | Regime | `regime/changepoint.py`, `regime/volatility_regime.py` | Changepoint + vol regimes |

## Phase 4 — Live Ops & Production (30 tasks)
→ [phase4_atomic_tasks.md](file:///C:/Users/rochv/.gemini/antigravity/brain/820c1653-4b98-4599-a441-ed3019872ea9/phase4_atomic_tasks.md)

| ID | Module | Files | Description |
|---|---|---|---|
| T-0401 | AltData | `alt_data/sentiment/news.py` | News ingestion + text extraction |
| T-0402 | AltData | `alt_data/sentiment/finbert.py` | FinBERT sentiment scoring |
| T-0403 | AltData | `alt_data/sentiment/filings.py` | SEC EDGAR 10-K/10-Q NLP |
| T-0404 | AltData | `alt_data/sentiment/social.py` | Reddit + StockTwits sentiment |
| T-0405 | AltData | `alt_data/sentiment/signals.py` | Composite sentiment signals |
| T-0406 | AltData | `alt_data/options_flow/unusual_activity.py` | UOA detection + classification |
| T-0407 | AltData | `alt_data/options_flow/gex.py` | GEX calculation + gamma flip |
| T-0408 | AltData | `alt_data/options_flow/max_pain.py`, `alt_data/options_flow/put_call.py` | Max pain + P/C ratio |
| T-0409 | Live | `live/paper.py` | Paper trading loop with simulated fills |
| T-0410 | Live | `live/trader.py` | Main live trading event loop |
| T-0411 | Live | `live/watchdog.py` | Heartbeat + dead man's switch |
| T-0412 | Live | `live/reconciliation.py` | Position/P&L/cash reconciliation |
| T-0413 | Live | emergency procedures | Kill switch + auto-halt triggers |
| T-0414 | Dash | `dashboard/app.py` | Dash app skeleton + auth + dark theme |
| T-0415 | Dash | `dashboard/panels/portfolio.py` | P&L, equity curve, positions panel |
| T-0416 | Dash | `dashboard/panels/strategy.py` | Strategy P&L, signals, trades panel |
| T-0417 | Dash | `dashboard/panels/risk.py` | VaR gauges, Greeks heatmap, drawdown |
| T-0418 | Dash | `dashboard/panels/market.py` | Regime, vol surface, breadth |
| T-0419 | Dash | `dashboard/panels/execution.py` | VWAP perf, slippage, TCA |
| T-0420 | Dash | `dashboard/alerts.py` | Alert rules engine + Slack/email |
| T-0421 | Infra | `Dockerfile`, `docker-compose.yml` | Multi-stage Docker + full stack compose |
| T-0422 | Infra | `k8s/` manifests | K8s deployments, services, HPA |
| T-0423 | Infra | `api/main.py`, `api/routes/` | FastAPI REST layer |
| T-0424 | Infra | `.github/workflows/ci.yml` | CI/CD pipeline |
| T-0425 | Perf | hot-path modules | Numba JIT + profiling |
| T-0426 | Comply | `compliance/audit_log.py` | Immutable audit trail + hash chain |
| T-0427 | Comply | `compliance/rules.py` | Pre-trade compliance rules |
| T-0428 | Comply | `compliance/reporting.py` | Trade blotter + tax reports |
| T-0429 | Research | `research/alpha_research.py` | Deflated Sharpe, multiple testing correction |
| T-0430 | Research | `research/experiment_tracker.py` | Experiment logging + A/B framework |

## Backtest Engine (Phase 1 continuation, 6 tasks)

| ID | Module | Files | Description |
|---|---|---|---|
| T-0142 | Backtest | `backtest/engine.py` | Event-driven simulator core |
| T-0143 | Backtest | `backtest/simulation.py` | Fill, slippage, commission models |
| T-0144 | Backtest | `backtest/validation.py` | Lookahead/survivorship bias detection |
| T-0145 | Backtest | `backtest/walk_forward.py` | Walk-forward + CPCV |
| T-0146 | Backtest | `backtest/analytics.py` | Performance analytics + HTML report |
| T-0147 | Rust | `rust_core/` | Order book, indicators, matching engine |

## Stats & Info Theory (Phase 1 continuation, 2 tasks)

| ID | Module | Files | Description |
|---|---|---|---|
| T-0148 | Stats | `math_models/statistics/information_theory.py` | Entropy, KL, MI, Transfer Entropy |
| T-0149 | Prob | `math_models/bayesian/particle_filter.py` | SMC particle filter |

---

**Grand Total: 140 atomic tasks**

| Phase | Tasks | Focus |
|---|---|---|
| P1 | 49 | Data, math, backtest, Rust |
| P2 | 30 | Signals, ML, strategies |
| P3 | 31 | Risk, execution, portfolio, regime |
| P4 | 30 | Live ops, dashboard, compliance, research |
