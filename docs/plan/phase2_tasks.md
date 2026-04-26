# Phase 2 — Signals & Strategies Tasks

**Depends on:** Phase 1 complete (data infra, math models, basic backtest)

---

## AGENT-P2-01: Technical / Price-Based Signals

**Goal:** Full library of statistically-tested technical indicators.

### Tasks
1. Implement `quantcore/signals/technical/trend.py`:
   - Moving averages: SMA, EMA, DEMA, TEMA, Hull MA, Kaufman AMA
   - MACD (line, signal, histogram) with configurable periods
   - Parabolic SAR
   - ADX (Average Directional Index) with +DI / -DI
   - Aroon indicator (up/down/oscillator)
   - Supertrend (ATR-based)
   - Linear regression slope + R² as trend strength
   - Breakout detection: Donchian channel breaks with volume confirmation
   - Each indicator returns z-score normalized signal strength
2. Implement `quantcore/signals/technical/mean_reversion.py`:
   - Bollinger Bands (dynamic bandwidth) with %B and bandwidth signals
   - RSI with configurable period, overbought/oversold thresholds
   - Stochastic Oscillator (%K, %D)
   - Williams %R
   - CCI (Commodity Channel Index)
   - Z-score of distance from MA (multiple windows: 5, 10, 21, 63)
   - Half-life of mean reversion (fit OU process → extract θ → t½)
3. Implement `quantcore/signals/technical/volume.py`:
   - VWAP deviation signal
   - OBV (On Balance Volume) with divergence detection
   - Volume Price Trend
   - Accumulation/Distribution Line
   - Chaikin Money Flow
   - MFI (Money Flow Index)
   - Volume profile (POC, value area)
   - Unusual volume detection (volume/ADV z-score)
4. Implement `quantcore/signals/technical/volatility.py`:
   - ATR (multiple windows)
   - Historical volatility (5d, 10d, 21d, 63d, 252d)
   - Volatility regime detection (HMM on realized vol)
   - VIX term structure signal (contango/backwardation)
   - Realized vs implied vol spread
   - Volatility of volatility
5. Implement `quantcore/signals/technical/momentum.py`:
   - Cross-sectional momentum (percentile rank in universe)
   - Time-series momentum (absolute return lookback)
   - Dual momentum (Antonacci: absolute + relative)
   - Idiosyncratic momentum (residual from factor model)
   - 52-week high proximity
   - Momentum with vol scaling (signal / realized_vol)
6. Each signal: statistical significance test (t-test of IC vs zero), decay analysis (IC at lag 1, 5, 21)
7. Write tests: verify indicator values against known examples (e.g., 14-day RSI of known series)

**Acceptance:** All indicators match established implementations (TA-Lib/pandas-ta). Each signal has documented IC and decay profile.

---

## AGENT-P2-02: Statistical Arbitrage Signals

**Goal:** Pairs, factor, and cross-asset arbitrage signals.

### Tasks
1. Implement `quantcore/signals/statistical/pairs.py`:
   - Universe-wide cointegration scanner (Johansen on all N*(N-1)/2 pairs)
   - Filter: require cointegration at 1% significance for 3+ rolling windows
   - Kalman Filter dynamic hedge ratio (online updates)
   - Spread z-score entry/exit signal (±2σ entry, 0 exit, ±4σ stop)
   - Half-life-based position sizing
   - Transaction cost adjusted thresholds
2. Implement `quantcore/signals/statistical/basket.py`:
   - ETF vs underlying basket arbitrage (SPY vs S&P500 constituents)
   - Index arbitrage (futures vs cash)
   - Cross-listed arbitrage (dual-listed stocks)
   - ADR arbitrage
3. Implement `quantcore/signals/statistical/factors.py`:
   - Value: P/E, P/B, P/S, EV/EBITDA, FCF yield (from fundamental data)
   - Quality: ROE, ROA, ROIC, gross margin, accruals quality
   - Growth: revenue growth, earnings growth, guidance revision
   - Sentiment: analyst revision breadth, short interest ratio, 13F ownership change
   - Each factor: cross-sectional z-score, quintile spreads, factor return series
4. Write tests: pairs scanner finds known cointegrated pairs (e.g., KO/PEP). Factor quintile spreads are monotonic on historical data.

**Acceptance:** Pairs scanner processes 500 symbols in <5 minutes. Factor signals have statistically significant ICs on historical data.

---

## AGENT-P2-03: ML Signal Pipeline

**Goal:** End-to-end ML pipeline for alpha signal generation.

### Tasks
1. Implement `quantcore/signals/ml/features.py`:
   - Feature engineering pipeline: 500+ features from price/volume, inter-market, calendar, macro
   - Rolling window feature computation (avoid lookahead)
   - Cross-sectional rank features
   - Interaction features (products, ratios)
   - Feature store: compute once, cache in TimescaleDB
2. Implement `quantcore/signals/ml/preprocessing.py`:
   - Winsorization (clip at ±3σ)
   - Cross-sectional z-score normalization
   - Missing feature imputation (median, KNN)
   - Feature selection: mutual information, LASSO path, Boruta
3. Implement `quantcore/signals/ml/linear_models.py`:
   - LASSO (L1), Ridge (L2), Elastic Net
   - Quantile regression (predict return distribution tails)
   - Proper time-series CV (expanding or rolling window, embargo)
   - Hyperparameter tuning via time-series CV
4. Implement `quantcore/signals/ml/tree_models.py`:
   - XGBoost, LightGBM, CatBoost wrappers
   - Random Forest, Extra Trees
   - SHAP feature importance
   - Purged K-fold CV (Lopez de Prado)
   - Hyperparameter search (Bayesian optimization)
5. Implement `quantcore/signals/ml/deep_learning.py`:
   - LSTM model (PyTorch): sequence of features → next-period return prediction
   - GRU variant
   - Temporal Convolutional Network (TCN)
   - Transformer with attention for market sequences
   - Training loop: early stopping, learning rate scheduling, gradient clipping
   - Autoencoder for anomaly/regime detection
6. Implement `quantcore/signals/ml/reinforcement.py`:
   - DQN for discrete trade actions (buy/hold/sell)
   - PPO for continuous portfolio allocation
   - SAC for continuous action spaces
   - Custom market environment (OpenAI Gym interface)
   - Reward: risk-adjusted returns (Sharpe-based)
7. Implement `quantcore/signals/ml/evaluation.py`:
   - Information Coefficient (IC) and Rank IC
   - IC decay curve
   - Turnover analysis
   - Alpha decay half-life
   - Out-of-sample degradation ratio
8. Write tests: models overfit on training data (sanity check), OOS IC is positive on historical data

**Acceptance:** At least one ML model produces statistically significant OOS IC (>0.02) with proper purged CV on S&P500 universe.

---

## AGENT-P2-04: Options-Specific Signals

**Goal:** Volatility, flow, and Greeks-based trading signals.

### Tasks
1. Implement `quantcore/signals/options_flow/volatility_signals.py`:
   - IV Rank: current IV percentile in trailing 252-day window
   - IV Percentile: % of days IV was lower
   - IV vs HV spread (volatility risk premium signal)
   - Vol surface anomalies: smile steepness, skew z-score
   - Term structure signal: front-month vs back-month IV ratio
   - VIX futures basis (contango/backwardation strength)
2. Implement `quantcore/signals/options_flow/flow_signals.py`:
   - Unusual options activity: volume > 2× avg OI, premium > threshold
   - Sweep vs block classification
   - Put/Call ratio (volume, OI, dollar-weighted)
   - Large trade detection and directional inference
   - Net premium by strike and expiry
3. Implement `quantcore/signals/options_flow/greeks_signals.py`:
   - GEX (Gamma Exposure): aggregate net dealer gamma by strike
   - Gamma flip level identification
   - Charm flow estimation near expiry
   - Vanna exposure and flow prediction
   - Pin risk / max pain calculation
   - Delta hedging pressure signal
4. Write tests: GEX calculation matches known examples. IV Rank correctly identifies extremes.

**Acceptance:** Volatility signals have meaningful correlation with future realized vol. GEX levels align with observed market support/resistance.

---

## AGENT-P2-05: Signal Combination Engine

**Goal:** Combine multiple alpha signals into composite scores.

### Tasks
1. Implement `quantcore/signals/combiner.py`:
   - Equal-weight combination (z-score each, average)
   - IC-weighted combination (weight by trailing IC)
   - MVO on signals (treat signals as "assets", optimize combination)
   - Stacking / meta-learning (train model on signal outputs)
   - Dynamic weighting (regime-conditional weights)
   - Decay-aware weighting (downweight stale signals)
2. Signal conflict resolution: when signals disagree, rules for net signal
3. Signal strength bucketing: strong/moderate/weak for position sizing input
4. Write tests: combined signal IC > best individual signal IC

**Acceptance:** Combiner demonstrably improves IC over best single signal on backtest.

---

## AGENT-P2-06: Equity Strategies

**Goal:** Complete strategy implementations with entry/exit logic.

### Tasks
1. Define `quantcore/strategies/base.py`:
   - Abstract `Strategy` base class: `on_bar()`, `generate_signals()`, `generate_orders()`, `on_fill()`
   - Strategy state management (positions, P&L tracking)
   - Config loading from per-strategy TOML
   - Logging and audit trail
2. Implement `quantcore/strategies/equity/momentum.py`:
   - Cross-sectional 12-1 month momentum
   - Long top decile, short bottom decile
   - Inverse-vol weighting, monthly rebalance
   - Transaction cost model, turnover constraint
3. Implement `quantcore/strategies/equity/stat_arb.py`:
   - Multi-pair cointegration strategy
   - Dynamic pair selection (rolling Johansen)
   - Kalman hedge ratios, z-score signals
   - Kelly-sized positions, correlation portfolio construction
4. Implement `quantcore/strategies/equity/mean_reversion.py`:
   - Short-term equity mean reversion (1-3 day hold)
   - Multi-indicator confirmation (RSI + Bollinger + volume)
   - Intraday entry, end-of-day sizing
5. Implement `quantcore/strategies/equity/trend_following.py`:
   - Multi-timeframe trend (fast/medium/slow signals)
   - Breakout confirmation filters
   - ATR position sizing, pyramiding rules
   - 50+ instrument universe
6. Implement `quantcore/strategies/equity/market_making.py`:
   - Two-sided quoting with spread
   - Inventory management (skew quotes based on position)
   - Adverse selection filter (cancel on momentum detection)
   - Dynamic spread: wider in high-vol, tighter in low-vol
7. Backtest each strategy on 10+ years of data. Verify positive risk-adjusted returns.

**Acceptance:** Each strategy has positive Sharpe (>0.5) on out-of-sample period with realistic costs.

---

## AGENT-P2-07: Options Strategies

**Goal:** Complete options strategy implementations.

### Tasks
1. Implement `quantcore/strategies/options/delta_neutral.py`:
   - Sell straddles/strangles when IVR > 50
   - Continuous delta hedging (rehedge when |delta| > threshold)
   - Gamma scalping on long vol positions
   - Vega-neutral portfolio construction
   - P&L attribution by Greek
2. Implement `quantcore/strategies/options/iron_condor.py`:
   - Sell 16-delta put spread + call spread
   - 30-45 DTE entry, 21 DTE or 50% profit exit
   - Position sizing by max loss
   - Adjustment rules (roll untested side)
3. Implement `quantcore/strategies/options/calendar.py`:
   - Long back-month / short front-month same strike
   - Entry: front IV > back IV (term structure inversion)
   - Theta/vega balanced
   - Roll mechanics at front-month expiry
4. Implement `quantcore/strategies/options/tail_hedge.py`:
   - Continuous OTM SPX put buying
   - Strike selection: maximize convexity per dollar
   - VVIX-based timing (buy when cheap)
   - Dynamic hedge ratio
5. Implement `quantcore/strategies/options/earnings_vol.py`:
   - Compare implied vs historical earnings moves
   - Sell straddle 7 days before if IV overstates historically
   - Iron condor structure for defined risk
   - Exit day after earnings
6. Backtest each on 5+ years of options data

**Acceptance:** Each options strategy has positive expected value accounting for bid-ask spread and commissions.

---

## AGENT-P2-08: Multi-Asset / Macro Strategies

**Goal:** Cross-asset strategies spanning equities, bonds, commodities, FX, crypto.

### Tasks
1. Implement `quantcore/strategies/macro/risk_parity.py`:
   - Equal risk contribution across 5 asset classes
   - Vol-weighted allocation (inverse vol sizing)
   - Leverage to target volatility (e.g., 10% annualized)
   - Regime detection overlay (reduce risk in crisis)
2. Implement `quantcore/strategies/macro/cta_trend.py`:
   - 50+ futures markets (equity indices, rates, commodities, FX, crypto)
   - Multi-speed trend: fast (10d), medium (50d), slow (200d)
   - EWMA vol scaling for each instrument
   - Correlation-adjusted portfolio (cap correlated exposure)
3. Implement `quantcore/strategies/macro/carry.py`:
   - FX carry: long high-yield, short low-yield currencies
   - Commodity carry: futures curve roll yield
   - Equity carry: dividend yield vs financing cost
   - Volatility carry: VRP (sell variance swaps conceptually)
   - Equal-risk-weighted combination
4. Backtest each across multiple regimes (bull, bear, crisis)

**Acceptance:** Risk parity has lower max drawdown than 60/40. CTA positive in 2008/2020 crises. Carry positive ex-crisis.
