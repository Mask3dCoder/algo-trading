# Phase 2 — Signals & Strategies (Atomic Tasks)

---

### T-0201 · Trend Signals
**Files:** `quantcore/signals/technical/trend.py`
**Deps:** T-0106, T-0147(optional Rust)
**Steps:**
1. `sma(series, period)`, `ema(series, period)`, `dema`, `tema`, `hull_ma`, `kaufman_ama`: all MA variants
2. `macd(series, fast=12, slow=26, signal=9) → (macd_line, signal_line, histogram)`
3. `adx(high, low, close, period=14) → (adx, plus_di, minus_di)`
4. `supertrend(high, low, close, period=10, mult=3)`
5. `linear_regression_signal(series, period) → (slope, r_squared)`
6. `breakout_signal(high, low, period=20) → signal`: Donchian with significance test
7. Each returns z-score normalized signal in [-1, +1] range
8. Test: EMA of constant = constant. MACD crossover detected on known pattern.
**Accept:** All indicators match pandas-ta / TA-Lib values on SPY daily data. Each signal z-scored.

---

### T-0202 · Mean Reversion Signals
**Files:** `quantcore/signals/technical/mean_reversion.py`
**Deps:** T-0126, T-0201
**Steps:**
1. `bollinger_bands(series, period=20, std=2) → (upper, mid, lower, pct_b, bandwidth)`
2. `rsi(series, period=14)`, `stochastic(high, low, close, k=14, d=3)`, `williams_r`, `cci`
3. `ma_zscore(series, period) → zscore`: distance from MA in σ
4. `ou_halflife_signal(series, lookback=252) → (halflife, signal)`: fit OU, compute z-score with halflife-aware decay
5. All return signal in [-1, +1]: -1=oversold (buy signal), +1=overbought (sell signal)
6. Test: RSI of uptrend > 50. Bollinger %B of mean-reverting series oscillates around 0.5.
**Accept:** RSI matches known calculation. OU half-life on synthetic mean-reverting series within 20% of true value.

---

### T-0203 · Volume Signals
**Files:** `quantcore/signals/technical/volume.py`
**Deps:** T-0106
**Steps:**
1. `vwap_deviation(price, volume, period) → zscore`
2. `obv(close, volume)`, `volume_price_trend(close, volume)`, `ad_line(high, low, close, volume)`
3. `chaikin_money_flow(high, low, close, volume, period=21)`, `mfi(high, low, close, volume, period=14)`
4. `unusual_volume(volume, lookback=20, threshold=2.0) → bool_mask`: volume/ADV z-score
5. `obv_divergence(close, obv, lookback=20) → signal`: price up + OBV down = bearish
6. Test: OBV monotonically increases in pure uptrend. CMF positive when close near high.
**Accept:** All calculations match published formulas. Unusual volume correctly flags known high-volume events.

---

### T-0204 · Volatility Signals
**Files:** `quantcore/signals/technical/volatility.py`
**Deps:** T-0136
**Steps:**
1. `atr(high, low, close, period=14)`, `historical_vol(close, windows=[5,10,21,63,252])`
2. `vol_regime(vol_series) → regime_label`: HMM-based (low/normal/high)
3. `vix_term_structure(vix_spot, vix_futures) → signal`: contango=complacent, backwardation=fear
4. `rv_iv_spread(realized_vol, implied_vol) → signal`: positive=sell vol, negative=buy vol
5. `vol_of_vol(vol_series, period=21) → float`
6. Test: ATR during high-vol period > ATR during low-vol. Vol regime detects 2020 as "high".
**Accept:** HV matches manual calculation. Vol regime transitions align with known market events.

---

### T-0205 · Momentum Signals
**Files:** `quantcore/signals/technical/momentum.py`
**Deps:** T-0106
**Steps:**
1. `cross_sectional_momentum(returns_matrix, lookback=252, skip=21) → ranks`: percentile rank by 12-1 month return
2. `time_series_momentum(returns, lookback=252) → signal`: sign of cumulative return
3. `dual_momentum(returns, benchmark_returns, lookback=252) → signal`: absolute + relative
4. `idiosyncratic_momentum(returns, factor_returns, lookback=252) → signal`: residual momentum
5. `high_52w_proximity(close) → signal`: close / 52-week high
6. `vol_scaled_momentum(returns, vol, lookback=252) → signal`: return / vol
7. Test: momentum signal positive for trending-up stock. Cross-sectional ranks sum correctly.
**Accept:** Top-decile momentum portfolio outperforms bottom-decile on historical data. Signals in [-1,1].

---

### T-0206 · Pairs Trading Signals
**Files:** `quantcore/signals/statistical/pairs.py`
**Deps:** T-0137, T-0141
**Steps:**
1. `scan_cointegrated_pairs(returns, significance=0.01) → list[Pair]`: Johansen on all combinations
2. `kalman_hedge_ratio(y, x) → (ratio_series, spread_series)`: online KF hedge ratio
3. `spread_zscore(spread, lookback=60) → zscore`
4. `pairs_signal(zscore, entry=2.0, exit=0.5, stop=4.0) → signal`: ±entry opens, ±exit closes, ±stop stops
5. `halflife_size(halflife, base_size) → position_size`: scale by mean-reversion speed
6. `cost_adjusted_threshold(entry, spread_cost) → adj_entry`: widen entry by transaction cost
7. Test: synthetic cointegrated pair detected. Signal correctly enters at ±2σ, exits at 0.
**Accept:** Scanner finds 10+ significant pairs in S&P500. Kalman ratio tracks true hedge ratio on synthetic data.

---

### T-0207 · Basket Arbitrage Signals
**Files:** `quantcore/signals/statistical/basket.py`
**Deps:** T-0137
**Steps:**
1. `etf_arb_signal(etf_price, nav_estimate) → zscore`: ETF premium/discount
2. `index_arb_signal(futures_price, spot_index, cost_of_carry) → spread`: futures basis
3. `adr_arb_signal(adr_price, foreign_price, fx_rate) → spread`
4. `fair_value(constituents, weights) → nav_estimate`: compute NAV from components
5. Test: ETF arb signal near zero during normal markets. Futures basis = risk-free rate (approx).
**Accept:** Basis calculation matches theoretical cost-of-carry. Signals mean-revert to zero.

---

### T-0208 · Factor Signals
**Files:** `quantcore/signals/statistical/factors.py`
**Deps:** T-0138
**Steps:**
1. Value: `pe_signal`, `pb_signal`, `ev_ebitda_signal`, `fcf_yield_signal` → cross-sectional z-score
2. Quality: `roe_signal`, `roic_signal`, `gross_margin_signal`, `accruals_signal`
3. Growth: `revenue_growth_signal`, `earnings_growth_signal`
4. Sentiment: `analyst_revision_signal`, `short_interest_signal`
5. `composite_factor(signals, weights) → combined_signal`
6. `factor_return(signal, forward_returns) → IC, factor_return_series`
7. Test: value factor has positive IC on 10-year backtest. Quintile spread monotonic.
**Accept:** Each factor produces cross-sectionally normalized scores. IC statistically significant (t>2) for at least 3 factors.

---

### T-0209 · ML Feature Engineering
**Files:** `quantcore/signals/ml/features.py`
**Deps:** T-0201–T-0205
**Steps:**
1. `TechnicalFeatures`: all technical indicators at multiple windows (5,10,21,63 days) → ~200 features
2. `InterMarketFeatures`: rolling correlations with SPY, sector ETFs, VIX → ~50 features
3. `CalendarFeatures`: day_of_week, month, days_to_opex, days_to_earnings → ~20 features
4. `CrossSectionalFeatures`: percentile ranks of all features within universe → doubles feature count
5. `FeatureStore`: compute features, store in TimescaleDB, retrieve by date range
6. `feature_matrix(universe, date_range) → DataFrame`: aligned features + forward returns
7. Test: no NaN in output (after warm-up). Features are point-in-time (no lookahead).
**Accept:** 500+ features generated. No lookahead verified by checking feature timestamps < return timestamps.

---

### T-0210 · ML Preprocessing
**Files:** `quantcore/signals/ml/preprocessing.py`
**Deps:** T-0209
**Steps:**
1. `winsorize(features, limits=3.0)`: clip at ±3σ cross-sectionally
2. `cross_sectional_zscore(features)`: normalize per date
3. `impute_missing(features, method="median")`: median/KNN imputation
4. `select_features(features, target, method="mutual_info", n=100) → selected`: MI, LASSO path, or Boruta
5. `TimeSeriesSplit(n_splits, embargo_pct)`: proper time-series CV splitter (no leakage)
6. Test: z-scored features have mean≈0, std≈1 per cross-section. No future data in splits.
**Accept:** Feature selection reduces 500→100 features. CV splits have correct embargo gap.

---

### T-0211 · Linear ML Models
**Files:** `quantcore/signals/ml/linear_models.py`
**Deps:** T-0210
**Steps:**
1. `LassoSignal`: LASSO with time-series CV, return feature importances (non-zero coefficients)
2. `RidgeSignal`: Ridge regression with optimal λ via CV
3. `ElasticNetSignal`: combine L1+L2
4. `QuantileRegression`: predict 5th and 95th percentile of returns (tail risk signal)
5. All: `fit(features, returns)`, `predict(features) → signal`, `feature_importance()`
6. Test: LASSO selects <50% of features. Ridge uses all. Predictions have positive IC on test set.
**Accept:** At least one model has OOS IC > 0.01. LASSO coefficient sparsity > 50%.

---

### T-0212 · Tree-Based ML Models
**Files:** `quantcore/signals/ml/tree_models.py`
**Deps:** T-0210
**Steps:**
1. `XGBoostSignal`: XGBRegressor with purged time-series CV, early stopping
2. `LightGBMSignal`: LGBMRegressor with same CV framework
3. `CatBoostSignal`: CatBoostRegressor
4. `RandomForestSignal`: RF with proper temporal CV
5. All: hyperparameter tuning via Bayesian optimization (optuna)
6. `shap_importance(model, features) → DataFrame`: SHAP values for interpretability
7. Test: tree models overfit on training (sanity). OOS IC positive. SHAP values sum to prediction.
**Accept:** Best tree model OOS IC > 0.02. SHAP analysis identifies sensible top features.

---

### T-0213 · Sequence Deep Learning (LSTM/GRU/TCN)
**Files:** `quantcore/signals/ml/deep_learning.py`
**Deps:** T-0210
**Steps:**
1. `LSTMSignal`: PyTorch LSTM (2 layers, 64 hidden), input=sequence of features, output=next return
2. `GRUSignal`: GRU variant (faster training)
3. `TCNSignal`: Temporal Convolutional Network (dilated causal convolutions)
4. Training: AdamW optimizer, ReduceLROnPlateau, early stopping, gradient clipping
5. `SequenceDataset`: sliding window dataset with proper train/val/test splits (temporal)
6. Test: model loss decreases over epochs. Predictions not all same value.
**Accept:** At least one DL model has OOS IC > 0.01. Training converges in <100 epochs.

---

### T-0214 · Transformer + Autoencoders
**Files:** `quantcore/signals/ml/deep_learning.py` (append)
**Deps:** T-0213
**Steps:**
1. `TransformerSignal`: multi-head self-attention on feature sequences, positional encoding
2. `Autoencoder`: encode features → latent → decode. Reconstruction error = anomaly score.
3. `VAE`: variational autoencoder, latent space for regime generation
4. Test: transformer attention weights are non-uniform. Autoencoder reconstruction error low on normal, high on anomalous.
**Accept:** Transformer attends to recent timesteps more. Autoencoder anomaly score spikes during known market stress.

---

### T-0215 · Reinforcement Learning
**Files:** `quantcore/signals/ml/reinforcement.py`
**Deps:** T-0142
**Steps:**
1. `TradingEnv(gym.Env)`: observation=features, action=position_change, reward=risk-adjusted return
2. `DQNAgent`: discrete actions (long/flat/short), experience replay, target network
3. `PPOAgent`: continuous portfolio weights, clipped objective
4. `SACAgent`: continuous actions, entropy regularization
5. Training loop with episode logging
6. Test: DQN learns to buy-and-hold in pure uptrend. PPO converges to non-random policy.
**Accept:** RL agent achieves Sharpe > 0 on training data. Policy is non-trivial (not all-cash).

---

### T-0216 · ML Signal Evaluation
**Files:** `quantcore/signals/ml/evaluation.py`
**Deps:** T-0211, T-0212
**Steps:**
1. `information_coefficient(signal, forward_returns) → (ic, t_stat, p_value)`: Pearson correlation
2. `rank_ic(signal, forward_returns) → (rank_ic, t_stat)`: Spearman rank correlation
3. `ic_decay(signal, returns, lags=[1,5,10,21,63]) → dict`: IC at each horizon
4. `turnover(signal_series) → float`: average daily signal change
5. `alpha_decay_halflife(ic_series) → float`: fit exponential decay
6. `oos_degradation(is_ic, oos_ic) → float`: ratio indicating overfitting
7. Test: perfect signal has IC=1. Random signal has IC≈0 (not significant).
**Accept:** Evaluation correctly ranks models by predictive power. Degradation ratio flags overfitted models (>2×).

---

### T-0217 · Volatility Trading Signals
**Files:** `quantcore/signals/options_flow/volatility_signals.py`
**Deps:** T-0131
**Steps:**
1. `iv_rank(current_iv, iv_history_1y) → float [0,1]`: percentile rank
2. `iv_percentile(current_iv, iv_history_1y) → float [0,1]`: % of days lower
3. `vrp_signal(realized_vol, implied_vol) → signal`: volatility risk premium
4. `skew_signal(put_25d_iv, call_25d_iv) → float`: 25-delta risk reversal
5. `term_structure_signal(front_iv, back_iv) → signal`: contango/backwardation
6. Test: IVR=1.0 when current IV is 1-year max. VRP positive on average.
**Accept:** IVR correctly computed on historical VIX data. VRP signal has positive mean (premium exists).

---

### T-0218 · Options Flow Signals
**Files:** `quantcore/signals/options_flow/flow_signals.py`
**Deps:** T-0106
**Steps:**
1. `detect_unusual_activity(trades, avg_oi) → list[UOATrade]`: volume > 2× OI, premium > threshold
2. `classify_sweep_vs_block(trades) → classified`: multi-exchange=sweep, single=block
3. `put_call_ratio(volume_data, method="volume"|"oi"|"dollar") → ratio`
4. `pc_ratio_signal(ratio, lookback=20) → zscore`: extreme readings
5. Test: known large trade flagged as UOA. P/C ratio of all-calls market = low.
**Accept:** UOA detection catches trades with volume > 2× OI. P/C ratio matches CBOE published values.

---

### T-0219 · Greeks-Based Flow Signals
**Files:** `quantcore/signals/options_flow/greeks_signals.py`
**Deps:** T-0130
**Steps:**
1. `calculate_gex(options_chain, spot) → GEXProfile`: net dealer gamma per strike
2. `gamma_flip_level(gex_profile) → float`: where GEX crosses zero
3. `charm_flow(options_chain, spot) → float`: expected delta decay → hedging flow
4. `vanna_flow(options_chain, spot) → float`: dDelta/dVol exposure
5. `max_pain(options_chain) → float`: strike minimizing total holder P&L
6. Test: GEX positive at current spot for typical market (dealer long gamma). Max pain near ATM.
**Accept:** GEX profile shows expected dealer gamma shape. Max pain within 2% of manual calculation.

---

### T-0220 · Signal Combiner
**Files:** `quantcore/signals/combiner.py`
**Deps:** T-0201–T-0219
**Steps:**
1. `equal_weight_combine(signals: dict[str, Series]) → Series`: z-score each, average
2. `ic_weighted_combine(signals, forward_returns, lookback=63) → Series`: weight by trailing IC
3. `stack_combine(signals, forward_returns) → Series`: train meta-model (Ridge) on signal outputs
4. `dynamic_combine(signals, regime, weights_per_regime) → Series`: regime-conditional weights
5. `signal_conflict_report(signals) → DataFrame`: show agreement/disagreement matrix
6. Test: combined IC ≥ median individual IC. Stacking IC > equal-weight IC.
**Accept:** At least one combination method improves IC over best single signal. Weights sum to 1.

---

### T-0221 · Strategy Base + Registry
**Files:** `quantcore/strategies/base.py`, `quantcore/strategies/registry.py`
**Deps:** T-0106, T-0108
**Steps:**
1. `Strategy(ABC)`: `name`, `config`, `on_bar(bar)`, `generate_signals() → list[Signal]`, `generate_orders(signals) → list[Order]`, `on_fill(fill)`, `get_state() → dict`, `load_state(dict)`
2. `StrategyState`: positions, pending_orders, pnl, metadata
3. `StrategyRegistry`: `register(name, cls)`, `create(name, config) → Strategy`, `list_strategies()`
4. Auto-discovery: scan `strategies/` subpackages, register all `Strategy` subclasses
5. Config loading: each strategy reads from `config/strategies/{name}.toml`
6. Test: register mock strategy, create instance, call lifecycle methods.
**Accept:** Registry discovers and instantiates strategies. Config loads correctly. State save/load round-trips.

---

### T-0222 · Momentum Strategy
**Files:** `quantcore/strategies/equity/momentum.py`, `config/strategies/momentum.toml`
**Deps:** T-0205, T-0221
**Steps:**
1. Universe: S&P500. Lookback: 12-1 month returns (skip most recent month).
2. Rank: long top decile, short bottom decile.
3. Weighting: inverse volatility within each leg.
4. Rebalance: monthly (first trading day).
5. Transaction cost model: 10bps round-trip.
6. Turnover constraint: max 200% annual.
7. Config: all parameters in TOML.
8. Backtest: 2005-2024, compute Sharpe, max DD.
**Accept:** Sharpe > 0.3 gross, > 0.1 net of costs. Monthly rebalance executes correctly.

---

### T-0223 · Statistical Arbitrage Strategy
**Files:** `quantcore/strategies/equity/stat_arb.py`, `config/strategies/stat_arb.toml`
**Deps:** T-0206, T-0221
**Steps:**
1. Pair selection: rolling Johansen test, top 20 pairs by cointegration strength.
2. Kalman filter hedge ratios, rolling spread z-score.
3. Entry: |z| > 2. Exit: |z| < 0.5. Stop: |z| > 4.
4. Sizing: fractional Kelly based on historical win rate.
5. Portfolio: max 20 pairs, max 5% per pair.
6. Config: all parameters in TOML.
7. Backtest: 2010-2024.
**Accept:** Sharpe > 0.5. Win rate > 55%. Max drawdown < 15%.

---

### T-0224 · Mean Reversion Strategy
**Files:** `quantcore/strategies/equity/mean_reversion.py`, `config/strategies/mean_reversion.toml`
**Deps:** T-0202, T-0221
**Steps:**
1. Scan universe for oversold (RSI<30 + BB %B<0.05 + unusual volume).
2. Enter long at close. Target: 1-3 day hold.
3. Exit: RSI>50 or 3-day timeout or -2% stop-loss.
4. Size: ATR-based (risk 0.5% per trade).
5. Max positions: 10 concurrent.
6. Backtest: 2010-2024.
**Accept:** Win rate > 55%. Average hold < 5 days. Positive expectancy per trade.

---

### T-0225 · Trend Following Strategy
**Files:** `quantcore/strategies/equity/trend_following.py`, `config/strategies/trend_following.toml`
**Deps:** T-0201, T-0221
**Steps:**
1. Multi-speed signals: fast (10/30 EMA cross), medium (50/100), slow (100/200).
2. Combined signal: vote across speeds.
3. Entry: all 3 agree on direction. Exit: 2/3 flip.
4. ATR-based position sizing. Pyramid: add 50% at +1 ATR in direction.
5. Trailing stop: 3× ATR.
6. Universe: 50+ liquid instruments (equities, ETFs, futures proxies).
7. Backtest: 2005-2024.
**Accept:** Positive returns in 2008 and 2020 (crisis alpha). Sharpe > 0.4.

---

### T-0226 · Delta-Neutral Vol Strategy
**Files:** `quantcore/strategies/options/delta_neutral.py`, `config/strategies/delta_neutral.toml`
**Deps:** T-0130, T-0217, T-0221
**Steps:**
1. Sell ATM straddles when IVR > 50.
2. Delta hedge when |net_delta| > 0.10 per underlying.
3. Gamma scalp: when long vol, trade underlying at ±1σ moves.
4. Vega-neutral: offset vega across positions.
5. P&L attribution: decompose into delta, gamma, theta, vega components.
6. Backtest on SPY options (2015-2024).
**Accept:** Theta P&L positive. Delta P&L near zero. Net Sharpe > 0.5.

---

### T-0227 · Iron Condor Strategy
**Files:** `quantcore/strategies/options/iron_condor.py`, `config/strategies/iron_condor.toml`
**Deps:** T-0130, T-0217, T-0221
**Steps:**
1. Sell 16-delta put spread + 16-delta call spread.
2. Entry: 30-45 DTE. Exit: 21 DTE OR 50% profit OR tested wing (delta > 0.30).
3. Adjustment: roll untested side closer when tested.
4. Size: max loss ≤ 2% of account per position.
5. Max concurrent: 5 positions.
6. Backtest: SPY, QQQ, IWM (2015-2024).
**Accept:** Win rate > 70%. Average winner > average loser in dollar terms. Positive expectancy.

---

### T-0228 · Earnings Vol + Tail Hedge
**Files:** `quantcore/strategies/options/earnings_vol.py`, `quantcore/strategies/options/tail_hedge.py`
**Deps:** T-0130, T-0217, T-0221
**Steps:**
1. Earnings: compare implied move (straddle price) to historical realized moves.
2. Sell straddle 7 days before if IV consistently overstates (>60% of last 8 quarters).
3. Iron condor structure. Exit day after earnings.
4. Tail hedge: continuously buy 5-10 delta OTM SPX puts, 60-90 DTE.
5. Strike: maximize convexity per dollar (gamma×multiplier/premium).
6. Sizing: 0.5-1% of portfolio per month on hedges.
7. Backtest both: 2015-2024.
**Accept:** Earnings: positive avg P&L per event. Tail hedge: large positive P&L in 2020-03, overall cost reasonable.

---

### T-0229 · Risk Parity Strategy
**Files:** `quantcore/strategies/macro/risk_parity.py`, `config/strategies/risk_parity.toml`
**Deps:** T-0221, T-0325
**Steps:**
1. Asset classes: US equity (SPY), US bonds (TLT), commodities (DBC), gold (GLD), TIPS (TIP).
2. Equal risk contribution weights (inverse vol, then optimize for exact ERC).
3. Leverage to target 10% vol.
4. Monthly rebalance.
5. Regime overlay: reduce leverage 50% when HMM detects crisis regime.
6. Backtest: 2005-2024.
**Accept:** Max DD < 60/40 portfolio. Sharpe > 0.5. Vol near target.

---

### T-0230 · CTA Trend + Carry
**Files:** `quantcore/strategies/macro/cta_trend.py`, `quantcore/strategies/macro/carry.py`
**Deps:** T-0201, T-0221
**Steps:**
1. CTA: 50+ instruments (use ETF proxies). Multi-speed EWMA trend signals.
2. EWMA vol scaling per instrument. Correlation-adjusted portfolio (cap at 3 correlated positions).
3. Carry: FX (interest rate diff), commodity (roll yield), equity (div yield - financing).
4. Equal-risk-weighted combination.
5. Backtest both: 2005-2024.
**Accept:** CTA positive in 2008. Carry positive in calm markets, negative in crises. Combined Sharpe > 0.4.
