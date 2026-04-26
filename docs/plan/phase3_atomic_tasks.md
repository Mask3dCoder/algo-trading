# Phase 3 — Risk, Execution, Portfolio, Regime (Atomic Tasks)

---

### T-0301 · Historical Simulation VaR
**Files:** `quantcore/risk/var.py`
**Deps:** T-0106
**Steps:**
1. `historical_var(returns, confidence=0.95, lookback=252) → float`: percentile of historical P&L
2. Support multiple horizons (1d, 10d via √T or direct simulation)
3. Lookback options: 1Y, 2Y, 5Y
4. Confidence levels: 95%, 99%, 99.5%, 99.9%
5. Test: VaR of N(0,1) at 95% ≈ 1.645. VaR of actual portfolio > 0.
**Accept:** Correct percentile calculation. 10-day VaR > 1-day VaR. Matches manual spreadsheet.

---

### T-0302 · Parametric + Cornish-Fisher VaR
**Files:** `quantcore/risk/var.py` (append)
**Deps:** T-0301
**Steps:**
1. `parametric_var(mu, sigma, confidence) → float`: Gaussian VaR
2. `ewma_covariance(returns, lambda_=0.94) → matrix`: RiskMetrics-style
3. `portfolio_parametric_var(weights, cov, confidence) → float`
4. `cornish_fisher_var(mu, sigma, skew, kurt, confidence) → float`: adjust for non-normality
5. Test: parametric = historical for Normal data. CF > parametric for fat-tailed data.
**Accept:** Parametric VaR matches formula. CF adjustment increases VaR when kurtosis > 3.

---

### T-0303 · Monte Carlo VaR + Filtered Historical
**Files:** `quantcore/risk/var.py` (append)
**Deps:** T-0129, T-0136, T-0301
**Steps:**
1. `monte_carlo_var(returns, n_scenarios=100000, confidence=0.95) → (var, std_error)`: simulate from fitted distribution
2. `filtered_historical_var(returns, confidence) → float`: GARCH-filter returns, historical sim on standardized residuals, re-scale
3. Support copula-based MC for multi-asset portfolios
4. Test: MC VaR converges to parametric for Normal data. Filtered > unfiltered in high-vol regime.
**Accept:** MC VaR within 5% of analytical for Normal. 100K scenarios run in <10s. Filtered captures vol clustering.

---

### T-0304 · CVaR (Expected Shortfall)
**Files:** `quantcore/risk/var.py` (append)
**Deps:** T-0301
**Steps:**
1. `cvar(returns, confidence=0.975) → float`: average loss beyond VaR
2. `marginal_cvar(returns, weights, confidence) → array`: ∂CVaR/∂w for each position
3. `component_cvar(returns, weights, confidence) → array`: position contributions (sum = portfolio CVaR)
4. `incremental_cvar(returns, weights, new_trade, confidence) → float`: impact of adding trade
5. Test: CVaR > VaR always. Component CVaR sums to portfolio CVaR.
**Accept:** CVaR at 97.5% matches Basel III FRTB methodology. Components sum correctly (< 1% error).

---

### T-0305 · VaR Backtesting
**Files:** `quantcore/risk/var_backtest.py`
**Deps:** T-0301
**Steps:**
1. `kupiec_pof_test(violations, n_obs, confidence) → (stat, pvalue)`: proportion of failures
2. `christoffersen_test(violations) → (stat, pvalue)`: conditional coverage (independence)
3. `traffic_light(violations, n_obs) → "green"|"yellow"|"red"`: Basel zone classification
4. `var_backtest_report(returns, var_series) → BacktestReport`: violations, p-values, chart
5. Test: 95% VaR on Normal data has ~5% violation rate. Traffic light is green.
**Accept:** Kupiec test does not reject correct VaR model (p > 0.05). Traffic light green for well-calibrated VaR.

---

### T-0306 · Historical Stress Scenarios
**Files:** `quantcore/risk/stress.py`
**Deps:** T-0106
**Steps:**
1. `ScenarioLibrary`: dict of named scenarios, each = dict of asset_class → shock (e.g., "GFC_2008": {"equity": -0.40, "vol": +3.0, "credit_spread": +0.05})
2. Scenarios: GFC 2008, COVID 2020, Dot-com 2000, Black Monday 1987, LTCM 1998, Rate Shock 2022
3. `apply_scenario(portfolio, scenario) → StressResult`: compute shocked portfolio value, P&L, new Greeks
4. `stress_report(portfolio, scenarios) → DataFrame`: all scenarios in one table
5. Test: long-only equity portfolio loses ~40% under GFC scenario.
**Accept:** All 6 scenarios defined with reasonable shocks. Application produces correct P&L for simple portfolios.

---

### T-0307 · Hypothetical + Factor Stress Tests
**Files:** `quantcore/risk/stress.py` (append)
**Deps:** T-0306
**Steps:**
1. `custom_scenario(shocks: dict) → Scenario`: user-defined shocks (VIX +30, SPX -20%, etc.)
2. `correlation_stress(portfolio, target_corr=0.8) → StressResult`: set all pairwise correlations to target
3. `factor_stress(portfolio, factor_exposures, factor_shock) → StressResult`: e.g., momentum factor -3σ
4. `liquidity_stress(portfolio, spread_multiplier=5) → StressResult`: widen all spreads
5. Test: correlation stress produces larger loss than normal-correlation scenario.
**Accept:** Custom scenarios are composable. Factor stress correctly uses portfolio's factor loadings.

---

### T-0308 · Reverse Stress Testing
**Files:** `quantcore/risk/stress.py` (append)
**Deps:** T-0306
**Steps:**
1. `reverse_stress(portfolio, loss_threshold=-0.20) → list[Scenario]`: find minimal scenarios causing threshold loss
2. Optimization: minimize Σ|shock_i|² subject to portfolio_loss ≤ threshold
3. Return top-5 most plausible scenarios
4. `concentration_report(reverse_results) → str`: identify which positions/factors are most vulnerable
5. Test: reverse stress of concentrated portfolio identifies the concentrated position.
**Accept:** Optimization converges. Identified scenarios actually produce the threshold loss when applied.

---

### T-0309 · Greeks Aggregation
**Files:** `quantcore/risk/greeks.py`
**Deps:** T-0130
**Steps:**
1. `aggregate_greeks(positions) → GreeksSnapshot`: net delta, gamma, vega, theta per underlying and total
2. `delta_adjusted_exposure(stock_positions, option_positions) → dict`: total delta per underlying
3. `gamma_by_tenor(option_positions, buckets=[7,30,90]) → dict`: gamma in each DTE bucket
4. `vega_by_tenor(option_positions, buckets) → dict`
5. `greeks_stress(positions, spot_shocks=[-3,-2,-1,0,1,2,3]) → DataFrame`: Greeks at each shock level
6. Test: delta of 100 shares = 100. Gamma of stock = 0. ATM straddle delta ≈ 0.
**Accept:** Aggregation correct for mixed stock+options portfolio. Stress table shows gamma effect on delta.

---

### T-0310 · Risk Limits + Circuit Breakers
**Files:** `quantcore/risk/limits.py`
**Deps:** T-0309, T-0304
**Steps:**
1. `RiskLimits`: from config — max_position_pct_adv, max_dollar_exposure, max_net_delta, max_gross_gamma, max_vega, max_sector_pct, max_concentration, max_var_utilization
2. `check_pre_trade(order, portfolio, limits) → Approved|Rejected(reason)`: check all limits before order
3. `check_post_trade(portfolio, limits) → list[Breach]`: check all limits after fill
4. `CircuitBreaker`: if any limit breached by >20%, auto-flatten that position/strategy
5. Limits configurable in `config/default.toml` under `[risk_limits]`
6. Test: order exceeding position limit rejected. Circuit breaker fires on 25% over-limit.
**Accept:** Pre-trade check runs in <1ms. All limit types enforced. Circuit breaker logs and flattens correctly.

---

### T-0311 · Drawdown Management
**Files:** `quantcore/risk/drawdown.py`
**Deps:** T-0106
**Steps:**
1. `DrawdownTracker`: running peak, current drawdown, duration, max drawdown
2. `update(nav) → DrawdownState`: update with new NAV, return current state
3. Threshold actions: -5% → reduce 25%, -10% → reduce 50% + alert, -15% → reduce 75%, -20% → flatten + halt
4. `drawdown_at_risk(returns, confidence=0.95) → float`: probabilistic max DD estimate
5. `conditional_dar(returns, confidence) → float`
6. `recovery_estimate(current_dd, avg_return) → int`: expected days to recover
7. Test: tracker correctly identifies max DD on known equity curve. Thresholds trigger at right levels.
**Accept:** Max DD matches manual calculation. Threshold actions fire correctly. DaR > max historical DD (conservative).

---

### T-0312 · Risk-Adjusted Metrics
**Files:** `quantcore/risk/metrics.py`
**Deps:** T-0106
**Steps:**
1. `sharpe(returns, rf=0) → float`, `sortino(returns, mar=0)`, `calmar(returns)`
2. `omega(returns, threshold=0)`, `information_ratio(returns, benchmark)`, `treynor(returns, beta, rf)`
3. `jensens_alpha(returns, market_returns, rf)`, `m_squared(returns, market_returns, rf)`
4. `ulcer_index(returns)`, `pain_index(returns)`, `sterling_ratio(returns)`
5. All metrics: `rolling(returns, metric_fn, window=252) → Series`
6. `bootstrap_ci(returns, metric_fn, n_boot=1000, ci=0.95) → (lower, upper)`
7. Test: Sharpe of known return series = manual calculation. Bootstrap CI contains point estimate.
**Accept:** All metrics match quantstats/empyrical. Bootstrap produces valid confidence intervals.

---

### T-0313 · Risk Reports
**Files:** `quantcore/risk/reports.py`
**Deps:** T-0301, T-0309, T-0311, T-0312
**Steps:**
1. `daily_risk_report(portfolio, market_data) → HTMLReport`: P&L, VaR, Greeks, drawdown, limit utilization
2. `weekly_summary(portfolio, week_returns) → HTMLReport`: performance attribution, factor exposures, regime
3. Charts: equity curve, drawdown, VaR vs actual P&L, Greeks heatmap
4. Export: HTML file with embedded Plotly charts
5. Test: report generates without error for sample portfolio.
**Accept:** Reports render correctly in browser. All sections populated. Charts interactive.

---

### T-0314 · Order Management System
**Files:** `quantcore/execution/oms.py`
**Deps:** T-0106, T-0108
**Steps:**
1. `OrderManager`: manage order lifecycle — new → submitted → partial → filled → cancelled/rejected
2. Order types: market, limit, stop, stop_limit, trailing_stop, bracket (entry+TP+SL), OCO, OTO
3. `submit(order) → order_id`, `cancel(order_id)`, `modify(order_id, new_params)`
4. State machine: valid transitions only (can't fill a cancelled order)
5. Audit trail: every state change logged with timestamp
6. Idempotency: duplicate submissions detected and rejected
7. Test: full lifecycle test (submit → partial fill → full fill). Invalid transitions raise error.
**Accept:** State machine enforces valid transitions. Audit log captures all events. Duplicate rejected.

---

### T-0315 · TWAP Algorithm
**Files:** `quantcore/execution/algos/twap.py`
**Deps:** T-0314
**Steps:**
1. `TWAPAlgo`: split parent order into N child slices over time window
2. Equal-time intervals with ±10% random jitter (anti-detection)
3. Min/max slice size constraints
4. Cancel remaining slices on timeout or parent cancel
5. Track performance: actual avg price vs TWAP benchmark
6. Test: 1000-share order over 60 min → 10 slices of 100 each. Timing within jitter bounds.
**Accept:** Slices evenly distributed. Jitter randomizes timing. Partial cancellation works correctly.

---

### T-0316 · VWAP Algorithm
**Files:** `quantcore/execution/algos/vwap.py`
**Deps:** T-0314
**Steps:**
1. `VWAPAlgo`: use historical volume profile (30-min buckets) to size child orders
2. `load_volume_profile(symbol) → dict[time_bucket, pct_volume]`: from historical data
3. Size each slice proportional to expected volume in that bucket
4. Track: actual fill VWAP vs market VWAP
5. Intraday adjustment: if actual volume deviates from profile, adjust remaining slices
6. Test: more volume during open/close, less midday. VWAP performance ≈ market VWAP.
**Accept:** Volume-profile-weighted slicing correct. VWAP performance within 5bps of benchmark on backtest.

---

### T-0317 · POV Algorithm
**Files:** `quantcore/execution/algos/pov.py`
**Deps:** T-0314
**Steps:**
1. `POVAlgo`: trade at fixed percentage of real-time market volume
2. `participation_rate`: configurable (e.g., 5% of volume)
3. Monitor real-time volume, compute target shares for current interval
4. Urgency parameter: scale participation rate up/down
5. Test: at 10% participation, fills 10% of total market volume.
**Accept:** Actual participation rate within 2% of target. Completes within time window.

---

### T-0318 · Implementation Shortfall (Almgren-Chriss)
**Files:** `quantcore/execution/algos/implementation_shortfall.py`
**Deps:** T-0314, T-0319
**Steps:**
1. `AlmgrenChriss`: minimize E[cost] + λ·Var[cost] for linear impact model
2. Analytical optimal trajectory: closed-form for linear temporary + permanent impact
3. `compute_trajectory(total_shares, T, sigma, eta, gamma, lambda_) → schedule`
4. Incorporate alpha decay: if signal decays, front-load execution
5. Dynamic adjustment: recalculate trajectory based on actual fills
6. Test: higher λ (risk aversion) → more front-loading. Zero impact → uniform trajectory (= TWAP).
**Accept:** Trajectory matches Almgren-Chriss analytical solution. Alpha decay front-loads correctly.

---

### T-0319 · Market Impact Models
**Files:** `quantcore/execution/impact.py`
**Deps:** T-0106
**Steps:**
1. `sqrt_impact(shares, adv, sigma) → bps`: ΔP/P = σ × √(Q/ADV)
2. `linear_impact(shares, adv, kyle_lambda) → bps`
3. `amihud_illiquidity(returns, volume) → float`: |return|/dollar_volume
4. `roll_spread(returns) → float`: estimated bid-ask from autocov
5. `predict_cost(order, model="sqrt") → CostEstimate`: total expected cost in bps
6. Test: impact increases with order size. Zero order → zero impact.
**Accept:** Impact model calibrates to historical data. Predicted cost within 50% of actual TCA.

---

### T-0320 · Transaction Cost Analysis
**Files:** `quantcore/execution/tca.py`
**Deps:** T-0319
**Steps:**
1. `tca_trade(fills, arrival_price) → TCAResult`: total cost = arrival_price − avg_fill_price
2. Decompose: spread_cost + impact_cost + timing_cost + opportunity_cost
3. `tca_strategy(fills, strategy_id) → StrategySummary`: aggregate TCA by strategy
4. `tca_report(fills, period) → HTMLReport`: charts of cost by strategy, venue, time_of_day
5. Test: buying at market → positive cost. Limit order at bid → negative spread cost.
**Accept:** Cost decomposition sums to total cost. Report generates correctly.

---

### T-0321 · Abstract Broker + Paper Broker
**Files:** `quantcore/execution/brokers/base.py`, `quantcore/execution/brokers/paper.py`
**Deps:** T-0314, T-0143
**Steps:**
1. `BrokerAdapter(ABC)`: `submit_order()`, `cancel_order()`, `get_positions()`, `get_account()`, `stream_fills(callback)`
2. Standardized models: map broker-specific to internal `Order`/`Fill`/`Position`
3. `PaperBroker(BrokerAdapter)`: in-memory matching using backtest simulation models
4. Configurable latency (default 50ms), commission model, slippage model
5. Simulated margin, buying power, account value
6. Test: submit market order → fill callback within latency. Position updates correctly.
**Accept:** Paper broker simulates realistic trading. Account balance tracks correctly. Short selling works.

---

### T-0322 · Alpaca Broker Adapter
**Files:** `quantcore/execution/brokers/alpaca.py`
**Deps:** T-0321
**Steps:**
1. `AlpacaBroker(BrokerAdapter)`: REST API for orders + WebSocket for updates
2. `submit_order`: map internal Order → Alpaca order, handle response
3. `cancel_order`, `get_positions`, `get_account`: implement all
4. Stream fills via WebSocket trade_updates
5. Paper mode: use `paper-api.alpaca.markets` endpoint
6. Rate limiting, retry with backoff
7. Test: submit/cancel on paper account. Positions match after fills.
**Accept:** Full order lifecycle works on Alpaca paper. Positions reconcile correctly.

---

### T-0323 · Interactive Brokers Adapter
**Files:** `quantcore/execution/brokers/ibkr.py`
**Deps:** T-0321
**Steps:**
1. `IBKRBroker(BrokerAdapter)`: via `ibapi` (TWS API)
2. Connection management: connect to TWS/Gateway, handle disconnects, auto-reconnect
3. Order placement: stocks and options (including combo orders)
4. Real-time position, P&L, account streaming
5. Margin and buying power queries
6. Test: connect to IB paper trading, submit/cancel order.
**Accept:** Connects to TWS paper. Orders placed and confirmed. Positions stream correctly.

---

### T-0324 · CCXT Crypto Adapter
**Files:** `quantcore/execution/brokers/ccxt_adapter.py`
**Deps:** T-0321
**Steps:**
1. `CCXTBroker(BrokerAdapter)`: unified CCXT interface
2. Support Binance, Coinbase, Kraken (configurable)
3. Spot and futures trading
4. WebSocket for fills and position updates
5. Rate limiting per exchange
6. Test: place small limit order on testnet/sandbox.
**Accept:** Orders work on at least one exchange's testnet. Balance and position tracking correct.

---

### T-0325 · Position Sizing Methods
**Files:** `quantcore/portfolio/sizing.py`
**Deps:** T-0106
**Steps:**
1. `kelly_criterion(win_prob, avg_win, avg_loss) → fraction`: f* = (bp - q) / b
2. `fractional_kelly(kelly_f, fraction=0.25) → float`
3. `volatility_target(asset_vol, target_vol, n_assets) → weight`
4. `risk_parity_size(vols, correlations, target_vol) → weights`: equal risk contribution
5. `atr_size(account_value, risk_pct, atr) → shares`: (account × risk%) / ATR
6. `cvar_size(returns, max_cvar, account_value) → max_position`
7. Test: full Kelly is aggressive. Fractional Kelly < full Kelly. Vol target produces target vol.
**Accept:** Kelly formula correct. Vol-targeted portfolio has realized vol within 20% of target.

---

### T-0326 · MVO + Black-Litterman
**Files:** `quantcore/portfolio/optimization.py`
**Deps:** T-0325
**Steps:**
1. `mean_variance(returns, target="max_sharpe"|"min_variance"|"target_return") → weights`: Markowitz
2. Constraints: long-only, box, sector, turnover
3. `black_litterman(market_caps, cov, views, view_confidence) → posterior_returns`
4. `bl_optimize(posterior_returns, cov) → weights`
5. Efficient frontier: `compute_frontier(returns, n_points=50) → list[(risk, return, weights)]`
6. Test: 2-asset MVO matches analytical. BL with no views = market-cap weights.
**Accept:** Optimizer produces valid weights (sum=1, within constraints). BL posterior shifts toward views.

---

### T-0327 · HRP + NCO + Risk Parity
**Files:** `quantcore/portfolio/optimization.py` (append)
**Deps:** T-0326
**Steps:**
1. `hrp(returns) → weights`: hierarchical clustering (Ward) → quasi-diagonalization → recursive bisection
2. `nco(returns, n_clusters) → weights`: cluster → intra-cluster optimize → inter-cluster optimize
3. `risk_parity(cov, target_vol=None) → weights`: scipy root finding for ERC
4. Stability test: perturb covariance by 10%, measure weight change (HRP < MVO)
5. Test: HRP weights sum to 1. Risk contributions approximately equal for risk parity.
**Accept:** HRP more stable than MVO under covariance perturbation. Risk parity contributions within 10% of equal.

---

### T-0328 · Rebalance Engine
**Files:** `quantcore/portfolio/rebalance.py`
**Deps:** T-0326
**Steps:**
1. `RebalanceEngine`: trigger = calendar (daily/weekly/monthly) OR drift (weight drift > threshold)
2. `compute_trades(current_weights, target_weights, portfolio_value) → list[Order]`
3. Cost-aware: only trade if |drift| × value > min_trade_value AND benefit > estimated cost
4. Turnover budget: max turnover per rebalance = configurable %
5. Test: rebalance correctly computes trades. Cost filter prevents small trades.
**Accept:** Trades bring weights within tolerance of target. Turnover stays within budget.

---

### T-0329 · Tax-Lot Tracking
**Files:** `quantcore/portfolio/tax.py`
**Deps:** T-0106
**Steps:**
1. `TaxLotTracker`: track each purchase lot (date, cost, shares)
2. Methods: FIFO, LIFO, specific identification
3. `realize(symbol, shares, method="FIFO") → list[TaxLot]`: select lots, compute gain/loss
4. `short_vs_long(lots) → (short_term_gain, long_term_gain)`: by holding period
5. `harvest_candidates(positions, threshold=-100) → list`: positions with unrealized losses > threshold
6. Wash sale detection: flag if repurchased within 30 days
7. Test: FIFO sells oldest lot first. Wash sale detected correctly.
**Accept:** Lot tracking matches manual calculation. Wash sale rule correctly applied.

---

### T-0330 · HMM Regime Detection
**Files:** `quantcore/regime/hmm.py`, `quantcore/regime/features.py`
**Deps:** T-0136
**Steps:**
1. `RegimeHMM(n_states)`: fit Gaussian HMM (hmmlearn) on returns or features
2. 2-state (bull/bear), 3-state (bull/bear/sideways), 4-state (trend/mean-revert/high-vol/low-vol)
3. `predict_regime(data) → (state, probabilities)`: online Viterbi
4. `smoothed_probabilities(data) → DataFrame`: forward-backward
5. `features.py`: compute regime features — realized vol, cross-asset correlation, ADX, Hurst, breadth, credit spread, yield curve slope
6. Model selection: AIC/BIC for number of states
7. Test: 2-state HMM on synthetic regime-switching data recovers states >80%.
**Accept:** HMM detects 2008/2020 as different regime within 5 days of onset. State persistence > 20 days average.

---

### T-0331 · Changepoint + Vol Regimes
**Files:** `quantcore/regime/changepoint.py`, `quantcore/regime/volatility_regime.py`
**Deps:** T-0136
**Steps:**
1. `pelt_changepoints(series, penalty) → list[int]`: via `ruptures` library
2. `bocpd(series) → run_length_distribution`: Bayesian Online Changepoint Detection
3. `cusum(series, threshold, drift) → list[int]`: CUSUM control chart
4. `vix_regime(vix_level) → str`: <15=low, 15-25=normal, 25-35=elevated, >35=crisis
5. `garch_regime(returns) → regime_series`: Markov-switching GARCH
6. `vol_percentile_regime(realized_vol, lookback=252) → regime`: percentile-based
7. Test: PELT detects known structural break. VIX regime labels match intuition for 2020.
**Accept:** PELT finds true changepoints in synthetic data (precision > 80%). Vol regimes consistent across methods.
