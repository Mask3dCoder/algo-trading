# Phase 3 — Risk, Execution & Portfolio Tasks

**Depends on:** Phase 2 complete (signals, strategies, options pricing)

---

## AGENT-P3-01: VaR / CVaR Engine

**Goal:** Multi-method Value-at-Risk and Expected Shortfall computation.

### Tasks
1. Implement `quantcore/risk/var.py`:
   - **Historical Simulation VaR**: configurable lookback (1Y, 2Y, 5Y), confidence levels (95%, 99%, 99.5%, 99.9%), horizons (1d, 10d, 1M)
   - **Parametric VaR**: variance-covariance method with exponentially weighted covariance (λ=0.94 RiskMetrics)
   - **Monte Carlo VaR**: generate 100K portfolio return scenarios from fitted distribution + copula, compute percentile loss
   - **Cornish-Fisher VaR**: adjust Normal VaR for skewness and kurtosis
   - **Filtered Historical Simulation**: GARCH-filter returns, then historical simulation on standardized residuals
   - Scaling: √T scaling for parametric, direct simulation for MC
2. Implement CVaR / Expected Shortfall:
   - CVaR at 97.5% (Basel III FRTB standard)
   - **Marginal CVaR**: sensitivity of portfolio CVaR to each position weight
   - **Component CVaR**: decompose portfolio CVaR into position contributions (sum = portfolio CVaR)
   - **Incremental CVaR**: impact of adding a proposed trade
3. Implement `quantcore/risk/var_backtest.py`:
   - Kupiec POF test (proportion of failures)
   - Christoffersen conditional coverage test (independence of violations)
   - Traffic light test (Basel green/yellow/red zones)
   - VaR violation report with P&L on breach days
4. All methods operate on `PortfolioSnapshot` → return `RiskResult` pydantic model
5. Write tests: VaR of Normal portfolio matches analytical. VaR backtesting on historical data produces expected violation rates.

**Acceptance:** MC VaR converges within 1% of analytical for Normal case. Backtest shows correct violation frequency (≈5% for 95% VaR).

---

## AGENT-P3-02: Stress Testing

**Goal:** Historical, hypothetical, and reverse stress testing framework.

### Tasks
1. Implement `quantcore/risk/stress.py`:
2. **Historical scenario library** — store as config:
   - 2008 GFC (Sep-Nov 2008): equities -40%, vol +300%, credit spreads +500bps, correlations →1
   - COVID crash (Feb-Mar 2020): equities -34%, vol +400%, oil -60%
   - Dot-com bust (2000-2002): tech -80%, broad market -50%
   - 1987 Black Monday: -22% single day
   - LTCM/Russia 1998: EM -50%, spreads blow out
   - 2022 rate shock: bonds -20%, equities -25%, crypto -70%
3. **Hypothetical scenarios** (configurable shocks):
   - VIX +30 points
   - S&P500 -20% over 5 days
   - Yield curve inversion (10Y-2Y → -100bps)
   - Correlation spike (all pairwise correlations → 0.8)
   - Liquidity crisis (spreads 5× normal)
   - Sector rotation (tech -30%, utilities +10%)
4. **Factor stress tests**:
   - Momentum factor crash (-3σ)
   - Value factor crash
   - Volatility regime change (low → high)
   - Apply factor shocks via portfolio factor exposures
5. **Reverse stress testing**:
   - Given loss threshold (e.g., -20%), find scenarios that produce it
   - Optimization: minimize scenario "severity" subject to loss ≥ threshold
   - Report most plausible paths to catastrophic loss
6. Scenario application engine: take current portfolio + scenario → compute P&L, new Greeks, margin impact
7. Write tests: stress test of a long-only equity portfolio under 2008 scenario produces ~-40% loss

**Acceptance:** All historical scenarios reproduce approximate actual losses. Reverse stress test identifies correct risk concentrations.

---

## AGENT-P3-03: Greeks & Position-Level Risk

**Goal:** Real-time aggregated Greeks and position limits.

### Tasks
1. Implement `quantcore/risk/greeks.py`:
   - Aggregate delta, gamma, vega, theta across all options positions per underlying
   - Net delta across stocks + options (delta-adjusted)
   - Gamma by tenor bucket (0-7 DTE, 7-30 DTE, 30-90 DTE, 90+ DTE)
   - Vega by tenor bucket
   - Cross-gamma (multi-underlying)
   - Theta P&L forecast (daily time decay)
   - Greeks stress: recompute under ±1σ, ±2σ, ±3σ spot moves
2. Implement `quantcore/risk/limits.py`:
   - Position limits: max % of ADV, max dollar exposure per name, max loss per position
   - Portfolio limits: max net delta, max gross gamma, max vega, max sector exposure
   - Concentration limits: max % of portfolio in single name, sector, asset class
   - Correlation limits: reject trades that push correlation above threshold
   - Margin utilization limit (% of buying power)
   - **Circuit breakers**: auto-flatten if any limit breached by >20%
   - Pre-trade limit check: `check_order(order) → Approved | Rejected(reason)`
   - Post-trade limit recheck on every fill
3. Implement limit configuration in `config/default.toml` under `[risk_limits]`
4. Write tests: limit checker correctly rejects oversized orders, circuit breaker fires on breach

**Acceptance:** Pre-trade checks execute in <1ms. Circuit breaker correctly triggers on simulated limit breaches.

---

## AGENT-P3-04: Drawdown Management & Risk Metrics

**Goal:** Drawdown monitoring with automatic risk reduction, plus full metrics dashboard.

### Tasks
1. Implement `quantcore/risk/drawdown.py`:
   - Real-time drawdown tracking (peak-to-trough)
   - Maximum drawdown calculation (running)
   - Drawdown duration tracking
   - **Drawdown thresholds** with automatic actions:
     - -5%: reduce position sizes by 25%
     - -10%: reduce by 50%, alert
     - -15%: reduce by 75%, escalate
     - -20%: flatten all positions, halt trading
   - Drawdown-at-Risk (DaR): probabilistic max drawdown estimate
   - Conditional DaR (CDaR)
   - Calmar ratio monitoring (return / max DD, rolling)
   - Recovery time estimation
2. Implement `quantcore/risk/metrics.py` — compute all risk-adjusted metrics:
   - Sharpe, Sortino (MAR=0 and MAR=Rf), Calmar, Omega (threshold=0)
   - Information Ratio (vs benchmark), Treynor Ratio, Jensen's Alpha
   - M² (Modigliani), Sterling Ratio, Burke Ratio
   - Ulcer Index, Pain Index, Pain Ratio
   - Skewness, Kurtosis, Jarque-Bera normality test
   - Max DD, Avg DD, DD duration stats
   - All metrics with rolling windows (63d, 126d, 252d)
   - Bootstrap confidence intervals (1000 resamples)
3. Implement `quantcore/risk/reports.py`:
   - Daily risk report: P&L, VaR, Greeks, drawdown, limit utilization
   - Weekly summary: performance attribution, factor exposures, regime
   - HTML report generation with Plotly charts
4. Write tests: metrics match known values (e.g., Sharpe of S&P500 ≈ 0.4-0.6 historically)

**Acceptance:** Drawdown circuit breakers fire correctly in backtest simulation. All metrics match quantlib/empyrical implementations.

---

## AGENT-P3-05: Execution Algorithms

**Goal:** TWAP, VWAP, Implementation Shortfall, and adaptive execution.

### Tasks
1. Implement `quantcore/execution/oms.py`:
   - Order Management System: order lifecycle (new → submitted → partial → filled → cancelled)
   - Order types: market, limit, stop, stop-limit, trailing stop, MIT, iceberg, pegged
   - Bracket orders (entry + take-profit + stop-loss)
   - OCO (one-cancels-other), OTO (one-triggers-other)
   - Order state machine with audit trail
   - Idempotent order submission (prevent duplicates)
2. Implement `quantcore/execution/algos/twap.py`:
   - Slice parent order evenly over time window
   - Randomize child order timing (±10% jitter)
   - Min/max slice size constraints
   - Cancel remaining on timeout
3. Implement `quantcore/execution/algos/vwap.py`:
   - Historical volume profile by time-of-day (30min buckets)
   - Size participation to match volume curve
   - Track performance vs actual VWAP benchmark
   - Intraday volume forecast adjustment
4. Implement `quantcore/execution/algos/pov.py`:
   - Percentage-of-Volume: trade at fixed % of real-time volume
   - Dynamic rate adjustment based on urgency parameter
   - Volume detection from real-time feed
5. Implement `quantcore/execution/algos/implementation_shortfall.py`:
   - Almgren-Chriss optimal execution framework
   - Minimize: E[cost] + λ·Var[cost]
   - Analytical solution for linear temporary + permanent impact
   - Incorporate alpha decay (front-load if alpha decays fast)
   - Dynamic trajectory adjustment
6. Implement `quantcore/execution/impact.py`:
   - Square-root impact model: ΔP = σ · √(Q/ADV)
   - Kyle's Lambda estimation from trade data
   - Amihud illiquidity measure
   - Roll bid-ask spread estimator
   - Predicted cost for a given order
7. Implement `quantcore/execution/tca.py`:
   - Transaction Cost Analysis: arrival price vs fill price
   - Slippage breakdown: spread + impact + timing
   - Cost per strategy, per venue, per time-of-day
   - TCA report generation
8. Write tests: TWAP produces uniform distribution. VWAP participation matches volume profile. Impact model produces reasonable costs.

**Acceptance:** Execution algos correctly slice orders. TCA shows improvement over naive market orders. Impact model calibrates to historical data.

---

## AGENT-P3-06: Broker Integration

**Goal:** Production broker connections for paper and live trading.

### Tasks
1. Define `quantcore/execution/brokers/base.py`:
   - Abstract `BrokerAdapter`: `submit_order()`, `cancel_order()`, `get_positions()`, `get_account()`, `stream_fills()`
   - Standardized order/fill/position models (map broker-specific to internal)
2. Implement `quantcore/execution/brokers/alpaca.py`:
   - REST API: orders, positions, account
   - WebSocket: order updates, trade confirmations
   - Paper trading mode (use paper endpoint)
   - Fractional shares support
   - Rate limiting and retry logic
3. Implement `quantcore/execution/brokers/ibkr.py`:
   - IB TWS/Gateway API via `ibapi`
   - Connection management (reconnect on drop)
   - Order placement, modification, cancellation
   - Real-time position and P&L streaming
   - Options order support (combo orders)
   - Margin and buying power queries
4. Implement `quantcore/execution/brokers/ccxt_adapter.py`:
   - CCXT unified API for crypto exchanges
   - Support: Binance, Coinbase, Kraken (configurable)
   - Spot and futures trading
   - WebSocket for fills and positions
5. Implement `quantcore/execution/brokers/paper.py`:
   - In-memory simulated broker for testing
   - Realistic fill simulation (use backtest fill models)
   - Simulated account with margin
   - Latency simulation (configurable delay)
6. Write integration tests: submit/cancel order lifecycle on paper accounts

**Acceptance:** Can place, modify, cancel orders on each broker's paper/sandbox. Position reconciliation matches broker's reported positions.

---

## AGENT-P3-07: Portfolio Construction & Optimization

**Goal:** Full position sizing and portfolio optimization engine.

### Tasks
1. Implement `quantcore/portfolio/sizing.py`:
   - Kelly Criterion: `f* = (bp - q) / b` with edge and odds
   - Fractional Kelly (configurable fraction, default 0.25)
   - Volatility targeting: `weight = target_vol / (asset_vol * √n_assets)`
   - Risk parity sizing: equal risk contribution via optimization
   - CVaR sizing: constrain position so incremental CVaR < limit
   - ATR-based sizing: `shares = (account × risk%) / ATR`
   - Optimal-f (Vince): maximize geometric growth rate
2. Implement `quantcore/portfolio/optimization.py`:
   - Markowitz MVO: min variance, max Sharpe, efficient frontier
   - Black-Litterman: equilibrium + views → posterior → optimize
   - HRP: hierarchical clustering → quasi-diag → recursive bisection
   - NCO: nested clusters → intra-cluster optimization → inter-cluster
   - Risk Parity: scipy root finding for ERC weights
   - Max Diversification: maximize diversification ratio
   - CVaR optimization via CVXPY (linear programming)
   - Robust optimization: uncertainty sets on expected returns
   - All with constraints: long-only, box, sector, turnover, cardinality
3. Implement `quantcore/portfolio/rebalance.py`:
   - Rebalance trigger: calendar (daily/weekly/monthly), threshold (drift > X%)
   - Transaction-cost-aware rebalancing (only trade if benefit > cost)
   - Multi-period optimization (look-ahead rebalancing)
   - Turnover budget management
4. Implement `quantcore/portfolio/tax.py`:
   - Tax-lot tracking (FIFO, LIFO, specific identification)
   - Tax-loss harvesting: identify losses to realize, wash sale rule compliance
   - Short-term vs long-term capital gains tracking
5. Write tests: MVO on 2-asset case matches analytical. HRP weights more stable than MVO under covariance perturbation.

**Acceptance:** All optimizers produce valid weights. HRP demonstrably more stable than MVO in Monte Carlo stability test.

---

## AGENT-P3-08: Market Regime Detection

**Goal:** Statistical regime identification for strategy adaptation.

### Tasks
1. Implement `quantcore/regime/hmm.py`:
   - 2-state HMM (bull/bear) with Gaussian emissions on returns
   - 3-state HMM (bull/bear/sideways)
   - 4-state HMM (trending/mean-reverting/high-vol/low-vol)
   - Student-t emissions for fat tails
   - Online Viterbi for real-time state estimation
   - Forward-backward for smoothed state probabilities
   - Model selection: AIC/BIC for number of states
   - Calibration on configurable lookback window
2. Implement `quantcore/regime/features.py`:
   - Volatility level and trend (GARCH forecast vs realized)
   - Cross-asset correlation regime (rolling correlation, DCC)
   - Trend strength: ADX, Hurst exponent
   - Liquidity: bid-ask spread, turnover ratio
   - Market breadth: advance/decline ratio, new highs vs lows
   - Credit conditions: HY-IG spread, TED spread
   - Yield curve: slope (10Y-2Y), curvature (2×5Y - 2Y - 10Y)
3. Implement `quantcore/regime/changepoint.py`:
   - PELT algorithm (via `ruptures` library)
   - Bayesian Online Changepoint Detection (BOCPD) — implement from paper
   - CUSUM control chart
   - Sequential Probability Ratio Test (SPRT)
4. Implement `quantcore/regime/volatility_regime.py`:
   - VIX-based regimes: <15 (low), 15-25 (normal), 25-35 (elevated), >35 (crisis)
   - GARCH regime (Markov-switching GARCH via `statsmodels`)
   - Realized vol regime (rolling percentile rank)
5. Implement regime-strategy mapping in config:
   - Which strategies to run/pause per regime
   - Position size multiplier per regime
   - Risk limit adjustment per regime
6. Write tests: HMM correctly identifies bull/bear on synthetic regime-switching data. Changepoint detects known structural breaks.

**Acceptance:** HMM assigns >80% probability to correct state on synthetic data. Regime detection identifies 2008/2020 as crisis within 5 trading days of onset.
