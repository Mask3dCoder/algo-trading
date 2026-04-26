# Phase 1B — Math Models, Backtest, Rust

---

### T-0124 · Geometric Brownian Motion
**Files:** `quantcore/math_models/stochastic/gbm.py`
**Deps:** T-0103
**Steps:**
1. `simulate_gbm(S0, mu, sigma, T, dt, n_paths)`: Euler-Maruyama + analytical solution
2. Milstein scheme (higher-order correction term)
3. `estimate_params(prices) → (mu, sigma)`: MLE from log returns
4. Vectorized NumPy (all paths simultaneously)
5. Test: MC call price within 0.5% of BSM analytical for 100K paths
**Accept:** Analytical and Euler converge. MLE recovers known μ,σ from synthetic data within 10%.

---

### T-0125 · Heston Stochastic Volatility
**Files:** `quantcore/math_models/stochastic/heston.py`
**Deps:** T-0124
**Steps:**
1. `simulate_heston(S0, v0, mu, kappa, theta, xi, rho, T, dt, n_paths)`: full truncation scheme
2. Characteristic function (semi-analytical) for European option pricing
3. `calibrate(market_prices, strikes, expiries) → HestonParams`: Levenberg-Marquardt on market options
4. Test: Heston reduces to BSM when ξ→0. Calibration fits market smile within 0.5 vol points.
**Accept:** No negative variance in simulation. Characteristic function matches MC within 1%.

---

### T-0126 · Ornstein-Uhlenbeck Process
**Files:** `quantcore/math_models/stochastic/ou_process.py`
**Deps:** T-0124
**Steps:**
1. `simulate_ou(X0, theta, mu, sigma, T, dt, n_paths)`: exact discretization
2. `analytical_mean_var(X0, theta, mu, sigma, t)`: closed-form moments
3. `estimate_params(series) → OUParams`: MLE (regression-based)
4. `half_life(theta) → float`: `ln(2)/theta`
5. Test: simulated distribution matches analytical mean/variance. Half-life correct for known θ.
**Accept:** MLE recovers θ within 15% on 1000-point synthetic series. Half-life used in pairs trading sizing.

---

### T-0127 · Jump Diffusion + SABR
**Files:** `quantcore/math_models/stochastic/jump_diffusion.py`, `quantcore/math_models/stochastic/sabr.py`
**Deps:** T-0124
**Steps:**
1. Merton jump-diffusion: GBM + compound Poisson (jump intensity λ, jump size ~N(μ_J, σ_J))
2. Simulation: detect jumps via Poisson, apply log-normal jump sizes
3. SABR: `simulate_sabr(F0, sigma0, alpha, beta, rho, T, dt, n_paths)`
4. Hagan formula for implied vol: `implied_vol_sabr(F, K, T, alpha, beta, rho, nu)`
5. Calibration to market smile
6. Test: JD with λ=0 equals GBM. SABR Hagan matches known tabulated values.
**Accept:** Jump diffusion produces fatter tails than GBM (kurtosis > 3). SABR calibration fits ATM vol within 0.1%.

---

### T-0128 · Fractional Brownian Motion
**Files:** `quantcore/math_models/stochastic/fractional_bm.py`
**Deps:** T-0124
**Steps:**
1. `simulate_fbm(H, T, n_steps)`: Cholesky decomposition of covariance matrix
2. Hosking method (faster for long series)
3. `estimate_hurst_rs(series)`: R/S analysis
4. `estimate_hurst_dfa(series)`: Detrended Fluctuation Analysis
5. Test: H=0.5 produces standard BM statistics. H=0.7 shows positive autocorrelation.
**Accept:** Estimated H within 0.05 of true H on 5000-point synthetic series for H ∈ {0.3, 0.5, 0.7}.

---

### T-0129 · Monte Carlo Engine
**Files:** `quantcore/math_models/stochastic/monte_carlo.py`
**Deps:** T-0124
**Steps:**
1. `MonteCarloEngine`: configurable n_paths, n_steps, seed
2. Antithetic variates: pair each path with its mirror
3. Control variates: use correlated known-expectation variable
4. Importance sampling: tilt distribution toward important region
5. Quasi-MC: Sobol sequences via `scipy.stats.qmc.Sobol`
6. `compute_price(payoff_fn, paths) → (price, std_error, ci)`
7. Convergence diagnostic: running mean ± 2σ plot
8. Test: European call price matches BSM. Variance reduction lowers std_error by >2×.
**Accept:** 100K-path MC runs in <2s. Antithetic reduces variance by ~50%. Sobol converges faster than pseudo-random.

---

### T-0130 · Black-Scholes-Merton + All Greeks
**Files:** `quantcore/math_models/options/bsm.py`
**Deps:** T-0103
**Steps:**
1. `bsm_call(S, K, T, r, sigma)`, `bsm_put(...)`: full formula with d1, d2
2. Analytical Greeks: delta, gamma, theta, vega, rho (both call and put)
3. Higher Greeks: vanna (∂²V/∂S∂σ), volga (∂²V/∂σ²), charm (∂Δ/∂t), speed (∂Γ/∂S), color (∂Γ/∂t), ultima (∂³V/∂σ³)
4. `numerical_greek(price_fn, param, epsilon)`: finite difference verification
5. Vectorized: accept arrays for S, K, T
6. Test: match textbook values (Hull Ch.19 examples). Put-call parity: C - P = S - Ke^(-rT).
**Accept:** Prices match to 10 decimal places. All Greeks correct sign. Put-call parity holds to machine precision.

---

### T-0131 · Implied Volatility Solvers
**Files:** `quantcore/math_models/options/implied_vol.py`
**Deps:** T-0130
**Steps:**
1. `implied_vol_newton(market_price, S, K, T, r, is_call)`: Newton-Raphson using vega
2. `implied_vol_brent(...)`: Brent's method fallback (no derivative needed)
3. Rational approximation initial guess (Jaeckel "Let's Be Rational")
4. Edge cases: deep ITM (intrinsic value), deep OTM (near zero vega), near-expiry
5. `implied_vol(...)`: auto-select method, return IV or NaN with warning
6. Test: round-trip: BSM(S,K,T,r,σ) → price → IV(price) → σ. Must match within 1e-8.
**Accept:** Converges in <10 iterations for 99.9% of cases. Round-trip accuracy < 1e-8. Handles σ range [0.01, 5.0].

---

### T-0132 · Volatility Surface
**Files:** `quantcore/math_models/options/vol_surface.py`
**Deps:** T-0131
**Steps:**
1. `SVIParams`: a, b, rho, m, sigma. `svi_total_variance(k, params)`: raw SVI formula
2. `fit_svi(strikes, ivs, forward) → SVIParams`: least squares fit
3. `SSVISurface`: extend SVI across multiple expiries
4. Arbitrage checks: `check_butterfly(surface)`, `check_calendar_spread(surface)`
5. Interpolation: linear in total variance across expiries
6. `plot_surface(surface)`: 3D Plotly visualization
7. Test: fit SVI to synthetic smile, verify no arbitrage. Interpolated IV smooth.
**Accept:** SVI fit RMSE < 0.5% on market-like smile. Arbitrage checks pass. Surface renders in Plotly.

---

### T-0133 · Exotic Options
**Files:** `quantcore/math_models/options/exotics.py`
**Deps:** T-0130, T-0129
**Steps:**
1. `asian_geometric(S, K, T, r, sigma, n)`: analytical (geometric average)
2. `asian_arithmetic(...)`: Monte Carlo (arithmetic average)
3. `barrier_option(S, K, H, T, r, sigma, barrier_type, option_type)`: analytical for simple barriers
4. `american_option_lsm(S, K, T, r, sigma, n_paths, n_steps)`: Longstaff-Schwartz
5. `binary_option(S, K, T, r, sigma, is_call)`: analytical (cash-or-nothing)
6. Test: geometric Asian matches analytical. American put ≥ European put. Binary converges to step function.
**Accept:** Asian geometric matches formula to 6 decimals. LSM American exercise boundary is reasonable. All exotics > 0.

---

### T-0134 · Stationarity Tests
**Files:** `quantcore/math_models/statistics/stationarity.py`
**Deps:** T-0103
**Steps:**
1. `adf_test(series, maxlag) → TestResult`: wrap statsmodels, return (stat, pvalue, critical_values, interpretation)
2. `kpss_test(series) → TestResult`
3. `phillips_perron_test(series) → TestResult`
4. `variance_ratio_test(series, lags) → TestResult`: Lo-MacKinlay
5. `test_stationarity(series) → StationarityReport`: run all, summarize
6. Test: random walk fails ADF (p>0.05). Stationary AR(1) with φ=0.5 passes ADF (p<0.05).
**Accept:** All tests agree on synthetic stationary/nonstationary series. P-values match statsmodels directly.

---

### T-0135 · Hurst + Serial Correlation
**Files:** `quantcore/math_models/statistics/hurst.py`, `quantcore/math_models/statistics/serial_correlation.py`
**Deps:** T-0103
**Steps:**
1. `hurst_rs(series)`, `hurst_dfa(series)`: return H estimate
2. `acf(series, nlags)`, `pacf(series, nlags)`: compute + optional plot
3. `ljung_box(series, lags) → (stat, pvalue)`: serial correlation test
4. `arch_test(series, lags) → (stat, pvalue)`: Engle's LM test for ARCH effects
5. Test: white noise H≈0.5, ACF≈0 at all lags, Ljung-Box p>0.05
**Accept:** Hurst within 0.05 of true value. Ljung-Box detects AR(1) serial correlation. ARCH test detects GARCH(1,1) effects.

---

### T-0136 · GARCH Family
**Files:** `quantcore/math_models/statistics/garch.py`
**Deps:** T-0135
**Steps:**
1. `fit_garch(returns, p=1, q=1) → GARCHResult`: wrap `arch` library, return params + forecasts
2. `fit_egarch(returns)`, `fit_gjr_garch(returns)`, `fit_tgarch(returns)`: asymmetric variants
3. `fit_figarch(returns)`: long-memory
4. `har_rv(realized_vol_series) → HARResult`: HAR-RV with 1d, 5d, 22d components
5. `compare_models(returns) → ModelComparison`: fit all, rank by AIC/BIC
6. `forecast(model, horizon) → VolForecast`: 1-step and multi-step ahead
7. Test: GARCH(1,1) on S&P500 returns has significant params. Forecast is positive.
**Accept:** All models converge. AIC selects correct model on synthetic GARCH data. Forecast variance is always positive.

---

### T-0137 · Cointegration & Pairs
**Files:** `quantcore/math_models/statistics/cointegration.py`
**Deps:** T-0134
**Steps:**
1. `engle_granger(y, x) → CointResult`: two-step (OLS residual → ADF)
2. `johansen_test(data, det_order, k) → JohansenResult`: trace + max eigenvalue
3. `vecm(data, coint_rank, k) → VECMResult`: Vector Error Correction Model
4. `scan_pairs(universe_returns, method="johansen") → list[CointPair]`: test all N*(N-1)/2, filter by significance
5. Rolling cointegration: `rolling_coint(y, x, window) → series of p-values`
6. Test: two synthetic cointegrated series detected. Random series not detected.
**Accept:** Scanner finds known pairs (e.g., KO/PEP) at 1% significance. Runs 500-symbol universe in <5 min.

---

### T-0138 · Factor Models
**Files:** `quantcore/math_models/statistics/factor_models.py`
**Deps:** T-0103
**Steps:**
1. `pca_returns(returns_matrix, n_components) → PCAResult`: eigenvalues, loadings, explained variance, scree plot
2. `ica_returns(returns_matrix, n_components) → ICAResult`
3. `fama_french(returns, factors_df, n_factors=5) → FFResult`: regression, alpha, betas, R²
4. `carhart(returns, factors_df) → FFResult`: 4-factor (add momentum)
5. `download_ff_factors(start, end) → DataFrame`: Kenneth French data library
6. Test: PCA on correlated stocks produces <5 significant components. FF alpha of SPY ≈ 0.
**Accept:** PCA explains >80% variance with first 5 components on 100-stock universe. FF betas have correct signs.

---

### T-0139 · Distribution Fitting + EVT
**Files:** `quantcore/math_models/distributions/fitting.py`, `quantcore/math_models/distributions/evt.py`
**Deps:** T-0103
**Steps:**
1. `fit_distribution(returns, dist_name) → FitResult`: fit Normal, Student-t, Skewed-t, NIG. Return params, loglik, AIC, BIC
2. `compare_distributions(returns) → list[FitResult]`: fit all, rank by AIC
3. `qq_plot(returns, dist)`: QQ plot against fitted distribution
4. `fit_gpd(tail_losses, threshold) → GPDResult`: Generalized Pareto (peaks over threshold)
5. `fit_gev(block_maxima) → GEVResult`: block maxima approach
6. `evt_var(gpd_result, confidence) → float`, `evt_cvar(...)`: tail-based risk measures
7. Test: Student-t beats Normal on S&P500 returns by AIC. GPD fits tail well.
**Accept:** Student-t has lower AIC than Normal on equity returns. EVT VaR > Parametric VaR at 99.9%.

---

### T-0140 · Copulas
**Files:** `quantcore/math_models/distributions/copulas.py`
**Deps:** T-0139
**Steps:**
1. `GaussianCopula`: fit from correlation matrix, simulate, compute PDF/CDF
2. `StudentTCopula`: fit (correlation + dof), captures tail dependence
3. `ClaytonCopula`, `GumbelCopula`, `FrankCopula`: Archimedean family
4. `fit_copula(u, v, family) → CopulaResult`: MLE calibration on uniform marginals
5. `simulate_copula(copula, n_samples) → array`: generate correlated uniform samples
6. `tail_dependence(copula) → (lower, upper)`: compute coefficients
7. Test: Gaussian copula with ρ=0.8 produces ~0.8 rank correlation. t-copula has positive tail dependence.
**Accept:** Simulated samples preserve input correlation structure. Tail dependence > 0 for t-copula, = 0 for Gaussian.

---

### T-0141 · Kalman Filter + MCMC
**Files:** `quantcore/math_models/bayesian/kalman.py`, `quantcore/math_models/bayesian/mcmc.py`
**Deps:** T-0103
**Steps:**
1. `KalmanFilter`: predict/update cycle, `filter(observations) → (states, covariances)`
2. `ExtendedKalmanFilter`: linearize via Jacobian
3. `UnscentedKalmanFilter`: sigma points, no Jacobian needed
4. Application: `dynamic_hedge_ratio(y, x) → series of betas` (time-varying regression)
5. `MetropolisHastings`: `sample(log_posterior, init, n_samples, proposal_std) → chain`
6. `convergence_diagnostics(chains) → (gelman_rubin, ess)`: R-hat, effective sample size
7. Test: KF on linear system matches true state. MH samples from known Normal posterior.
**Accept:** KF hedge ratio tracks true time-varying beta. MH posterior mean within 2σ of truth. R-hat < 1.1.

---

### T-0142 · Backtest Engine Core
**Files:** `quantcore/backtest/engine.py`
**Deps:** T-0106, T-0108
**Steps:**
1. Event types: `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`
2. `BacktestEngine`: event queue (heapq by timestamp), main loop processes events in order
3. `DataHandler`: feeds historical bars, emits `MarketEvent`
4. Strategy interface: `on_market(event) → list[SignalEvent]`
5. Portfolio: tracks positions, cash, value; processes fills
6. Strict no-lookahead: strategy only sees data up to current bar
7. Test: buy-and-hold SPY backtest returns match actual buy-and-hold.
**Accept:** Events processed in correct time order. No future data accessible. SPY backtest within 1% of actual.

---

### T-0143 · Fill Simulation
**Files:** `quantcore/backtest/simulation.py`
**Deps:** T-0142
**Steps:**
1. `MarketOrderFill`: fill at next bar open (or configurable)
2. `LimitOrderFill`: fill if bar low ≤ limit (buy) or bar high ≥ limit (sell)
3. Slippage models: `FixedSlippage(bps)`, `VolumeSlippage(participation_rate)`, `VolatilitySlippage(vol_mult)`
4. Commission models: `PerShare(rate)`, `PerTrade(flat)`, `Tiered(brackets)`
5. `ShortBorrowCost(annual_rate)`, `MarginCost(annual_rate)`
6. Test: slippage always worsens fill. Commission reduces P&L.
**Accept:** Fill prices are always worse than or equal to ideal. All cost models produce positive costs.

---

### T-0144 · Bias Detection
**Files:** `quantcore/backtest/validation.py`
**Deps:** T-0142
**Steps:**
1. `check_lookahead(strategy_code) → list[Warning]`: static analysis for future data access patterns
2. `check_survivorship(universe, data) → SurvivorReport`: verify delisted stocks are included
3. `overfitting_score(in_sample_sharpe, oos_sharpe) → float`: ratio of degradation
4. `minimum_backtest_length(sharpe, n_trials) → int`: minimum years needed for significance
5. Test: intentionally biased strategy flagged. Clean strategy passes.
**Accept:** Lookahead checker catches `df.shift(-1)` patterns. Overfitting score >2 flags overfitted strategies.

---

### T-0145 · Walk-Forward & CPCV
**Files:** `quantcore/backtest/walk_forward.py`
**Deps:** T-0142
**Steps:**
1. `rolling_walk_forward(data, train_size, test_size, strategy_factory) → list[WFResult]`
2. `anchored_walk_forward(data, initial_train, test_size, ...)`: expanding train window
3. `purged_kfold_cv(data, n_splits, embargo_pct) → list[Fold]`: Lopez de Prado CPCV
4. Embargo period: gap between train end and test start (prevent leakage)
5. Aggregate OOS results into single equity curve
6. Test: no overlap between train and test sets. Embargo correctly applied.
**Accept:** Train/test splits have zero temporal overlap. Embargo gap = ceil(embargo_pct × test_size). OOS curve is continuous.

---

### T-0146 · Performance Analytics
**Files:** `quantcore/backtest/analytics.py`
**Deps:** T-0142
**Steps:**
1. `PerformanceReport`: compute all return metrics (total, annual, monthly distribution, rolling)
2. Risk metrics: Sharpe, Sortino, Calmar, Omega, max DD, avg DD
3. Trade analytics: win rate, profit factor, expectancy, avg hold period
4. Drawdown table: top-10 episodes (start, trough, recovery, depth, duration)
5. `bootstrap_confidence(returns, metric_fn, n_boot=1000) → (lower, upper)`
6. Factor attribution: regress strategy returns on FF factors
7. Monte Carlo equity curves: resample trades, plot percentile bands
8. `generate_html_report(report) → str`: full Plotly-based HTML report
9. Test: Sharpe of known return series matches manual calculation.
**Accept:** All metrics match empyrical/quantstats for same data. HTML report renders with charts.

---

### T-0147 · Rust Core
**Files:** `rust_core/Cargo.toml`, `rust_core/src/lib.rs`, `rust_core/src/orderbook.rs`, `rust_core/src/indicators.rs`
**Deps:** T-0103
**Steps:**
1. Cargo project with PyO3 + maturin
2. `OrderBook`: BTreeMap-based, `apply_update()`, `best_bid/ask()`, `spread()`, `midprice()`, `microprice()`, `book_imbalance(depth)`
3. `indicators`: `sma(data, period)`, `ema(data, period)`, `rsi(data, period)`, `atr(h,l,c, period)` on `&[f64]`
4. Python bindings: `#[pyfunction]` for all public functions
5. `maturin develop` builds + installs
6. Benchmark script: compare Rust vs Python (NumPy) implementations
7. Test: Rust outputs match Python implementations to f64 precision
**Accept:** `maturin develop` succeeds. Rust order book 10×+ faster than Python. Indicators match NumPy exactly.

---

### T-0148 · Information Theory
**Files:** `quantcore/math_models/statistics/information_theory.py`
**Deps:** T-0103
**Steps:**
1. `shannon_entropy(series, bins=50) → float`: discretize + compute H
2. `kl_divergence(p, q) → float`: KL(P||Q)
3. `mutual_information(x, y, method="knn") → float`: KNN-based MI for continuous vars
4. `transfer_entropy(source, target, lag=1) → float`: directional information flow
5. Test: entropy of uniform > entropy of peaked. MI of identical series = entropy. TE of independent series ≈ 0.
**Accept:** Entropy of N(0,1) samples ≈ 1.42 nats. MI symmetric. TE detects known causal relationship.

---

### T-0149 · Particle Filter
**Files:** `quantcore/math_models/bayesian/particle_filter.py`
**Deps:** T-0141
**Steps:**
1. `ParticleFilter`: `init(n_particles, state_dim)`, `predict(transition_fn)`, `update(observation, likelihood_fn)`, `resample()`
2. Systematic resampling (low variance)
3. `effective_sample_size() → float`: ESS diagnostic
4. Application: `online_volatility(returns) → vol_series` (stochastic volatility model)
5. Test: track mean of random walk, compare to Kalman Filter (should be similar).
**Accept:** ESS stays above n_particles/2 with resampling. Tracks known state within 2σ. Matches KF on linear problem.
