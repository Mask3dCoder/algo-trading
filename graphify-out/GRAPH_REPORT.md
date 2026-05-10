# Graph Report - algo sw  (2026-04-30)

## Corpus Check
- 345 files · ~279,014 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 6576 nodes · 23137 edges · 86 communities detected
- Extraction: 33% EXTRACTED · 67% INFERRED · 0% AMBIGUOUS · INFERRED: 15506 edges (avg confidence: 0.54)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]
- [[_COMMUNITY_Community 97|Community 97]]
- [[_COMMUNITY_Community 98|Community 98]]
- [[_COMMUNITY_Community 101|Community 101]]
- [[_COMMUNITY_Community 107|Community 107]]
- [[_COMMUNITY_Community 123|Community 123]]
- [[_COMMUNITY_Community 124|Community 124]]
- [[_COMMUNITY_Community 125|Community 125]]

## God Nodes (most connected - your core abstractions)
1. `Side` - 777 edges
2. `Order` - 762 edges
3. `Position` - 743 edges
4. `OrderType` - 535 edges
5. `PortfolioSnapshot` - 525 edges
6. `Fill` - 467 edges
7. `RiskLimits` - 392 edges
8. `BacktestEngine` - 315 edges
9. `StrategyConfig` - 296 edges
10. `OrderStatus` - 294 edges

## Surprising Connections (you probably didn't know these)
- `Fill simulation models: slippage, commissions, borrowing costs.` --uses--> `Side`  [INFERRED]
  backtest\simulation.py → quantcore\core\types.py
- `Abstract slippage model.` --uses--> `Side`  [INFERRED]
  backtest\simulation.py → quantcore\core\types.py
- `Fixed basis-point slippage.` --uses--> `Side`  [INFERRED]
  backtest\simulation.py → quantcore\core\types.py
- `Volume-proportional slippage: participation_rate * sigma.` --uses--> `Side`  [INFERRED]
  backtest\simulation.py → quantcore\core\types.py
- `Volatility-adjusted slippage.` --uses--> `Side`  [INFERRED]
  backtest\simulation.py → quantcore\core\types.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.01
Nodes (696): BrokerStateSnapshot, CanonicalOrderRecord, FillSeqStatus, Broker Emulator — canonical truth layer that can intentionally diverge from SUT., Submit an order to the canonical broker. Returns ack success., Cancel an order. Returns True if successful., Reject an order on the canonical side., Generate a canonical fill for an order. Returns None if order unknown. (+688 more)

### Community 1 - "Community 1"
Cohesion: 0.01
Nodes (346): ABC, _extract_fill_fields(), FrameworkStrategyAdapter, Framework strategy adapter — bridges quantcore.strategies.base.Strategy to the b, Adapts a framework ``Strategy`` to the paper trader's     ``generate_signals(fee, Called by the paper trader's main loop.          Accepts the latest bar events (, Preload historical bars to warm up the strategy.          Feed bars in chronolog, Forward fill events to the framework strategy.          Handles IBKR Fill object (+338 more)

### Community 2 - "Community 2"
Cohesion: 0.01
Nodes (323): cmd_api(), cmd_backtest(), cmd_live(), cmd_report(), cmd_risk(), main(), Start live trading (paper or live)., Start the REST API server. (+315 more)

### Community 3 - "Community 3"
Cohesion: 0.01
Nodes (174): LiveStrategyAdapter, create_hybrid(), HybridFeed, Hybrid configuration: Alpaca data feed + IBKR execution.  This is the recommende, Factory for creating a hybrid Alpaca data + IBKR execution setup.      Reads Alp, Delegates market data to Alpaca, wraps IBKR for execution.      Usage:         h, Connect both Alpaca feed and IBKR broker. Fail fast if either fails., Disconnect both feed and broker. (+166 more)

### Community 4 - "Community 4"
Cohesion: 0.01
Nodes (197): IcebergAlgo, IcebergConfig, IcebergResult, IcebergSlice, Iceberg (reserve) order execution algorithm.  Shows only a small visible portion, Replenish a visible slice after it has been filled.          Call this when a vi, Compute the visible slice size.          Adjusts based on:         - Base visibl, Iceberg algorithm configuration. (+189 more)

### Community 5 - "Community 5"
Cohesion: 0.03
Nodes (146): AutoencoderAnomaly, Deep learning models: LSTM, GRU, TCN, Transformer, Autoencoder, VAE.  Note: Thes, Generate predictions., Simplified GRU-like signal. Same as LSTM but with different activation., Train with sigmoid gate approximation., Simplified TCN using 1D exponential smoothing as convolution proxy., Create sliding-window sequences from feature matrix for DL models.      Each sam, Simple autoencoder for anomaly detection using PCA-like reconstruction.      Rec (+138 more)

### Community 6 - "Community 6"
Cohesion: 0.01
Nodes (194): _bond_term_premium(), _commodity_carry_raw(), compute_carry_matrix(), _equity_carry_raw(), _load_close_prices(), _load_dividends(), Compute non-price-based carry signals for each asset type.  Sources:   Equities:, Term premium = ^TNX (10Y yield) - ^IRX (3M yield).      Returns daily Series ind (+186 more)

### Community 7 - "Community 7"
Cohesion: 0.01
Nodes (161): ClaytonCopula, fit_copula(), FrankCopula, GaussianCopula, GumbelCopula, Copula models: Gaussian, t, Clayton, Gumbel, Frank., Generate via conditional inverse method., Bivariate Gaussian copula. (+153 more)

### Community 8 - "Community 8"
Cohesion: 0.01
Nodes (171): Compute all technical features. Returns dict of feature_name → array., detect_sweeps(), Detect sweep orders: trades with unusually large size.      A sweep = single ord, benchmark_signal_generation(), Unit tests for technical signals: volume, volatility, momentum., Price rising but OBV falling should be bearish divergence., Constant returns should give zero volatility., Parkinson estimator should be more efficient (less noisy) than close-to-close. (+163 more)

### Community 9 - "Community 9"
Cohesion: 0.02
Nodes (77): PortfolioState, BrokerAdapter, _ccxt_order_to_internal(), CCXTBroker, PaperBroker, _extract_cash(), _extract_positions(), _normalize_positions() (+69 more)

### Community 10 - "Community 10"
Cohesion: 0.02
Nodes (116): brinson_attribution(), capm_attribution(), factor_attribution(), PerformanceAttribution, pnl_explain(), Performance and risk attribution: Brinson, factor-based, and risk decomposition., Brinson performance attribution.      Decomposes active return into allocation,, Factor-based return attribution.      R_portfolio = Σ β_k * F_k + ε  (idiosyncra (+108 more)

### Community 11 - "Community 11"
Cohesion: 0.01
Nodes (156): Detect unusual options activity via volume vs open interest ratio.      VOI = vo, unusual_options_activity(), charm_exposure(), dealer_positioning_signal(), gamma_exposure(), gamma_flip_level(), gex_profile(), pin_risk_score() (+148 more)

### Community 12 - "Community 12"
Cohesion: 0.03
Nodes (92): compute_filing_sentiment_trend(), FilingDocument, FilingIngestor, FilingType, LMDictionary, SEC EDGAR filing ingestion with Loughran-McDonald sentiment dictionary., Score text for all LM sentiment categories., Net sentiment = (positive - negative) / total words. (+84 more)

### Community 13 - "Community 13"
Cohesion: 0.02
Nodes (95): deflated_sharpe(), DeflatedSharpeResult, fdr_correction(), haircut_sharpe(), holm_correction(), _min_backtest_length(), minimum_backtest_length(), multiple_testing_summary() (+87 more)

### Community 14 - "Community 14"
Cohesion: 0.02
Nodes (118): Unit tests for BSM pricing, Greeks, implied vol, and put-call parity., test_parity_holds(), TestBSMPricing, TestGreeks, TestImpliedVol, TestPutCallParity, Unit tests for stochastic models: GBM, OU, Heston, jump diffusion, fBM, MC engin, Monte Carlo call price should be within 0.5% of BSM analytical. (+110 more)

### Community 15 - "Community 15"
Cohesion: 0.03
Nodes (107): Unit tests for yfinance_loader — all network calls mocked., load_multiple should skip symbols that fail and return successes., load_multiple returns empty dict when all symbols fail., load_combined should concatenate and sort by timestamp + symbol., load_combined returns empty DataFrame when all symbols fail., When cache exists, load_bars returns it without calling yfinance., When yfinance returns empty, load_bars returns empty DataFrame., TestCachePath (+99 more)

### Community 16 - "Community 16"
Cohesion: 0.05
Nodes (57): BrokerEmulator, CanonicalFillRecord, ChaosConfig, ChaosEvent, EventChaosGenerator, EventType, FailureEvent, FailureInjector (+49 more)

### Community 17 - "Community 17"
Cohesion: 0.02
Nodes (105): bocpd(), ChangepointResult, cusum(), pelt_changepoints(), Changepoint detection: PELT, BOCPD, CUSUM, SPRT., CUSUM (Cumulative Sum) control chart for changepoint detection.      Detects upw, Sequential Probability Ratio Test for mean shift detection.      Tests H0: mean, PELT algorithm via the ruptures library.      Args:         series: Time series (+97 more)

### Community 18 - "Community 18"
Cohesion: 0.03
Nodes (72): options_flow_composite(), put_call_ratio(), put_call_ratio_signal(), Options flow signals: unusual options activity, sweeps, put/call ratio., Put/call ratio z-scored against rolling history.      Contrarian interpretation:, Combine options flow signals into a composite.      Weighted average of VOI, vol, Volume-weighted delta: net call vs put volume, normalized.      Positive = more, Raw put/call ratio. Higher = more bearish sentiment. (+64 more)

### Community 19 - "Community 19"
Cohesion: 0.03
Nodes (67): bootstrap_confidence(), calmar_ratio(), compute_returns(), drawdown_analysis(), generate_html_report(), generate_performance_report(), max_drawdown(), omega_ratio() (+59 more)

### Community 20 - "Community 20"
Cohesion: 0.04
Nodes (28): AuditEntry, AuditLog, _compute_hash(), Append-only audit trail with hash chain for tamper detection., Export all entries to a JSON file., Import entries from JSON and validate hash chain., Generate a tamper detection report for an audit log., Immutable entry in the audit log with cryptographic hash chaining. (+20 more)

### Community 21 - "Community 21"
Cohesion: 0.05
Nodes (31): DataSource, AlphaVantageDataSource, FREDDataSource, Alpha Vantage and FRED data source adapters., Fetch a time series from FRED by series ID., Fetch US Treasury yield curve data., Alpha Vantage data source for fundamentals and earnings., Fetch daily OHLCV from Alpha Vantage. (+23 more)

### Community 22 - "Community 22"
Cohesion: 0.08
Nodes (38): active_count(), Alert, AlertChannel, AlertEngine, AlertRule, AlertSeverity, build_default_alert_rules(), critical_count() (+30 more)

### Community 23 - "Community 23"
Cohesion: 0.05
Nodes (39): Unit tests for data quality engines: outlier detection, missing data, stale quot, TestMissingData, TestOutlierDetection, TestStaleQuotes, Cross-validation convenience re-export., detect_gaps(), em_impute(), forward_fill() (+31 more)

### Community 24 - "Community 24"
Cohesion: 0.09
Nodes (30): apply_fill(), fill_application_to_position(), FillApplication, _new_position(), EconomicEngineCore: single canonical economic kernel for the entire system.  Thi, Convert a FillApplication to a typed Position model, if needed.      Args:, Result of applying a single fill to a position.      Attributes:         positio, Create a serializable Position-like dict.      Using dict for zero-dependency ke (+22 more)

### Community 25 - "Community 25"
Cohesion: 0.07
Nodes (10): IBKRBroker, main(), mock_ib(), _mock_trade(), _summary_item(), TestIBKRBrokerAccount, TestIBKRBrokerConstruction, TestIBKRBrokerOrders (+2 more)

### Community 26 - "Community 26"
Cohesion: 0.07
Nodes (19): DatabaseConfig, DatabaseManager, Async database access layer for TimescaleDB with connection pooling., Query OHLCV data for a symbol and date range., Bulk insert using COPY protocol for maximum throughput., Async database manager with connection pooling (asyncpg backend)., Create connection pool., Close connection pool. (+11 more)

### Community 27 - "Community 27"
Cohesion: 0.09
Nodes (17): generate_best_execution_report(), generate_regulatory_report(), Regulatory compliance reporting: VaR backtesting, best execution, audit trails., Generate a best execution compliance report (SEC Rule 606 / MiFID II).      Args, Complete regulatory compliance report., Generate a complete regulatory compliance report.      Args:         returns: Da, RegulatoryReport, christoffersen_test() (+9 more)

### Community 28 - "Community 28"
Cohesion: 0.09
Nodes (15): CacheConfig, CacheManager, Redis cache layer with msgpack serialization, pub/sub, and TTL., Delete keys matching a pattern (cache busting)., Async Redis cache manager with msgpack serialization., Create Redis connection., Close Redis connection., Serialize data with msgpack. (+7 more)

### Community 29 - "Community 29"
Cohesion: 0.1
Nodes (15): dynamic_hedge_ratio(), ExtendedKalmanFilter, KalmanFilter, KFState, Kalman Filter: standard, extended, unscented with dynamic hedge ratio applicatio, Kalman Filter state at a given time step., Unscented Kalman Filter — no Jacobians, uses sigma points., Generate sigma points and weights. (+7 more)

### Community 30 - "Community 30"
Cohesion: 0.1
Nodes (17): atr_size(), cvar_size(), fractional_kelly(), kelly_criterion(), optimal_f(), Position sizing methods: Kelly, vol targeting, risk parity, ATR, CVaR., Risk parity portfolio weights (equal risk contribution).      Iterative solve fo, ATR-based position sizing.      shares = (account_value × risk_pct) / (ATR × mul (+9 more)

### Community 31 - "Community 31"
Cohesion: 0.11
Nodes (18): Event, EventBus, EventType, Async-compatible event bus for inter-component communication., Base event carried on the bus., Lightweight async pub/sub event bus.      Handlers may be sync or async. All han, Register a handler for an event type., Remove a handler for an event type. (+10 more)

### Community 32 - "Community 32"
Cohesion: 0.13
Nodes (10): online_volatility(), ParticleFilter, Sequential Monte Carlo (Particle Filter) for nonlinear state-space models., Particle filter with systematic resampling., Online volatility estimation using particle filter.      Stochastic volatility m, Initialize particles from a Gaussian distribution., Propagate particles through the state transition model.          transition_fn:, Update particle weights using the observation likelihood.          likelihood_fn (+2 more)

### Community 33 - "Community 33"
Cohesion: 0.14
Nodes (18): check_butterfly_arbitrage(), check_calendar_spread(), fit_svi(), plot_surface(), SVI/SSVI volatility surface parameterization and arbitrage checks., Interpolate IV at arbitrary (k, T) via linear interpolation in total variance., Check butterfly arbitrage: g(k) = (1 - k*dw/dk)^2 - (-d²w/dk²)*w ≥ 0.      Retur, Raw SVI parameters: w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2)). (+10 more)

### Community 34 - "Community 34"
Cohesion: 0.14
Nodes (17): carhart(), download_ff_factors(), fama_french(), FFResult, ica_returns(), ICAResult, pca_returns(), PCAResult (+9 more)

### Community 35 - "Community 35"
Cohesion: 0.15
Nodes (13): get_config(), Dynaconf-based configuration loader with environment layering., Return the singleton Dynaconf settings object.      Layering order (later wins):, get_logger(), Structured logging setup — JSON in production, pretty console in dev., Initialize structlog configuration based on current environment., Return a bound structlog logger for the given module name., setup_logging() (+5 more)

### Community 36 - "Community 36"
Cohesion: 0.17
Nodes (15): is_market_open(), is_trading_day(), next_market_open(), Timezone-aware time utilities for market calendar operations., Convert any timezone-aware datetime to UTC., Convert any timezone-aware datetime to US/Eastern., Check if the given datetime falls within NYSE regular trading hours., Check if a date is a valid NYSE trading day. (+7 more)

### Community 37 - "Community 37"
Cohesion: 0.15
Nodes (15): evt_cvar(), evt_var(), fit_gev(), fit_gpd(), GEVResult, GPDResult, mean_excess_plot(), Extreme Value Theory: GPD, GEV, and EVT-based risk measures. (+7 more)

### Community 38 - "Community 38"
Cohesion: 0.2
Nodes (15): adf_test(), kpss_test(), phillips_perron_test(), Stationarity tests: ADF, KPSS, Phillips-Perron, variance ratio., Lo-MacKinlay variance ratio test for random walk.      H0: series is a random wa, Result of a statistical test., Run all stationarity tests and report consensus., Results from all stationarity tests. (+7 more)

### Community 39 - "Community 39"
Cohesion: 0.17
Nodes (8): compute_daily_borrow_cost(), compute_daily_margin_interest(), Financing costs for backtest: short borrow costs, margin interest, forced liquid, Compute daily borrow cost for a short position.      Uses 360-day convention sta, Compute daily margin interest.      margin_balance: the amount of borrowed money, Tracks a short position's borrow cost accrual., ShortPosition, TestCostComputations

### Community 40 - "Community 40"
Cohesion: 0.14
Nodes (13): ActionType, adjust_for_dividend(), adjust_for_split(), AdjustmentEvent, get_adjustment_factors(), handle_delisting(), handle_merger(), Corporate actions: split/dividend/merger adjustments, delisting handling. (+5 more)

### Community 41 - "Community 41"
Cohesion: 0.22
Nodes (13): compare_distributions(), fit_normal(), fit_skewed_t(), fit_student_t(), FitResult, qq_plot_data(), Distribution fitting and comparison: Normal, t, NIG, Skewed-t., Return (theoretical_quantiles, sample_quantiles) for QQ plot. (+5 more)

### Community 42 - "Community 42"
Cohesion: 0.15
Nodes (13): differential_entropy_normal(), kl_divergence(), _mi_knn(), mutual_information(), Information theory measures: entropy, KL divergence, mutual information, transfe, Shannon entropy of a continuous random variable (discretized).      H(X) = -sum(, Transfer entropy from source to target.      TE(X→Y) = H(Y_t|Y_{t-lag}) - H(Y_t|, Analytical differential entropy of N(0, sigma): 0.5*ln(2*pi*e*sigma^2). (+5 more)

### Community 43 - "Community 43"
Cohesion: 0.16
Nodes (13): _autocovariance_fgn(), estimate_hurst_dfa(), estimate_hurst_rs(), _hosking_sample(), Fractional Brownian Motion: simulation and Hurst exponent estimation., Estimate Hurst exponent via R/S (rescaled range) analysis., Estimate Hurst exponent via Detrended Fluctuation Analysis (DFA)., Simulate fBM via Cholesky decomposition of the covariance matrix.      Cov(W^H_s (+5 more)

### Community 44 - "Community 44"
Cohesion: 0.14
Nodes (5): Tests for the /health endpoint., Validate the /health endpoint returns correct schema and values., Health endpoint does not require authentication., Verify the endpoint reflects state updates in real time., TestHealthEndpoint

### Community 45 - "Community 45"
Cohesion: 0.2
Nodes (11): arch_test(), ARCHTestResult, compute_acf(), compute_pacf(), ljung_box_test(), LjungBoxResult, Serial correlation analysis: ACF, PACF, Ljung-Box, ARCH test., Compute autocorrelation function. (+3 more)

### Community 46 - "Community 46"
Cohesion: 0.18
Nodes (11): calibrate_heston(), heston_call_price(), heston_characteristic_function(), HestonParams, Heston stochastic volatility model: simulation, characteristic function, calibra, European call price via Heston characteristic function (FFT-free integration)., Heston model parameters., Calibrate Heston model to market option prices.      Uses Levenberg-Marquardt (L (+3 more)

### Community 47 - "Community 47"
Cohesion: 0.18
Nodes (8): effective_sample_size(), gelman_rubin(), MetropolisHastings, MCMC samplers: Metropolis-Hastings, convergence diagnostics., Metropolis-Hastings MCMC sampler., Run MH sampler.          Returns array of shape (n_samples - burn_in, dim)., Gelman-Rubin R-hat statistic for convergence diagnosis.      R-hat < 1.1 indicat, Effective sample size for a single MCMC chain.      Uses autocorrelation-based e

### Community 48 - "Community 48"
Cohesion: 0.29
Nodes (9): analyze_metrics(), count_log_events(), find_latest_log(), find_latest_metrics(), main(), Find the most recent paper trade log file., Count errors, warnings, and critical events in a log file., Find the most recent metrics CSV file. (+1 more)

### Community 50 - "Community 50"
Cohesion: 0.25
Nodes (4): ImpactResult, Market impact configuration and computation for backtest fills.  Wires execution, Compute market impact for a single fill.          Args:             shares: Numb, Result of a market impact calculation for a single fill.

### Community 51 - "Community 51"
Cohesion: 0.25
Nodes (7): jd_call_price(), jd_characteristic_function(), Merton jump-diffusion model: simulation and characteristic function., Simulate Merton jump-diffusion: GBM + compound Poisson jumps.      dS/S = (mu -, Characteristic function of log-price under Merton jump-diffusion., European call price via Merton series expansion.      C = sum_{k=0}^{inf} P(N_T=, simulate_jump_diffusion()

### Community 52 - "Community 52"
Cohesion: 0.25
Nodes (7): calibrate_sabr(), implied_vol_sabr(), SABR model: simulation, Hagan implied vol approximation, calibration., Calibrate SABR to market implied vols.      Returns (alpha, rho, sigma0).     be, Simulate SABR model: dF = sigma*F^beta*dW1, dsigma = alpha*sigma*dW2.      Retur, Hagan et al. (2002) approximation for SABR implied volatility.      Valid for AT, simulate_sabr()

### Community 53 - "Community 53"
Cohesion: 0.33
Nodes (5): create_app(), Dash application skeleton: multi-tab monitoring dashboard with dark theme., Launch the dashboard server., Create the Dash application with dark theme and multi-tab layout., run_dashboard()

### Community 54 - "Community 54"
Cohesion: 0.33
Nodes (5): hurst_dfa(), hurst_rs(), Hurst exponent estimation via R/S analysis and DFA., Estimate Hurst exponent via Detrended Fluctuation Analysis., Estimate Hurst exponent via R/S (rescaled range) analysis.      H < 0.5: mean-re

### Community 55 - "Community 55"
Cohesion: 0.5
Nodes (4): build_execution_tab(), _exec_metric(), Execution dashboard panel: VWAP performance, slippage histogram, fill quality, T, Build the execution quality monitoring tab layout.

### Community 56 - "Community 56"
Cohesion: 0.5
Nodes (4): build_portfolio_tab(), _metric_card(), Portfolio dashboard panel: P&L chart, equity curve, positions table., Build the portfolio monitoring tab layout.

### Community 57 - "Community 57"
Cohesion: 0.5
Nodes (4): build_risk_tab(), Risk dashboard panel: VaR gauges, Greeks heatmap, drawdown chart, correlations., Build the risk monitoring tab layout., _var_card()

### Community 58 - "Community 58"
Cohesion: 0.5
Nodes (3): build_market_tab(), Market dashboard panel: regime bars, vol surface, term structure, sector heatmap, Build the market data and regime tab layout.

### Community 59 - "Community 59"
Cohesion: 0.5
Nodes (3): build_strategy_tab(), Strategy dashboard panel: strategy P&L, active signals, recent trades., Build the strategy monitoring tab layout.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (2): check_files(), main()

### Community 61 - "Community 61"
Cohesion: 0.67
Nodes (1): Diagnostic Summary: Consolidate all 6 diagnostic findings.  Aggregates results f

### Community 62 - "Community 62"
Cohesion: 0.67
Nodes (1): Pytest configuration and shared fixtures.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Strategy-level monitoring: signal quality, regime detection, performance drift.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Alternative data: sentiment analysis and options flow intelligence.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): REST API routes for all QuantCore modules.

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Trading constants and market parameters.

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Dashboard panel builders for each monitoring tab.

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Paper trading module.

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Technical signal indicators: trend, mean-reversion, volume, volatility, momentum

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): OrderBook

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Total cash (settled + unsettled).

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Restore a CashLedger from a snapshot dict.

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): Fetch OHLCV data for a symbol and date range.

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Fetch options chain data for a symbol.

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): Return list of supported asset symbols/identifiers.

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (1): Dead man's switch: returns False if watchdog has been triggered.

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (1): ESS = 1 / sum(w_i^2).

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (1): Weighted mean of particles.

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): Weighted covariance of particles.

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (1): Convert standard normal samples to GBM price paths.

### Community 98 - "Community 98"
Cohesion: 1.0
Nodes (1): Convert Sobol uniform samples to standard normal via inverse CDF.

### Community 101 - "Community 101"
Cohesion: 1.0
Nodes (1): Process new market data and emit signals.          Args:             data: di

### Community 107 - "Community 107"
Cohesion: 1.0
Nodes (1): Load CTA universe data.

### Community 123 - "Community 123"
Cohesion: 1.0
Nodes (1): Current position in a symbol.

### Community 124 - "Community 124"
Cohesion: 1.0
Nodes (1): Trading signal from a strategy or indicator.

### Community 125 - "Community 125"
Cohesion: 1.0
Nodes (1): Point-in-time portfolio state.

## Knowledge Gaps
- **1082 isolated node(s):** `Strategy-level monitoring: signal quality, regime detection, performance drift.`, `Alternative data: sentiment analysis and options flow intelligence.`, `Gamma Exposure (GEX) calculation, gamma flip levels, charm/vanna flows.`, `Compute Gamma Exposure from an options chain.`, `Compute GEX from a list of option contracts.          Each option dict must have` (+1077 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 60`** (3 nodes): `check_files()`, `main()`, `deployment_report.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (3 nodes): `main()`, `diagnostic_summary.py`, `Diagnostic Summary: Consolidate all 6 diagnostic findings.  Aggregates results f`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (3 nodes): `anyio_backend()`, `conftest.py`, `Pytest configuration and shared fixtures.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (2 nodes): `__init__.py`, `Strategy-level monitoring: signal quality, regime detection, performance drift.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (2 nodes): `Alternative data: sentiment analysis and options flow intelligence.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (2 nodes): `__init__.py`, `REST API routes for all QuantCore modules.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (2 nodes): `Trading constants and market parameters.`, `constants.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (2 nodes): `Dashboard panel builders for each monitoring tab.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (2 nodes): `Paper trading module.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (2 nodes): `__init__.py`, `Technical signal indicators: trend, mean-reversion, volume, volatility, momentum`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (2 nodes): `orderbook.rs`, `OrderBook`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Total cash (settled + unsettled).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Restore a CashLedger from a snapshot dict.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `Fetch OHLCV data for a symbol and date range.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Fetch options chain data for a symbol.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `Return list of supported asset symbols/identifiers.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (1 nodes): `Dead man's switch: returns False if watchdog has been triggered.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (1 nodes): `ESS = 1 / sum(w_i^2).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (1 nodes): `Weighted mean of particles.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `Weighted covariance of particles.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 97`** (1 nodes): `Convert standard normal samples to GBM price paths.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 98`** (1 nodes): `Convert Sobol uniform samples to standard normal via inverse CDF.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 101`** (1 nodes): `Process new market data and emit signals.          Args:             data: di`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 107`** (1 nodes): `Load CTA universe data.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 123`** (1 nodes): `Current position in a symbol.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 124`** (1 nodes): `Trading signal from a strategy or indicator.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 125`** (1 nodes): `Point-in-time portfolio state.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Side` connect `Community 0` to `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 9`, `Community 15`, `Community 16`, `Community 19`, `Community 20`, `Community 27`?**
  _High betweenness centrality (0.101) - this node is a cross-community bridge._
- **Why does `Order` connect `Community 0` to `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 9`, `Community 15`, `Community 16`, `Community 20`, `Community 25`, `Community 27`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `StrategyConfig` connect `Community 1` to `Community 0`, `Community 9`, `Community 3`, `Community 6`?**
  _High betweenness centrality (0.065) - this node is a cross-community bridge._
- **Are the 774 inferred relationships involving `Side` (e.g. with `Execution API routes: OMS, algorithmic orders, market impact, TCA.` and `List all orders in the OMS.`) actually correct?**
  _`Side` has 774 INFERRED edges - model-reasoned connections that need verification._
- **Are the 759 inferred relationships involving `Order` (e.g. with `EventType` and `Event`) actually correct?**
  _`Order` has 759 INFERRED edges - model-reasoned connections that need verification._
- **Are the 740 inferred relationships involving `Position` (e.g. with `EventType` and `Event`) actually correct?**
  _`Position` has 740 INFERRED edges - model-reasoned connections that need verification._
- **Are the 532 inferred relationships involving `OrderType` (e.g. with `EventType` and `Event`) actually correct?**
  _`OrderType` has 532 INFERRED edges - model-reasoned connections that need verification._