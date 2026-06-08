# QuantCore System Halt Log

## HALT EVENT: 2026-06-08

### Status: 🔴 ALL TRADING HALTED

### Trigger
Forensic audit revealed the entire QuantCore system operates on unvalidated backtests.

### Findings

1. **Options strategies in `live.yaml` are unvalidated on real data.**
   - `options_delta_neutral`: Claims OOS Sharpe 2.48 — tested only on synthetic IV/greeks
   - `options_put_write`: Claims OOS Sharpe 2.10 — tested only on synthetic IV/greeks
   - `options_iron_condor`: Claims OOS Sharpe 1.90 — tested only on synthetic IV/greeks
   - These strategies had **zero real-data walk-forward validation** when promoted to live.yaml.

2. **Walk-forward test engine uses `np.random` synthetic data.**
   - `test_walk_forward_strategies.py` generates prices via `rng.normal(0.02, 0.5)`
   - OOS Sharpe floor is -5.0 (proves strategies don't crash, proves nothing about alpha)
   - 37 strategies in RESEARCH status pass these tests but have no real-data validation

3. **Honest sweep on real data shows all portfolios are negative.**
   - Static allocation: Sharpe -0.81, MaxDD -16.45%, Return -10.45%
   - HRP: Sharpe -0.78, MaxDD -15.18%, Return -7.05%
   - Black-Litterman: Sharpe -0.61, MaxDD -25.37%, Return -11.62%

4. **3 honest survivors are excluded from live.yaml.**
   - `macro_cross_sectional_momentum`: Real-data OOS Sharpe +0.81 — NOT in config
   - `equity_tsmom`: Real-data OOS Sharpe +0.56 — NOT in config
   - `equity_cross_sectional_momentum`: Real-data OOS Sharpe +0.05 — NOT in config

5. **5 strategies produce zero trades on real data after bias removal.**
   - `macro_state_selective_momentum`, `macro_factor_residual_v3`, `equity_trend_following`,
     `macro_vrp` — look-ahead bias was the only source of apparent signal

6. **7 look-ahead biases were fixed on 2026-05-28.**
   - Causal rolling z-scores, expanding-window OLS, trailing IC estimation, fit/transform splits,
     pipeline serialization, ADV window limits, log1p equity curves

### Actions Taken

- [x] All strategy weights set to 0.0 in `config/live.yaml`
- [x] Paper trading suspended (all weights zero — no orders will be generated)
- [x] Hard gate comment added to live.yaml with restart criteria
- [x] This halt log created

### Restart Criteria

Trading may NOT resume until ALL of the following are satisfied:

1. **`scripts/honest_sweep.py` runs on real yfinance data (2018-2024) for all strategies**
2. **NullStrategy engine integrity check passes** (ci_lower < 0, oos_sharpe ≈ 0.0 ± 0.2)
3. **Each strategy in live.yaml shows ci_lower > 0 on real data**
4. **Options strategies pass real-data frictions backtest** (spread, slippage, commission, delta-hedge costs)
5. **Portfolio-level honest sweep shows at least one configuration with positive Sharpe and ci_lower > 0**
6. **No two strategies with weight > 0 are correlated > 0.7** (must have orthogonal alpha)

### If Restart Criteria Cannot Be Met

If no portfolio configuration shows positive Sharpe with ci_lower > 0 on honest real data:

> **QuantCore does not have a viable product. Do not deploy. Report the finding honestly.**
> A fund running on synthetic backtests is not a fund — it is a confidence game.

### Phase Schedule

| Phase | Task | Timeline | Status |
|-------|------|----------|--------|
| 0 | Emergency halt | Day 1 | ✅ COMPLETE |
| 1 | Honest sweep engine (real data) | Week 1-2 | 🔄 IN PROGRESS |
| 2 | Options real-data validation | Week 2-3 | ⏳ PENDING |
| 3 | Research strategy triage (bulk) | Week 3-4 | ⏳ PENDING |
| 4 | Portfolio reconstruction | Week 4-5 | ⏳ PENDING |
| 5 | Infrastructure hardening | Week 5-6 | ⏳ PENDING |
| 6 | Long-term alpha development | Post-recovery | ⏳ PENDING |

### Sign-off

- **Auditor**: QuantCore Claude Code audit (2026-06-08)
- **Authority**: User-initiated emergency halt
- **Next review**: After Phase 1 completion (honest sweep results available)
