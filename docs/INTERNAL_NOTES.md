# QuantCore Internal Notes — Q1 FIX FOUNDATION

**Classification:** Internal  
**Date:** 2026-05-10  
**Author:** QuantCore Team (Claude Code rebuild)  
**Phase Status:** Q1 FIX FOUNDATION — COMPLETE
**Reference audit:** [FORENSIC_AUDIT_REPORT.md](FORENSIC_AUDIT_REPORT.md)

---

## 1. Audit Summary (6 Fatal Findings → All Resolved)

The forensic audit (2026-05-09) identified the following fatal defects. All have been resolved in Q1:

| # | Finding | Severity | Resolution | File(s) |
|---|---|---|---|---|
| 1 | Backtest engine look-ahead bias | CRITICAL | Barrier pattern: `fill_sim.set_bars(bar, None)` before `on_bar()` | `quantcore/backtest/engine.py` |
| 2 | Validation bypasses execution stack | CRITICAL | verify_honest_foundation.py uses BacktestEngine.run(); real_data_backtest.py v2 added | `scripts/verify_honest_foundation.py`, `scripts/real_data_backtest.py` |
| 3 | Metrics bugs (Sharpe→0, Calmar wrong) | HIGH | Sharpe NaN for flat, all drawdown metrics equity-based, rf geometric | `quantcore/risk/metrics.py` |
| 4 | Purged CV crashes at runtime | HIGH | `total`→`n_bars` fix; purged CV test suite green | `quantcore/backtest/walk_forward.py` |
| 5 | No survivorship-bias correction | HIGH | universe_history.csv with 24 S&P500 PIT changes; HistoricalUniverseLoader CSV loading | `quantcore/data/sources/historical_universe.py`, `universe_history.csv` |
| 6 | Factor residual degenerates to 1 asset | HIGH | DISABLED; residual_momentum.py skeleton; Q2 FF5 rebuild planned | `quantcore/strategies/macro/factor_residual.py`, `quantcore/alpha/residual_momentum.py` |

All previously reported Sharpe ratios and performance figures must be discarded.

---

## 2. Q1 FIX FOUNDATION — Complete Changes List

### 2.1 Measurement Infrastructure (Phase 1)

| Task | Description | Status |
|---|---|---|
| 1.1 | Look-ahead bias barrier in BacktestEngine | DONE |
| 1.2 | Validation uses full execution stack | DONE — verify script uses BacktestEngine; real_data_backtest.py has v2 path |
| 1.3 | Fix all metrics bugs | DONE — sharpe/calmar/sterling/ulcer/pain/burke/drawdown_stats all fixed |
| 1.4 | Purged CV crash fix | DONE — `total`→`n_bars`, test suite green |
| 1.5 | Point-in-time universe | DONE — universe_history.csv (24 changes 2013-2024), HistoricalUniverseLoader CSV loading |
| 1.6 | 55 ETF universe validation | DONE — ETF_MACRO universe validated |

### 2.2 Strategy Cleanup (Phase 2)

| Task | Description | Status |
|---|---|---|
| 2.1 | StrategyStatus enum + DISABLED marking | DONE — DISABLED, RESEARCH, DEVELOPMENT, PAPER_TRADING, LIVE |
| 2.2 | Residual momentum skeleton | DONE — `quantcore/alpha/residual_momentum.py` |
| 2.3 | CSM v2 safety caps | DONE — max_leverage=1.0, correlation stress 50%, DD control |
| 2.4 | Bootstrap CI integration | DONE — block bootstrap with n^(1/3) block size, integrated in both validation paths |

### 2.3 Risk Engine Upgrades (Phase 3)

| Task | Description | Status |
|---|---|---|
| 3.1 | Student-t parametric VaR | DONE — MLE `_estimate_t_dof()`, `parametric_var(returns_for_t=..., nu=...)`, method="parametric_t" |
| 3.2 | Cornish-Fisher n<100 warning | DONE — `warnings.warn()`, method="unstable_cornish_fisher" for n<100 |
| 3.3 | Vol-adjusted drawdown guard | DONE — `VolAdjustedDrawdownGuard` state machine: normal→reduce→flatten→recovery_gate→normal |
| 3.4 | Liquidity-aware sizing | DONE — `max_position_size()` with participation_rate=0.05, 20% capital cap, unlimited sentinel |

### 2.4 Testing & Validation (Phase 4)

| Task | Description | Status |
|---|---|---|
| 4.1 | Honest backtesting test suite | DONE — 9 test classes, 26 tests covering all Q1 fixes |
| 4.2 | Verification script | DONE — `scripts/verify_honest_foundation.py` |
| 4.3 | Historical universe tests | DONE — `tests/unit/data/test_historical_universe.py` |

### 2.5 Documentation (Phase 5)

| Task | Description | Status |
|---|---|---|
| 5.1 | README.md cleanup | DONE — removed false +0.911 claims, honest strategy table, disclaimer |
| 5.2 | INTERNAL_NOTES.md | DONE — this document |

---

## 3. Files Modified (Q1 FIX FOUNDATION)

### Modified
| File | Change |
|---|---|
| `quantcore/backtest/engine.py` | Look-ahead bias fix (barrier pattern before on_bar) |
| `quantcore/backtest/walk_forward.py` | Fix `total`→`n_bars` crash; purged_kfold_cv out-of-bounds break |
| `quantcore/risk/metrics.py` | Fix sharpe (NaN), calmar/sterling/burke/ulcer/pain/drawdown_stats (equity-based), rf geometric |
| `quantcore/risk/var.py` | Student-t parametric VaR with MLE; Cornish-Fisher n<100 warning |
| `quantcore/data/sources/universe.py` | 55 ETF universe; survivorship bias warnings |
| `quantcore/data/sources/historical_universe.py` | CSV loading method; auto-load in get_loader() |
| `quantcore/strategies/base.py` | StrategyStatus enum |
| `quantcore/strategies/__init__.py` | Export StrategyStatus |
| `quantcore/strategies/macro/cta_trend.py` | DISABLED status |
| `quantcore/strategies/macro/risk_parity.py` | DISABLED status |
| `quantcore/strategies/macro/factor_residual.py` | DISABLED status; deprecation notice |
| `quantcore/strategies/equity/momentum.py` | DISABLED status |
| `quantcore/strategies/macro/cross_sectional_momentum_v2.py` | Max lev 1.0; corr stress 50% |
| `quantcore/risk/drawdown_guard.py` | VolAdjustedDrawdownGuard state machine |
| `quantcore/risk/liquidity.py` | ADV-based max_position_size |
| `scripts/real_data_backtest.py` | Bootstrap CI; threshold raises; v2 BacktestEngine path |
| `README.md` | Disclaimer; honest strategy table; removed false claims |

### Created
| File | Purpose |
|---|---|
| `tests/test_backtest_no_lookahead.py` | Look-ahead barrier test; purged CV crash test |
| `tests/unit/backtest/test_honest_backtesting.py` | 26 tests: drawdown, Sharpe, Student-t VaR, CF warning, purged CV, bootstrap CI, DD guard, liquidity, WF acceptance |
| `tests/unit/data/test_historical_universe.py` | PIT universe loader tests |
| `tests/unit/risk/test_metrics_fixed.py` | Sharpe NaN, Calmar equity, CI bounds |
| `quantcore/data/sources/universe_history.csv` | 24 S&P500 constituent changes 2013-2024 |
| `quantcore/alpha/__init__.py` | Alpha sub-package |
| `quantcore/alpha/residual_momentum.py` | Factor-neutral residual momentum skeleton |
| `scripts/verify_honest_foundation.py` | Honest foundation verification (BacktestEngine) |
| `INTERNAL_NOTES.md` | This document |

---

## 4. Known Limitations

1. **`_compute_signal_returns()` still exists** in `scripts/real_data_backtest.py`. The `run_full_period_backtest_v2()` wrapper uses BacktestEngine.run(), but the original path remains for backward compatibility. Full migration planned for Q2.

2. **Equity universe PIT coverage is limited.** `universe_history.csv` covers 24 S&P500 changes (2013-2024) and NASDAQ100 has 3 snapshots. Full CRSP/Compustat integration is required for institutionally valid equity backtests prior to 2013.

3. **ADV limits return unlimited sentinel** when ADV data is unavailable. `quantcore/risk/liquidity.py` needs real 30-day ADV data to enforce capacity constraints in production.

4. **yfinance data is OHLCV only** — no tick-level or order book validation. All backtests use daily bars from Yahoo Finance, which is sufficient for strategy validation but not for execution simulation.

5. **Verification script uses synthetic data fallback** when yfinance is unavailable. This is useful for CI but produces meaningless Sharpe ratios.

6. **Factor residual v3** still uses `_compute_signal_returns()` in real_data_backtest.py — needs BacktestEngine migration for full execution stack validation.

---

## 5. Q2 Alpha Rebuild Plan

Based on audit Tier-1 alpha sources:

### Priority 1: Factor-Neutral Residual Momentum
- Connect `quantcore/alpha/residual_momentum.py` to Fama-French 5-factor data (Ken French library)
- Universe: `ETF_MACRO` (55 assets, survivorship-bias minimal)
- Target: OOS Sharpe CI lower ≥ 0.3 on 2015-2024 data

### Priority 2: Equity Low-Volatility / Betting-Against-Beta
- Implement `equity/low_volatility.py` using BAB framework (Frazzini & Pedersen)
- Universe: S&P100 with PIT correction
- Risk: capacity limited; needs ADV data

### Priority 3: Cross-Sectional Momentum v2 Full Validation
- Full 2010-2024 backtest with 8+ folds on corrected engine
- Must reach OOS Sharpe CI lower ≥ 0.3 before paper trading

### Priority 4: Options Delta-Neutral Volatility Arbitrage
- Requires options data source (Polygon, CBOE) — research phase only

---

## 6. Verification

```bash
# Run honest foundation verification
python scripts/verify_honest_foundation.py
# Output: HONEST_FOUNDATION_REPORT.json

# Run full test suite
python -m pytest tests/ -v --tb=short

# Run Q1-specific tests
python -m pytest tests/unit/backtest/test_honest_backtesting.py -v
python -m pytest tests/unit/data/test_historical_universe.py -v
python -m pytest tests/test_backtest_no_lookahead.py -v
python -m pytest tests/unit/risk/test_metrics_fixed.py -v
```

Expected result: All Q1 infrastructure tests pass. Verification script reports FAIL for alpha performance (expected — alpha rebuild is Q2). The system is honest about its current limitations.

---

## Data provenance note (2026-08-04)

`data/sp500_pit_universe_2018_2024.csv` is TOTAL-RETURN-ADJUSTED (AMZN at $58.60 in
2018 = $1,180/20 back-adjusted), NOT raw/unadjusted as its filename and the
2026-08-04 plan text implied. Resolved as a non-issue for the gate: the pilot and the
decisive run source raw Massive/Polygon bars; the CSV remains triage-only. See
`docs/cross_sectional_factor_protocol.md` Section 3.

## Alpaca delisted-name data probe (2026-08-04)

Bounded free test before paying for Massive: daily bars 2018-01-01..2024-12-31 via
`https://data.alpaca.markets` (paper keys in `.env`).

| Symbol | Bars HTTP | n bars | first | last | asset endpoint |
|---|---|---|---|---|---|
| ATVI | 200 | 1456 | 2018-01-02 | 2023-10-13 | inactive, tradable=False (Activision Blizzard) |
| AIV | 200 | 1761 | 2018-01-02 | 2024-12-31 | active, tradable=True |
| ADS | 200 | 1071 | 2018-01-02 | 2022-04-01 | HTTP 404 (last bar 2022-04-01 = correct final ADS session; ticker changed to BFH on 2022-04-04) |
| AAPL (control) | 200 | 1761 | 2018-01-02 | 2024-12-31 | active, tradable=True |

Corporate actions: `GET /v2/stocks/{symbol}/corporate_actions` (AMZN splits 2022, ATVI
2023) -> HTTP 404 Not Found on the data API.

Reading: Alpaca returned clean, correctly-bounded daily bars for both true ticker-death
cases tested (ATVI through its merger close; ADS through the BFH ticker change) - better
than the plan assumed. What it does NOT provide (probe scope): a delisted/inactive
reference set to DISCOVER deaths (assets 404 for ADS; no survivorship-free universe
endpoint), corporate actions for the pilot's split/dividend/delisting cases (404 on the
data API), or a coverage guarantee beyond hand-picked names. Verdict: bars-only partial
coverage - the PIT pilot and decisive run still need Massive/Polygon (corp actions,
delisted reference set, reusable schema-complete pipeline). Alpaca stays execution/
live-feed only. IBKR data API not tested (frozen by decision; execution-only).

Side finding: `.env` `ALPACA_DATA_URL` holds the WebSocket streaming URL
(`wss://stream.data.alpaca.markets`), not a REST base URL - flagged, not changed.

## Data vendor alternatives survey (2026-08-04)

Question asked: "any alternative to Massive (Polygon) for the PIT backfill?" Surveyed the
current landscape. Requirements: survivorship-free PIT history (delisted names), raw EOD
bars, corporate actions (splits/dividends with dates), 5+ years, 500+ names,
bulk-friendly.

| Provider | Delisted reference set | Corporate actions | Approx cost | Fit for this repo |
|---|---|---|---|---|
| Massive (Polygon) | Yes (`active=false` flag) | Yes | $29/mo | Baseline - pipeline built for its schema |
| EODHD | Yes (`delisted=1` symbol list) | Yes (splits/dividends endpoints) | $19.99/mo EOD; +$29.99 constituents add-on | Capable; new adapter + pilot re-validation |
| Sharadar (via Nasdaq Data Link) | Yes (no survivorship bias; 20k+ US names) | Yes | Tiered; verify at signup | Capable; new adapter; professional vs non-professional licensing |
| Norgate | Yes (survivorship-free, backtest-native) | Yes | Tiered; verify at signup | Exact index constituents; desktop/CSV-oriented delivery |
| Databento | Corporate-actions dataset covers listed + delisted | Yes | Usage-based | Institutional-grade; more complex than needed now |
| Tiingo | Partial ("not yet recycled") | Splits API | Low; verify | Weak delisted guarantee |
| Alpaca (probed 2026-08-04) | No | 404 on data API | Free | Bars OK for known delisted names; execution/live only |
| IBKR | No (contract resolution) | - | - | Execution only (frozen by decision) |
| yfinance / Alpha Vantage | No (survivorship bias) | No | Free | Excluded - the original problem |

Decision: stay with Massive Starter for the pilot and the decisive run. Engineering and
calendar time to verdict are the expensive things; the ~$10/mo delta with EODHD is dwarfed
by a new adapter plus pilot re-validation. Revisit only if Massive proves deficient during
the pilot. One protocol nuance: exact S&P 500 membership history (the universe rule in
Section 3) is available from Norgate / Sharadar / EODHD add-on, whereas Massive gives
listing-date-based membership - noted as a possible future protocol amendment, not a
Step-7 blocker.

---

*End of INTERNAL_NOTES.md*
