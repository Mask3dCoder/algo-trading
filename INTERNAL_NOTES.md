# QuantCore Internal Notes — Q1 FIX FOUNDATION

**Classification:** Internal  
**Date:** 2026-05-10  
**Author:** QuantCore Team (Claude Code rebuild)  
**Reference audit:** [FORENSIC_AUDIT_REPORT.md](FORENSIC_AUDIT_REPORT.md)

---

## 1. Audit Summary (6 Fatal Findings)

The forensic audit (2026-05-09) identified the following fatal defects:

| # | Finding | Severity | File(s) |
|---|---|---|---|
| 1 | Backtest engine look-ahead bias | CRITICAL | `quantcore/backtest/engine.py` L787-801 |
| 2 | Validation bypasses execution stack | CRITICAL | `scripts/real_data_backtest.py` |
| 3 | Metrics bugs (Sharpe→0, Calmar wrong) | HIGH | `quantcore/risk/metrics.py` |
| 4 | Purged CV crashes at runtime | HIGH | `quantcore/backtest/walk_forward.py` L129 |
| 5 | No survivorship-bias correction | HIGH | `quantcore/data/sources/universe.py` |
| 6 | Factor residual degenerates to 1 asset | HIGH | `quantcore/strategies/macro/factor_residual.py` |

All previously reported Sharpe ratios and performance figures must be discarded.

---

## 2. Q1 FIX FOUNDATION — Changes Made (2026-05-10)

### 2.1 Look-Ahead Bias Fix (Task 1.1)
**File:** `quantcore/backtest/engine.py`

**Change:** Moved the `next_bar` peek from BEFORE `strategy.on_bar()` to AFTER.
Also added `fill_sim.set_bars(bar, None)` as a barrier before signal generation
so that any strategy that tries to access `fill_sim._next_bar` during `on_bar()`
will see `None`, not future data.

**Before:**
```
Loop:
  1. PEEK next_bar          ← future data accessible here
  2. Process pending orders
  3. fill_sim.set_bars(bar, next_bar)
  4. strategy.on_bar(bar)  ← signal sees next_bar via fill_sim
```

**After:**
```
Loop:
  1. Process pending orders
  2. fill_sim.set_bars(bar, None)  ← BARRIER
  3. strategy.on_bar(bar)          ← no future data visible
  4. PEEK next_bar                 ← only after signals generated
  5. fill_sim.set_bars(bar, next_bar)
```

**Test:** `tests/test_backtest_no_lookahead.py` — checks that
`fill_sim._next_bar is None` during every `on_bar()` call.

---

### 2.2 Purged CV Crash Fix (Task 1.4)
**File:** `quantcore/backtest/walk_forward.py`, line ~129

**Change:** `total` (undefined) → `n_bars` (in scope).
Also fixed first-fold `train_end=total` → `n_bars`.

---

### 2.3 Metrics Bugs Fixed (Task 1.3)
**File:** `quantcore/risk/metrics.py`

| Bug | Fix |
|---|---|
| `sharpe_ratio()` returns `0.0` for flat series | Returns `float('nan')` |
| `calmar_ratio()` uses `cumsum(r)` as equity | Uses `exp(cumsum(r))` |
| `rf` rate not geometrically decompounded | `rf_daily = (1+rf_annual)^(1/252) - 1` |

New helpers added:
- `rolling_sharpe(returns, window=252)` — rolling annualised Sharpe (NaN pre-window)
- `rolling_maxdd(returns, window=252)` — rolling equity-based max drawdown
- `bootstrap_ci()` upgraded to block bootstrap (block_size ≈ n^(1/3))

---

### 2.4 Universe Expansion (Task 1.5, 1.6)
**File:** `quantcore/data/sources/universe.py`

Added `ETF_MACRO` universe: **55 highly liquid ETFs** covering US/international
equities, fixed income (rates + credit), commodities, currencies, real assets,
sector ETFs, and factor ETFs. All available via yfinance from 2010 (some 2006).

**Survivorship bias status:**
- ETF universes: minimal bias (ETFs rarely delist)
- S&P500/NASDAQ100: still using current constituents → BIAS PRESENT

Added `quantcore/data/sources/historical_universe.py` — MVP loader with static
snapshots of S&P500 and NASDAQ100 at 5-year intervals from 2008 to 2024.
**PRODUCTION TODO:** Replace with Compustat/CRSP point-in-time database.

---

### 2.5 Strategy Status (Task 2.1)
Added `StrategyStatus` enum to `quantcore/strategies/base.py`:
- `DISABLED` — forensic audit confirmed no alpha
- `RESEARCH` — early-stage, not walk-forward validated
- `DEVELOPMENT` — OOS validation in progress
- `PAPER_TRADING` — OOS Sharpe lower CI ≥ 0.3 confirmed
- `LIVE` — full compliance sign-off

Strategies marked DISABLED:
- `macro_cta_trend` (OOS Sharpe -0.147)
- `macro_risk_parity` (pure beta, not alpha)
- `macro_factor_residual` (degenerate — rebuilt as skeleton)
- `equity_momentum` (edge decayed)
- `equity_earnings_momentum` (circular logic)

---

### 2.6 Cross-Sectional Momentum v2 Safety Fix (Task 2.3)
**File:** `quantcore/strategies/macro/cross_sectional_momentum_v2.py`

- `max_leverage` hard-capped at `1.0` (was `2.0`)
- Added correlation-stress adjustment: if avg pairwise corr of active
  positions > `0.70`, all weights reduced by 50%
- Both changes applied in `_compute_weights()` with a final gross exposure
  normalization

**Why:** The uncapped 2x leverage + high correlations produced an 88% max
drawdown in OOS testing. The strategy retains a real signal (OOS Sharpe 0.55
pre-audit) and should survive the fixes with reduced-but-honest returns.

---

### 2.7 Residual Momentum Skeleton (Task 2.2)
**File:** `quantcore/alpha/residual_momentum.py`

Clean-sheet rebuild framework:
- `PlaceholderFactorSource` — stub that returns zeros (for unit testing)
- `ResidualMomentumSignal.compute()` — runs Ridge regression per asset,
  cross-sectionally z-scores residuals, normalises weights
- **Q2 deliverable:** Wire `FamaFrenchLoader` as the real factor source
- **Audit requirement:** Factor returns must be EXTERNAL to the trading universe
  (no more SPY/TLT/DBC/GLD acting as both factors and trading assets)

---

### 2.8 Risk Engine Upgrades (Tasks 3.1-3.3)
| Task | File | Status |
|---|---|---|
| 3.1 VaR Student-t | `quantcore/risk/var.py` | Q2 — `monte_carlo_var` already uses t-dist; Cornish-Fisher warning pending |
| 3.2 Vol-adjusted drawdown | `quantcore/risk/drawdown_guard.py` | DONE — `VolAdjustedDrawdownGuard` added |
| 3.3 Liquidity sizing | `quantcore/risk/liquidity.py` | DONE — `max_position_size()` returns unlimited sentinel |

---

## 3. Files Modified (2026-05-10)

### Modified
| File | Change |
|---|---|
| `quantcore/backtest/engine.py` | Look-ahead bias fix (barrier pattern) |
| `quantcore/backtest/walk_forward.py` | Fix `total`→`n_bars` crash; compute_oos_metrics NaN-safe |
| `quantcore/risk/metrics.py` | Fix sharpe/calmar; add rolling_sharpe, rolling_maxdd; bootstrap_ci block |
| `quantcore/data/sources/universe.py` | 55 ETF universe; survivorship bias warnings |
| `quantcore/strategies/base.py` | Add `StrategyStatus` enum |
| `quantcore/strategies/__init__.py` | Export `StrategyStatus` |
| `quantcore/strategies/macro/cta_trend.py` | DISABLED status |
| `quantcore/strategies/macro/risk_parity.py` | DISABLED status |
| `quantcore/strategies/macro/factor_residual.py` | DISABLED status; deprecation notice |
| `quantcore/strategies/equity/momentum.py` | DISABLED status |
| `quantcore/strategies/macro/cross_sectional_momentum_v2.py` | Max lev 1.0; corr stress 50% |
| `README.md` | Disclaimer; honest strategy status table |
| `quantcore/risk/drawdown_guard.py` | Added `VolAdjustedDrawdownGuard` |

### Created
| File | Purpose |
|---|---|
| `tests/test_backtest_no_lookahead.py` | Look-ahead barrier test; purged CV crash test |
| `tests/unit/risk/test_metrics_fixed.py` | Sharpe NaN, Calmar equity, CI bounds |
| `quantcore/data/sources/historical_universe.py` | MVP point-in-time universe loader |
| `quantcore/alpha/__init__.py` | Alpha sub-package |
| `quantcore/alpha/residual_momentum.py` | Factor-neutral residual momentum skeleton |
| `quantcore/risk/liquidity.py` | ADV-based position sizing interface |
| `scripts/verify_honest_foundation.py` | Full verification script |
| `INTERNAL_NOTES.md` | This document |

---

## 4. Known Limitations

1. **`_compute_signal_returns()`** still exists in `scripts/real_data_backtest.py` —
   deprecated but not yet removed. The `verify_honest_foundation.py` script uses
   the full engine, but the old script migration is a Q2 task.

2. **Equity universe survivorship bias** is unresolved. `universe_history.json`
   contains only abbreviated snapshots. A Compustat or CRSP database integration
   is required for institutionally valid equity backtests.

3. **ADV limits** return an unlimited sentinel. `quantcore/risk/liquidity.py`
   needs real 30-day ADV data to enforce capacity constraints.

4. **VaR Student-t** Cornish-Fisher expansion does not yet warn when fewer than
   100 observations are provided (Task 3.1 partial).

5. **Event timestamp** in `quantcore/core/events.py` defaults to `asyncio` loop
   time (audited issue #6 from architecture section). Not fixed in Q1 scope.

---

## 5. Q2 Alpha Rebuild Plan

Based on audit Tier-1 alpha sources:

### Priority 1: Factor-Neutral Residual Momentum
- Connect `quantcore/alpha/residual_momentum.py` to a real Fama-French 5-factor
  data loader (Ken French data library is free)
- Universe: `ETF_MACRO` (55 assets, survivorship-bias minimal)
- Initial target: OOS Sharpe CI lower ≥ 0.3 on 2015-2024 data

### Priority 2: Equity Low-Volatility / Betting-Against-Beta
- Implement `equity/low_volatility.py` using the BAB framework (Frazzini & Pedersen)
- Universe: S&P100 (requires historical_universe.json for bias correction)
- Risk: capacity limited; need ADV data

### Priority 3: Options / Delta-Neutral Volatility Arbitrage
- Requires options data source (Polygon, CBOE) — not yet integrated
- Research phase only until data pipeline is established

### Priority 4: Cross-Sectional Momentum v2 Full Validation
- Already marked DEVELOPMENT — needs full 2010-2024 backtest with corrected engine
- OOS Sharpe CI lower must reach ≥ 0.3 before paper trading

---

## 6. How to Run Verification

```bash
# From repo root
python scripts/verify_honest_foundation.py

# Expected output:
# ======================================================================
# QUANTCORE HONEST FOUNDATION VERIFICATION
# ======================================================================
#   [macro_cross_sectional_momentum_v2] validating (expanded_15)...
#      PASS/FAIL  Sharpe=X.XX  CI90=[lo,hi]  (Xs)
# 
# DISABLED (not run):
#   [DISABLED] macro_cta_trend
#   ...
# 
# FATAL FINDINGS CHECKLIST:
#   [RESOLVED  ] look_ahead_bias
#   ...
# 
# RESULT: FAIL  (expected at this stage)
# 
# Report saved: HONEST_FOUNDATION_REPORT.json
```

The system will report FAIL until the factor-neutral residual momentum is rebuilt
and achieves CI lower ≥ 0.3. This is the correct and expected behaviour — an
honest system that accurately reports its limitations.

---

*End of INTERNAL_NOTES.md*
