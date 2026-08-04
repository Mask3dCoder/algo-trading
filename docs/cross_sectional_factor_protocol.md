# Cross-Sectional Factor Experiment — Pre-Registered Protocol

**Status: PRE-REGISTERED and LOCKED on 2026-08-04, before any factor results were computed.**

This document fixes the decisive experiment in advance. It is written *before* the first-pass
smoke test and *before* the real backfill, so the design cannot be tailored to results.
Amendments require a dated, justified note (below). The **gate thresholds are not amendable
after results are seen.**

---

## 1. Research question

Do cross-sectional factor premia (**momentum**, **low-volatility**) on a point-in-time US
equity universe, implemented **long-only, monthly-rebalanced, net of realistic costs**,
produce returns that **beat SPY** over the same window, with significance that survives
multiple-testing correction (DSR p < 0.05)?

## 2. Why this thesis

- Cross-sectional momentum (Jegadeesh–Titman) and low-volatility (Ang et al.) are the most
  documented cross-sectional premia. They are price-only and testable on the data tier this
  project can afford.
- The prior best cost-adjusted result in this codebase was `equity_full_universe_xs_momentum`
  at Sharpe 1.27, CI_lower −0.12 (a coin flip), computed on approximate data. This experiment
  re-tests that family **pre-registered, on survivorship-reduced real data, with a fixed gate**.
- Long-only + monthly rebalance is chosen to minimise turnover/costs and be honestly
  implementable at retail scale. Shorting and leverage are explicitly **out of scope**.

## 3. Data (fixed)

- **Decisive dataset:** Massive/Polygon PIT backfill, 500+ universe, survivorship-free
  (delisted tail retained), raw unadjusted bars + separate corporate-actions table, PIT
  membership (names included only from their actual listing / index-entry date; excluded
  after delisting/removal). Sourced via `bin/polygon_ingest.py` into the PIT store.
- **Evaluation window (rule, not a chosen number):** the full clean post-formation period
  the backfill yields, with the first 18 months reserved as formation/training burn-in.
  The exact start/end are recorded in the results file at run time — they are set by data
  availability, not by what makes the numbers look good.
- **Smoke-test panel (triage ONLY, excluded from the gate):** `data/sp500_pit_universe_2018_2024.csv`
  — real Alpaca data, survivorship-reduced (612 names incl. delisted), total-return-adjusted,
  approximate Wikipedia membership. Used only to catch gross pipeline errors.
- **Benchmark:** SPY **total return** over the same evaluation window, computed from SPY bars
  in the backfill + dividends through the engine's corporate-action handling, cross-checked
  against the committed Yahoo fixture (`tests/fixtures/spy_raw_daily_2020_2024.csv`).

## 4. Universe rules (fixed)

- Point-in-time 500: membership as of each rebalance date T. Delisted/removed names held until
  their last traded day, then dropped. Not-yet-listed names never enter.
- Liquidity filter applied **as of T** (no lookahead): price ≥ $2 and market cap ≥ $1B.
  Exact filter thresholds recorded at run time.

## 5. Factors (≤ 3 variants, fixed, price-only)

| Variant | Definition (trading days) |
|---|---|
| `momentum` | 12-month cumulative return, skipping the most recent month: `close[T-21] / close[T-273] - 1` |
| `low_vol` | Trailing 6-month realized volatility of daily returns (126 days), ranked ascending |
| `composite` | Equal-weight average of the two sign-corrected cross-sectional z-scores (higher = better) |

- **No parameter search.** Lookbacks are fixed above. No other factor definitions, no
  tuning, no alternative variants may be added after results are seen.

## 6. Portfolio construction (fixed)

- Long-only **top quintile** (top 20% of the liquid universe by factor rank) at each monthly
  rebalance (21 trading days).
- **Equal-weighted**, fully invested in risky assets. No leverage, no shorting.
- Rebalance on the close of the formation date; hold for the next 21 trading days.

## 7. Costs (fixed)

- Per the existing validation harness (`scripts/validate_all_strategies.py`): **20 bps per
  unit of turnover** (5 bps half-spread + 5 bps spread + 10 bps impact, one-way) applied to
  the value traded at each rebalance (actual names added/removed × 20 bps).
- Commission: $0.005/share (harness default).
- No financing (long-only, no margin). Cost configuration recorded verbatim in the run.

## 8. Evaluation and the gate (hard, pre-committed — ALL must hold)

1. **Anchored walk-forward** over the evaluation window; OOS Sharpe annualized per fold and
   pooled.
2. **OOS Sharpe 90% block-bootstrap CI lower bound ≥ 0.5** (2000 reps, block_len = 63).
3. **Deflated Sharpe Ratio (Bailey & López de Prado) p-value < 0.05**, computed across the
   full variant family tested (corrects for multiple testing).
4. **Net-of-cost total return over the OOS window > SPY total return** over the same window.
5. **Margin rule (gray zone, lock point 2026-08-04):** a variant is only a PASS if it
   clears every threshold with margin, not just on the correct side of it:
   - DSR p <= 0.04 (not just < 0.05)
   - OOS Sharpe CI_lower >= 0.6 (not just >= 0.5)
   - Net-of-cost annualized return exceeds SPY by >= 1.5 percentage points over the full
     window (not merely > 0)
   Any variant that clears the raw threshold but fails the margin is recorded as
   **FAIL - insufficient margin**, not as a pass under review. No exceptions, no re-running
   with adjusted parameters.

- A variant failing **any** criterion is **dead**. No re-tuning, no cherry-picking, no
  re-running with different windows.
- **If ALL variants fail the gate → the thesis FAILS**, and the pre-committed decision
  applies: stop the alpha search, do not raise capital, invest personal capital in
  SPY / low-cost factor ETFs. The system remains a research + monitoring platform.

## 9. If the gate passes (pre-committed next steps)

- The single best variant (highest OOS Sharpe CI_lower) proceeds to a **12-month
  paper-trading phase, fixed at the lock point (2026-08-04; see amendment log)**, on **IBKR**
  (primary broker) or Alpaca paper (already wired), with daily reconciliation, the
  compliance hash-chain audit trail, and monthly investor-grade reporting.
- **Fund formation (legal/registration, structure, LP docs) is a separate, user-owned track**
  that begins only after the 12-month paper phase and the personal-live pilot below.
- Quarterly live re-test; halt the strategy on any 3-month rolling Sharpe < 0 (existing
  kill-switch convention).

### Lock-point execution decisions (2026-08-04)

- **Paper phase:** 12 months (fixed at lock; a 6-month phase was considered and rejected
  because it adds almost no statistical power on top of ~60 OOS observations). Paper
  notional: $10,000. Daily reconciliation; monthly reporting.
- **Personal-live pilot (overlaps the paper phase; resolved 2026-08-04):** starts after
  the first 6 clean paper months (no 3-month rolling Sharpe < 0, reconciliation clean)
  and runs concurrently with paper months 7-12: $25,000 of personal capital, 6 months of
  live evidence. Total post-PASS runway to external-capital readiness: 12 months, not 18.
  No external capital before the full 12-month paper phase completes clean AND the live
  pilot is clean (net-of-cost >= SPY over the live window, no 3-month rolling Sharpe < 0)
  at fund-formation time.
- **Fund-formation triggers:** (1) 12-month paper phase clean; (2) 6 months of live
  personal-capital evidence clean at fund-formation time; (3) legal/registration track
  resolved; (4) only then begin any external raise, sized conservatively and structured
  under the applicable exemption with counsel.
- **Legal/registration track (user-owned, started 2026-08-04):** research
  SEC/FINRA-exempt vs registered-adviser status and fund structure NOW, in parallel
  with the data work, so a PASS cannot be blocked later by a surprise. This is not code
  work.

## 10. Pre-registration integrity

- This document was drafted as a pre-registration on 2026-08-04. See the amendment log for
  the chronology and integrity statement (the verifiable git lock point).
- Any amendment to data window, factor definition, cost assumption, or universe rule requires
  a dated, justified note below. Amendments to **Section 8 (the gate)** are prohibited after
  results are seen.
- The decisive run must dump `reports/cross_sectional_factor_results.json` containing, per
  variant: full OOS return series, OOS Sharpe + CI, DSR p, net-of-cost vs SPY, turnover,
  trade count, per-year Sharpe.

## 11. Declared limitations

- 5-year data horizon (Starter tier) limits statistical power; DSR is the guardrail.
- One formation per month (no overlapping-portfolio averaging).
- Daily bars only; market impact approximated by the cost model, not simulated fills.
- Tests ONE thesis. Options/VRP, intraday, and alt-data are out of scope by decision
  (would need the $199 tier + IV plumbing + short-vol operational risk).

---

### Amendment log
- **2026-08-04 (lock point) - chronology and integrity statement:** This document was
  found untracked in git with no verifiable authorship timestamp. We cannot
  cryptographically prove it was finalized before any informal smoke-test results
  existed. To remove ambiguity going forward: the git commit made today, hash
  `ed52bae`, is the actual pre-registration checkpoint. No thresholds, variants,
  universe, or window are altered after this commit. Any prior smoke-test output
  (if it exists) was triage-only, used to catch gross errors, and did not inform the
  variant selection or thresholds below.
- **2026-08-04 (lock point) - gray-zone / margin rule:** Section 8 criterion 5 added
  before any backfill. Thresholds tightened at lock; the gate remains unamendable
  after results are seen.
- **2026-08-04 (lock point) - execution-phase decisions:** paper phase fixed at 12
  months; paper notional $10,000; personal-live pilot $25,000 for 6 months;
  fund-formation triggers and the legal/registration track recorded in Section 9.
- **2026-08-04 (lock point) - pilot sequencing resolved:** the $25,000 personal-live
  pilot OVERLAPS paper months 7-12 (starts after 6 clean paper months) instead of
  running sequentially after month 12. Total post-PASS runway to external-capital
  readiness: 12 months, not 18. The live pilot proves execution, not alpha; overlap
  keeps the raise on a deliberate 12-month clock while preserving the kill-switch
  safeguards.
