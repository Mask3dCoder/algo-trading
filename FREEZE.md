# FREEZE — Effective 2026-08-02

The 2026-08-02 forensic audit found **zero deployable alpha** (DSR p=1.0 across
all 45 strategies) and identified a single bottleneck: **no real, point-in-time
market data**. Everything upstream of that is fiction until it is fixed.

## DSR instrument correction — 2026-08-05 (integrity note)

The "DSR p=1.0" results that underpin the audit's zero-alpha conclusion were
computed with a **broken statistic**. The legacy
`quantcore.research.deflated_sharpe` implementation was fed *annualized*
Sharpe ratios into a formula that requires *per-observation* Sharpe ratios
(plus three smaller formula deviations: kurtosis term, `sqrt(T-0.25)`, and a
missing variance scaling in the multiple-testing benchmark). On the paper's
canonical case the published formula gives DSR p ≈ 0.48 where the legacy path
returned p ≈ 1.0. The corrected, verified implementation is
`quantcore.research.dsr_published` (commit `41981da`, audit in
`docs/dsr_audit.md`).

Consequence, stated plainly:

- Every prior DSR p=1.0 result — the "0/46 KEEP, DSR p=1.0 unanimous" outcome
  of the May, June, and August 2026 audits — is **uninformative, not
  confirmatory** of "no alpha". It was based on a since-corrected calculation.
- This does **not** retract those audits, and it does **not** re-open them:
  their data was yfinance/ETF-derived and is unusable for a verdict either
  way, and re-scanning 46 strategies is explicitly **out of scope**.
- This finding does **not** trigger a re-scan of the 46 strategies. The freeze
  on new strategies, signals, and experiments stays in force; the three
  pre-registered protocol variants stay locked.
- The consequence is exactly one change in state: the one pre-registered
  decisive experiment (`docs/cross_sectional_factor_protocol.md`) will now run
  on a **correct** DSR, on real point-in-time data, for the first time. The
  verdict of that run — pass or fail — is the first DSR-based verdict this
  project can trust.

This note exists so that anyone reading the audit trail later (including an
investor) does not treat the old numbers as clean evidence, and does not
wonder why a gate-breaking bug did not reopen everything.

## Frozen — do NOT write until unfrozen

- ❌ No new strategies
- ❌ No new signals / alpha generators
- ❌ No new backtest experiments in the daily-bar ETF search space
- ❌ No new one-off scripts
- ❌ No validation of synthetic or yfinance-derived data as if it were real

## Allowed — the only work that changes the outcome

- ✅ Real data acquisition: Polygon.io subscription, PIT market data
  (500+ liquid equities), real options NBBO chains, corporate actions
- ✅ PIT data ingestion pipeline (`quantcore/data/pit/`,
  `quantcore/data/sources/polygon.py`)
- ✅ Backtest engine cost model: real bid-ask spread, market impact,
  borrow/financing, calibrated commissions + slippage
- ✅ The `buy_and_hold(SPY)` sanity gate — until it passes, all backtests are fiction
- ✅ Mechanical codebase triage under the `pre-forensic-audit` tag

## Unfrozen-but-segregated work

- 🔀 Broker consolidation WIP (duplicate `quantcore/brokers/` vs
  `quantcore/execution/brokers/`, plus `live/` + `risk/` touch points) is
  preserved on the **`broker-consolidation-wip`** branch (`b9c8dee`). It was
  swept there un-reviewed to keep the data-pipeline line clean. Do not merge
  into `main` without review; treat it as a parallel line of work.

## Unfreeze gate

New strategy/signal code may resume only after:

1. Real PIT data for a 500+ ticker universe is loading.
2. The backtest cost model passes the `buy_and_hold(SPY)` sanity check.
3. Walk-forward + purged k-fold + DSR p < 0.05 are enforced as a hard gate.

Rollback point: `git tag pre-forensic-audit` (HEAD `61daebe`).
