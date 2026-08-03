# FREEZE — Effective 2026-08-02

The 2026-08-02 forensic audit found **zero deployable alpha** (DSR p=1.0 across
all 45 strategies) and identified a single bottleneck: **no real, point-in-time
market data**. Everything upstream of that is fiction until it is fixed.

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
