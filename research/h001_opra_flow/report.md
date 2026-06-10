# H001 OPRA Flow Event Study — Phase 4 Deferral Note

Date: 2026-06-09

## Status

**DATA_BLOCKED — no further H001 event-study work is permitted until a reliable OPRA data source is available.**

## Blocked data dependencies

The H001 OPRA flow event-study pipeline cannot be executed because all candidate OPRA sources are currently unavailable or invalid:

- Alpaca OPRA agreement: not signed.
- Polygon API key: invalid.
- IBKR gateway: unreachable.

## Locked discipline

Until a valid OPRA feed is available and documented:

1. Do not run the event study.
2. Do not run ingestion commands against OPRA, Polygon OPRA, Alpaca OPRA, or IBKR option-flow endpoints.
3. Do not create new H001 strategy code under `quantcore/`.
4. Do not tune or validate any H001-related strategy thresholds using future or test-window data.
5. Documentation and infrastructure cleanup are allowed.

## Current action

The H001 pipeline is deferred to a future data-access phase. The Phase 4 rollout will proceed only with registered, walk-forward-tested strategies that do not depend on OPRA flow data.

## Re-entry criteria

H001 may resume only after all of the following are true:

- A signed OPRA data agreement or approved substitute data source is available.
- The data source has been validated for timestamp correctness, symbol mapping, trade/aggressor fields, and coverage.
- A fresh Phase 3/4 validation plan is approved before any event-study execution.
