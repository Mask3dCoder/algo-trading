# H001: OPRA Flow Hedging Pressure

- Status: **locked before analysis**
- Phase: **Phase 3 hypothesis gate**
- Date locked: **2026-06-09**
- Purpose: define the structural mechanism and falsification rules before any Phase 3 strategy or event-study code is written.

## Gate rule

This document is the Phase 3 gate. No OPRA-flow strategy code, parameter search, or event-study implementation may be written before this hypothesis is committed.

## A. One-sentence mechanism

Large non-consensus option trades create short-horizon directional pressure because they force market makers to delta-hedge mechanically into the underlying before the hedge demand is fully absorbed by price.

## B. Tradable implication

If the mechanism is real, the data must show the following before costs and then after costs:

1. After large unusual call-heavy flow, the underlying should drift upward over the next tested horizons.
2. After large unusual put-heavy flow, the underlying should drift downward over the next tested horizons.
3. The effect should be monotonic with trade size.
4. The effect should be stronger for delta-sensitive, near-ATM contracts than for deep ITM or deep OTM contracts.

Holding horizons locked before looking at the data:

- +30 minutes
- +2 hours
- End of day
- Next open
- T+2 open

This is an event-study requirement, not a strategy requirement.

## C. Information edge

The observed edge is trade-level OPRA options flow that is public but computationally expensive to aggregate and interpret quickly. The market has not fully priced the signal when the observed flow implies near-term dealer delta-hedging demand into the underlying.

The edge is attention plus speed: identify delta-hedge demand before it is fully reflected in the underlying order book.

Capacity constraints:

- Target underlying market capitalization: **$500M to $10B**.
- Target underlying dollar ADV: **>= $5M**.
- Target options activity: the underlying must have active listed options liquidity, defined as **30-day option chain ADV >= 500 contracts**.
- Exclude names so liquid that the hedge-demand absorption lag is expected to be effectively zero.
- Exclude names so illiquid that the hypothesis cannot be detected.

## D. Counterparty: who is wrong and why

The market maker who sold the unusual option contract is not assumed to be wrong; they may have priced the option correctly.

The counterparty that is wrong is the rest of the market, because it does not immediately incorporate the implied dealer hedge-demand signal into the underlying price. They may not be observing the OPRA flow, may not be decoding it fast enough, or may not be able to act before hedging pressure moves price.

This is the weakest point. The more liquid the underlying, the faster the market absorbs hedge demand. Therefore, the hypothesis targets names where the absorption lag is measurable. If the lag is zero in the data, the hypothesis is falsified.

## E. Exclusion rules

These exclusions are locked before any analysis. No exclusion may be added after seeing event-study results.

Exclude:

1. Underlyings with dollar ADV < **$5M**.
2. Contracts with bid/ask spread > **10% of mid**.
3. Trades with zero or negative premium.
4. Trades where delta or implied volatility is unavailable.
5. Deep ITM and deep OTM contracts, defined as delta near 0 or 1, because they create minimal incremental hedging pressure.
6. The first and last **15 minutes** of regular trading hours, due to auction distortion.
7. Earnings windows of **±2 trading days**, unless the hypothesis is explicitly about earnings flow.
8. Identifiable spread legs if trade classification is available.
9. Contracts expiring the same trading day, because pinning behavior distorts flow interpretation.

If a subgroup does not work, that is a finding, not a reason to exclude it.

## F. Primary test design: event study, not backtest

The first Phase 3 code must be an event study. It must test whether the mechanism exists before asking whether it is tradable.

Event identification:

1. Identify unusual flow events on the training set.
2. Use `unusual_volume_ratio > threshold` and `large_order_flag = true`.
3. Lock the primary threshold at `unusual_volume_ratio >= 5.0`.
4. Classify each event as call-heavy, put-heavy, or neutral using signed delta-adjusted notional.
5. Align every event by `valid_from` timestamp, not by bar close.

Delta-adjusted notional:

```text
delta_adjusted_notional = signed_side × option_delta × contract_multiplier × underlying_price × number_of_contracts
```

Event classification:

- **Call-heavy**: positive signed delta-adjusted notional.
- **Put-heavy**: negative signed delta-adjusted notional.
- **Neutral-flow control group**: events near zero signed delta-adjusted notional or matched non-flow periods.

Forward return horizons:

- +30 minutes
- +2 hours
- End of day
- Next open
- T+2 open

Required breakdowns:

1. Call-heavy vs put-heavy vs neutral-flow control group.
2. Moneyness bucket.
3. Expiry bucket.
4. Underlying ADV bucket.
5. Market-cap tier.

Cost gate:

- The mean forward return must exceed **2× estimated round-trip cost** to be considered viable.
- If top-quintile call-heavy events do not show positive forward returns against the control group on the training set before parameter tuning, the hypothesis is falsified.

## G. Falsification conditions

The hypothesis is rejected if any of the following are true on the training-set event study:

1. Call-heavy flow does not predict positive forward returns after costs at any tested horizon.
2. Put-heavy flow does not predict negative forward returns after costs at any tested horizon.
3. The effect disappears after controlling for same-day underlying momentum.
4. The effect is not monotonic with delta-adjusted notional size.
5. The effect is not stronger for near-ATM contracts than for deep ITM or deep OTM contracts.
6. The effect is concentrated only in earnings windows.
7. The effect reverses within T+1, indicating microstructure noise rather than hedging pressure.
8. The DSR p-value is > **0.05** after accounting for all hypotheses tested across the full rebuild.

If any condition is met: reject H001, document the finding, and write H002.

## H. Data split

Alpaca OPRA history starts in February 2024. The split is locked now.

- **Training set:** February 2024 through September 2024, approximately 60%.
  - Use for signal development and the first event study.
- **Validation set:** October 2024 through January 2025, approximately 20%.
  - Use for parameter confirmation only.
- **Test set:** February 2025 through present, approximately 20%.
  - Locked. Do not touch until Phase 4 final evaluation.

If the test-set period is too short for statistical significance, accept a longer paper-trading period in Phase 4 rather than reopening the test set early.

## I. Transaction cost model

These cost assumptions are locked before testing and may not be adjusted after seeing results.

- Equity spread: **1 tick**, assumed as **$0.01** for names above $10 and proportional for lower-priced names.
- Commission: Alpaca paper is modeled as **0**, but live trading must model **0.5 bp per leg**.
- Market impact: assume **10% of order size moves price** for mid-cap names.
- Slippage buffer: add **50%** to spread estimates as conservatism.
- Round-trip cost threshold: the signal must produce forward returns greater than **2× estimated round-trip cost** to be considered viable.

Costs must be modeled pessimistically, not optimistically. The audit's four surviving strategies all failed this test; the best flipped from **+1.10** to **-0.13** Sharpe after costs.
