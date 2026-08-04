# Hypothesis H004: Volatility Risk Premium (VRP)

## 1. Core Hypothesis
Implied Volatility (IV) consistently overstates subsequent Realized Volatility (HV). The spread (VRP = IV - HV) represents a risk premium paid by hedgers. Selling options when VRP > 3% generates positive expected returns.

## 2. Economic Rationale
- **Risk premium:** Options buyers pay a premium for insurance (convexity). Sellers collect this premium structurally.
- **Structural demand:** Corporate hedgers, retail investors, and pension funds permanently demand put protection, creating persistent supply/demand imbalance.
- **Academic support:** Jackwerth & Rubinstein (1996), Carr & Wu (2009). VRP is well-documented in the academic literature.

## 3. Why It May Persist
The demand for downside protection is structural and nearly inelastic. Pension funds and corporations must hedge regardless of the cost. This flow is permanent.

## 4. Asset Universe
SPY / SPX options. Liquid strikes within ±2 standard deviations of spot.

## 5. Time Period
Train: 2010-2018, OOS Holdout: 2019-2024.

## 6. Required Data
**CRITICAL:** Daily closing IV from options chain (CBOE VIX or per-strike surface). Without this data, VRP cannot be evaluated. The repository does NOT currently have reliable IV data.

## 7. Expected Holding Period
1 month (selling 30-DTE options, exiting at 50% profit or 21 DTE).

## 8. Expected Failure Modes
- **Tail events:** Short volatility strategies suffer catastrophic losses during market crises (2008, 2020, 2022). Max drawdown can exceed 50%.
- **VIX spikes:** When VIX > 40, the strategy should be turned off. Selling into a volatility spike is capital destruction.
- **Transaction costs:** 10% round-trip spread assumed in original code — this alone destroys the 3% edge. Must be re-calibrated.

## 9. Transaction Cost Assumptions
Realistic spread: 2-5% of premium per leg. This must be empirically validated with historical bid/ask data.

## 10. Evaluation Criteria
- Positive VRP signal predicts positive next-month option seller P&L
- IC > 0.03 (options edge must be larger to survive spread costs)
- IC must be robust across non-crisis periods

## 11. Falsification Conditions
- If transaction-cost-adjusted IC < 0.02, the edge is insufficient. Reject.
- **Current Status: DATA BLOCKER** — This hypothesis cannot be evaluated without IV data. Do NOT advance to backtesting until real options data is obtained.
