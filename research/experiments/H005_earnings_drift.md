# Hypothesis H005: Post-Earnings Announcement Drift (PEAD)

## 1. Core Hypothesis
Stocks with large positive earnings surprises (vs. analyst consensus) continue to drift upward for 30-60 days after the announcement. The signal is the standardized unexpected earnings (SUE = actual EPS - expected EPS / std dev of past surprises).

## 2. Economic Rationale
- **Underreaction:** Analysts are slow to update their models. Investors anchor to prior estimates. The market gradually incorporates the full earnings information over weeks.
- **Academic support:** Ball & Brown (1968), Bernard & Thomas (1989, 1990). PEAD is one of the most robust anomalies in academic finance (50+ years of evidence).

## 3. Why It May Persist
Institutional mandates prevent rapid capital reallocation. Sell-side analyst conservatism creates persistent mispricing. Arbitrage requires earnings data which is expensive to procure.

## 4. Asset Universe
U.S. equities with quarterly earnings releases. Requires analyst consensus estimates (Bloomberg or FactSet).

## 5. Time Period
Train: 2005-2015, OOS Holdout: 2016-2024.

## 6. Required Data
**CRITICAL:** Standardized analyst consensus EPS estimates. This requires a vendor subscription (Refinitiv, Bloomberg, FactSet) or the free I/B/E/S dataset.

## 7. Expected Holding Period
30-60 trading days post-announcement.

## 8. Expected Failure Modes
- **Data cost:** Analyst consensus is expensive. If using free sources, data quality degrades.
- **Market efficiency improvement:** High-frequency traders and ML models have made PEAD weaker in the post-2015 period.

## 9. Transaction Cost Assumptions
Low turnover (quarterly events). Round-trip 15-25 bps.

## 10. Evaluation Criteria
- IC > 0.03 using SUE as signal
- IC decays slowly (positive for at least 30 trading days post-announcement)

## 11. Falsification Conditions
- If IC is not positive in the 30-day window following earnings, reject.
- **Current Status: DATA BLOCKER** — Requires earnings consensus data not currently in the repository.
