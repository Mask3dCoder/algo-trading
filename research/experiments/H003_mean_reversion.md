# Hypothesis H003: Short-Term Mean Reversion

## 1. Core Hypothesis
Stocks that have deviated significantly from their 20-day rolling mean (high z-score) will revert toward the mean over 1-5 trading days. Signal = negative z-score (buy the dip, sell the spike).

## 2. Economic Rationale
- **Overreaction:** Retail and algorithmic traders overreact to news or order flow imbalances. The excess move corrects as markets digest information.
- **Liquidity provision:** Market makers and stat-arb desks exploit short-term deviations. This strategy is essentially providing liquidity.
- **Academic support:** De Bondt & Thaler (1985), Lehmann (1990), Lo & MacKinlay (1990).

## 3. Why It May Persist
Behavioral overreaction is persistent. High turnover creates barriers (most institutional capital cannot trade at this frequency profitably after costs).

## 4. Asset Universe
Large-cap U.S. equities with tight bid-ask spreads. Poor for small-caps (execution costs destroy edge).

## 5. Time Period
Train: 2012-2019, OOS Holdout: 2020-2024.

## 6. Required Data
Daily OHLCV. Intraday VWAP for more refined signal.

## 7. Expected Holding Period
1-5 trading days.

## 8. Expected Failure Modes
- **Trending markets:** High z-score continuation (momentum) destroys this strategy during persistent trends.
- **Transaction costs:** Daily turnover means execution costs are critical. Large-cap only.

## 9. Transaction Cost Assumptions
Round-trip 20-40 bps (higher frequency). Strategy requires IC > 0.05 to survive these costs.

## 10. Evaluation Criteria
- Pearson IC > 0.03 (higher bar due to transaction cost burden)
- IC positive at 1-day lag, decays quickly (consistent with reversion hypothesis)
- IC must be negative by day 10 (signal should be fully exhausted)

## 11. Falsification Conditions
- If IC at lag 1 is positive (instead of the reversion signal), the market regime is trending and this hypothesis fails.
- If IC does not decay to zero by 10 days, the signal is not capturing mean reversion.
