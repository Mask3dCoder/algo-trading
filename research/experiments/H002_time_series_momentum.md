# Hypothesis H002: Time-Series Momentum (TSMOM)

## 1. Core Hypothesis
Each asset's own sign of 12-month trailing return predicts the sign of its next month return. Long assets trending up, short assets trending down. Position size is scaled inversely by realized volatility to maintain equal risk contribution.

## 2. Economic Rationale
- **Trend following:** Price trends emerge from serial autocorrelation in asset returns driven by investor herding and slow information diffusion.
- **Academic Support:** Moskowitz, Ooi, Pedersen (2012): documented across 58 futures markets over 25 years. AQR's core managed futures edge.

## 3. Why It May Persist
Managed futures / CTA industry is a natural enforcer of this trend. The strategy has negative skewness costs (occasional large reversals) that discourage arbitrageurs.

## 4. Asset Universe
Initially: SPY as single-asset test. Extension: multi-asset (equities + futures) cross-sectional application.

## 5. Time Period
Train: 2007-2017, OOS Holdout: 2018-2024.

## 6. Required Data
Daily adjusted close prices. Annualized realized volatility (60-day window).

## 7. Expected Holding Period
1 month (21 trading days), rebalanced monthly.

## 8. Expected Failure Modes
- **Whipsaw:** Sideways, choppy markets generate false trend signals and incur repeated losses.
- **Momentum crashes:** Same as H001 — sharp reversals punish trend followers.

## 9. Transaction Cost Assumptions
Turnover lower than H001 (signal only flips sign, not rank). Round-trip 10-20 bps.

## 10. Evaluation Criteria
- Pearson IC > 0.02
- IC remains positive in trending (bull) regime
- IC does NOT need to remain positive in all regimes (trend-following is regime-specific)

## 11. Falsification Conditions
- If IC is not positive in trending markets, reject.
- If signal flips sign more than 30% of months (high turnover), the edge is likely insufficient to cover costs.
