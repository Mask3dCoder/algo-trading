# Hypothesis H001: Cross-Sectional Equity Momentum

## 1. Core Hypothesis
Stocks that have outperformed their peers over the past 12 months (excluding the most recent month) will continue to outperform over the next 1-3 months. The signal is the rank of trailing 12-1 month return within the cross-section.

## 2. Economic Rationale
- **Underreaction:** Investors underreact to positive firm-level news (earnings, guidance upgrades). Price slowly drifts upward as the market reprices the stock over months.
- **Herding:** Institutional investors chase performance, creating momentum in price.
- **Academic Support:** Jegadeesh & Titman (1993), Carhart (1997), Asness et al. (2013). Momentum is one of the most robustly documented anomalies across asset classes and geographies.

## 3. Why It May Persist
Momentum requires sustained uncomfortable shorting of recent losers (career risk for institutions). Tax-loss harvesting creates artificial selling that reinforces momentum. Behavioral biases are persistent.

## 4. Asset Universe
U.S. equities: S&P 500 / Russell 1000 constituents. Requires price > $5, average daily dollar volume > $5M.

## 5. Time Period
Train: 2007-2017, OOS Holdout: 2018-2024.

## 6. Required Data
Daily adjusted OHLCV. Corporate actions (splits, dividends) required.

## 7. Expected Holding Period
1 month (21 trading days), rebalanced monthly.

## 8. Expected Failure Modes
- **Momentum crashes:** Historically happen during sharp reversals from bear markets (March 2009, March 2020). Can lose 30-50% in a single month.
- **Low dispersion:** Flat market returns compress the spread between winners and losers.

## 9. Transaction Cost Assumptions
Round-trip cost ≈ 15-30 bps (equity large-cap). Turnover ≈ 30% per month.

## 10. Evaluation Criteria
- Pearson IC > 0.02
- Monotonic quintiles (Q5 - Q1 > 0)
- Stable across rolling 6-month windows
- IC remains positive in both bull and bear regimes

## 11. Falsification Conditions
- If Q5 (top quintile) returns do not exceed Q1 (bottom quintile) returns consistently, reject.
- If IC degrades by > 50% in OOS vs IS (degradation ratio > 2.0), suspect overfitting.
