# H005: Quality Factor (Risk-Adjusted Return)

**Generated:** 2026-06-30 23:55 UTC
**Factors:** quality

## Economic Rationale

Asness, Frazzini, Pedersen (2013): Quality Minus Junk. High-quality firms (profitable, stable, low-leverage) earn higher risk-adjusted returns than theory predicts. Our proxy uses Sharpe-as-factor (return/vol) as a simple quality measure.

**Prior Expectation:** Weak to moderate positive IC. Close to low_vol factor.

## Findings

| Factor | IC | Rank IC | ICIR | Hit Rate | Stat. Sig. | CI (95%) | Verdict |
|--------|------:|--------:|-----:|---------:|-----------:|----------|---------|
| quality | +0.0039 | +0.0119 | 0.06 | 52.5% | ✗ | [+0.0063, +0.0293] | WEAK |

## Summary

**Weak/Inconclusive factors:** quality

## Regime Stability

- **quality:** Robust across regimes (✓)

## Capacity & Cost Viability

- **quality:** ✗ Not cost-viable (Cost: 784.4 bps/yr, Capacity: $95M)

## Failure Analysis

⚠ **No factors confirmed statistically significant.**

Potential root causes:
- Quality proxy too simple — real QMJ uses ROE, earnings stability, leverage
- High correlation with low_vol → likely redundant
- Data-mined: trailing Sharpe is a noisy proxy for true quality

## Lessons Learned

1. Signal extraction is correct, but no factor survives statistical validation.
2. This does NOT mean the economic rationale is wrong — it may mean:
   - The signal is too weak for this universe size (25 stocks)
   - The 10-year sample includes regime shifts that dilute average IC
   - Transaction costs and implementation lag erase the predicted edge
3. **Larger universe (100+ stocks) is needed for reliable cross-sectional IC estimation.**

## Next Steps

1. Expand universe to 100+ stocks for statistical power.
2. Obtain point-in-time data to eliminate survivorship bias.
3. Consider alternative signal construction (nonlinear, interaction).
4. Do NOT deploy any strategy until at least one factor is confirmed with cost viability.
