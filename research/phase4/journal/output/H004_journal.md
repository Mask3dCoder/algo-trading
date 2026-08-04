# H004: Low Volatility Anomaly

**Generated:** 2026-06-30 23:55 UTC
**Factors:** low_vol

## Economic Rationale

Blitz & Van Vliet (2007): Low-beta stocks earn higher risk-adjusted returns. Causes: leverage constraints, benchmarking, lottery preference. Institutional investors constrained to hold high-beta stocks, overpaying for risk.

**Prior Expectation:** Positive IC (low vol → higher future returns), slow signal, low turnover.

## Findings

| Factor | IC | Rank IC | ICIR | Hit Rate | Stat. Sig. | CI (95%) | Verdict |
|--------|------:|--------:|-----:|---------:|-----------:|----------|---------|
| low_vol | -0.0156 | -0.0176 | -0.08 | 46.3% | ✗ | [-0.0391, -0.0147] | WEAK |

## Summary

**Weak/Inconclusive factors:** low_vol

## Regime Stability

- **low_vol:** Directionally stable but not consistently positive

## Capacity & Cost Viability

- **low_vol:** ✗ Not cost-viable (Cost: 519.2 bps/yr, Capacity: $95M)

## Failure Analysis

⚠ **No factors confirmed statistically significant.**

Potential root causes:
- IC vs. Sharpe trade-off: high Sharpe but low absolute return
- Underperforms in strong bull markets (missing upside)
- Concentrated in utilities/REITs — sector bet disguised as factor

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
