# H002: Short-Term Mean Reversion

**Generated:** 2026-06-30 23:55 UTC
**Factors:** rev_1w, rev_1m, zscore_20, rsi_14

## Economic Rationale

Overreaction: investors overreact to short-term news, creating predictable reversion. Liquidity provision: market makers demand compensation for absorbing order flow.

**Prior Expectation:** Negative 1-week IC (reversal), stronger in high-turnover stocks.

## Findings

| Factor | IC | Rank IC | ICIR | Hit Rate | Stat. Sig. | CI (95%) | Verdict |
|--------|------:|--------:|-----:|---------:|-----------:|----------|---------|
| rev_1w | +0.0046 | +0.0058 | 0.02 | 50.4% | ✗ | [-0.0037, +0.0169] | WEAK |
| rev_1m | +0.0030 | +0.0046 | 0.02 | 49.8% | ✗ | [-0.0070, +0.0143] | WEAK |
| zscore_20 | +0.0031 | +0.0052 | 0.02 | 50.7% | ✗ | [-0.0055, +0.0142] | WEAK |
| rsi_14 | +0.0179 | +0.0243 | 0.02 | 50.6% | ✗ | [-0.0058, +0.0149] | WEAK |

## Summary

**Weak/Inconclusive factors:** rev_1w, rev_1m, zscore_20, rsi_14

## Regime Stability

- **rev_1w:** Fragile — IC varies by regime (✗)
- **rev_1m:** Fragile — IC varies by regime (✗)
- **zscore_20:** Fragile — IC varies by regime (✗)
- **rsi_14:** Directionally stable but not consistently positive

## Capacity & Cost Viability

- **rev_1w:** ✗ Not cost-viable (Cost: 4642.9 bps/yr, Capacity: $95M)
- **rev_1m:** ✗ Not cost-viable (Cost: 2359.6 bps/yr, Capacity: $95M)
- **zscore_20:** ✗ Not cost-viable (Cost: 3658.9 bps/yr, Capacity: $95M)
- **rsi_14:** ✗ Not cost-viable (Cost: 1609.9 bps/yr, Capacity: $95M)

## Failure Analysis

⚠ **No factors confirmed statistically significant.**

Potential root causes:
- Reversal profits dominated by bid-ask bounce (not captured in daily close data)
- Costs from high turnover may exceed signal profits
- Signal decay too fast — 1-week IC may not survive execution lag

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
