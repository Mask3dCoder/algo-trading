# H001: Cross-Sectional Momentum (Jegadeesh-Titman)

**Generated:** 2026-06-30 23:55 UTC
**Factors:** mom_1m, mom_3m, mom_6m, mom_12_1, ts_mom

## Economic Rationale

Underreaction: investors gradually incorporate new information into prices. Herding: trend-following reinforces momentum until reversal. The 12-1 month skip avoids contamination from short-term reversal.

**Prior Expectation:** Positive IC, stronger in 3-12m horizons, weaker in 1m due to reversal contamination.

## Findings

| Factor | IC | Rank IC | ICIR | Hit Rate | Stat. Sig. | CI (95%) | Verdict |
|--------|------:|--------:|-----:|---------:|-----------:|----------|---------|
| mom_1m | -0.0030 | -0.0046 | -0.02 | 50.2% | ✗ | [-0.0143, +0.0070] | WEAK |
| mom_3m | -0.0005 | +0.0035 | 0.02 | 51.9% | ✗ | [-0.0044, +0.0158] | WEAK |
| mom_6m | +0.0055 | +0.0118 | 0.05 | 51.8% | ✗ | [+0.0055, +0.0281] | WEAK |
| mom_12_1 | +0.0077 | +0.0175 | 0.07 | 53.6% | ✗ | [+0.0102, +0.0352] | WEAK |
| ts_mom | -0.0064 | -0.0062 | -0.04 | 48.5% | ✗ | [-0.0185, +0.0015] | WEAK |

## Summary

**Weak/Inconclusive factors:** mom_1m, mom_3m, mom_6m, mom_12_1, ts_mom

## Regime Stability

- **mom_1m:** Fragile — IC varies by regime (✗)
- **mom_3m:** Fragile — IC varies by regime (✗)
- **mom_6m:** Robust across regimes (✓)
- **mom_12_1:** Robust across regimes (✓)
- **ts_mom:** Fragile — IC varies by regime (✗)

## Capacity & Cost Viability

- **mom_1m:** ✗ Not cost-viable (Cost: 2359.6 bps/yr, Capacity: $95M)
- **mom_3m:** ✗ Not cost-viable (Cost: 1407.9 bps/yr, Capacity: $95M)
- **mom_6m:** ✗ Not cost-viable (Cost: 1026.0 bps/yr, Capacity: $95M)
- **mom_12_1:** ✗ Not cost-viable (Cost: 749.5 bps/yr, Capacity: $95M)
- **ts_mom:** ✗ Not cost-viable (Cost: 973.7 bps/yr, Capacity: $95M)

## Failure Analysis

⚠ **No factors confirmed statistically significant.**

Potential root causes:
- Momentum crash during sharp reversals (Mar 2009, Mar 2020)
- High turnover increases costs beyond IC benefit
- IC concentrated in small/mid-cap — weak in large-cap universe

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
