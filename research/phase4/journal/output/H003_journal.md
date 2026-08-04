# H003: Volatility Risk Premium (VRP Proxy)

**Generated:** 2026-06-30 23:55 UTC
**Factors:** vrp_proxy, vol_regime

## Economic Rationale

Implied vol systematically exceeds realized vol due to: (1) risk aversion — investors overpay for downside protection, (2) volatility selling as insurance analog. The VRP is one of the most documented risk premia in options literature.

**Prior Expectation:** Positive IC (VIX > RV → premium available to sellers), regime-conditioned.

## Findings

| Factor | IC | Rank IC | ICIR | Hit Rate | Stat. Sig. | CI (95%) | Verdict |
|--------|------:|--------:|-----:|---------:|-----------:|----------|---------|
| vrp_proxy | -0.0126 | -0.0132 | -0.06 | 47.8% | ✗ | [-0.0295, -0.0078] | WEAK |
| vol_regime | +0.0041 | +0.0243 | — | 47.4% | ✗ | [-0.0000, +0.0000] | WEAK |

## Summary

**Weak/Inconclusive factors:** vrp_proxy, vol_regime

## Regime Stability

- **vrp_proxy:** Directionally stable but not consistently positive
- **vol_regime:** Robust across regimes (✓)

## Capacity & Cost Viability

- **vrp_proxy:** ✗ Not cost-viable (Cost: 1221.1 bps/yr, Capacity: $95M)
- **vol_regime:** ✗ Not cost-viable (Cost: 2260.4 bps/yr, Capacity: $95M)

## Failure Analysis

⚠ **No factors confirmed statistically significant.**

Potential root causes:
- VIX is SPX-specific — poor proxy for individual stock IV
- Equity proxy cannot capture the true options-based VRP
- Tail events (vol spikes) can wipe months of premium

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
