# Promotion Committee Report

**Date:** 2026-07-01 17:41 UTC
**Universe:** 100 large-cap US equities (S&P 100)
**Period:** 2013-01-01 → 2024-12-31
**Horizons evaluated:** 1-day, 5-day, 21-day forward returns
**Significance Level:** α = 0.10 (t-test + permutation + BH-FDR)
**Multi-Horizon Gate:** Factor must pass gate at any single horizon to PROMOTE/REFINE

## Summary

| Decision | Count | Factors |
|----------|------:|---------|
| 🟢 PROMOTE | 0 | — |
| 🟡 REFINE | 0 | — |
| ⚪ HOLD | 6 | mom_6m, mom_12_1, rev_1w, zscore_20, vrp_proxy, vol_regime |
| 🔵 ARCHIVE | 7 | mom_1m, mom_3m, ts_mom, rev_1m, rsi_14, low_vol, quality |
| 🔴 REJECT | 0 | — |

**Total factors evaluated:** 13
**Promoted:** 0 | **Refinable:** 0 | **Pending:** 13 | **Rejected:** 0

---

## Horizon Comparison Matrix

Each factor evaluated at h=1, h=5, h=21 independently. Decision at each horizon uses the 3-way AND-gate (t-test p<0.10 AND permutation p<0.10 AND bootstrap CI lower > 0).

| Factor | h | IC | Perm p | Min IC | Cost Viable | Decision ||------|--|------|--------|--------|-------------|----------|
| low_vol | 1 | -0.0042 | 0.960 | 1.4025 | ✗ | 🔵 ARCHIVE |
| low_vol | 5 | -0.0108 | 0.166 | 0.2987 | ✗ | 🔵 ARCHIVE |
| low_vol | 21 | -0.0232 | 1.000 | 0.0667 | ✗ | 🔵 ARCHIVE |
| mom_12_1 | 1 | +0.0009 | 0.976 | 2.0485 | ✗ | ⚪ HOLD |
| mom_12_1 | 5 | +0.0012 | 0.862 | 0.4145 | ✗ | ⚪ HOLD |
| mom_12_1 | 21 | +0.0019 | 0.940 | 0.1009 | ✗ | ⚪ HOLD |
| mom_1m | 1 | -0.0003 | 0.646 | 6.3891 | ✗ | 🔵 ARCHIVE |
| mom_1m | 5 | +0.0002 | 0.242 | 1.2912 | ✗ | ⚪ HOLD |
| mom_1m | 21 | -0.0039 | 0.154 | 0.3086 | ✗ | 🔵 ARCHIVE |
| mom_3m | 1 | -0.0020 | 0.234 | 3.7442 | ✗ | 🔵 ARCHIVE |
| mom_3m | 5 | -0.0025 | 0.246 | 0.7504 | ✗ | 🔵 ARCHIVE |
| mom_3m | 21 | -0.0012 | 0.168 | 0.1836 | ✗ | 🔵 ARCHIVE |
| mom_6m | 1 | +0.0010 | 0.860 | 2.7286 | ✗ | ⚪ HOLD |
| mom_6m | 5 | +0.0030 | 0.826 | 0.5529 | ✗ | ⚪ HOLD |
| mom_6m | 21 | +0.0068 | 0.586 | 0.1351 | ✗ | ⚪ HOLD |
| quality | 1 | -0.0002 | 0.396 | 2.0388 | ✗ | 🔵 ARCHIVE |
| quality | 5 | -0.0009 | 0.962 | 0.4142 | ✗ | 🔵 ARCHIVE |
| quality | 21 | -0.0016 | 0.966 | 0.1015 | ✗ | 🔵 ARCHIVE |
| rev_1m | 1 | +0.0003 | 0.900 | 6.3891 | ✗ | ⚪ HOLD |
| rev_1m | 5 | -0.0002 | 0.584 | 1.2912 | ✗ | 🔵 ARCHIVE |
| rev_1m | 21 | +0.0039 | 1.000 | 0.3086 | ✗ | ⚪ HOLD |
| rev_1w | 1 | +0.0049 | 0.990 | 6.6667 | ✗ | ⚪ HOLD |
| rev_1w | 5 | +0.0078 | 1.000 | 2.6923 | ✗ | ⚪ HOLD |
| rev_1w | 21 | +0.0014 | 0.900 | 0.6480 | ✗ | ⚪ HOLD |
| rsi_14 | 1 | -0.0001 | 0.956 | 6.6667 | ✗ | 🔵 ARCHIVE |
| rsi_14 | 5 | -0.0016 | 0.914 | 1.6135 | ✗ | 🔵 ARCHIVE |
| rsi_14 | 21 | -0.0006 | 1.000 | 0.3955 | ✗ | 🔵 ARCHIVE |
| ts_mom | 1 | +0.0001 | 0.128 | 2.5751 | ✗ | ⚪ HOLD |
| ts_mom | 5 | -0.0013 | 0.500 | 0.5143 | ✗ | 🔵 ARCHIVE |
| ts_mom | 21 | -0.0033 | 0.368 | 0.1365 | ✗ | 🔵 ARCHIVE |
| vol_regime | 1 | +0.0070 | 0.384 | 6.6667 | ✗ | ⚪ HOLD |
| vol_regime | 5 | +0.0248 | 0.876 | 1.4268 | ✗ | ⚪ HOLD |
| vol_regime | 21 | +0.0472 | 0.936 | 0.2885 | ✗ | ⚪ HOLD |
| vrp_proxy | 1 | +0.0023 | 0.010 | 6.6667 | ✗ | ⚪ HOLD |
| vrp_proxy | 5 | +0.0004 | 0.906 | 1.4504 | ✗ | ⚪ HOLD |
| vrp_proxy | 21 | +0.0005 | 0.778 | 0.3358 | ✗ | ⚪ HOLD |
| zscore_20 | 1 | +0.0008 | 0.894 | 6.6667 | ✗ | ⚪ HOLD |
| zscore_20 | 5 | +0.0030 | 0.122 | 2.0665 | ✗ | ⚪ HOLD |
| zscore_20 | 21 | +0.0016 | 0.274 | 0.4936 | ✗ | ⚪ HOLD |

---

## Detailed Per-Factor Decisions

### 🔵 low_vol: ARCHIVE

**Economic Rationale:** Strong — Blitz & Van Vliet 2007, leverage constraints theory

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | -0.0042 | 0.960 | 0.099 | [-0.0161, +0.0013] | ✗ | ✗ | ✗ | ARCHIVE |
| 5 | -0.0108 | 0.166 | 0.000 | [-0.0252, -0.0081] | ✓ | ✗ | ✗ | ARCHIVE |
| 21 | -0.0232 | 1.000 | 0.000 | [-0.0433, -0.0268] | ✓ | ✗ | ✗ | ARCHIVE |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Acquire options IV surface data (ORATS) for true VRP evaluation
- Re-evaluate when paid data sources are available
- Longer horizon (h=21) may rescue this factor once turnover is reduced

### ⚪ mom_12_1: HOLD

**Economic Rationale:** Very strong — flagship academic anomaly, hundreds of replications

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | +0.0009 | 0.976 | 0.033 | [+0.0006, +0.0202] | ✗ | ✗ | ✗ | HOLD |
| 5 | +0.0012 | 0.862 | 0.064 | [-0.0005, +0.0183] | ✗ | ✗ | ✗ | HOLD |
| 21 | +0.0019 | 0.940 | 0.429 | [-0.0050, +0.0128] | ✗ | ✗ | ✗ | HOLD |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Expand universe to S&P 500 for more statistical power
- Obtain point-in-time data (Polygon/FirstRate) to fix survivorship bias
- Test sector-neutral variant

### 🔵 mom_1m: ARCHIVE

**Economic Rationale:** Moderate — well-documented but weak in large-caps

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | -0.0003 | 0.646 | 0.291 | [-0.0116, +0.0034] | ✗ | ✗ | ✗ | ARCHIVE |
| 5 | +0.0002 | 0.242 | 0.101 | [-0.0135, +0.0010] | ✗ | ✗ | ✗ | HOLD |
| 21 | -0.0039 | 0.154 | 0.004 | [-0.0182, -0.0037] | ✓ | ✗ | ✗ | ARCHIVE |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Acquire options IV surface data (ORATS) for true VRP evaluation
- Re-evaluate when paid data sources are available
- Longer horizon (h=21) may rescue this factor once turnover is reduced

### 🔵 mom_3m: ARCHIVE

**Economic Rationale:** Strong — Jegadeesh-Titman, cross-sectional persistence documented

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | -0.0020 | 0.234 | 0.939 | [-0.0077, +0.0081] | ✗ | ✗ | ✗ | ARCHIVE |
| 5 | -0.0025 | 0.246 | 0.370 | [-0.0113, +0.0044] | ✗ | ✗ | ✗ | ARCHIVE |
| 21 | -0.0012 | 0.168 | 0.249 | [-0.0122, +0.0034] | ✗ | ✗ | ✗ | ARCHIVE |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Acquire options IV surface data (ORATS) for true VRP evaluation
- Re-evaluate when paid data sources are available
- Longer horizon (h=21) may rescue this factor once turnover is reduced

### ⚪ mom_6m: HOLD

**Economic Rationale:** Persistent intermediate-term momentum

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | +0.0010 | 0.860 | 0.179 | [-0.0025, +0.0145] | ✗ | ✗ | ✗ | HOLD |
| 5 | +0.0030 | 0.826 | 0.216 | [-0.0028, +0.0142] | ✗ | ✗ | ✗ | HOLD |
| 21 | +0.0068 | 0.586 | 0.111 | [-0.0015, +0.0160] | ✗ | ✗ | ✗ | HOLD |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Expand universe to S&P 500 for more statistical power
- Obtain point-in-time data (Polygon/FirstRate) to fix survivorship bias
- Test sector-neutral variant

### 🔵 quality: ARCHIVE

**Economic Rationale:** Moderate — Asness 2013, our proxy is too simple

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | -0.0002 | 0.396 | 0.108 | [-0.0011, +0.0167] | ✗ | ✗ | ✗ | ARCHIVE |
| 5 | -0.0009 | 0.962 | 0.389 | [-0.0049, +0.0131] | ✗ | ✗ | ✗ | ARCHIVE |
| 21 | -0.0016 | 0.966 | 0.451 | [-0.0123, +0.0056] | ✗ | ✗ | ✗ | ARCHIVE |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Acquire options IV surface data (ORATS) for true VRP evaluation
- Re-evaluate when paid data sources are available
- Longer horizon (h=21) may rescue this factor once turnover is reduced

### 🔵 rev_1m: ARCHIVE

**Economic Rationale:** Short-term reversal, may be bid-ask bounce artifact

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | +0.0003 | 0.900 | 0.291 | [-0.0034, +0.0116] | ✗ | ✓ | ✗ | HOLD |
| 5 | -0.0002 | 0.584 | 0.101 | [-0.0010, +0.0135] | ✗ | ✓ | ✗ | ARCHIVE |
| 21 | +0.0039 | 1.000 | 0.004 | [+0.0037, +0.0182] | ✓ | ✓ | ✗ | HOLD |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Acquire options IV surface data (ORATS) for true VRP evaluation
- Re-evaluate when paid data sources are available
- Longer horizon (h=21) may rescue this factor once turnover is reduced

### ⚪ rev_1w: HOLD

**Economic Rationale:** Weak — dominated by microstructure at daily frequency

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | +0.0049 | 0.990 | 0.026 | [+0.0009, +0.0156] | ✗ | ✓ | ✗ | HOLD |
| 5 | +0.0078 | 1.000 | 0.000 | [+0.0058, +0.0198] | ✓ | ✗ | ✗ | HOLD |
| 21 | +0.0014 | 0.900 | 0.179 | [-0.0023, +0.0117] | ✗ | ✗ | ✗ | HOLD |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Expand universe to S&P 500 for more statistical power
- Obtain point-in-time data (Polygon/FirstRate) to fix survivorship bias
- Test sector-neutral variant

### 🔵 rsi_14: ARCHIVE

**Economic Rationale:** Weak — popular but academically weak, often over-fitted

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | -0.0001 | 0.956 | 0.332 | [-0.0034, +0.0099] | ✗ | ✗ | ✗ | ARCHIVE |
| 5 | -0.0016 | 0.914 | 0.387 | [-0.0035, +0.0099] | ✗ | ✗ | ✗ | ARCHIVE |
| 21 | -0.0006 | 1.000 | 0.202 | [-0.0016, +0.0109] | ✗ | ✗ | ✗ | ARCHIVE |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Acquire options IV surface data (ORATS) for true VRP evaluation
- Re-evaluate when paid data sources are available
- Longer horizon (h=21) may rescue this factor once turnover is reduced

### 🔵 ts_mom: ARCHIVE

**Economic Rationale:** Moskowitz 2012 — trend-following with economic rationale

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | +0.0001 | 0.128 | 0.670 | [-0.0051, +0.0084] | ✗ | ✗ | ✗ | HOLD |
| 5 | -0.0013 | 0.500 | 0.671 | [-0.0086, +0.0056] | ✗ | ✗ | ✗ | ARCHIVE |
| 21 | -0.0033 | 0.368 | 0.031 | [-0.0151, -0.0006] | ✗ | ✗ | ✗ | ARCHIVE |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Acquire options IV surface data (ORATS) for true VRP evaluation
- Re-evaluate when paid data sources are available
- Longer horizon (h=21) may rescue this factor once turnover is reduced

### ⚪ vol_regime: HOLD

**Economic Rationale:** Weak — regime indicator, not a standalone alpha source

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | +0.0070 | 0.384 | 0.596 | [-0.0000, +0.0000] | ✗ | ✗ | ✗ | HOLD |
| 5 | +0.0248 | 0.876 | 0.901 | [-0.0000, +0.0000] | ✗ | ✗ | ✗ | HOLD |
| 21 | +0.0472 | 0.936 | 0.972 | [-0.0000, +0.0000] | ✗ | ✓ | ✗ | HOLD |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not regime-robust (IC not consistent across regimes)
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Expand universe to S&P 500 for more statistical power
- Obtain point-in-time data (Polygon/FirstRate) to fix survivorship bias
- Test sector-neutral variant

### ⚪ vrp_proxy: HOLD

**Economic Rationale:** Moderate — VRP exists but this is a noisy proxy

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | +0.0023 | 0.010 | 0.353 | [-0.0027, +0.0079] | ✗ | ✓ | ✗ | HOLD |
| 5 | +0.0004 | 0.906 | 0.888 | [-0.0049, +0.0060] | ✗ | ✗ | ✗ | HOLD |
| 21 | +0.0005 | 0.778 | 0.337 | [-0.0024, +0.0082] | ✗ | ✓ | ✗ | HOLD |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Expand universe to S&P 500 for more statistical power
- Obtain point-in-time data (Polygon/FirstRate) to fix survivorship bias
- Test sector-neutral variant

### ⚪ zscore_20: HOLD

**Economic Rationale:** Mean reversion with economic basis but fragile at large N

| Horizon | IC | Perm p | t p | Bootstrap CI | FDR | Reg-Rob | Cost OK | Decision |
|---------|----:|--------:|----:|--------------|-----|---------|---------|----------|
| 1 | +0.0008 | 0.894 | 0.229 | [-0.0029, +0.0114] | ✗ | ✓ | ✗ | HOLD |
| 5 | +0.0030 | 0.122 | 0.036 | [+0.0007, +0.0140] | ✗ | ✗ | ✗ | HOLD |
| 21 | +0.0016 | 0.274 | 0.064 | [-0.0003, +0.0125] | ✗ | ✗ | ✗ | HOLD |

**Blocking Issues:**
- IC not statistically significant (t-test + permutation fail)
- Failed FDR multiple-testing correction
- Not cost-viable (IC below min_IC for this holding horizon)

**Next Steps:**
- Expand universe to S&P 500 for more statistical power
- Obtain point-in-time data (Polygon/FirstRate) to fix survivorship bias
- Test sector-neutral variant

---

## Overall Assessment

⚠ **No factors promoted or refinable.**
   This evaluation confirms: no deployable edge with current data/methodology.
   All factors suffer from either IC non-significance or prohibitive turnover costs.

### Recommendations

1. **Do NOT deploy any strategy.**
2. Expand universe to S&P 500 (100 stocks is still underpowered).
3. Acquire point-in-time data (Polygon/FirstRate) for survivorship-free research.
4. Acquire ORATS IV surface for true per-stock VRP evaluation.
6. Re-run Phase 4 pipeline when data infrastructure is upgraded.
