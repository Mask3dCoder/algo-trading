# Signal Correlation Report

## IC Correlation Matrix

Pairwise correlation of daily IC time series between factors.
High correlation (>0.7) indicates redundancy — same risk premium.

| | mom_1m | mom_3m | mom_6m | mom_12_1 | ts_mom | rev_1m | rev_1w | zscore_20 | rsi_14 | low_vol | quality | vrp_proxy | vol_regime |
|---|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| **mom_1m** | 1.00 | 0.59 | 0.40 | 0.02 | 0.21 | -1.00 | -0.53 | -0.78 | -0.82 | 0.16 | 0.25 | 0.20 | 0.00 |
| **mom_3m** | 0.59 | 1.00 | 0.75 | 0.38 | 0.33 | -0.59 | -0.26 | -0.43 | -0.50 | 0.19 | 0.53 | 0.08 | 0.00 |
| **mom_6m** | 0.40 | 0.75 | 1.00 | 0.70 | 0.40 | -0.40 | -0.15 | -0.29 | -0.36 | 0.20 | 0.78 | -0.01 | -0.01 |
| **mom_12_1** | 0.02 | 0.38 | 0.70 | 1.00 | 0.37 | -0.02 | 0.06 | 0.01 | -0.03 | 0.13 | 0.94 | -0.09 | 0.01 |
| **ts_mom** | 0.21 | 0.33 | 0.40 | 0.37 | 1.00 | -0.21 | -0.11 | -0.25 | -0.26 | 0.78 | 0.56 | -0.10 | 0.00 |
| **rev_1m** | -1.00 | -0.59 | -0.40 | -0.02 | -0.21 | 1.00 | 0.53 | 0.78 | 0.82 | -0.16 | -0.25 | -0.20 | -0.00 |
| **rev_1w** | -0.53 | -0.26 | -0.15 | 0.06 | -0.11 | 0.53 | 1.00 | 0.78 | 0.60 | -0.11 | -0.07 | -0.07 | 0.00 |
| **zscore_20** | -0.78 | -0.43 | -0.29 | 0.01 | -0.25 | 0.78 | 0.78 | 1.00 | 0.89 | -0.20 | -0.20 | -0.14 | -0.00 |
| **rsi_14** | -0.82 | -0.50 | -0.36 | -0.03 | -0.26 | 0.82 | 0.60 | 0.89 | 1.00 | -0.21 | -0.25 | -0.18 | -0.01 |
| **low_vol** | 0.16 | 0.19 | 0.20 | 0.13 | 0.78 | -0.16 | -0.11 | -0.20 | -0.21 | 1.00 | 0.30 | -0.12 | 0.01 |
| **quality** | 0.25 | 0.53 | 0.78 | 0.94 | 0.56 | -0.25 | -0.07 | -0.20 | -0.25 | 0.30 | 1.00 | -0.04 | 0.01 |
| **vrp_proxy** | 0.20 | 0.08 | -0.01 | -0.09 | -0.10 | -0.20 | -0.07 | -0.14 | -0.18 | -0.12 | -0.04 | 1.00 | 0.03 |
| **vol_regime** | 0.00 | 0.00 | -0.01 | 0.01 | 0.00 | -0.00 | 0.00 | -0.00 | -0.01 | 0.01 | 0.01 | 0.03 | 1.00 |

---

## Hierarchical Clusters

Factors grouped by IC correlation distance (1 - |corr|):

**Cluster 1:** mom_1m, rev_1m, rev_1w, zscore_20, rsi_14
**Cluster 2:** mom_3m, mom_6m, mom_12_1, ts_mom, low_vol, quality
**Cluster 3:** vrp_proxy
**Cluster 4:** vol_regime

---

## Redundancy Analysis

Threshold: |IC correlation| > 70%

| Factor A | Factor B | IC Correlation | Verdict |
|----------|----------|---------------:|---------|
| mom_1m | rev_1m | -1.000 | REDUNDANT |
| mom_12_1 | quality | +0.938 | REDUNDANT |
| zscore_20 | rsi_14 | +0.886 | REDUNDANT |
| rev_1m | rsi_14 | +0.824 | REDUNDANT |
| mom_1m | rsi_14 | -0.824 | REDUNDANT |
| rev_1w | zscore_20 | +0.782 | REDUNDANT |
| mom_6m | quality | +0.782 | REDUNDANT |
| mom_1m | zscore_20 | -0.781 | REDUNDANT |
| rev_1m | zscore_20 | +0.781 | REDUNDANT |
| ts_mom | low_vol | +0.778 | REDUNDANT |
| mom_3m | mom_6m | +0.754 | REDUNDANT |
| mom_6m | mom_12_1 | +0.704 | REDUNDANT |

---

## Independent Factor Set

After removing redundant factors (keeping higher-IC from each pair):

- mom_3m
- mom_12_1
- rev_1m
- rev_1w
- low_vol
- vrp_proxy
- vol_regime

**6 factor(s) removed as redundant.**