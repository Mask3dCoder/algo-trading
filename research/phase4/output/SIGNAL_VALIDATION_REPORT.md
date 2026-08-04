# Signal Validation Report

## IC Summary (Full Sample)

| Factor | IC | Rank IC | ICIR | Hit Rate | Coverage | Persistence | Econ ★ |
|--------|------:|--------:|-----:|---------:|---------:|------------:|-------:|
| low_vol | -0.0156 | -0.0176 | -0.08 | 46.3% | 98.0% | 0.98 | ★★★★ |
| mom_12_1 | +0.0077 | +0.0175 | 0.07 | 53.6% | 91.7% | 0.99 | ★★★★★ |
| mom_1m | -0.0030 | -0.0046 | -0.02 | 50.2% | 99.3% | 0.92 | ★★★ |
| mom_3m | -0.0005 | +0.0035 | 0.02 | 51.9% | 97.9% | 0.97 | ★★★★ |
| mom_6m | +0.0055 | +0.0118 | 0.05 | 51.8% | 95.8% | 0.98 | ★★★★ |
| quality | +0.0039 | +0.0119 | 0.06 | 52.5% | 91.7% | 0.99 | ★★★ |
| rev_1m | +0.0030 | +0.0046 | 0.02 | 49.8% | 99.3% | 0.92 | ★★★ |
| rev_1w | +0.0046 | +0.0058 | 0.02 | 50.4% | 99.8% | 0.74 | ★★ |
| rsi_14 | +0.0179 | +0.0243 | 0.02 | 50.6% | 99.5% | 0.92 | ★★ |
| ts_mom | -0.0064 | -0.0062 | -0.04 | 48.5% | 91.7% | 0.94 | ★★★★ |
| vol_regime | +0.0041 | +0.0243 | — | 47.4% | 91.7% | 0.94 | ★★ |
| vrp_proxy | -0.0126 | -0.0132 | -0.06 | 47.8% | 99.3% | 0.94 | ★★★ |
| zscore_20 | +0.0031 | +0.0052 | 0.02 | 50.7% | 99.4% | 0.83 | ★★★ |

---

## Statistical Validation

| Factor | t-stat | p-value | Perm p | 95% CI | FDR ✓ | Reject H₀ |
|--------|-------:|--------:|-------:|--------|------:|-----------:|
| low_vol | -4.24 | 0.0000 | 1.0000 | [-0.0391, -0.0147] | ✓ | ✗ |
| mom_12_1 | 3.74 | 0.0002 | 0.9500 | [+0.0102, +0.0352] | ✓ | ✗ |
| mom_1m | -0.87 | 0.3817 | 0.7700 | [-0.0143, +0.0070] | ✗ | ✗ |
| mom_3m | 1.10 | 0.2736 | 0.4400 | [-0.0044, +0.0158] | ✗ | ✗ |
| mom_6m | 2.63 | 0.0086 | 0.5700 | [+0.0055, +0.0281] | ✓ | ✗ |
| quality | 2.90 | 0.0038 | 1.0000 | [+0.0063, +0.0293] | ✓ | ✗ |
| rev_1m | 0.87 | 0.3817 | 0.3900 | [-0.0070, +0.0143] | ✗ | ✗ |
| rev_1w | 1.14 | 0.2538 | 0.9900 | [-0.0037, +0.0169] | ✗ | ✗ |
| rsi_14 | 1.00 | 0.3162 | 0.9000 | [-0.0058, +0.0149] | ✗ | ✗ |
| ts_mom | -1.92 | 0.0550 | 0.0100 | [-0.0185, +0.0015] | ✗ | ✗ |
| vol_regime | 0.47 | 0.6379 | 0.6600 | [-0.0000, +0.0000] | ✗ | ✗ |
| vrp_proxy | -3.29 | 0.0010 | 1.0000 | [-0.0295, -0.0078] | ✓ | ✗ |
| zscore_20 | 1.09 | 0.2749 | 0.7800 | [-0.0055, +0.0142] | ✗ | ✗ |

---

## Economic Rationale Scores

| Factor | Score | Rationale |
|--------|------:|-----------|
| low_vol | ★★★★☆ | Strong — Blitz & Van Vliet 2007, leverage constraints theory |
| mom_12_1 | ★★★★★ | Very strong — flagship academic anomaly, hundreds of replications |
| mom_1m | ★★★☆☆ | Moderate — well-documented but weak in large-caps |
| mom_3m | ★★★★☆ | Strong — Jegadeesh-Titman, cross-sectional persistence documented |
| mom_6m | ★★★★☆ | Strong — persistent intermediate-term momentum |
| quality | ★★★☆☆ | Moderate — Asness 2013 but our proxy is too simple |
| rev_1m | ★★★☆☆ | Moderate — exists but may be bid-ask bounce artifact |
| rev_1w | ★★☆☆☆ | Weak — dominated by microstructure effects at daily frequency |
| rsi_14 | ★★☆☆☆ | Weak — popular but academically weak, often overfitted |
| ts_mom | ★★★★☆ | Strong — Moskowitz 2012, trend-following economic basis |
| vol_regime | ★★☆☆☆ | Weak — regime indicator, not a standalone alpha source |
| vrp_proxy | ★★★☆☆ | Moderate — VRP is real but VIX is SPX-specific, poor stock proxy |
| zscore_20 | ★★★☆☆ | Moderate — mean reversion with economic basis but fragile |

---

**Summary:** 0/13 factors reject H₀. 5/13 survive FDR correction.