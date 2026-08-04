# Regime Stability Report

## Regime Classification

- **Date Range:** 2013-01-02 → 2024-12-30
- **Trend:** Bull = SPY 12-month rolling return > 0; Bear ≤ 0
- **Vol:** HighVol = VIX ≥ 75th percentile; LowVol = VIX ≤ 25th percentile

### Regime Distribution

| Regime | Dates | % of Total |
|--------|-------:|-----------:|
| Bull+MidVol | 1087 | 36.0% |
| Bull+LowVol | 869 | 28.8% |
| Bull+HighVol | 539 | 17.9% |
| Bear+MidVol | 280 | 9.3% |
| Bear+HighVol | 161 | 5.3% |
| Bear+LowVol | 83 | 2.7% |

---

## Factor Regime Performance

| Factor | Bull+LowVol | Bull+HighVol | Bear+LowVol | Bear+HighVol | Robust | Stable |
|--------|-------:|-------:|-------:|-------:|--------|--------|
| low_vol | -0.0531 | -0.0083 | -0.0895 | -0.1094 | ✗ | ✓ |
| mom_12_1 | -0.0035 | +0.0379 | -0.0868 | -0.1167 | ✗ | ✓ |
| mom_1m | -0.0068 | +0.0252 | -0.0429 | -0.0679 | ✗ | ✓ |
| mom_3m | -0.0054 | +0.0287 | -0.0290 | -0.1481 | ✗ | ✓ |
| mom_6m | +0.0005 | +0.0580 | -0.0564 | -0.1349 | ✗ | ✗ |
| quality | -0.0070 | +0.0402 | -0.0551 | -0.1417 | ✗ | ✓ |
| rev_1m | +0.0068 | -0.0252 | +0.0429 | +0.0679 | ✓ | ✓ |
| rev_1w | -0.0146 | -0.0080 | -0.0061 | +0.0440 | ✗ | ✓ |
| rsi_14 | -0.0007 | -0.0294 | +0.0161 | +0.0423 | ✗ | ✗ |
| ts_mom | -0.0175 | +0.0057 | +0.0096 | -0.0479 | ✗ | ✗ |
| vol_regime | +0.0000 | -0.0000 | +0.0000 | +0.0000 | ✓ | ✓ |
| vrp_proxy | +0.0111 | +0.0046 | +0.0263 | -0.0092 | ✓ | ✓ |
| zscore_20 | -0.0036 | -0.0188 | +0.0102 | +0.0344 | ✗ | ✗ |

---

## Detailed Per-Factor Regime Analysis

### low_vol

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0531 | -0.0486 | 868 | 29.5% |  |
| Bull+HighVol | -0.0083 | -0.0057 | 535 | 18.2% |  |
| Bear+LowVol | -0.0895 | -0.0733 | 83 | 2.8% |  |
| Bear+HighVol | -0.1094 | -0.1144 | 161 | 5.5% |  |
| Bull | -0.0260 | -0.0233 | 2474 | 84.2% |  |
| Bear | -0.0828 | -0.0783 | 464 | 15.8% |  |
| HighVol | -0.0317 | -0.0308 | 696 | 23.7% |  |
| LowVol | -0.0563 | -0.0507 | 951 | 32.4% |  |

### mom_12_1

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0035 | -0.0058 | 843 | 30.7% |  |
| Bull+HighVol | +0.0379 | +0.0346 | 500 | 18.2% |  |
| Bear+LowVol | -0.0868 | -0.0861 | 83 | 3.0% |  |
| Bear+HighVol | -0.1167 | -0.1211 | 160 | 5.8% |  |
| Bull | +0.0113 | +0.0064 | 2348 | 85.5% |  |
| Bear | -0.0415 | -0.0434 | 398 | 14.5% |  |
| HighVol | +0.0004 | -0.0031 | 660 | 24.0% |  |
| LowVol | -0.0110 | -0.0130 | 926 | 33.7% |  |

### mom_1m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0068 | -0.0078 | 868 | 29.2% |  |
| Bull+HighVol | +0.0252 | +0.0314 | 535 | 18.0% |  |
| Bear+LowVol | -0.0429 | -0.0499 | 83 | 2.8% |  |
| Bear+HighVol | -0.0679 | -0.0702 | 161 | 5.4% |  |
| Bull | -0.0040 | -0.0032 | 2474 | 83.1% |  |
| Bear | -0.0433 | -0.0553 | 503 | 16.9% |  |
| HighVol | +0.0037 | +0.0079 | 696 | 23.4% |  |
| LowVol | -0.0099 | -0.0115 | 951 | 31.9% |  |

### mom_3m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0054 | -0.0074 | 868 | 29.6% |  |
| Bull+HighVol | +0.0287 | +0.0329 | 535 | 18.2% |  |
| Bear+LowVol | -0.0290 | -0.0395 | 83 | 2.8% |  |
| Bear+HighVol | -0.1481 | -0.1497 | 161 | 5.5% |  |
| Bull | +0.0110 | +0.0108 | 2474 | 84.3% |  |
| Bear | -0.0883 | -0.1024 | 461 | 15.7% |  |
| HighVol | -0.0122 | -0.0093 | 696 | 23.7% |  |
| LowVol | -0.0075 | -0.0102 | 951 | 32.4% |  |

### mom_6m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0005 | -0.0010 | 868 | 30.2% |  |
| Bull+HighVol | +0.0580 | +0.0531 | 535 | 18.6% |  |
| Bear+LowVol | -0.0564 | -0.0593 | 83 | 2.9% |  |
| Bear+HighVol | -0.1349 | -0.1338 | 160 | 5.6% |  |
| Bull | +0.0194 | +0.0160 | 2474 | 86.1% |  |
| Bear | -0.0699 | -0.0726 | 398 | 13.9% |  |
| HighVol | +0.0136 | +0.0101 | 695 | 24.2% |  |
| LowVol | -0.0044 | -0.0060 | 951 | 33.1% |  |

### quality

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0070 | -0.0088 | 843 | 30.7% |  |
| Bull+HighVol | +0.0402 | +0.0390 | 500 | 18.2% |  |
| Bear+LowVol | -0.0551 | -0.0613 | 83 | 3.0% |  |
| Bear+HighVol | -0.1417 | -0.1473 | 160 | 5.8% |  |
| Bull | +0.0052 | +0.0017 | 2348 | 85.5% |  |
| Bear | -0.0544 | -0.0588 | 398 | 14.5% |  |
| HighVol | -0.0039 | -0.0062 | 660 | 24.0% |  |
| LowVol | -0.0114 | -0.0135 | 926 | 33.7% |  |

### rev_1m

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0068 | +0.0078 | 868 | 29.2% |  |
| Bull+HighVol | -0.0252 | -0.0314 | 535 | 18.0% |  |
| Bear+LowVol | +0.0429 | +0.0499 | 83 | 2.8% |  |
| Bear+HighVol | +0.0679 | +0.0702 | 161 | 5.4% |  |
| Bull | +0.0040 | +0.0032 | 2474 | 83.1% |  |
| Bear | +0.0433 | +0.0553 | 503 | 16.9% |  |
| HighVol | -0.0037 | -0.0079 | 696 | 23.4% |  |
| LowVol | +0.0099 | +0.0115 | 951 | 31.9% |  |

### rev_1w

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0146 | -0.0118 | 868 | 29.0% |  |
| Bull+HighVol | -0.0080 | -0.0064 | 535 | 17.9% |  |
| Bear+LowVol | -0.0061 | +0.0025 | 83 | 2.8% |  |
| Bear+HighVol | +0.0440 | +0.0512 | 161 | 5.4% |  |
| Bull | +0.0015 | +0.0028 | 2474 | 82.7% |  |
| Bear | +0.0206 | +0.0267 | 519 | 17.3% |  |
| HighVol | +0.0040 | +0.0070 | 696 | 23.2% |  |
| LowVol | -0.0139 | -0.0106 | 951 | 31.8% |  |

### rsi_14

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0007 | +0.0000 | 868 | 29.1% |  |
| Bull+HighVol | -0.0294 | -0.0326 | 535 | 17.9% |  |
| Bear+LowVol | +0.0161 | +0.0210 | 83 | 2.8% |  |
| Bear+HighVol | +0.0423 | +0.0484 | 161 | 5.4% |  |
| Bull | +0.0004 | +0.0006 | 2474 | 82.9% |  |
| Bear | +0.0235 | +0.0331 | 510 | 17.1% |  |
| HighVol | -0.0128 | -0.0139 | 696 | 23.3% |  |
| LowVol | +0.0008 | +0.0019 | 951 | 31.9% |  |

### ts_mom

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0175 | -0.0147 | 843 | 30.7% |  |
| Bull+HighVol | +0.0057 | +0.0048 | 500 | 18.2% |  |
| Bear+LowVol | +0.0096 | -0.0145 | 83 | 3.0% |  |
| Bear+HighVol | -0.0479 | -0.0591 | 160 | 5.8% |  |
| Bull | -0.0064 | -0.0058 | 2348 | 85.5% |  |
| Bear | -0.0190 | -0.0287 | 398 | 14.5% |  |
| HighVol | -0.0073 | -0.0107 | 660 | 24.0% |  |
| LowVol | -0.0150 | -0.0147 | 926 | 33.7% |  |

### vol_regime

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0000 | — | 819 | 32.2% |  |
| Bull+HighVol | -0.0000 | — | 478 | 18.8% |  |
| Bear+LowVol | +0.0000 | — | 80 | 3.1% |  |
| Bear+HighVol | +0.0000 | — | 148 | 5.8% |  |
| Bull | -0.0000 | — | 2193 | 86.1% |  |
| Bear | +0.0000 | — | 353 | 13.9% |  |
| HighVol | -0.0000 | — | 626 | 24.6% |  |
| LowVol | +0.0000 | — | 899 | 35.3% |  |

### vrp_proxy

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0111 | +0.0094 | 868 | 29.0% |  |
| Bull+HighVol | +0.0046 | +0.0043 | 535 | 17.9% |  |
| Bear+LowVol | +0.0263 | +0.0279 | 83 | 2.8% |  |
| Bear+HighVol | -0.0092 | +0.0102 | 161 | 5.4% |  |
| Bull | +0.0008 | -0.0011 | 2474 | 82.8% |  |
| Bear | +0.0111 | +0.0147 | 514 | 17.2% |  |
| HighVol | +0.0014 | +0.0056 | 696 | 23.3% |  |
| LowVol | +0.0125 | +0.0110 | 951 | 31.8% |  |

### zscore_20

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0036 | -0.0027 | 868 | 29.1% |  |
| Bull+HighVol | -0.0188 | -0.0200 | 535 | 18.0% |  |
| Bear+LowVol | +0.0102 | +0.0168 | 83 | 2.8% |  |
| Bear+HighVol | +0.0344 | +0.0396 | 161 | 5.4% |  |
| Bull | +0.0024 | +0.0028 | 2474 | 83.0% |  |
| Bear | +0.0247 | +0.0316 | 505 | 17.0% |  |
| HighVol | -0.0065 | -0.0062 | 696 | 23.4% |  |
| LowVol | -0.0024 | -0.0010 | 951 | 31.9% |  |

---

## Summary Verdicts

**Robust (positive IC ≥3/4 regimes):** rev_1m, vrp_proxy, vol_regime
**Regime-Dependent (stable direction but not consistently positive):** mom_1m, mom_3m, mom_12_1, rev_1w, low_vol, quality
**Fragile (direction flips, no consistency):** mom_6m, ts_mom, zscore_20, rsi_14