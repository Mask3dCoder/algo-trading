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
| low_vol | -0.0228 | -0.0106 | -0.0214 | -0.0670 | ✗ | ✓ |
| mom_12_1 | -0.0207 | +0.0520 | -0.0437 | -0.0423 | ✗ | ✓ |
| mom_1m | +0.0088 | -0.0130 | -0.0248 | -0.0405 | ✗ | ✗ |
| mom_3m | -0.0093 | +0.0103 | -0.0695 | -0.0725 | ✗ | ✓ |
| mom_6m | -0.0199 | +0.0389 | -0.0332 | -0.0591 | ✗ | ✓ |
| quality | -0.0152 | +0.0396 | -0.0445 | -0.0578 | ✗ | ✓ |
| rev_1m | -0.0088 | +0.0130 | +0.0248 | +0.0405 | ✓ | ✗ |
| rev_1w | -0.0093 | +0.0045 | -0.0094 | +0.0713 | ✗ | ✗ |
| rsi_14 | -0.0139 | +0.0026 | -0.0303 | +0.0319 | ✗ | ✗ |
| ts_mom | -0.0100 | +0.0035 | -0.0132 | -0.0072 | ✗ | ✓ |
| vol_regime | -0.0000 | +0.0000 | +0.0000 | -0.0000 | ✗ | ✗ |
| vrp_proxy | +0.0088 | -0.0105 | +0.0343 | -0.0154 | ✗ | ✗ |
| zscore_20 | -0.0129 | +0.0104 | -0.0140 | +0.0436 | ✗ | ✗ |

---

## Detailed Per-Factor Regime Analysis

### low_vol

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0228 | -0.0189 | 869 | 29.4% |  |
| Bull+HighVol | -0.0106 | -0.0131 | 538 | 18.2% |  |
| Bear+LowVol | -0.0214 | -0.0155 | 83 | 2.8% |  |
| Bear+HighVol | -0.0670 | -0.0687 | 161 | 5.5% |  |
| Bull | -0.0113 | -0.0081 | 2490 | 84.3% |  |
| Bear | -0.0462 | -0.0414 | 464 | 15.7% |  |
| HighVol | -0.0236 | -0.0259 | 699 | 23.7% |  |
| LowVol | -0.0227 | -0.0186 | 952 | 32.2% |  |

### mom_12_1

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0207 | -0.0220 | 844 | 30.6% |  |
| Bull+HighVol | +0.0520 | +0.0541 | 503 | 18.2% |  |
| Bear+LowVol | -0.0437 | -0.0381 | 83 | 3.0% |  |
| Bear+HighVol | -0.0423 | -0.0382 | 160 | 5.8% |  |
| Bull | +0.0140 | +0.0158 | 2364 | 85.6% |  |
| Bear | -0.0214 | -0.0172 | 398 | 14.4% |  |
| HighVol | +0.0293 | +0.0318 | 663 | 24.0% |  |
| LowVol | -0.0227 | -0.0235 | 927 | 33.6% |  |

### mom_1m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0088 | +0.0090 | 869 | 29.0% |  |
| Bull+HighVol | -0.0130 | -0.0106 | 538 | 18.0% |  |
| Bear+LowVol | -0.0248 | -0.0274 | 83 | 2.8% |  |
| Bear+HighVol | -0.0405 | -0.0403 | 161 | 5.4% |  |
| Bull | -0.0039 | -0.0029 | 2490 | 83.2% |  |
| Bear | -0.0169 | -0.0262 | 503 | 16.8% |  |
| HighVol | -0.0193 | -0.0175 | 699 | 23.4% |  |
| LowVol | +0.0058 | +0.0059 | 952 | 31.8% |  |

### mom_3m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0093 | -0.0127 | 869 | 29.4% |  |
| Bull+HighVol | +0.0103 | +0.0110 | 538 | 18.2% |  |
| Bear+LowVol | -0.0695 | -0.0731 | 83 | 2.8% |  |
| Bear+HighVol | -0.0725 | -0.0673 | 161 | 5.5% |  |
| Bull | +0.0048 | +0.0030 | 2490 | 84.4% |  |
| Bear | -0.0494 | -0.0533 | 461 | 15.6% |  |
| HighVol | -0.0088 | -0.0071 | 699 | 23.7% |  |
| LowVol | -0.0145 | -0.0180 | 952 | 32.3% |  |

### mom_6m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0199 | -0.0193 | 869 | 30.1% |  |
| Bull+HighVol | +0.0389 | +0.0385 | 538 | 18.6% |  |
| Bear+LowVol | -0.0332 | -0.0378 | 83 | 2.9% |  |
| Bear+HighVol | -0.0591 | -0.0537 | 160 | 5.5% |  |
| Bull | +0.0124 | +0.0136 | 2490 | 86.2% |  |
| Bear | -0.0376 | -0.0378 | 398 | 13.8% |  |
| HighVol | +0.0164 | +0.0174 | 698 | 24.2% |  |
| LowVol | -0.0210 | -0.0209 | 952 | 33.0% |  |

### quality

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0152 | -0.0163 | 844 | 30.6% |  |
| Bull+HighVol | +0.0396 | +0.0411 | 503 | 18.2% |  |
| Bear+LowVol | -0.0445 | -0.0407 | 83 | 3.0% |  |
| Bear+HighVol | -0.0578 | -0.0546 | 160 | 5.8% |  |
| Bull | +0.0100 | +0.0119 | 2364 | 85.6% |  |
| Bear | -0.0318 | -0.0285 | 398 | 14.4% |  |
| HighVol | +0.0161 | +0.0180 | 663 | 24.0% |  |
| LowVol | -0.0178 | -0.0184 | 927 | 33.6% |  |

### rev_1m

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0088 | -0.0090 | 869 | 29.0% |  |
| Bull+HighVol | +0.0130 | +0.0106 | 538 | 18.0% |  |
| Bear+LowVol | +0.0248 | +0.0274 | 83 | 2.8% |  |
| Bear+HighVol | +0.0405 | +0.0403 | 161 | 5.4% |  |
| Bull | +0.0039 | +0.0029 | 2490 | 83.2% |  |
| Bear | +0.0169 | +0.0262 | 503 | 16.8% |  |
| HighVol | +0.0193 | +0.0175 | 699 | 23.4% |  |
| LowVol | -0.0058 | -0.0059 | 952 | 31.8% |  |

### rev_1w

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0093 | -0.0088 | 869 | 28.9% |  |
| Bull+HighVol | +0.0045 | +0.0046 | 538 | 17.9% |  |
| Bear+LowVol | -0.0094 | -0.0002 | 83 | 2.8% |  |
| Bear+HighVol | +0.0713 | +0.0803 | 161 | 5.3% |  |
| Bull | +0.0061 | +0.0069 | 2490 | 82.8% |  |
| Bear | +0.0441 | +0.0499 | 519 | 17.2% |  |
| HighVol | +0.0199 | +0.0220 | 699 | 23.2% |  |
| LowVol | -0.0093 | -0.0080 | 952 | 31.6% |  |

### rsi_14

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0139 | -0.0163 | 869 | 29.0% |  |
| Bull+HighVol | +0.0026 | +0.0014 | 538 | 17.9% |  |
| Bear+LowVol | -0.0303 | -0.0296 | 83 | 2.8% |  |
| Bear+HighVol | +0.0319 | +0.0384 | 161 | 5.4% |  |
| Bull | +0.0006 | -0.0012 | 2490 | 83.0% |  |
| Bear | +0.0149 | +0.0252 | 510 | 17.0% |  |
| HighVol | +0.0094 | +0.0100 | 699 | 23.3% |  |
| LowVol | -0.0153 | -0.0175 | 952 | 31.7% |  |

### ts_mom

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0100 | -0.0065 | 844 | 30.6% |  |
| Bull+HighVol | +0.0035 | +0.0026 | 503 | 18.2% |  |
| Bear+LowVol | -0.0132 | -0.0160 | 83 | 3.0% |  |
| Bear+HighVol | -0.0072 | -0.0119 | 160 | 5.8% |  |
| Bull | -0.0003 | +0.0032 | 2364 | 85.6% |  |
| Bear | -0.0093 | -0.0107 | 398 | 14.4% |  |
| HighVol | +0.0009 | -0.0009 | 663 | 24.0% |  |
| LowVol | -0.0103 | -0.0073 | 927 | 33.6% |  |

### vol_regime

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0000 | — | 820 | 32.0% |  |
| Bull+HighVol | +0.0000 | — | 481 | 18.8% |  |
| Bear+LowVol | +0.0000 | — | 80 | 3.1% |  |
| Bear+HighVol | -0.0000 | — | 148 | 5.8% |  |
| Bull | -0.0000 | — | 2206 | 86.2% |  |
| Bear | +0.0000 | — | 353 | 13.8% |  |
| HighVol | +0.0000 | — | 629 | 24.6% |  |
| LowVol | -0.0000 | — | 900 | 35.2% |  |

### vrp_proxy

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0088 | +0.0082 | 869 | 28.9% |  |
| Bull+HighVol | -0.0105 | -0.0076 | 538 | 17.9% |  |
| Bear+LowVol | +0.0343 | +0.0309 | 83 | 2.8% |  |
| Bear+HighVol | -0.0154 | -0.0137 | 161 | 5.4% |  |
| Bull | -0.0024 | -0.0027 | 2490 | 82.9% |  |
| Bear | +0.0138 | +0.0112 | 514 | 17.1% |  |
| HighVol | -0.0116 | -0.0090 | 699 | 23.3% |  |
| LowVol | +0.0111 | +0.0102 | 952 | 31.7% |  |

### zscore_20

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0129 | -0.0140 | 869 | 29.0% |  |
| Bull+HighVol | +0.0104 | +0.0091 | 538 | 18.0% |  |
| Bear+LowVol | -0.0140 | -0.0109 | 83 | 2.8% |  |
| Bear+HighVol | +0.0436 | +0.0521 | 161 | 5.4% |  |
| Bull | +0.0038 | +0.0026 | 2490 | 83.1% |  |
| Bear | +0.0237 | +0.0307 | 505 | 16.9% |  |
| HighVol | +0.0181 | +0.0190 | 699 | 23.3% |  |
| LowVol | -0.0130 | -0.0138 | 952 | 31.8% |  |

---

## Summary Verdicts

**Robust (positive IC ≥3/4 regimes):** rev_1m
**Regime-Dependent (stable direction but not consistently positive):** mom_3m, mom_6m, mom_12_1, ts_mom, low_vol, quality
**Fragile (direction flips, no consistency):** mom_1m, rev_1w, zscore_20, rsi_14, vrp_proxy, vol_regime