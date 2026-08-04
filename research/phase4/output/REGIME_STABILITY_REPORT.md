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
| low_vol | -0.0433 | -0.0234 | -0.0509 | -0.0322 | ✗ | ✓ |
| mom_12_1 | +0.0249 | +0.0328 | -0.0222 | +0.0262 | ✓ | ✓ |
| mom_1m | +0.0096 | +0.0014 | -0.0039 | -0.0517 | ✗ | ✗ |
| mom_3m | +0.0062 | +0.0248 | -0.0305 | -0.0143 | ✗ | ✗ |
| mom_6m | +0.0168 | +0.0317 | -0.0192 | +0.0070 | ✓ | ✓ |
| quality | +0.0187 | +0.0286 | -0.0086 | +0.0052 | ✓ | ✓ |
| rev_1m | -0.0096 | -0.0014 | +0.0039 | +0.0517 | ✗ | ✗ |
| rev_1w | -0.0136 | +0.0128 | -0.0170 | +0.0442 | ✗ | ✗ |
| rsi_14 | -0.0088 | -0.0055 | -0.0426 | +0.0401 | ✗ | ✓ |
| ts_mom | -0.0299 | -0.0066 | +0.0104 | +0.0141 | ✗ | ✗ |
| vol_regime | +0.0000 | +0.0000 | — | +0.0000 | ✓ | ✓ |
| vrp_proxy | -0.0294 | -0.0192 | -0.0464 | -0.0351 | ✗ | ✓ |
| zscore_20 | -0.0155 | +0.0052 | -0.0198 | +0.0517 | ✗ | ✗ |

---

## Detailed Per-Factor Regime Analysis

### low_vol

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0433 | -0.0279 | 869 | 29.4% |  |
| Bull+HighVol | -0.0234 | -0.0190 | 538 | 18.2% |  |
| Bear+LowVol | -0.0509 | -0.0387 | 83 | 2.8% |  |
| Bear+HighVol | -0.0322 | -0.0410 | 161 | 5.4% |  |
| Bull | -0.0247 | -0.0142 | 2494 | 84.3% |  |
| Bear | -0.0341 | -0.0336 | 464 | 15.7% |  |
| HighVol | -0.0255 | -0.0241 | 699 | 23.6% |  |
| LowVol | -0.0439 | -0.0289 | 952 | 32.2% |  |

### mom_12_1

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0249 | +0.0203 | 844 | 30.5% |  |
| Bull+HighVol | +0.0328 | +0.0370 | 503 | 18.2% |  |
| Bear+LowVol | -0.0222 | -0.0267 | 83 | 3.0% |  |
| Bear+HighVol | +0.0262 | +0.0193 | 160 | 5.8% |  |
| Bull | +0.0277 | +0.0275 | 2368 | 85.6% |  |
| Bear | +0.0020 | +0.0006 | 398 | 14.4% |  |
| HighVol | +0.0312 | +0.0328 | 663 | 24.0% |  |
| LowVol | +0.0207 | +0.0161 | 927 | 33.5% |  |

### mom_1m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0096 | +0.0032 | 869 | 29.0% |  |
| Bull+HighVol | +0.0014 | -0.0015 | 538 | 17.9% |  |
| Bear+LowVol | -0.0039 | +0.0048 | 83 | 2.8% |  |
| Bear+HighVol | -0.0517 | -0.0495 | 161 | 5.4% |  |
| Bull | -0.0009 | -0.0054 | 2494 | 83.2% |  |
| Bear | -0.0238 | -0.0191 | 503 | 16.8% |  |
| HighVol | -0.0108 | -0.0126 | 699 | 23.3% |  |
| LowVol | +0.0084 | +0.0034 | 952 | 31.8% |  |

### mom_3m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0062 | +0.0003 | 869 | 29.4% |  |
| Bull+HighVol | +0.0248 | +0.0232 | 538 | 18.2% |  |
| Bear+LowVol | -0.0305 | -0.0343 | 83 | 2.8% |  |
| Bear+HighVol | -0.0143 | -0.0125 | 161 | 5.5% |  |
| Bull | +0.0108 | +0.0057 | 2494 | 84.4% |  |
| Bear | -0.0188 | -0.0198 | 461 | 15.6% |  |
| HighVol | +0.0158 | +0.0149 | 699 | 23.6% |  |
| LowVol | +0.0030 | -0.0027 | 952 | 32.2% |  |

### mom_6m

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0168 | +0.0109 | 869 | 30.0% |  |
| Bull+HighVol | +0.0317 | +0.0313 | 538 | 18.6% |  |
| Bear+LowVol | -0.0192 | -0.0181 | 83 | 2.9% |  |
| Bear+HighVol | +0.0070 | -0.0033 | 160 | 5.5% |  |
| Bull | +0.0188 | +0.0169 | 2494 | 86.2% |  |
| Bear | -0.0023 | -0.0041 | 398 | 13.8% |  |
| HighVol | +0.0260 | +0.0234 | 698 | 24.1% |  |
| LowVol | +0.0137 | +0.0084 | 952 | 32.9% |  |

### quality

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0187 | +0.0154 | 844 | 30.5% |  |
| Bull+HighVol | +0.0286 | +0.0316 | 503 | 18.2% |  |
| Bear+LowVol | -0.0086 | -0.0104 | 83 | 3.0% |  |
| Bear+HighVol | +0.0052 | +0.0023 | 160 | 5.8% |  |
| Bull | +0.0209 | +0.0213 | 2368 | 85.6% |  |
| Bear | -0.0019 | -0.0016 | 398 | 14.4% |  |
| HighVol | +0.0230 | +0.0245 | 663 | 24.0% |  |
| LowVol | +0.0163 | +0.0131 | 927 | 33.5% |  |

### rev_1m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0096 | -0.0032 | 869 | 29.0% |  |
| Bull+HighVol | -0.0014 | +0.0015 | 538 | 17.9% |  |
| Bear+LowVol | +0.0039 | -0.0048 | 83 | 2.8% |  |
| Bear+HighVol | +0.0517 | +0.0495 | 161 | 5.4% |  |
| Bull | +0.0009 | +0.0054 | 2494 | 83.2% |  |
| Bear | +0.0238 | +0.0191 | 503 | 16.8% |  |
| HighVol | +0.0108 | +0.0126 | 699 | 23.3% |  |
| LowVol | -0.0084 | -0.0034 | 952 | 31.8% |  |

### rev_1w

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0136 | -0.0074 | 869 | 28.8% |  |
| Bull+HighVol | +0.0128 | +0.0169 | 538 | 17.9% |  |
| Bear+LowVol | -0.0170 | -0.0140 | 83 | 2.8% |  |
| Bear+HighVol | +0.0442 | +0.0507 | 161 | 5.3% |  |
| Bull | +0.0021 | +0.0066 | 2494 | 82.8% |  |
| Bear | +0.0260 | +0.0278 | 519 | 17.2% |  |
| HighVol | +0.0201 | +0.0247 | 699 | 23.2% |  |
| LowVol | -0.0139 | -0.0079 | 952 | 31.6% |  |

### rsi_14

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0088 | -0.0062 | 869 | 28.9% |  |
| Bull+HighVol | -0.0055 | -0.0027 | 538 | 17.9% |  |
| Bear+LowVol | -0.0426 | -0.0574 | 83 | 2.8% |  |
| Bear+HighVol | +0.0401 | +0.0322 | 161 | 5.4% |  |
| Bull | +0.0008 | +0.0034 | 2494 | 83.0% |  |
| Bear | +0.0266 | +0.0193 | 510 | 17.0% |  |
| HighVol | +0.0050 | +0.0054 | 699 | 23.3% |  |
| LowVol | -0.0118 | -0.0107 | 952 | 31.7% |  |

### ts_mom

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0299 | -0.0153 | 844 | 30.5% |  |
| Bull+HighVol | -0.0066 | -0.0005 | 503 | 18.2% |  |
| Bear+LowVol | +0.0104 | +0.0022 | 83 | 3.0% |  |
| Bear+HighVol | +0.0141 | +0.0058 | 160 | 5.8% |  |
| Bull | -0.0110 | -0.0007 | 2368 | 85.6% |  |
| Bear | -0.0025 | -0.0068 | 398 | 14.4% |  |
| HighVol | -0.0016 | +0.0010 | 663 | 24.0% |  |
| LowVol | -0.0263 | -0.0138 | 927 | 33.5% |  |

### vol_regime

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0000 | — | 177 | 21.8% |  |
| Bull+HighVol | +0.0000 | — | 205 | 25.2% |  |
| Bear+LowVol | — | — | 11 | 1.4% | Insufficient dates (11 < 50) |
| Bear+HighVol | +0.0000 | — | 65 | 8.0% |  |
| Bull | -0.0000 | — | 688 | 84.6% |  |
| Bear | +0.0000 | — | 125 | 15.4% |  |
| HighVol | +0.0000 | — | 270 | 33.2% |  |
| LowVol | +0.0000 | — | 188 | 23.1% |  |

### vrp_proxy

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0294 | -0.0143 | 869 | 29.0% |  |
| Bull+HighVol | -0.0192 | -0.0153 | 538 | 17.9% |  |
| Bear+LowVol | -0.0464 | -0.0418 | 83 | 2.8% |  |
| Bear+HighVol | -0.0351 | -0.0439 | 161 | 5.4% |  |
| Bull | -0.0192 | -0.0093 | 2494 | 83.2% |  |
| Bear | -0.0196 | -0.0187 | 503 | 16.8% |  |
| HighVol | -0.0229 | -0.0219 | 699 | 23.3% |  |
| LowVol | -0.0309 | -0.0167 | 952 | 31.8% |  |

### zscore_20

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0155 | -0.0119 | 869 | 29.0% |  |
| Bull+HighVol | +0.0052 | +0.0118 | 538 | 17.9% |  |
| Bear+LowVol | -0.0198 | -0.0266 | 83 | 2.8% |  |
| Bear+HighVol | +0.0517 | +0.0468 | 161 | 5.4% |  |
| Bull | +0.0007 | +0.0052 | 2494 | 83.2% |  |
| Bear | +0.0298 | +0.0269 | 505 | 16.8% |  |
| HighVol | +0.0159 | +0.0199 | 699 | 23.3% |  |
| LowVol | -0.0159 | -0.0132 | 952 | 31.7% |  |

---

## Summary Verdicts

**Robust (positive IC ≥3/4 regimes):** mom_6m, mom_12_1, quality, vol_regime
**Regime-Dependent (stable direction but not consistently positive):** rsi_14, low_vol, vrp_proxy
**Fragile (direction flips, no consistency):** mom_1m, mom_3m, ts_mom, rev_1m, rev_1w, zscore_20