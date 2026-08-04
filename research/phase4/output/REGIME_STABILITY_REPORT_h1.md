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
| low_vol | -0.0151 | -0.0041 | -0.0120 | -0.0221 | ✗ | ✓ |
| mom_12_1 | -0.0030 | +0.0284 | -0.0136 | +0.0073 | ✗ | ✗ |
| mom_1m | +0.0075 | -0.0050 | -0.0225 | -0.0183 | ✗ | ✗ |
| mom_3m | -0.0013 | +0.0163 | -0.0393 | -0.0157 | ✗ | ✓ |
| mom_6m | -0.0050 | +0.0241 | -0.0113 | -0.0047 | ✗ | ✓ |
| quality | -0.0003 | +0.0228 | -0.0152 | +0.0002 | ✗ | ✗ |
| rev_1m | -0.0075 | +0.0050 | +0.0225 | +0.0183 | ✓ | ✗ |
| rev_1w | -0.0049 | +0.0152 | +0.0202 | +0.0158 | ✓ | ✗ |
| rsi_14 | -0.0090 | +0.0055 | -0.0083 | +0.0114 | ✗ | ✗ |
| ts_mom | -0.0075 | +0.0074 | -0.0032 | +0.0135 | ✗ | ✗ |
| vol_regime | +0.0000 | +0.0000 | -0.0000 | -0.0000 | ✗ | ✗ |
| vrp_proxy | +0.0064 | +0.0074 | +0.0090 | +0.0026 | ✓ | ✓ |
| zscore_20 | -0.0113 | +0.0105 | +0.0074 | +0.0078 | ✓ | ✗ |

---

## Detailed Per-Factor Regime Analysis

### low_vol

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0151 | -0.0068 | 869 | 29.4% |  |
| Bull+HighVol | -0.0041 | -0.0003 | 538 | 18.2% |  |
| Bear+LowVol | -0.0120 | -0.0059 | 83 | 2.8% |  |
| Bear+HighVol | -0.0221 | -0.0235 | 161 | 5.4% |  |
| Bull | -0.0047 | +0.0024 | 2494 | 84.3% |  |
| Bear | -0.0225 | -0.0175 | 464 | 15.7% |  |
| HighVol | -0.0083 | -0.0057 | 699 | 23.6% |  |
| LowVol | -0.0148 | -0.0067 | 952 | 32.2% |  |

### mom_12_1

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0030 | -0.0010 | 844 | 30.5% |  |
| Bull+HighVol | +0.0284 | +0.0302 | 503 | 18.2% |  |
| Bear+LowVol | -0.0136 | -0.0135 | 83 | 3.0% |  |
| Bear+HighVol | +0.0073 | +0.0143 | 160 | 5.8% |  |
| Bull | +0.0126 | +0.0165 | 2368 | 85.6% |  |
| Bear | -0.0020 | +0.0014 | 398 | 14.4% |  |
| HighVol | +0.0233 | +0.0264 | 663 | 24.0% |  |
| LowVol | -0.0040 | -0.0021 | 927 | 33.5% |  |

### mom_1m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0075 | +0.0047 | 869 | 29.0% |  |
| Bull+HighVol | -0.0050 | -0.0044 | 538 | 17.9% |  |
| Bear+LowVol | -0.0225 | -0.0207 | 83 | 2.8% |  |
| Bear+HighVol | -0.0183 | -0.0262 | 161 | 5.4% |  |
| Bull | -0.0033 | -0.0040 | 2494 | 83.2% |  |
| Bear | -0.0083 | -0.0142 | 503 | 16.8% |  |
| HighVol | -0.0081 | -0.0094 | 699 | 23.3% |  |
| LowVol | +0.0048 | +0.0025 | 952 | 31.8% |  |

### mom_3m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0013 | -0.0034 | 869 | 29.4% |  |
| Bull+HighVol | +0.0163 | +0.0187 | 538 | 18.2% |  |
| Bear+LowVol | -0.0393 | -0.0409 | 83 | 2.8% |  |
| Bear+HighVol | -0.0157 | -0.0133 | 161 | 5.5% |  |
| Bull | +0.0037 | +0.0033 | 2494 | 84.4% |  |
| Bear | -0.0180 | -0.0206 | 461 | 15.6% |  |
| HighVol | +0.0089 | +0.0113 | 699 | 23.6% |  |
| LowVol | -0.0046 | -0.0067 | 952 | 32.2% |  |

### mom_6m

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0050 | -0.0038 | 869 | 30.0% |  |
| Bull+HighVol | +0.0241 | +0.0241 | 538 | 18.6% |  |
| Bear+LowVol | -0.0113 | -0.0161 | 83 | 2.9% |  |
| Bear+HighVol | -0.0047 | -0.0015 | 160 | 5.5% |  |
| Bull | +0.0086 | +0.0111 | 2494 | 86.2% |  |
| Bear | -0.0097 | -0.0097 | 398 | 13.8% |  |
| HighVol | +0.0175 | +0.0182 | 698 | 24.1% |  |
| LowVol | -0.0055 | -0.0049 | 952 | 32.9% |  |

### quality

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0003 | +0.0032 | 844 | 30.5% |  |
| Bull+HighVol | +0.0228 | +0.0256 | 503 | 18.2% |  |
| Bear+LowVol | -0.0152 | -0.0150 | 83 | 3.0% |  |
| Bear+HighVol | +0.0002 | +0.0060 | 160 | 5.8% |  |
| Bull | +0.0097 | +0.0145 | 2368 | 85.6% |  |
| Bear | -0.0057 | -0.0032 | 398 | 14.4% |  |
| HighVol | +0.0174 | +0.0209 | 663 | 24.0% |  |
| LowVol | -0.0016 | +0.0015 | 927 | 33.5% |  |

### rev_1m

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0075 | -0.0047 | 869 | 29.0% |  |
| Bull+HighVol | +0.0050 | +0.0044 | 538 | 17.9% |  |
| Bear+LowVol | +0.0225 | +0.0207 | 83 | 2.8% |  |
| Bear+HighVol | +0.0183 | +0.0262 | 161 | 5.4% |  |
| Bull | +0.0033 | +0.0040 | 2494 | 83.2% |  |
| Bear | +0.0083 | +0.0142 | 503 | 16.8% |  |
| HighVol | +0.0081 | +0.0094 | 699 | 23.3% |  |
| LowVol | -0.0048 | -0.0025 | 952 | 31.8% |  |

### rev_1w

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0049 | -0.0023 | 869 | 28.8% |  |
| Bull+HighVol | +0.0152 | +0.0154 | 538 | 17.9% |  |
| Bear+LowVol | +0.0202 | +0.0182 | 83 | 2.8% |  |
| Bear+HighVol | +0.0158 | +0.0236 | 161 | 5.3% |  |
| Bull | +0.0053 | +0.0080 | 2494 | 82.8% |  |
| Bear | +0.0240 | +0.0299 | 519 | 17.2% |  |
| HighVol | +0.0153 | +0.0173 | 699 | 23.2% |  |
| LowVol | -0.0027 | -0.0005 | 952 | 31.6% |  |

### rsi_14

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0090 | -0.0089 | 869 | 28.9% |  |
| Bull+HighVol | +0.0055 | +0.0043 | 538 | 17.9% |  |
| Bear+LowVol | -0.0083 | -0.0161 | 83 | 2.8% |  |
| Bear+HighVol | +0.0114 | +0.0143 | 161 | 5.4% |  |
| Bull | +0.0019 | +0.0014 | 2494 | 83.0% |  |
| Bear | +0.0112 | +0.0137 | 510 | 17.0% |  |
| HighVol | +0.0069 | +0.0066 | 699 | 23.3% |  |
| LowVol | -0.0089 | -0.0095 | 952 | 31.7% |  |

### ts_mom

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0075 | -0.0001 | 844 | 30.5% |  |
| Bull+HighVol | +0.0074 | +0.0114 | 503 | 18.2% |  |
| Bear+LowVol | -0.0032 | -0.0072 | 83 | 3.0% |  |
| Bear+HighVol | +0.0135 | +0.0139 | 160 | 5.8% |  |
| Bull | +0.0018 | +0.0085 | 2368 | 85.6% |  |
| Bear | +0.0002 | -0.0007 | 398 | 14.4% |  |
| HighVol | +0.0089 | +0.0120 | 663 | 24.0% |  |
| LowVol | -0.0071 | -0.0007 | 927 | 33.5% |  |

### vol_regime

- **Robust:** No — IC not consistently positive across regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0000 | — | 820 | 32.0% |  |
| Bull+HighVol | +0.0000 | — | 481 | 18.8% |  |
| Bear+LowVol | -0.0000 | — | 80 | 3.1% |  |
| Bear+HighVol | -0.0000 | — | 148 | 5.8% |  |
| Bull | -0.0000 | — | 2210 | 86.2% |  |
| Bear | -0.0000 | — | 353 | 13.8% |  |
| HighVol | +0.0000 | — | 629 | 24.5% |  |
| LowVol | -0.0000 | — | 900 | 35.1% |  |

### vrp_proxy

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** Yes — IC direction consistent across ≥3/4 regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | +0.0064 | +0.0053 | 869 | 28.9% |  |
| Bull+HighVol | +0.0074 | +0.0098 | 538 | 17.9% |  |
| Bear+LowVol | +0.0090 | +0.0050 | 83 | 2.8% |  |
| Bear+HighVol | +0.0026 | +0.0007 | 161 | 5.3% |  |
| Bull | +0.0006 | -0.0002 | 2494 | 82.9% |  |
| Bear | +0.0125 | +0.0072 | 514 | 17.1% |  |
| HighVol | +0.0063 | +0.0077 | 699 | 23.2% |  |
| LowVol | +0.0067 | +0.0053 | 952 | 31.6% |  |

### zscore_20

- **Robust:** Yes — positive IC in ≥3/4 regimes
- **Stable:** No — IC direction flips across regimes

| Regime | IC | Rank IC | N Dates | Coverage | Note |
|--------|------:|--------:|--------:|---------:|------|
| Bull+LowVol | -0.0113 | -0.0091 | 869 | 29.0% |  |
| Bull+HighVol | +0.0105 | +0.0108 | 538 | 17.9% |  |
| Bear+LowVol | +0.0074 | +0.0005 | 83 | 2.8% |  |
| Bear+HighVol | +0.0078 | +0.0114 | 161 | 5.4% |  |
| Bull | +0.0022 | +0.0040 | 2494 | 83.2% |  |
| Bear | +0.0150 | +0.0174 | 505 | 16.8% |  |
| HighVol | +0.0099 | +0.0110 | 699 | 23.3% |  |
| LowVol | -0.0097 | -0.0082 | 952 | 31.7% |  |

---

## Summary Verdicts

**Robust (positive IC ≥3/4 regimes):** rev_1m, rev_1w, zscore_20, vrp_proxy
**Regime-Dependent (stable direction but not consistently positive):** mom_3m, mom_6m, low_vol
**Fragile (direction flips, no consistency):** mom_1m, mom_12_1, ts_mom, rsi_14, quality, vol_regime