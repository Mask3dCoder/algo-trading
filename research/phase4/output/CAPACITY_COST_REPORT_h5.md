# Capacity & Cost Report

## Methodology

- **Capacity:** 5% of Average Daily Dollar Volume (ADV)
- **Turnover:** Estimated from signal rank changes, scaled by holding_days
- **Round-trip cost:** 10 bps (conservative for large-cap equities)
- **Min IC:** IC required to cover annual trading costs
- **Target vol:** 15% annual (institutional vol-target standard)

- **Holding days adjustment:** Turnover is scaled by 252/holding_days
- **min_IC scaling:** min_IC ∝ 1/holding_days — longer horizons lower the hurdle

---

## Summary Table (per horizon)

| Factor | h | ADV ($M) | Max Cap ($M) | Turnover (×/yr) | Ann Cost (bps) | Min IC | Viable |
|--------|--|--------:|-------------:|----------------:|---------------:|-------:|--------|
| low_vol | 5 | 844 | 42 | 44.8 | 448.0 | 0.2987 | NOT VIABLE |
| mom_12_1 | 5 | 844 | 42 | 62.2 | 621.8 | 0.4145 | NOT VIABLE |
| mom_1m | 5 | 844 | 42 | 193.7 | 1936.8 | 1.2912 | NOT VIABLE |
| mom_3m | 5 | 844 | 42 | 112.6 | 1125.6 | 0.7504 | NOT VIABLE |
| mom_6m | 5 | 844 | 42 | 82.9 | 829.4 | 0.5529 | NOT VIABLE |
| quality | 5 | 844 | 42 | 62.1 | 621.2 | 0.4142 | NOT VIABLE |
| rev_1m | 5 | 844 | 42 | 193.7 | 1936.8 | 1.2912 | NOT VIABLE |
| rev_1w | 5 | 844 | 42 | 403.8 | 4038.4 | 2.6923 | NOT VIABLE |
| rsi_14 | 5 | 844 | 42 | 242.0 | 2420.3 | 1.6135 | NOT VIABLE |
| ts_mom | 5 | 844 | 42 | 77.1 | 771.4 | 0.5143 | NOT VIABLE |
| vol_regime | 5 | 844 | 42 | 214.0 | 2140.2 | 1.4268 | NOT VIABLE |
| vrp_proxy | 5 | 844 | 42 | 217.6 | 2175.6 | 1.4504 | NOT VIABLE |
| zscore_20 | 5 | 844 | 42 | 310.0 | 3099.7 | 2.0665 | NOT VIABLE |

**0/13 factor-horizon combinations survive cost hurdle.**

---

## Detailed Analysis

### low_vol (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 44.8× (17.78% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 448.0 bps
- **Min IC to Survive:** 0.2987
- **Verdict:** ✗ NOT VIABLE

### mom_12_1 (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 62.2× (24.67% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 621.8 bps
- **Min IC to Survive:** 0.4145
- **Verdict:** ✗ NOT VIABLE

### mom_1m (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 193.7× (76.86% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 1936.8 bps
- **Min IC to Survive:** 1.2912
- **Verdict:** ✗ NOT VIABLE

### mom_3m (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 112.6× (44.67% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 1125.6 bps
- **Min IC to Survive:** 0.7504
- **Verdict:** ✗ NOT VIABLE

### mom_6m (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 82.9× (32.91% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 829.4 bps
- **Min IC to Survive:** 0.5529
- **Verdict:** ✗ NOT VIABLE

### quality (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 62.1× (24.65% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 621.2 bps
- **Min IC to Survive:** 0.4142
- **Verdict:** ✗ NOT VIABLE

### rev_1m (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 193.7× (76.86% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 1936.8 bps
- **Min IC to Survive:** 1.2912
- **Verdict:** ✗ NOT VIABLE

### rev_1w (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 403.8× (160.25% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 4038.4 bps
- **Min IC to Survive:** 2.6923
- **Verdict:** ✗ NOT VIABLE

### rsi_14 (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 242.0× (96.04% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 2420.3 bps
- **Min IC to Survive:** 1.6135
- **Verdict:** ✗ NOT VIABLE

### ts_mom (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 77.1× (30.61% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 771.4 bps
- **Min IC to Survive:** 0.5143
- **Verdict:** ✗ NOT VIABLE

### vol_regime (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 214.0× (84.93% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 2140.2 bps
- **Min IC to Survive:** 1.4268
- **Verdict:** ✗ NOT VIABLE

### vrp_proxy (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 217.6× (86.33% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 2175.6 bps
- **Min IC to Survive:** 1.4504
- **Verdict:** ✗ NOT VIABLE

### zscore_20 (h=5)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 310.0× (123.01% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 3099.7 bps
- **Min IC to Survive:** 2.0665
- **Verdict:** ✗ NOT VIABLE
