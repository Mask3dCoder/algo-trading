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
| low_vol | 1 | 844 | 42 | 210.4 | 2103.7 | 1.4025 | NOT VIABLE |
| mom_12_1 | 1 | 844 | 42 | 307.3 | 3072.8 | 2.0485 | NOT VIABLE |
| mom_1m | 1 | 844 | 42 | 958.4 | 9583.7 | 6.3891 | NOT VIABLE |
| mom_3m | 1 | 844 | 42 | 561.6 | 5616.4 | 3.7442 | NOT VIABLE |
| mom_6m | 1 | 844 | 42 | 409.3 | 4092.8 | 2.7286 | NOT VIABLE |
| quality | 1 | 844 | 42 | 305.8 | 3058.2 | 2.0388 | NOT VIABLE |
| rev_1m | 1 | 844 | 42 | 958.4 | 9583.7 | 6.3891 | NOT VIABLE |
| rev_1w | 1 | 844 | 42 | 1000.0 | 10000.0 | 6.6667 | NOT VIABLE |
| rsi_14 | 1 | 844 | 42 | 1000.0 | 10000.0 | 6.6667 | NOT VIABLE |
| ts_mom | 1 | 844 | 42 | 386.3 | 3862.7 | 2.5751 | NOT VIABLE |
| vol_regime | 1 | 844 | 42 | 1000.0 | 10000.0 | 6.6667 | NOT VIABLE |
| vrp_proxy | 1 | 844 | 42 | 1000.0 | 10000.0 | 6.6667 | NOT VIABLE |
| zscore_20 | 1 | 844 | 42 | 1000.0 | 10000.0 | 6.6667 | NOT VIABLE |

**0/13 factor-horizon combinations survive cost hurdle.**

---

## Detailed Analysis

### low_vol (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 210.4× (83.48% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 2103.7 bps
- **Min IC to Survive:** 1.4025
- **Verdict:** ✗ NOT VIABLE

### mom_12_1 (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 307.3× (121.94% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 3072.8 bps
- **Min IC to Survive:** 2.0485
- **Verdict:** ✗ NOT VIABLE

### mom_1m (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 958.4× (380.31% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 9583.7 bps
- **Min IC to Survive:** 6.3891
- **Verdict:** ✗ NOT VIABLE

### mom_3m (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 561.6× (222.87% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 5616.4 bps
- **Min IC to Survive:** 3.7442
- **Verdict:** ✗ NOT VIABLE

### mom_6m (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 409.3× (162.41% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 4092.8 bps
- **Min IC to Survive:** 2.7286
- **Verdict:** ✗ NOT VIABLE

### quality (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 305.8× (121.36% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 3058.2 bps
- **Min IC to Survive:** 2.0388
- **Verdict:** ✗ NOT VIABLE

### rev_1m (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 958.4× (380.31% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 9583.7 bps
- **Min IC to Survive:** 6.3891
- **Verdict:** ✗ NOT VIABLE

### rev_1w (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 1000.0× (396.83% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 10000.0 bps
- **Min IC to Survive:** 6.6667
- **Verdict:** ✗ NOT VIABLE

### rsi_14 (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 1000.0× (396.83% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 10000.0 bps
- **Min IC to Survive:** 6.6667
- **Verdict:** ✗ NOT VIABLE

### ts_mom (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 386.3× (153.28% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 3862.7 bps
- **Min IC to Survive:** 2.5751
- **Verdict:** ✗ NOT VIABLE

### vol_regime (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 1000.0× (396.83% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 10000.0 bps
- **Min IC to Survive:** 6.6667
- **Verdict:** ✗ NOT VIABLE

### vrp_proxy (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 1000.0× (396.83% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 10000.0 bps
- **Min IC to Survive:** 6.6667
- **Verdict:** ✗ NOT VIABLE

### zscore_20 (h=1)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 1000.0× (396.83% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 10000.0 bps
- **Min IC to Survive:** 6.6667
- **Verdict:** ✗ NOT VIABLE
