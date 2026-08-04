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
| low_vol | 21 | 844 | 42 | 10.0 | 100.0 | 0.0667 | NOT VIABLE |
| mom_12_1 | 21 | 844 | 42 | 15.1 | 151.3 | 0.1009 | NOT VIABLE |
| mom_1m | 21 | 844 | 42 | 46.3 | 462.9 | 0.3086 | NOT VIABLE |
| mom_3m | 21 | 844 | 42 | 27.5 | 275.4 | 0.1836 | NOT VIABLE |
| mom_6m | 21 | 844 | 42 | 20.3 | 202.6 | 0.1351 | NOT VIABLE |
| quality | 21 | 844 | 42 | 15.2 | 152.3 | 0.1015 | NOT VIABLE |
| rev_1m | 21 | 844 | 42 | 46.3 | 462.9 | 0.3086 | NOT VIABLE |
| rev_1w | 21 | 844 | 42 | 97.2 | 972.0 | 0.6480 | NOT VIABLE |
| rsi_14 | 21 | 844 | 42 | 59.3 | 593.2 | 0.3955 | NOT VIABLE |
| ts_mom | 21 | 844 | 42 | 20.5 | 204.7 | 0.1365 | NOT VIABLE |
| vol_regime | 21 | 844 | 42 | 43.3 | 432.7 | 0.2885 | NOT VIABLE |
| vrp_proxy | 21 | 844 | 42 | 50.4 | 503.7 | 0.3358 | NOT VIABLE |
| zscore_20 | 21 | 844 | 42 | 74.0 | 740.3 | 0.4936 | NOT VIABLE |

**0/13 factor-horizon combinations survive cost hurdle.**

---

## Detailed Analysis

### low_vol (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 10.0× (3.97% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 100.0 bps
- **Min IC to Survive:** 0.0667
- **Verdict:** ✗ NOT VIABLE

### mom_12_1 (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 15.1× (6.00% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 151.3 bps
- **Min IC to Survive:** 0.1009
- **Verdict:** ✗ NOT VIABLE

### mom_1m (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 46.3× (18.37% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 462.9 bps
- **Min IC to Survive:** 0.3086
- **Verdict:** ✗ NOT VIABLE

### mom_3m (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 27.5× (10.93% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 275.4 bps
- **Min IC to Survive:** 0.1836
- **Verdict:** ✗ NOT VIABLE

### mom_6m (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 20.3× (8.04% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 202.6 bps
- **Min IC to Survive:** 0.1351
- **Verdict:** ✗ NOT VIABLE

### quality (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 15.2× (6.04% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 152.3 bps
- **Min IC to Survive:** 0.1015
- **Verdict:** ✗ NOT VIABLE

### rev_1m (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 46.3× (18.37% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 462.9 bps
- **Min IC to Survive:** 0.3086
- **Verdict:** ✗ NOT VIABLE

### rev_1w (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 97.2× (38.57% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 972.0 bps
- **Min IC to Survive:** 0.6480
- **Verdict:** ✗ NOT VIABLE

### rsi_14 (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 59.3× (23.54% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 593.2 bps
- **Min IC to Survive:** 0.3955
- **Verdict:** ✗ NOT VIABLE

### ts_mom (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 20.5× (8.12% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 204.7 bps
- **Min IC to Survive:** 0.1365
- **Verdict:** ✗ NOT VIABLE

### vol_regime (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 43.3× (17.17% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 432.7 bps
- **Min IC to Survive:** 0.2885
- **Verdict:** ✗ NOT VIABLE

### vrp_proxy (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 50.4× (19.99% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 503.7 bps
- **Min IC to Survive:** 0.3358
- **Verdict:** ✗ NOT VIABLE

### zscore_20 (h=21)

- **Average ADV:** $844M
- **Max Capacity:** $42M (5% participation)
- **Annual Turnover:** 74.0× (29.38% per day)
- **Round-trip Cost:** 10.0 bps
- **Annual Cost:** 740.3 bps
- **Min IC to Survive:** 0.4936
- **Verdict:** ✗ NOT VIABLE
