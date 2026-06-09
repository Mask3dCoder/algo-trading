# Pre-Mortem: options_put_write
## Strategy: Short Put (Cash-Secured Put Writing on SPY)

### This strategy will fail when:
- **Tail risk materializes:** A gap-down > strike distance (e.g., -15% single-day gap on SPY) wipes out months/years of premium.
- **Persistent low-vol grind:** If realized vol consistently > implied vol (negative VRP), the strategy loses its edge.
- **Correlation regime change:** If the VIX/SPY inverse correlation breaks (e.g., both fall together during a slow grind), the put-write signal becomes mis-calibrated.

### The first sign of decay will be:
1. **Put skew steepening:** If the put skew (difference between 80% and 90% moneyness put IV) increases > 20% month-over-month.
2. **Win rate on monthly cycles drops below 50%** for 3 consecutive months.
3. **Rolling 21-day realized vol > 1-month implied vol** for >40% of days (negative VRP persistence).

### We will disable it when:
- DSR p-value > 0.1 (fails multiple-testing correction)
- Rolling 60-day Max DD exceeds 25%
- Kelly fraction (based on forward-looking realized vol) drops below 0.05 (edge erosion)
- Put skew index > 1.5 (tail risk mispricing)

### Review checklist (monthly):
- [ ] DSR p < 0.1?
- [ ] Max DD 60-day rolling < 25%?
- [ ] Win rate on last 3 months > 50%
- [ ] Tail-risk overlay (2% OTM puts) not triggered?