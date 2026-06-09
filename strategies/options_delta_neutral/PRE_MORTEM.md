# Pre-Mortem: options_delta_neutral
## Strategy: Delta-Neutral Short Volatility (Straddles/Strangles)

### This strategy will fail when:
- **Volmageddon-type event:** A sudden jump in realized vol that exceeds implied vol (e.g., Feb 2018). The delta-hedge frequency (daily) may be too slow for a gap-opening followed by sustained momentum.
- **VIX flash > 30 with sustained elevation:** If VIX > 30 for >5 consecutive days, the strategy's vol regime filter reduces size 50%, but the remaining exposure still bleeds theta faster than gamma scalping can recover.
- **Negative skew regime shift:** If market skew steepens Burke sharply (e.g., March 2020), the straddle/strangle short suffers asymmetric losses on the put wing even with delta hedging.

### The first sign of decay will be:
1. **Gamma/vega P&L turns consistently negative for >10 consecutive days** (the P&L attribution in the hardened version will flag this).
2. **Delta-hedge frequency becomes inadequate:** If the average absolute delta after re-hedge exceeds 0.10 regularly, the hedge is mis-calibrated.
3. **Win rate on trades drops below 55%** (from historical ~65%) for two consecutive months.

### We will disable it when:
- OOS Sharpe CI_lower(90%) < 0.0 for 2 consecutive walk-forward windows
- Intraday Max DD exceeds 8% (MTM basis) in any rolling 60-day window
- Gamma/vega P&L negative for >20 consecutive trading days (structural decay signal)
- VIX regime filter triggers vol_ok = False for >20 consecutive trading days (prolonged high-vol regime)

### Review checklist (monthly):
- [ ] OOS Sharpe CI_lower > 0.0?
- [ ] Intraday Max DD < 8%?
- [ ] Gamma/vega P&L cumulative over 20 days > 0?
- [ ] Delta-hedge slippage cost per round-trip < 1.5 bps?