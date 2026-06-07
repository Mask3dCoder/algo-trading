# Pre-Mortem: options_iron_condor
## Strategy: Iron Condor — 16-delta wings, 30-45 DTE, IV-rank filtered

### This strategy will fail when:
- **Volatility crush during trend acceleration:** If VIX spikes >15 pts in a single session (e.g., rate shock, macro event), the short-gamma exposure creates large unrealized losses before IV has time to decay.
- **Wings get breached without adjustment:** If the underlying gaps through a short strike overnight, the 2:1 max-loss rule triggers a 2R loss before the adjustment logic can fire.
- **IV rank signal fails:** During low-vol regimes (VIX < 15), IV rank stays below entry threshold for months — no signals generated.
- **Post-earnings IV collapse too fast:** Iron condors entered before earnings face rapid IV crush — the short premium decays faster than expected, but wing risk remains elevated.

### The first sign of decay will be:
1. **Win rate drops below 72%** (from target 75-80%) — too many full-loss events.
2. **Average loss > 2.5× average win** —wing adjustments triggering too close to expiration.
3. **Adjustments triggered > 3× per week** — underlying repeatedly approaching short strikes.
4. **IV rank filter generating < 2 signals/month** — regime shift in vol behavior.

### We will disable it when:
- OOS Sharpe < 0.8 for 2 consecutive walk-forward windows
- Win rate < 70% for 2 consecutive quarters
- N adjustments/week > 5 (adjustment exhaustion)
- Average loss/R > 3.0 (wing management failing)

### Review checklist (monthly):
- [ ] OOS Sharpe > 0.8?
- [ ] Win rate last 60 days > 72%?
- [ ] Avg loss / avg win < 2.5?
- [ ] N adjustments week < 3?
- [ ] IV rank signal frequency > 2/month?