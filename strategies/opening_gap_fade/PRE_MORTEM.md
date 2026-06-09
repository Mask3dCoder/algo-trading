# Pre-Mortem: opening_gap_fade
## Strategy: Fade Overnight Gaps > 2% on SPY/QQQ

### This strategy will fail when:
- **News-driven gaps become non-reversible:** If macro/news events create persistent directional moves (e.g., rate surprise, war, pandemic), the mean-reversion assumption breaks.
- **Algorithmic gap exploitation:** If HFT/market makers widen spreads after large gaps, slippage consumes the fade profit.
- **Pre-market volume loses predictive power:** If institutional pre-market activity changes pattern (e.g., more 24-hour trading, extended hours), the volume filter becomes ineffective.

### The first sign of decay will be:
1. **Pre-market volume threshold no longer filters news gaps:** False positive rate > 30% on high-pre-market-volume days.
2. **Average fill time > 5 minutes:** If the opening auction becomes more volatile, the 2× slippage model may be underestimated.
3. **Win rate drops below 55%** for 2 consecutive months (from target 58-65%).

### We will disable it when:
- OOS Sharpe < 0.8 for 2 consecutive walk-forward windows
- Win rate < 55% for 2 consecutive months
- Average slippage > 4 bps (2× base) on opening fills
- Gap frequency drops below 1 per week (insufficient signal)

### Review checklist (monthly):
- [ ] OOS Sharpe > 0.8?
- [ ] Win rate last 60 days > 55%?
- [ ] Avg slippage per trade < 4 bps?
- [ ] Gap frequency > 1/week?