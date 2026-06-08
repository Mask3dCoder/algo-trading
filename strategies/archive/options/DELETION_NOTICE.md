# DELETION NOTICE — Options Strategies

**Date:** 2026-06-08
**Authority:** Honest Sweep Validation, Phase 0 Rebuild

## Reason

Options strategies were never validated on real market data. Backtests used synthetic
implied volatility (constant 25% with noise) and zero transaction costs. Claims of
Sharpe 1.90–2.48 are not supported by real-data evidence.

Specific defects:
- **Synthetic IV**: All options backtests used `np.full((n, 1), 0.25)` for implied vol
- **Zero bid-ask spread**: No modeling of the 1-2% typical options spread
- **No delta-hedge cost**: Continuous delta hedging was assumed frictionless
- **No gamma risk**: Intraday MTM P&L from gamma exposure was ignored
- **No real options chain**: Strategies operated on synthetic Greeks, not market data

## Action

Archived indefinitely. No allocation permitted. Not "pending." Not "research." **Dead.**

If real-data validation is attempted in future, must pass ALL of:
1. Real options chain data or documented IV proxy with verifiable source
2. Bid-ask spread modeling (minimum 1% of midprice for liquid, 2% for illiquid)
3. Delta-hedge cost inclusion (minimum 5bps per rebalance)
4. Intraday gamma risk (5-min MTM P&L stress test)
5. ci_lower > 0, DSR p < 0.1, MaxDD < 15%
6. Walk-forward OOS validation on ≥ 4 folds with realistic costs

## Status

**DELETED** from active registry. See `quantcore/strategies/registry.py`.

## Archived Strategies

| Strategy | Claimed Sharpe | Validation | Verdict |
|---|---|---|---|
| options_delta_neutral | ~2.0 | Synthetic IV, zero costs | DEAD |
| options_iron_condor | ~1.9 | Synthetic IV, zero costs | DEAD |
| options_put_write | ~2.3 | Synthetic IV, zero costs | DEAD |
| options_tail_hedge | ~2.5 | Synthetic IV, zero costs | DEAD |
| options_earnings_vol | ~1.8 | Synthetic IV, zero costs | DEAD |

## Rationale for Not Building Options Infrastructure Now

The 4 equity/macro survivors failed cost-aware validation. Building options chain
infrastructure for dead strategies is wasted effort. If a viable equity/macro product
is established, options infrastructure can be built then — as a diversification layer,
not a foundation.

---

*"The first principle is that you must not fool yourself — and you are the easiest
person to fool."* — Richard Feynman
