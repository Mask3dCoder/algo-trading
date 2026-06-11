# Atlas RAMFA Model-Risk Review

## Scope

Atlas RAMFA is a regime-adaptive multi-factor long/short equity research strategy.
This review is a research gate document only. It does not approve paper or live trading.

## Required data dependencies

- Point-in-time close prices for the full target universe.
- Point-in-time volume, spread, and liquidity fields.
- Optional but preferred: sector, implied volatility, fundamentals, ADV, float value,
  institutional ownership, and days-to-liquidate.
- Regime score or a reproducible regime proxy computed only from past data.

## Forward-looking bias controls

- Signal generation must use only data available at or before the signal timestamp.
- Walk-forward folds must include an embargo between training and test windows.
- Execution must be lagged by at least one bar after signal generation.
- No future universe membership, fundamentals, or corporate-action information may be used.

## Edge cases and failure modes

- Insufficient history returns no directional signals.
- Low liquidity or sparse universe can suppress entries.
- Regime score below the configured threshold blocks new entries and can trigger exits.
- Missing implied volatility should not crash the strategy; volatility carry becomes neutral.
- Sector neutralization is a no-op when sector labels are absent.
- IC weighting can be unstable with short histories or low cross-sectional dispersion.
- Single-name caps and gross/net exposure limits can clip intended positions.
- Short books require borrow availability and borrow-cost modeling before paper activation.

## Validation gates before production use

1. Point-in-time backtest with lagged execution.
2. Walk-forward validation with embargo.
3. Transaction-cost and borrow-cost sensitivity.
4. Paper-readiness configuration review.
5. Model-risk review approval.

## Production quarantine

Atlas RAMFA must remain absent from `config/paper.yaml` and `config/live.yaml` until all
gates are passed. The only approved status is `RESEARCH`.
