# Dataset Specification (MVQRP)

All research datasets used in the Signal Lab must adhere to the following schema to ensure compatibility with `evaluation.py`.

## Core Columns
- `timestamp`: UTC datetime (Index)
- `symbol`: String identifier
- `open`, `high`, `low`, `close`: Float (split/dividend adjusted)
- `volume`: Float
- `vwap`: Float (if available intraday)

## Evaluation Columns (Mandatory for Signal Lab)
- `returns`: Float, current bar's return `(close - pre_close) / pre_close`
- `forward_returns_1`: Float, `(next_close - close) / close`
- `forward_returns_5`: Float, 5-bar forward return
- `benchmark_returns`: Float, return of the broader index (e.g., SPY)

## Best Practices
- Ensure data is fully adjusted for corporate actions.
- Any alternative data must be timestamp-aligned to the `close` time to prevent look-ahead bias.
