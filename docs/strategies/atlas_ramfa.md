# Atlas RAMFA Strategy

## Status

- **Registry name:** `atlas_ramfa`
- **Phase:** Research implementation
- **Deployment status:** Not approved for paper or live trading
- **Required gates:** Point-in-time backtest, walk-forward validation, transaction-cost calibration, paper trading, model-risk review

## Strategy Summary

Atlas RAMFA is a regime-adaptive multi-factor long/short equity strategy. It combines:

1. Cross-sectional momentum
2. Time-series momentum
3. Quality
4. Value
5. Low-volatility / risk efficiency
6. Short-term reversal
7. Liquidity and flow
8. Implied-realized volatility carry

Signals are winsorized, z-scored, optionally sector-neutralized, and combined with fixed or trailing information-coefficient weights. Positions are selected from the top and bottom tails of the composite score and weighted by conviction divided by variance.

## Data Contract

Minimum required input:

```python
{
    "close": np.ndarray,       # shape: [time, asset]
}
```

Recommended inputs:

```python
{
    "close": np.ndarray,
    "volume": np.ndarray,
    "spread_bps": np.ndarray,
    "implied_vol": np.ndarray,
    "regime_score": np.ndarray,
    "symbols": np.ndarray,
    "sector": np.ndarray,
    "earnings": np.ndarray,
    "book_value": np.ndarray,
    "cash_flow": np.ndarray,
    "enterprise_value": np.ndarray,
}
```

All arrays must be point-in-time and aligned by timestamp and asset.

## Configuration

Primary config file:

```text
config/strategies/atlas_ramfa.toml
```

Important parameters:

| Parameter | Default | Purpose |
|---|---:|---|
| `rebalance_freq` | 5 | Rebalance every 5 trading days |
| `lookback_days` | 252 | Intermediate momentum lookback |
| `skip_days` | 21 | Skip most recent month for momentum |
| `vol_lookback` | 63 | Realized volatility window |
| `max_gross_exposure` | 1.5 | Maximum gross exposure |
| `min_net_exposure` | -0.25 | Minimum net exposure |
| `max_net_exposure` | 0.50 | Maximum net exposure |
| `single_name_max_pct` | 0.02 | Single-name cap |
| `min_regime_score` | 55.0 | New-entry regime gate |
| `min_liquidity_score` | 75.0 | Liquidity gate |

## Validation Checklist

Before any paper allocation:

- [ ] Point-in-time universe reconstruction completed
- [ ] Survivorship bias audit completed
- [ ] Look-ahead bias audit completed
- [ ] Corporate action adjustment validated
- [ ] Transaction-cost model calibrated
- [ ] Borrow-cost model calibrated
- [ ] Walk-forward test passed
- [ ] Deflated Sharpe / multiple-testing review completed
- [ ] Stress tests completed
- [ ] Paper trading run completed without operational failures

## Shutdown Conditions

The strategy should remain in research until these conditions are met:

1. 12-month information coefficient is positive and statistically significant.
2. Net Sharpe lower confidence bound meets committee threshold.
3. Maximum drawdown is within approved risk budget.
4. Transaction costs do not consume expected alpha.
5. Liquidity and borrow constraints are satisfied across the target universe.
