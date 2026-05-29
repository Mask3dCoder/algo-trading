# Options Data Roadmap — From VIX Proxy to Real Chains

**Created**: 2026-05-29
**Status**: Planning
**Owner**: QuantCore Team

---

## Current State

### What We Have
- **VIX as IV proxy**: The live Delta Neutral strategy uses the CBOE VIX closing price
  (`yfinance ^VIX`) as a flat implied volatility estimate for all strikes and expirations.
- **Synthetic IV model**: `generate_synthetic_iv()` in `quantcore/alpha/delta_neutral.py`
  blends realized vol with a 15% VRP premium and applies mean-reversion smoothing.
- **BSM pricing**: All option pricing uses Black-Scholes with the VIX-derived IV.
  No actual bid/ask spreads, no real Greeks, no real contract selection.
- **Alpaca paper broker**: Orders are submitted as equity trades (delta hedge shares),
  not as actual option contracts.

### Limitations of the VIX Proxy

| Dimension | VIX Proxy | Real Chain Data |
|---|---|---|
| **Skew** | None — flat IV across all strikes | Full skew surface (OTM puts richer) |
| **Term structure** | None — single 30-day IV | Multiple expirations (weekly, monthly) |
| **Contract pricing** | BSM theoretical (no bid/ask) | Real bid/ask with liquidity filters |
| **Greeks** | BSM-computed (delta, gamma, theta, vega) | Market-observable Greeks |
| **Fill simulation** | Assumed mid-price ± slippage | Actual bid/ask execution |
| **DSR p-value** | ~0.10 (Delta Neutral), ~0.15 (Iron Condor) | Expected < 0.05 with real data |

### Why This Matters
The Deflated Sharpe Ratio (DSR) test is sensitive to the null distribution of
Sharpe ratios under multiple testing.  Synthetic IV smoothing inflates the null
distribution because it removes the noise that distinguishes a lucky strategy
from a genuinely profitable one.  Real data introduces realistic noise — wider
bid/ask on illiquid strikes, gap risk, and execution slippage — which *tightens*
the null distribution and makes it easier to reject the null hypothesis.

**Current DSR p-values (synthetic IV)**:
- Delta Neutral: 0.00 (PASS — but this is with VIX proxy, which is unusually clean)
- Iron Condor: 0.83 (FAIL — too few trades, synthetic data inflates noise)

**Expected DSR p-values (real data)**:
- Delta Neutral: < 0.05 (likely still passes)
- Iron Condor: 0.10–0.30 (may pass with more trades)

---

## Target State

### Real Options Chains from Polygon.io

**Plan**: Polygon.io Options Starter ($29/month)

**What it provides**:
- Full US options chain snapshots (all strikes, all expirations)
- Real-time bid/ask quotes
- Greeks (delta, gamma, theta, vega, rho)
- Implied volatility per contract
- Volume and open interest
- Historical data back to 2003 (paid tiers)

**API limits** (Starter plan):
- 5 requests per minute (free tier)
- 100 requests per minute (Starter)
- Unlimited historical snapshots (Starter)

### What We Gain
1. **Skew signals**: Measure actual put skew (OTM put IV − ATM IV) and use it
   as a timing signal for entry/exit.
2. **Term structure**: Trade different expirations to capture term premium.
3. **Realistic fills**: Use actual bid/ask for entry credit and exit cost.
4. **Contract selection**: Filter by volume, open interest, and bid/ask spread
   to avoid illiquid strikes.
5. **Portfolio margin simulation**: Use real Greeks for margin estimation.

---

## Integration Steps

### Step 1: Set Up Polygon API Key

```bash
# In .env (never commit)
POLYGON_API_KEY=your_polygon_api_key_here
```

Verify connectivity:
```python
from quantcore.data.options_loader import PolygonOptionsDataLoader
loader = PolygonOptionsDataLoader()
chain = loader.download("SPY", "2024-01-01", "2024-01-05")
print(f"Loaded {len(chain)} contracts")
```

### Step 2: Download Historical Options Data

Download SPY options chains for the validation period (2022–2024):

```python
from quantcore.data.options_loader import PolygonOptionsDataLoader

loader = PolygonOptionsDataLoader(rate_limit_sleep=0.6)  # 100 req/min

# Download 3 years of SPY chains
# ~750 trading days × 1 request each = ~750 requests
# At 100 req/min: ~8 minutes
df = loader.download("SPY", "2022-01-01", "2024-12-31")

# Verify
print(f"Contracts: {len(df)}")
print(f"Dates: {df['date'].nunique()}")
print(f"Strikes: {df['strike'].nunique()}")
print(f"Expirations: {df['expiration'].nunique()}")
```

Storage: `data/options/SPY/{date}.parquet` (snappy compressed, ~50KB per day).

### Step 3: Re-run Validations with Real Data

Update the backtest engines to accept an `OptionsDataLoader`:

```python
# In quantcore/alpha/delta_neutral.py
strategy = DeltaNeutralVRP(
    close=spy_close,
    options_loader=loader,  # NEW: real data
    underlying="SPY",
)
equity, returns, turnover, logs = strategy.run_returns_backtest(
    start=pd.Timestamp("2022-01-01"),
    end=pd.Timestamp("2024-12-31"),
    training_start=pd.Timestamp("2020-01-01"),
)
```

The `DeltaNeutralVRP` class already supports `options_loader` — it falls back
to BSM when the loader returns no data.  The `IronCondorVRP` class has the same
interface.

### Step 4: Add Skew and Term Structure Signals

New signal generators for the VRP strategies:

```python
def compute_put_skew(loader, underlying, trade_date, otm_delta=0.10):
    """Measure put skew: OTM put IV - ATM IV."""
    atm_iv = loader.get_atm_iv(underlying, trade_date)
    otm_option = loader.get_option_by_delta(
        underlying, trade_date, otm_delta, "put"
    )
    if atm_iv is None or otm_option is None:
        return None
    return float(otm_option["implied_volatility"]) - atm_iv

def compute_term_structure(loader, underlying, trade_date):
    """Measure term structure: 60-day IV / 30-day IV."""
    iv_30 = loader.get_atm_iv(underlying, trade_date, days_to_expiry=30)
    iv_60 = loader.get_atm_iv(underlying, trade_date, days_to_expiry=60)
    if iv_30 is None or iv_60 is None:
        return None
    return iv_60 / iv_30
```

### Step 5: Update Live Strategy for Real Contracts

Once validated, update the `options_delta_neutral` strategy to submit actual
option orders through the broker (requires Alpaca options trading enabled):

```python
# In production_runner.py — future enhancement
if options_data_available:
    # Submit actual option order
    order = Order(
        symbol=selected_contract["option_symbol"],
        side=Side.SELL,  # short put
        type=OrderType.LIMIT,
        limit_price=Decimal(str(bid_price)),
        quantity=Decimal(str(contracts)),
    )
```

---

## Fallback: IBKR Options Data

If Polygon is unavailable, use Interactive Brokers' options data feed via the
existing IBKR gateway.

### Configuration

```yaml
# config/live.yaml
broker:
  type: ibkr
  ibkr:
    host: 127.0.0.1
    port: 7497      # TWS paper
    client_id: 1
    readonly: false
```

### IBKR Options Data API

```python
from quantcore.live.ibkr_gateway import IBKRExecutionGateway

gw = IBKRExecutionGateway(host="127.0.0.1", port=7497, client_id=1)

# Request options chain
chain = gw.get_options_chain("SPY", "2024-01-19")  # expiry date

# Request historical options data
hist = gw.get_options_historical(
    "SPY", "2024-01-19", "2024-01-15", "2024-01-19"
)
```

### IBKR Limitations
- Requires TWS or IB Gateway running locally (not cloud-hosted).
- Rate limits: ~50 requests/second for market data.
- Historical options data: limited to 1 year for most contracts.
- Paper trading mode: simulated fills, no real market data for options.

### Recommendation
Polygon is the better choice for backtesting (unlimited historical snapshots,
REST API, no local software).  IBKR is better for live trading (real-time
streaming, actual order execution for options contracts).

---

## Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| **Phase 1** | 1 day | Set up Polygon API key, download SPY chains 2022–2024 |
| **Phase 2** | 2 days | Re-run Delta Neutral and Iron Condor validations with real data |
| **Phase 3** | 3 days | Add skew and term structure signals to VRP strategies |
| **Phase 4** | 1 week | Walk-forward validate Put Write strategy with real data |
| **Phase 5** | 2 weeks | Enable options order submission in live paper trading |

**Total estimated time**: 2–3 weeks from API key setup to live options paper trading.

---

## Cost-Benefit Analysis

| Item | Cost | Benefit |
|---|---|---|
| Polygon Starter | $29/month | Real options data, skew signals, DSR improvement |
| IBKR data | Free (with account) | Real-time streaming, actual fills |
| Development time | ~2 weeks | Production-grade VRP system |
| Risk reduction | — | Realistic fill simulation eliminates synthetic data bias |

**Break-even**: If the improved DSR allows deploying an additional $100K at 15% annual
return (vs. 0% while waiting for data), the $29/month cost is recovered in < 1 day.

---

## Related Documents

- `reports/vrp_real_data_validation.json` — Current VIX-proxy validation results
- `reports/delta_neutral_realistic_backtest.json` — Walk-forward backtest results
- `quantcore/data/options_loader.py` — Alpaca and Polygon data loaders (already implemented)
- `quantcore/alpha/delta_neutral.py` — Delta Neutral VRP with `options_loader` support
- `quantcore/alpha/iron_condor.py` — Iron Condor VRP with `options_loader` support
- `notebooks/vrp_strategy_research.ipynb` — Put Write and Skew Trading research
