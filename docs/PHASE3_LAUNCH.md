# Phase 3 — Live Paper Trading Deployment Guide

> **DEPLOY strategies**: `macro_cross_sectional_momentum` (OOS Sharpe +0.85) and `macro_state_selective_momentum` (OOS Sharpe +0.81). Both passed CI_lower ≥ 0.3 on 10-year walk-forward validation.

---

## Overview

Phase 3 routes **paper orders** to your Alpaca paper broker, executing the same signal logic validated in Phase 2. No real money is at risk. The goal is live confirmation that the backtest signal pipeline is identical to production.

```
┌──────────────────────────────────────────────────────────────────────┐
│  LiveDataFeed (Alpaca IEX REST)  ──→  strategy.on_data()           │
│                                        │                            │
│                                    TradeSignal[]                    │
│                                        ↓                            │
│  LiveExecutionGateway  ◄──  Order (side, qty, notional)          │
│        │                                                         │
│  Alpaca Paper Broker (orders submitted — no real fills)           │
└──────────────────────────────────────────────────────────────────────┘
```

**One code path guarantee**: `strategy.on_data(data)` is called identically in backtest and live. No separate signal computation, no if-live conditions, no mock data in the strategy.

---

## Prerequisites

### 1. Alpaca Paper Account

1. Sign up at [alpaca.markets](https://app.alpaca.markets) (free, no credit card required)
2. Log in and go to **Paper Trading → Account**
3. Copy your **API Key** and **API Secret**

> Paper accounts reset daily at 4 PM ET. Your paper equity starts fresh each day.
> Paper fills are simulated at the next-bar open price with 10 bp slippage modelled.

### 2. Environment Variables

```bash
export ALPACA_API_KEY=PKXXXXXXXXXXXXXXXX
export ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export ALPACA_BASE_URL=https://paper-api.alpaca.markets   # default for paper
```

Or add them to `.env` (not committed to git):

```
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXX
ALPACA_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## Installation

```bash
# Install schedule dependency (required)
pip install schedule

# Verify you can import everything
python -c "from quantcore.live.orchestrator import make_paper_trader; print('OK')"
```

---

## Quick Start

```bash
# 1. Verify connectivity (no trading starts)
python run_paper_trading.py --check

# 2. Run cross_sectional_momentum (default DEPLOY strategy)
python run_paper_trading.py

# 3. Run state_selective_momentum
python run_paper_trading.py --strategy macro_state_selective_momentum

# 4. Custom starting cash
python run_paper_trading.py --cash 250000

# 5. Custom asset universe
python run_paper_trading.py --tickers SPY,QQQ,TLT,GLD
```

---

## Output Files

| File | What it contains |
|---|---|
| `reports/live_equity.csv` | Daily equity curve: date, cash, positions_value, total_equity, return |
| `reports/live_events.jsonl` | Every order event as newline JSON: submit / fill / close per order |

### Equity CSV format

```csv
date, cash, positions_value, total_equity, ret
2024-06-17,98923.50,1234.00,100157.50,0.00158
2024-06-18,98721.00,1876.00,100597.00,0.00439
```

### Event JSONL format

```json
{"timestamp":"2024-06-17T20:15:01Z","event":"submit","symbol":"SPYL","side":"buy","qty":14,"price":451.23,"notional":6321.00,"order_id":"A1B2C3D4","strategy":"CrossSectionalMomentumStrategy","trade_date":"2024-06-17","strength":0.085,"signal_size":8421.00}
{"timestamp":"2024-06-17T20:15:01Z","event":"close","symbol":"TMF","qty":12,"order_id":"E5F6G7H8","strategy":"CrossSectionalMomentumStrategy"}
```

---

## What the Trader Does Each Day

At **4:15 PM ET** (30 min after market close):

1. `get_daily_snapshot()` — fetches 252-day price matrix from Alpaca IEX REST for all assets
2. `strategy.on_data(data)` — same call used in walk-forward backtest → `list[TradeSignal]`
3. For each signal: compute order quantity from `signal.size` × portfolio equity, capped to available cash
4. `broker.submit_order()` — routes to Alpaca Paper Broker (simulated fills at next-bar open, 10 bp slippage)
5. Appends event JSON to `live_events.jsonl`
6. Appends equity snapshot to `live_equity.csv`

At **4:30 PM ET**:
- Reconciliation pass: syncs broker positions, logs cash/equity

---

## Monitoring

### Real-time equity

```bash
tail -f reports/live_equity.csv
```

### Watch for order events

```bash
tail -f reports/live_events.jsonl
```

### Count today's orders

```bash
grep "2024-06-17" reports/live_events.jsonl | wc -l
```

### Check for rejections

```bash
grep rejected reports/live_events.jsonl
```

---

## Strategy Parameters

### `macro_cross_sectional_momentum`

- **Signal**: cross-sectional momentum ranking (weighted z-score of 20/60/120-day returns)
- **Rebalance**: every 21 calendar days (configurable via `rebalance_freq`)
- **Risk controls**: vol targeting (12% target), regime filter, drawdown scale, beta-neutral, gross exposure cap
- **Position limits**: 10% max per asset, 100% max gross

### `macro_state_selective_momentum`

- **Signal**: instability-adjusted momentum (short 21d / medium 63d)
- **Rebalance**: every 5 calendar days
- **State machine**: STABLE → medium-term signal / TRANSITION → short-term / NOISE → flat
- **Position limits**: vol-scaled, min 5% threshold to emit signal

---

## Stopping Gracefully

```bash
# Ctrl+C — the trader will:
# 1. Log final equity snapshot
# 2. Cancel all open orders (via Alpaca API)
# 3. Write final event log
# 4. Exit cleanly
```

Or from another terminal:
```bash
kill $(lsof -ti:8000)   # if running as background service
```

---

## Troubleshooting

### "ALPACA_API_KEY and ALPACA_API_SECRET must be set"
```bash
# Verify env vars are set:
echo $ALPACA_API_KEY
echo $ALPACA_API_SECRET

# Reload if editing .bashrc:
source ~/.bashrc
```

### "Alpaca connectivity check failed"
- Verify you're using paper credentials (not live)
- Check `ALPACA_BASE_URL=https://paper-api.alpaca.markets`
- Log into alpaca.markets and verify the paper account status (sometimes needs re-activation)

### "No price data for 2024-XX-XX"
- The data feed needs market to be open on that date. Weekends/holidays return no bars
- Check `reports/live_equity.csv` shows data only for trading days

### "Insufficient cash" rejections in equity log
- Normal when the strategy emits short signals (no cash required for short sells in paper)
- In long-only scenarios, the signal size might exceed 95% available cash (the 5% buffer)

---

## Phase Gate Summary

| Gate | Criterion | Status |
|---|---|---|
| Phase A OOS | CI_lower(90%) ≥ 0.3, 10-yr expanding walk-forward | **PASS** (2 strategies) |
| Phase B Code Quality | All signals match backtest path, no separate live pipeline | **PASS** (same `on_data()` call) |
| Phase C Parity Test | LiveDataFeed vs yfinance: identical signals for same date | `pytest tests/test_live_parity.py` |
| Phase D Live Bootstrap | Run paper trading for ≥ 20 trading days without crash | **PENDING** |
| Phase E Live Sign-off | OOS paper Sharpe ≥ 0.3 over live period | **PENDING** |

---

## Exit Criteria

The Phase 3 live run can be stopped when **any** of:
1. Live OOS Sharpe > 0.0 for 60+ trading days (confirms signal quality)
2. Drawdown exceeds −10% peak equity (strategy risk controls may need review)
3. More than 5%% of orders rejected (systematic signal sizing error)
4. 90-day live run complete (standard validation window)

---

*Generated 2026-05-21 — Phase 3 launch*
*Strategies validated in walk-forward: `macro_cross_sectional_momentum` (Sharpe +0.85, CI_lower +0.33), `macro_state_selective_momentum` (Sharpe +0.81, CI_lower +0.40)*