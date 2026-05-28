# Phase 4 — Production Risk Controls & Graduated Live Trading

> **⚠️ REAL MONEY AT RISK — Only proceed after Phase 3 paper validation.**

---

## Overview

Phase 4 introduces **production risk controls**, **circuit breakers**, and **graduated live-trading deployment** for the two DEPLOY strategies:

- `macro_cross_sectional_momentum` (OOS Sharpe +0.85)
- `macro_state_selective_momentum` (OOS Sharpe +0.81)

Both strategies passed OOS validation with `CI_lower(90%) ≥ 0.3` on a 10-year walk-forward.

```
┌──────────────────────────────────────────────────────────────────────┐
│  LiveDataFeed  ──→  strategy.on_data()  ──→  Risk Gate            │
│                                                   │                 │
│                                           Circuit Breaker         │
│                                                   │                 │
│  IBKRExecutionGateway  ◄──  Order (scaled by mode)              │
│        │                                                            │
│  Interactive Brokers (TWS / IB Gateway)                            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## New Components

| File | Purpose |
|------|---------|
| `quantcore/live/ibkr_gateway.py` | IBKR execution gateway — sync facade over IBKRBroker implementing the Broker protocol. |
| `quantcore/live/risk_manager.py` | Pre-trade checks (position, leverage, daily loss, drawdown, single order size). Env-configurable limits. |
| `quantcore/live/circuit_breaker.py` | Graduated halt states: `NORMAL` → `DAILY_LOSS_HALT` → `CRITICAL_HALT`. |
| `quantcore/live/broker.py` + `cancel_all_orders()`, `health_check()` | Emergency order cancellation and broker health monitoring (Alpaca). |
| `quantcore/live/orchestrator.py` | Integrated kill switch, risk gate, circuit breaker, scale modes. |
| `scripts/paper_trade.py` | `--live`, `--scale`, `--broker ibkr` flags with guard rails. |
| `scripts/test_ibkr_connection.py` | Manual IBKR connection smoketest. |

---

## Risk Limits (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `QUANTCORE_RISK_MAX_POSITION_PCT` | 0.05 | Max % equity per ticker |
| `QUANTCORE_RISK_MAX_GROSS_EXPOSURE` | 2.0 | Max gross exposure (200%) |
| `QUANTCORE_RISK_MAX_DAILY_LOSS_PCT` | 0.02 | Max daily loss (2%) |
| `QUANTCORE_RISK_MAX_DRAWDOWN_PCT` | 0.10 | Max drawdown (10%) |
| `QUANTCORE_RISK_MAX_SINGLE_ORDER_PCT` | 0.05 | Max % equity per single order |
| `QUANTCORE_RISK_MAX_LEVERAGE` | 2.0 | Max gross / equity ratio |
| `QUANTCORE_RISK_CRITICAL_DRAWDOWN_PCT` | 0.10 | Circuit breaker critical drawdown |
| `QUANTCORE_RISK_CONSECUTIVE_REJECTIONS` | 5 | Consecutive rejections → halt |

Set in your shell or `.env` before launching.

---

## IBKR Broker Configuration

| Variable | Default | Description |
|---|---|---|
| `IBKR_HOST` | `127.0.0.1` | TWS/IB Gateway host address |
| `IBKR_PORT` | `7497` | Connection port (see below) |
| `IBKR_CLIENT_ID` | `1` | Client ID for the API connection |

### Connection Ports

| Port | Mode | Use Case |
|---|---|---|
| `7497` | TWS Paper Trading | Paper trading via TWS (default) |
| `7496` | TWS Live Trading | Live trading via TWS |
| `4002` | IB Gateway Paper | Paper trading via IB Gateway (headless) |
| `4001` | IB Gateway Live | Live trading via IB Gateway (headless) |

### Pre-flight Checklist

1. TWS or IB Gateway is running and API connections are enabled
2. In TWS: File → Global Configuration → API → Settings → "Enable ActiveX and Socket Clients" ✓
3. Port matches your `IBKR_PORT` environment variable
4. Run the connection smoketest:

```bash
python scripts/test_ibkr_connection.py
```

---

## Graduated Deployment

Phase 4 introduces **three scale modes** for graduated exposure to live markets:

| Mode | Notional | Use Case |
|---|---|---|
| `single_share` | 1 share per signal | Verify connectivity, latency, fill mechanics |
| `fractional` | 10% of full notional | Gradual ramp with real capital at reduced exposure |
| `full` | 100% of signal notional | Full production deployment |

### Paper (default, safe)

```bash
python scripts/paper_trade.py --broker mock --scale single_share
python scripts/paper_trade.py --broker mock --scale fractional
python scripts/paper_trade.py --broker mock --scale full           # paper fills, no real money
```

### Live — Interactive Brokers (REAL MONEY)

```bash
# Step 1: Verify IBKR connectivity
python scripts/test_ibkr_connection.py

# Step 2: Single-share live validation
python scripts/paper_trade.py --live --broker ibkr --scale single_share --framework macro_cross_sectional_momentum

# Step 3: Graduated to fractional
python scripts/paper_trade.py --live --broker ibkr --scale fractional --framework macro_cross_sectional_momentum

# Parallel launcher (both DEPLOY strategies)
python scripts/launch_paper_trading.py --live --broker ibkr --scale single_share
```

> **⚠️ `--live --scale full` is PROHIBITED** — must pass fractional validation first.

---

## Circuit Breaker States

### State Machine

```
NORMAL
  │
  ├── daily loss > 2%  →  DAILY_LOSS_HALT  (auto-resets next day)
  │
  ├── 5+ consecutive rejections  →  DAILY_LOSS_HALT
  │
  └── drawdown > 10%  →  CRITICAL_HALT  (requires manual reset)
```

### Operator Actions

| Action | Command / Procedure |
|---|---|
| Check state | Read logs — `grep "CIRCUIT BREAKER"` |
| Reset daily halt | Automatic at next calendar day |
| Reset critical halt | Call `circuit_breaker.reset()` or restart |
| Emergency stop | `touch KILL_SWITCH` |

---

## Kill Switch

Create a file named `KILL_SWITCH` in the project root to immediately halt all trading.

```bash
touch KILL_SWITCH   # halt
touch KILL_SWITCH   # remove to allow resuming
```

The orchestrator checks for this file **before every trading tick**.

---

## Broker Health Check

```python
from quantcore.live.ibkr_gateway import IBKRExecutionGateway

gw = IBKRExecutionGateway()
gw.connect()
health = gw.health_check()
# Returns: {'connected': True, 'paper': True, 'host': '127.0.0.1',
#           'port': 7497, 'cash': 100000.0, 'position_count': 2,
#           'healthy': True, ...}
```

---

## Monitoring

### Watch equity in real-time

```bash
tail -f reports/live_equity.csv
```

### Watch for circuit breaker events

```bash
tail -f reports/live_events.jsonl | grep -i "circuit\|kill\|risk reject"
```

### Broker health

```bash
python -c "from quantcore.live.ibkr_gateway import IBKRExecutionGateway; gw = IBKRExecutionGateway(); gw.connect(); print(gw.health_check())"
```

---

## Tests

```bash
# Run all Phase 4 risk tests
pytest tests/test_live_risk.py -v

# Specific test classes
pytest tests/test_live_risk.py::TestRiskManagerLive -v
pytest tests/test_live_risk.py::TestCircuitBreaker -v
pytest tests/test_live_risk.py::TestKillSwitch -v
```

---

## Exit Criteria

Phase 4 is considered **validated** when:

1. All `test_live_risk.py` tests pass ✅
2. Paper trading with `full` scale runs ≥ 20 trading days without crash
3. Paper Sharpe over the period ≥ 0.3
4. Live `single_share` mode runs ≥ 5 days with < 1% slippage variance
5. Live `fractional` mode runs ≥ 20 days with positive P&L
6. No `CRITICAL_HALT` triggered during paper or live runs
7. Kill switch tested and responded within 1 tick

---

*Generated 2026-05-21 — Phase 4 launch*
*Strategies: macro_cross_sectional_momentum (Sharpe +0.85), macro_state_selective_momentum (Sharpe +0.81)*
