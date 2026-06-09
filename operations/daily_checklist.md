# QuantCore Daily Operations Checklist

> 30-day paper trading validation run. Started: 2026-05-27.
> Target: validate 3 strategies in live market conditions before graduated live deployment.

## Monitor Script

Run **after** each market close (16:30 ET) and **before** reviewing anything else:

```powershell
python scripts/monitor_paper_trading.py
```

### Reading the Output

The monitor prints three sections:

#### 1. Per-strategy status lines

```
  options_delta_neutral: eq=$35,196.56 | orders=3 | fills=0 | last_pnl=+$0.00 | status=running
  options_put_write: eq=$57,994.33 | orders=0 | fills=0 | last_pnl=+$0.00 | status=running
  options_iron_condor: eq=$6,799.34 | orders=0 | fills=0 | last_pnl=+$0.00 | status=running
```

Each shows: equity allocation, order/fill count from the event log, latest daily P&L, and strategy status. If `bt_target_sharpe` is shown, a matching backtest file was loaded.

#### 2. Portfolio summary

The header status is one of:

| Status | Meaning | Action |
|--------|---------|--------|
| **INSUFFICIENT DATA** | < 5 trading days of data | **Log it and move on** — no action needed. Metrics are not meaningful yet. |
| **OK** | All metrics within backtest-expected ranges | Consider the run healthy. |
| **WARNING** | Some metrics below threshold (e.g. Sharpe < 0.3) | Investigate. Check if this is a trend or a single bad day. |
| **CRITICAL** | Metrics breached critical limits (loss > 2%, DD > 25%) | Stop and investigate immediately. |

#### 3. Fill-event summary

```
  Fills: 0 recent / 0 total
  WARNING: No fill events detected at all. Orders may not be executing.
```

**If this warning appears, investigate immediately.** It means orders are being logged but never filled by the broker. Common causes:
- Alpaca paper API base URL incorrect (should be `https://paper-api.alpaca.markets`)
- Risk manager rejecting orders before they reach Alpaca
- Orders submitted outside market hours (check timezone settings)
- Alpaca API credentials missing or expired

**If the output shows no `fills=` count for any strategy**, see "No Fills" section below.

## Morning Check (09:30 ET)

- [ ] Docker Desktop running: `docker ps` — container `quantcore-live` should be `Up`, health `healthy`
- [ ] Health endpoint: `curl http://localhost:8000/health` — status should be `ok`
- [ ] No KILL_SWITCH file present: `ls KILL_SWITCH` — should not exist
- [ ] No overnight alerts: `tail -5 reports/alerts.log`
- [ ] Market calendar check: verify today is a trading day
- [ ] Telegram test alert (optional): `curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe"`

## Post-Market Check (16:30 ET)

- [ ] Run monitor: `python scripts/monitor_paper_trading.py`
- [ ] If **INSUFFICIENT DATA**: log it, no action needed
- [ ] If **WARNING: No fill events**: investigate immediately (see "No Fills" below)
- [ ] If **CRITICAL**: stop all trading, investigate root cause
- [ ] If **OK** for all strategies: consider the run healthy ✓
- [ ] Verify per-strategy equity: `ls reports/live_equity_*.csv` — one file per active strategy
- [ ] No circuit breaker: check alerts.log for `CIRCUIT BREAKER`
- [ ] No risk rejections: check alerts.log for `RISK REJECT`
- [ ] Positions reasonable: `curl http://localhost:8000/status` — check positions count
- [ ] Update monitoring log review: `tail -5 reports/monitoring_log.csv`

## Weekly Review (Friday 17:00 ET)

- [ ] Cumulative P&L: review `reports/live_equity.csv` — total return since start
- [ ] Per-strategy attribution: review each `reports/live_equity_{name}.csv` — which strategy contributed most/least
- [ ] Fill rate check: how many submit→fill conversions occurred this week
- [ ] Execution costs: review TCA from event log — slippage within 5bps target
- [ ] Alert review: `cat reports/alerts.log` — any recurring warnings
- [ ] Backtest target freshness: check `reports/*_backtest.json` file modification dates
- [ ] Data quality: check for missing bars or stale data in logs

## Known Issues & Anomalies

### No Fills
If orders are submitted (visible in `reports/live_events.jsonl`) but never filled:

1. Check Alpaca connection directly:
   ```powershell
   curl -s -H "APCA-API-KEY-ID: $env:ALPACA_API_KEY" -H "APCA-API-SECRET-KEY: $env:ALPACA_API_SECRET" https://paper-api.alpaca.markets/v2/account
   ```
2. Check API logs inside container:
   ```powershell
   docker logs quantcore-live --tail 100 | grep -i "fill\|order\|error\|reject"
   ```
3. Verify `APCA_API_BASE_URL` is set to `https://paper-api.alpaca.markets` (not live)
4. Verify trading hours — US equities market hours are 09:30–16:00 ET
5. Check risk manager circuit breaker status in `reports/alerts.log`
6. Ensure `.env` file has valid credentials (keys starting with `PK` = paper)

### Circuit Breaker False Positives
The circuit breaker may fire with a stale equity value (e.g. $97,000 when actual is $99,990). Check the actual equity curve before assuming a real breach:
```powershell
tail -3 reports/live_equity.csv
```

### Strategy Errors
| Strategy | Known Error | Status |
|----------|------------|--------|
| `options_put_write` | "only 0-dimensional arrays can be converted to Python scalars" | Needs fix |
| `options_iron_condor` | Intermittent errors (3/5 threshold) | Needs investigation |

## Strategy Mapping

| Config Name | Weight | Data Source | Backtest File |
|-------------|--------|-------------|---------------|
| `options_delta_neutral` | 0.352 | yfinance_vix | `reports/delta_neutral_realistic_backtest.json` |
| `options_put_write` | 0.580 | yfinance_vix | Not found |
| `options_iron_condor` | 0.068 | yfinance_vix | Not found |

## Incident Response

### Kill Switch Activation
1. `curl -X POST http://localhost:8000/kill-switch/activate`
2. Investigate root cause in logs
3. Fix issue, then `curl -X POST http://localhost:8000/kill-switch/deactivate`

### Container Down
1. `docker compose logs --tail=50 quantcore` — check crash reason
2. `docker compose up -d` — restart
3. Verify health: `curl http://localhost:8000/health`

### Circuit Breaker Triggered
1. Check actual equity: `tail -1 reports/live_equity.csv` — verify against the reported $97,000
2. If legitimate loss, wait for next trading day (auto-resets)
3. If false alarm, check equity feed and risk config

### No Fill Events (Critical)
1. Check broker connectivity: `docker logs quantcore-live --tail 100 | grep -i "fill\|reject\|error"`
2. Verify `APCA_API_BASE_URL` environment variable
3. Check market hours — options strategies only trade during regular hours
4. If risk manager is blocking, check `circuit_breaker` status in alerts.log

## Graduation Criteria (After 30 Days)

| Metric | Threshold | Current |
|--------|-----------|---------|
| Paper Sharpe (annualized) | >= 0.3 | TBD (insufficient data) |
| Max drawdown | < 10% | TBD (insufficient data) |
| Days without crash | >= 20/30 | TBD |
| Circuit breaker trips | <= 3 | 0 (false positives excluded) |
| Fill rate | >= 95% | TBD (0 fills to date — investigate) |

## Key Files

| File | Purpose |
|------|---------|
| `config/live.yaml` | Strategy and broker configuration |
| `reports/live_equity.csv` | Portfolio equity curve |
| `reports/live_equity_{name}.csv` | Per-strategy equity curves |
| `reports/live_events.jsonl` | Order submission & fill event log |
| `reports/alerts.log` | Alert history |
| `reports/monitoring_log.csv` | Monitor run history |
| `reports/*_backtest.json` | Backtest targets for comparison |
| `KILL_SWITCH` | Emergency halt file (delete to resume) |