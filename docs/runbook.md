# QuantCore Operational Runbook

This document covers emergency and routine procedures. If you were paged, start at the
[Incident Response Checklist](#incident-response-checklist).

---

## Incident Response Checklist

1. Check the dashboard: `http://<host>:8000/` (or wherever the API is running)
2. Hit `/health`: `curl http://<host>:8000/health | jq`
3. Check `kill_switch_active` in the response — if `true`, the system is already halted
4. Check `last_data_timestamp` — if null or >30s old, the feed is stale
5. Check `open_positions_count` — if non-zero after a kill switch, verify the broker directly
6. Check `logs/` for the most recent paper/live log file
7. If the system is in an unrecoverable state, trigger the kill switch manually (see below)
8. If you cannot diagnose the issue within 15 minutes, escalate

---

## System Startup — Paper Mode

Paper trading simulates fills and uses a mock broker. It does not connect to a real
exchange and does not risk real capital.

### Start

```bash
python main.py live --broker paper
```

Or use the dedicated paper trading script (with duration control):

```bash
python scripts/paper_trade.py --duration 5d --strategy momentum --data-source yahoo
```

### Required Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `QUANTCORE_ENV` | No | `development` | Config environment (`development`, `production`) |
| `ALPACA_API_KEY` | No | — | Alpaca data (paper mode can use yahoo instead) |
| `ALPACA_SECRET_KEY` | No | — | Alpaca secret (paper mode can use yahoo instead) |
| `REDIS_URL` | No | `redis://localhost:6379` | Redis for caching/pub-sub |
| `DATABASE_URL` | No | `postgresql://localhost:5432/quantcore` | TimescaleDB |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity |

### Verify Startup

- Confirm `[QuantCore API] Starting up...` in console output
- Hit `http://localhost:8000/health` — expect `{"status": "ok"}`
- Hit `http://localhost:8000/docs` for the OpenAPI docs
- Check `logs/paper_trade_*.log` for data ingestion messages
- Wait 2-3 minutes, then check `/health` again — `last_data_timestamp` should be recent

---

## System Startup — Live Mode

Live mode connects to a real broker and trades real capital. **Double-check everything
before going live.**

### Start

```bash
python main.py live --broker alpaca
```

### Additional Required Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `ALPACA_API_KEY` | **Yes** | Alpaca live trading API key (not paper key) |
| `ALPACA_SECRET_KEY` | **Yes** | Alpaca live secret key |
| `ALPACA_LIVE_TRADING` | **Yes** | Set to `true` to confirm live trading intent |
| `QUANTCORE_ENV` | **Yes** | Set to `production` |

### Pre-Live Confirmation Checklist

- [ ] Capital: account value confirmed in broker portal, matches config
- [ ] Risk limits: `config/production.toml` reviewed — max concentration, max daily loss, max leverage
- [ ] Kill switch: tested in paper mode first, confirmed <1ms response
- [ ] Drawdown thresholds: -5% alert, -10% halt, -15% flatten — verified
- [ ] Data feed: 5 minutes of live data received, no gaps
- [ ] Audit trail: SHA-256 chain verified with empty state
- [ ] Alerts: Slack/email notification channels tested
- [ ] Dry run: at least 1 hour of paper trading with the same config, zero errors

### Verify Live Connection

- `curl http://localhost:8000/health` — check `open_positions_count` matches broker
- Check broker portal confirms the API client is connected
- Watch logs for `State: pre_market → trading`

---

## Kill Switch

The kill switch halts all trading activity immediately: cancels all open orders,
liquidates all positions, and transitions the system to EMERGENCY state.

### How to Trigger

**From the running process:**
- Press `Ctrl+C` (SIGINT) — the system catches this, triggers kill switch, then shuts down
- Send SIGTERM: `kill -TERM <pid>` — same graceful emergency shutdown

**From the API (if the API server is running separately):**
- There is no dedicated kill-switch endpoint yet. SIGINT/SIGTERM on the trader process
  is the primary mechanism.

**From the code (live loop):**
- The kill switch auto-triggers when:
  - Daily loss exceeds `max_daily_loss_pct` (default: 3%)
  - Drawdown exceeds the configured threshold (default: -15%)
  - Circuit breaker activates from post-trade risk checks

### What Happens

1. `kill_switch_active` is set to `true`
2. All pending orders in the OMS are cancelled
3. All open positions are liquidated via market orders
4. System state transitions to `EMERGENCY` / `HALTED`
5. The `/health` endpoint reports `kill_switch_active: true`
6. Critical alerts fire (dashboard, email, Slack)

### Verify Kill Switch Worked

- `curl http://localhost:8000/health` — `kill_switch_active` must be `true`
- `curl http://localhost:8000/health` — `open_positions_count` must be `0`
- Check broker portal — no open positions, no pending orders
- Check `logs/` — should see `[ALERT] Drawdown action: flatten` or emergency messages

### Re-Enabling After Kill Switch

- Kill switch is **latching** — the system does not auto-resume
- Restart the process: stop the current process, then start a new one
- Before restarting, verify in the broker portal that all positions are flat
- Verify the audit trail hash chain is intact
- Start in paper mode first if you want to test before going live again

---

## Stale Feed Response

### Detection

The system considers the feed stale when `data_stale_seconds > 30` (configurable in
alert rules, rule ID `data_feed_stale`). The `/health` endpoint reports `last_data_timestamp`
— if it's null or >30s old, the feed is stale.

### Automatic Response

- A CRITICAL alert fires (dashboard + Slack)
- The system **does not** automatically halt on stale feed (this is configurable)
- The system continues using the last known prices

### Operator Response

1. Check the data provider status:
   - Alpaca: https://status.alpaca.markets/
   - Polygon: https://status.polygon.io/
   - Yahoo Finance: verify manually at finance.yahoo.com
2. Check logs for connection errors: `grep -i "connection\|disconnect\|timeout" logs/paper_trade_*.log`
3. If the provider is up but the feed is down: restart the data feed handler
4. To restart just the feed without restarting the system:
   - The feed handler auto-reconnects with exponential backoff (up to 10 attempts)
   - If auto-reconnect fails, a full restart is required
5. Verify data is flowing: `curl http://localhost:8000/health | jq .last_data_timestamp`
6. If data is not flowing after 5 minutes, escalate

### Restart Procedure (Full System)

```bash
# Stop the running process
kill -TERM <pid>
# Wait for graceful shutdown (watch logs for "Shutting down")
# Verify positions are flat in broker
# Restart
python main.py live --broker <paper|alpaca>
```

---

## Order Rejected

### Common Reasons

| Reason | Cause | Action |
|---|---|---|
| Insufficient capital | Order size exceeds available cash | Reduce position size; check `sizing_pct` in config |
| Market closed | Trading outside market hours | Wait for market open; system handles this automatically |
| Risk limit breached | Pre-trade check failed (concentration, exposure, VaR) | Review risk limits in `config/`; the order was correctly blocked |
| Kill switch active | System in EMERGENCY state | All orders blocked until restart — see kill switch section |
| Symbol not found | Symbol not in universe or delisted | Verify symbol; check `data/sources/universe.py` |
| Broker rejection | Broker-specific reason (e.g., short-sale restriction) | Check broker logs; contact broker support |

### Finding the Rejection Reason

- In `logs/paper_trade_*.log`: grep for `REJECTED` or the order ID
- In the OMS audit trail: `oms.audit_trail` — filter by `to_state == "rejected"`
- In the risk logs: look for `LimitStatus.REJECTED` or circuit breaker activation

### When to Escalate

- Broker rejects that are not in the common reasons table above
- More than 10 rejected orders in 1 minute (possible misconfiguration)
- Rejected orders that were supposed to close a position (risk of loss)

---

## Restart Without Losing State

### Where State Lives

| State | Storage | Survives Restart? |
|---|---|---|
| Positions | Broker (source of truth) + cached in-memory | Yes (from broker) |
| Orders (filled) | Broker + OMS audit trail (in-memory) | Filled orders: yes (broker). In-flight: no |
| Orders (in-flight) | OMS (in-memory only) | **No** — cancelled on shutdown |
| Audit trail | Append-only log (in-memory) + hash chain | **No** — lost on restart (future: persist to DB) |
| Portfolio value | Broker (source of truth) | Yes |
| Kill switch state | In-memory | No — resets to `false` on restart |
| Health state | In-memory | No — resets on restart |

### Correct Restart Procedure

1. Send SIGTERM — triggers graceful shutdown: cancel open orders, sync positions
2. Wait for `[QuantCore API] Shutting down...` in logs
3. Verify in broker portal: no open orders, positions as expected
4. Verify audit trail hash chain integrity before shutdown
5. Restart: `python main.py live --broker <broker>`
6. System syncs positions from broker on startup (`_sync_portfolio()`)
7. Check `/health` — all fields populated
8. Verify audit trail chain is intact from the new session

### What Happens to In-Flight Orders

In-flight orders (submitted but not yet filled) are **cancelled** during shutdown.
After restart, the strategy re-evaluates market conditions and generates new signals.
No in-flight orders survive a restart.

### Crash Recovery

If the system crashes (unclean exit):
1. Check broker portal for actual positions and open orders
2. Cancel any stale orders directly in the broker portal
3. Verify positions match your expectations
4. Reconcile P&L from broker statements against the last saved metrics CSV
5. Start fresh — the system syncs from broker on startup
6. File an incident report if the crash occurred during live trading

---

## Emergency Contacts and Escalation

Fill in the table below before going live. Keep this section printed and accessible.

| Role | Name | Phone | Email | Notes |
|---|---|---|---|---|
| Lead Trader | | | | Primary decision-maker for manual intervention |
| DevOps | | | | Infrastructure, CI/CD, database, network |
| Broker Contact | | | | Broker emergency line |
| Data Provider | | | | Alpaca/Polygon support |
| Legal/Compliance | | | | Regulatory reporting, incident disclosure |

### Escalate If

- **The system is losing money faster than $1,000 per minute** (adjust threshold to your risk tolerance)
- **The audit trail hash chain is broken** — indicates data corruption or tampering
- **The kill switch does not respond within 5 seconds** — manual intervention required
- **Any error you cannot diagnose within 15 minutes** — do not guess in live trading
- **The broker reports positions that don't match the system's view** — possible sync bug
- **A critical alert (severity=CRITICAL) fires and does not auto-resolve within 5 minutes**

### After an Incident

1. Stop the system. Do not restart until root cause is understood.
2. Save all logs: `cp -r logs/ logs_incident_$(date +%Y%m%d_%H%M%S)/`
3. Run `scripts/benchmark.py` to verify system performance is unchanged
4. Run `pytest tests/ -v --tb=short` to verify no regressions
5. File a post-mortem: what happened, why, timeline, impact, remediation
6. Create a new git tag to mark the incident state
7. Apply fixes in paper mode first for at least 24 hours before going live again

---

## Quick Reference

```bash
# Health check
curl http://localhost:8000/health | jq

# Start paper trading (5-day dry run)
python scripts/paper_trade.py --duration 5d --strategy momentum --data-source yahoo

# Start live
python main.py live --broker alpaca

# Run benchmarks
python scripts/benchmark.py

# Run tests
pytest tests/ -v --tb=short -m "not integration"

# Run integration tests
pytest tests/integration/ -v --tb=short

# Check logs
tail -f logs/paper_trade_*.log

# Docker stack
docker-compose up -d
docker-compose logs -f
docker-compose down
```
