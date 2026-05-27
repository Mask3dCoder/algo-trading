# QuantCore Daily Operations Checklist

> 30-day paper trading validation run. Started: 2026-05-27.
> Target: validate 3 strategies in live market conditions before graduated live deployment.

## Morning Check (09:30 ET)

- [ ] Container running: `docker compose ps` — status should be `Up`, health `healthy`
- [ ] Health endpoint: `curl http://localhost:8000/health` — status should be `ok`
- [ ] No KILL_SWITCH file present: `ls KILL_SWITCH` — should not exist
- [ ] No overnight alerts: `tail -5 reports/alerts.log`
- [ ] Market calendar check: verify today is a trading day

## Post-Market Check (16:30 ET)

- [ ] Daily validation: `python scripts/daily_validation.py` — status should be OK
- [ ] Trading tick executed: `tail -20 reports/live_events.jsonl` — look for `submit` events today
- [ ] All 3 strategies ran: check logs for `macro_cross_sectional_momentum`, `macro_state_selective_momentum`, `equity_stat_arb`
- [ ] Equity logged: `tail -1 reports/live_equity.csv` — date should be today
- [ ] No circuit breaker: check logs for `Circuit breaker ACTIVE`
- [ ] No risk rejections: check logs for `RISK REJECT`
- [ ] Positions reasonable: `curl http://localhost:8000/status` — check positions count

## Weekly Review (Friday 17:00 ET)

- [ ] Cumulative P&L: review `reports/live_equity.csv` — total return since start
- [ ] Max drawdown: compute from equity curve — alert if > 5%
- [ ] Strategy attribution: which strategy contributed most/least to returns
- [ ] Execution costs: review TCA from event log — slippage within 5bps target
- [ ] Alert review: `cat reports/alerts.log` — any recurring warnings
- [ ] Data quality: check for missing bars or stale data in logs

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
1. Check daily loss: `tail -1 reports/live_equity.csv`
2. If legitimate loss, wait for next trading day (auto-resets)
3. If false alarm, check equity feed and risk config

## Graduation Criteria (After 30 Days)

| Metric | Threshold | Current |
|--------|-----------|---------|
| Paper Sharpe (annualized) | >= 0.3 | TBD |
| Max drawdown | < 10% | TBD |
| Days without crash | >= 20/30 | TBD |
| Circuit breaker trips | <= 3 | TBD |
| Fill rate | >= 95% | TBD |

## Strategy Mapping

| Config Name | Registry Name | Data Source | Weight |
|-------------|---------------|-------------|--------|
| `macro_cross_sectional_momentum` | `macro_cross_sectional_momentum` | Alpaca daily | 0.35 |
| `macro_state_selective_momentum` | `macro_state_selective_momentum` | Alpaca daily | 0.35 |
| `pairs_trading_intraday` | `equity_stat_arb` | Alpaca intraday (5min) | 0.30 |

## Key Files

| File | Purpose |
|------|---------|
| `config/live.yaml` | Strategy and broker configuration |
| `reports/live_equity.csv` | Daily equity curve |
| `reports/live_events.jsonl` | Order submission log |
| `reports/alerts.log` | Alert history |
| `KILL_SWITCH` | Emergency halt file (delete to resume) |
