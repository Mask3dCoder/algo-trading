# Phase 4 Validation Rollout Plan

Date: 2026-06-09  
System phase: Phase 4 — Production Risk Controls

## Executive summary

The system currently has **48 registered/importable strategies**. Strategy tests pass with **1671 passed, 13 skipped** and **185 strategy-related tests passed**. All **48 strategies** are covered by the walk-forward matrix after adding `vix_call_ladder`.

Only four strategies are live/paper-production enabled in `config/live.yaml`, and their weights remain unchanged and sum to **1.0**:

| Strategy | Live/paper status | Weight |
|---|---:|---:|
| `options_delta_neutral` | KEEP | 0.317 |
| `options_put_write` | MONITOR | 0.522 |
| `options_iron_condor` | MONITOR | 0.061 |
| `opening_gap_fade` | KEEP | 0.100 |

This Phase 4 change adds a **separate paper-validation config** for three non-live strategies. It does **not** alter `config/live.yaml` or live allocations.

## H001 OPRA flow deferral

`research/h001_opra_flow/report.md` records the H001 event-study pipeline as **DATA_BLOCKED**.

Blocked sources:

- Alpaca OPRA agreement not signed.
- Polygon API key invalid.
- IBKR gateway unreachable.

No H001 ingestion, event-study execution, or new H001 strategy code under `quantcore/` is allowed until a reliable OPRA data source is available.

## Candidate pool with existing config files

The following registered, walk-forward-tested strategies have config files and are not currently enabled in `config/live.yaml`:

| Strategy | Family | Expected return/risk profile | Rough correlation view |
|---|---|---|---|
| `equity_stat_arb` | Equity | Market-neutral pair/stat-arb returns; lower beta than directional equity strategies | Low correlation to short-vol options and intraday gap fade |
| `equity_momentum` | Equity | Directional cross-sectional momentum | Moderate equity-market exposure; can correlate with broad risk-on regimes |
| `equity_mean_reversion` | Equity | Short-horizon reversal/mean-reversion | Potentially diversifying vs momentum, but more path-dependent |
| `equity_trend_following` | Equity | Directional trend exposure | Can correlate with equity momentum in persistent trends |
| `macro_carry` | Macro | Carry premia across macro assets | Diversifying vs options VRP; different driver than equity factors |
| `macro_cta_trend` | Macro | Trend following across macro assets | Historically useful crisis diversifier, but real-data diagnostics require caution |
| `macro_risk_parity` | Macro | Risk-contribution allocation | Diversifying in calm regimes; lower expected alpha as standalone strategy |
| `macro_vrp` | Macro/options | Volatility risk premium signal | Overlaps with current short-vol live strategies |
| `options_earnings_vol` | Options | Earnings-vol event exposure | Options/event-driven; requires careful event calendar controls |
| `options_tail_hedge` | Options | Long-vol tail hedge | Expected negative/low correlation to current short-vol VRP mix |

## Selected paper-validation batch

Selected batch for the initial 6-week paper window:

| Strategy | Allocation in paper config | Risk-budget reservation | Rationale |
|---|---:|---:|---|
| `equity_stat_arb` | 33.3333% | 1.5% | Existing config; market-neutral equity/stat-arb profile; diversifies away from short-vol options |
| `macro_carry` | 33.3334% | 1.5% | Existing config; macro carry driver; diversifies away from equity/intraday/options VRP |
| `options_tail_hedge` | 33.3333% | 1.5% | Existing config; long-vol hedge profile; expected low/negative correlation to current short-vol live strategies |

Rationale:

- All three have existing config files.
- All three are registered and walk-forward tested.
- None are currently enabled in `config/live.yaml`.
- The batch adds equity/stat-arb, macro carry, and long-vol hedge exposure rather than increasing concentration in the current short-vol VRP mix.
- `macro_cta_trend` remains deferred for a later review because prior deployment notes flag real-data concerns despite walk-forward coverage.

Paper period:

- Start: 2026-06-09
- End: 2026-07-21
- Duration: 6 weeks
- Config: `config/paper.yaml`
- Mode: separate paper account, `single_share` scale mode, $10,000 paper capital

## Monitoring cadence

Daily:

1. Run `python scripts/strategy_risk_dashboard.py --paper`.
2. Confirm the kill-switch status is clean.
3. Review open positions, inferred weights, daily return, cumulative return, Sharpe, and max drawdown.
4. Check the timestamped paper attribution archive in `reports/paper/` and the paper kill-switch log at `reports/paper_kill_switch_events.jsonl` if any alert appears.
5. Optionally run `python scripts/correlation_monitor.py --paper` to check rolling strategy-return correlation alarms.
6. If a strategy breaches its drawdown limit, halt it and document the event. Do not tune parameters.

Paper bootstrap and monitoring commands:

```bash
# Validate the paper config before launch.
python scripts/config_preflight.py --config config/paper.yaml

# Start the paper-validation runner without changing config/live.yaml.
docker compose -f docker-compose.yml -f docker-compose.paper.yml up -d

# Confirm the runtime files exist and are updated after startup.
ls -l reports/paper_events.jsonl reports/paper_equity.csv reports/paper_attribution.csv
ls -l reports/paper/paper_attribution_YYYY-MM-DD.csv

# Daily monitoring after market close.
python scripts/strategy_risk_dashboard.py --paper
python scripts/correlation_monitor.py --paper
```

Runtime files to inspect:

- `reports/paper_events.jsonl` — order submit/fill/close events from the paper runner.
- `reports/paper_equity.csv` — portfolio-level paper equity curve.
- `reports/paper_attribution.csv` — current daily strategy attribution export used by the correlation monitor.
- `reports/paper/paper_attribution_YYYY-MM-DD.csv` — timestamped archive copy.
- `reports/paper_monitoring_log.csv` — daily monitor summary appended by `scripts/monitor_paper_trading.py --paper`.

Bootstrap note: 2026-06-10 — paper bootstrap wiring and monitoring hardening were implemented. Container startup must be completed only after Docker is available and `config_preflight.py --config config/paper.yaml` passes; do not mark the paper run as started until the runtime files above are created and updated.

Weekly:

1. Append a short note to the log below.
2. Include Sharpe, max drawdown, turnover/data-quality observations, attribution CSV paths, and any notable behavior.
3. Keep the note factual; do not reinterpret marginal results after seeing the paper P&L.

Weekly paper-validation log:

| Week ending | Sharpe | Max drawdown | Kill-switch status | Attribution paths | Notes |
|---|---:|---:|---|---|---|
| 2026-06-14 | TBD | TBD | TBD | `reports/paper/paper_attribution_YYYY-MM-DD.csv` | Initial monitoring week |
| 2026-06-21 | TBD | TBD | TBD |  |  |
| 2026-06-28 | TBD | TBD | TBD |  |  |
| 2026-07-05 | TBD | TBD | TBD |  |  |
| 2026-07-12 | TBD | TBD | TBD |  |  |
| 2026-07-21 | TBD | TBD | TBD |  | Final 6-week review |

Weekly paper-validation log template:

```markdown
### Week ending YYYY-MM-DD

Date range: Mon YYYY-MM-DD to Sun YYYY-MM-DD

Strategy performance:

| Strategy | Cumulative return | Sharpe | Max drawdown | Status |
|---|---:|---:|---:|---|
| equity_stat_arb | TBD | TBD | TBD | running / halted / review |
| macro_carry | TBD | TBD | TBD | running / halted / review |
| options_tail_hedge | TBD | TBD | TBD | running / halted / review |

Kill-switch events:

- None / strategy_name on YYYY-MM-DD — brief factual note.

Attribution CSV archive paths:

- reports/paper/paper_attribution_YYYY-MM-DD.csv
- reports/paper/paper_attribution_YYYY-MM-DD.csv
- reports/paper/paper_attribution_YYYY-MM-DD.csv
- reports/paper/paper_attribution_YYYY-MM-DD.csv
- reports/paper/paper_attribution_YYYY-MM-DD.csv
- reports/paper/paper_attribution_YYYY-MM-DD.csv
- reports/paper/paper_attribution_YYYY-MM-DD.csv

Anomalies / notes:

- Factual operational notes only. Do not tune parameters or reinterpret marginal results after seeing paper P&L.
```

## Paper live-graduation criteria

A paper strategy may be considered for live graduation only after all criteria below are satisfied. These criteria are fixed before observing paper results.

Minimum observation:

- At least 4 weeks of paper trading; 6 weeks is preferred.
- No parameter, threshold, or signal-code changes during the paper window.
- No validation/test-window peeking; the paper window is the out-of-sample period.

Performance and risk:

- Annualized Sharpe > 0.5, or clearly positive versus the assigned benchmark if Sharpe is not meaningful.
- Max drawdown < 15%.
- No -20% kill-switch breach.
- No unexplained P&L attribution gaps or missing event/equity logs.

Operational quality:

- No data-feed failures that corrupt signals.
- No execution failures that materially distort fills or costs.
- Average daily turnover remains within the strategy's documented expectation.
- Open positions reconcile to expected symbols/notionals.
- Any alert is reviewed and documented before graduation.

Live rollout rule:

- Even if all three paper strategies pass, graduate them one at a time.
- Start with a small live weight, re-balance the existing `config/live.yaml` weights so the total remains exactly 1.0, and monitor interaction with the current live mix before scaling.

## Next paper-validation pool

Do not add the next batch until the current 4–6 week paper window is complete. The next pool should stay serial and uncontaminated.

Candidate strategies with existing config files and walk-forward coverage:

| Strategy | Phase | Diversification rationale | Watch item |
|---|---|---|---|
| `equity_mean_reversion` | CANDIDATE | Short-horizon reversal can diversify away from momentum and short-vol VRP | Path dependency and crowded reversal regimes |
| `macro_risk_parity` | CANDIDATE | Risk-contribution allocation may diversify across macro regimes | Lower standalone alpha; needs real macro data checks |
| `options_earnings_vol` | CANDIDATE | Event-driven options exposure can diversify away from VRP carry | Calendar/event controls must be strict |
| `macro_cta_trend` | CANDIDATE | Crisis-diversifier profile; different driver from current VRP mix | Prior deployment notes flag real-data concerns |
| `equity_trend_following` | CANDIDATE | Directional trend exposure; simple and interpretable | Can correlate with equity momentum in persistent trends |
| `macro_vrp` | CANDIDATE | Existing macro/options config | Overlaps with current short-vol live strategies |
| `equity_momentum` | CANDIDATE | Existing config and walk-forward coverage | Can correlate with broad risk-on regimes |

Provisional next batch after the current paper window:

1. `equity_mean_reversion` — diversifies away from momentum/VRP.
2. `macro_risk_parity` — adds macro risk-budgeting behavior.
3. `options_earnings_vol` — adds event-driven options behavior with strict calendar controls.

If any current paper strategy breaches its kill-switch, keep the next batch blocked until the breach is reviewed and the paper-validation governance rule is reaffirmed.

## Risk controls added

1. **Per-strategy drawdown kill-switch**
   - New module: `quantcore/risk/strategy_kill_switch.py`
   - Default halt threshold: **-20% from peak equity**
   - Once halted, a strategy remains halted until explicit reset.
   - No parameter tuning is allowed after a breach.

2. **Runtime status gating**
   - New file: `strategy_status.yaml`
   - New helper: `quantcore/strategies/status.py`
   - `quantcore/live/production_runner.py` now checks `strategy_status.yaml` and skips strategies whose phase is not runtime-allowed.
   - Live config should load `LIVE` strategies only.
   - Paper config should load `PAPER` strategies only.

3. **Console risk dashboard / P&L attribution**
   - New script: `scripts/strategy_risk_dashboard.py`
   - Reads existing event and equity logs.
   - Exports daily attribution to CSV by default (`reports/strategy_attribution.csv`, or `reports/paper_attribution.csv` with `--paper`).
   - Also writes timestamped attribution archives under `reports/live/` or `reports/paper/`.
   - Appends structured kill-switch breach records to `reports/kill_switch_events.jsonl` or `reports/paper_kill_switch_events.jsonl`.
   - Reports:
     - open positions by strategy
     - inferred allocation weights
     - daily return
     - cumulative return
     - Sharpe
     - max drawdown
     - 20% kill-switch breaches

4. **Rolling correlation monitor**
   - New script: `scripts/correlation_monitor.py`
   - Reads attribution CSV exports and computes rolling pairwise strategy-return correlations.
   - Prints an alert when any pair exceeds the configured threshold, defaulting to 0.8 over 20 days.
   - This is an observability alarm only; it does not change allocations or strategy behavior.

5. **Paper config**
   - New file: `config/paper.yaml`
   - Separate from `config/live.yaml`
   - Paper allocation weights sum to 1.0 within the paper account.
   - `risk_budget_weight` documents each strategy's reserved share of total production risk budget and does not alter live weights.

## `vix_call_ladder` walk-forward status

`vix_call_ladder` has been added to the walk-forward test matrix. The targeted walk-forward test passes after fixing scalar extraction for synthetic multi-asset arrays. It remains `INACTIVE` in `strategy_status.yaml` and is not part of the Phase 4 paper-validation batch.

## Strategy status mapping

`strategy_status.yaml` maps all 48 registered strategies to a phase:

- `LIVE`: currently enabled in `config/live.yaml`
- `PAPER`: selected for the 6-week Phase 4 paper window
- `CANDIDATE`: registered/walk-forward-tested, deferred for later validation
- `INACTIVE`: registered/importable but not selected for immediate validation
- `DATA_BLOCKED`: H001 OPRA research pipeline only, not a registered strategy

## Security housekeeping

`.env` contains broker and alerting credentials and must never be committed, copied into logs, or embedded in documentation. If any credential is exposed outside the local environment, rotate it immediately and update the local `.env` copy only.

## Operating rules

- No strategy threshold or parameter tuning during paper trading.
- Paper trading is the out-of-sample validation period.
- If any paper strategy breaches its drawdown limit, halt it and review; do not tune.
- `config/live.yaml` live weights remain unchanged and must continue to sum to 1.0.
- H001 remains blocked until a reliable OPRA source is available.
