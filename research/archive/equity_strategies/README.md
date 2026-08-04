# Archived Equity Strategies - 2026-07-14

Reason for archival: both `equity_factor_neutral` and `equity_pairs` FAILED the
institutional gate (`scripts/institutional_gate.py`) on the REAL SP500 PIT panel
(2018-01-02..2024-12-31, 10 bps/leg, 5 anchored OOS folds). PIT-correct rolling
stats; no look-ahead; no synthetic data.

Verified results (from reports/*_gate_verdict.json):
- equity_pairs:    full gate run 2026-07-14 -> aggregate OOS Sharpe 0.4102,
                   MaxDD 39.3%, DSR p 1.0, 864 trades, gate_status FAIL.
                   Driver: F3 (2022) Sharpe -1.618 / DD 36.5% (regime break).
                   Per-fold Sharpe: F1 0.979, F2 0.349, F3 -1.618, F4 1.874, F5 0.992.
- equity_factor_neutral: gate run (coverage-only, --no-backtest) 2026-07-14 ->
                   0/5 folds evaluated, gate_status FAIL. A full real backtest was
                   NOT executed this session, so no aggregate Sharpe is recorded here.

Gate verdict: FAIL. Structural fragility (esp. 2022) is not fixable by parameter
tuning. No commit of these files; parked for reference only.

Next step: pivot to options VRP (data-gapped: real chains start 2021) or accept no
alpha. See reports/equity_pairs_gate_verdict.json for the full audit trail.
