# DSR Audit — 2026-08-04

## What was asked

The pre-registered factor gate hinges on `DSR p <= 0.04`
(`docs/cross_sectional_factor_protocol.md`, Section 8). Before the decisive
run touches it, the Deflated Sharpe implementation was audited against the
published reference (Bailey & Lopez de Prado, 2014, *Journal of Portfolio
Management* 40(5), 94-107; SSRN 2465675) and against worked reference values.

## Findings

### The legacy implementation (`quantcore/research/deflated_sharpe.py`) is not safe for the gate

Reading the code and its callers (e.g. `scripts/validate_all_strategies.py`,
`quantcore/research/walkforward.py`) found four deviations from the published
formula:

| # | Component | Published (B&LdP 2014) | Legacy implementation | Effect |
|---|---|---|---|---|
| 1 | Sharpe units | per-observation (daily/monthly) | callers pass **annualized** Sharpe | largest error; mixes scales |
| 2 | Kurtosis term | `(gamma4 - 1)/4` with raw kurtosis | `(gamma4_excess + 3)/4` | inflates the variance term |
| 3 | `sqrt(T-1)` | exact | `sqrt(T-0.25)` | small numeric drift |
| 4 | Benchmark `SR*` | `SR* = sigma_hat * E[max(N normals)]` | `E[max]` only (no sigma scaling) | under-corrects at high SR |

Worked check on the paper's canonical illustration (SR = 1.5 *annualized*,
T = 756 daily obs, N = 100 trials, skew = -0.15, kurtosis = 3):

- Published formula with per-observation SR (`1.5/sqrt(252) ~= 0.0945`):
  **DSR ~= 0.52, p ~= 0.48** — a coin flip after 100 trials.
- Legacy function fed the annualized 1.5: **p ~= 1.0** — always "not
  significant" for any realistic annualized Sharpe below ~2.5.

The legacy path is not merely conservative; the error is unit-dependent and
silent, so it would produce a confident wrong verdict. The decisive gate must
NOT use it.

### The published formula is now implemented and verified

`quantcore/research/dsr_published.py` implements the exact published form:

```
sigma_hat = sqrt( (1 - gamma3*SR + ((gamma4-1)/4)*SR^2) / (T-1) )
SR*       = SR0 + sigma_hat * E[max(N standard normals)]
DSR       = Phi( (SR - SR*) / sigma_hat )
```

where `E[max]` is computed exactly by quadrature (the paper's asymptotic
approximation under-corrects for N < 10; exact integration is identical to it
to well within 1e-3 for N >= 10 and is the conservative choice for small N).

Verification (see `tests/unit/research/test_dsr_published.py`):

- Standard error matches an external worked reference exactly
  (T=120, SR=0.1760, skew=-0.3649, raw kurtosis 6.129 -> sigma_hat = 0.0963;
  MetricGate B&LdP calculator, 2026).
- N=1 collapses to the Probabilistic Sharpe Ratio (closed-form check).
- Canonical paper case lands at DSR ~= 0.52 / p ~= 0.48, consistent with the
  paper's published narrative.
- Cross-check against an independent transcription of Eqs. 2-4 on 20 random
  inputs (agreement to 1e-10).
- `E[max]` exact quadrature validated against a 200k-sample Monte Carlo of the
  max of N standard normals (within 0.01).
- Monotonicity in trials / Sharpe / sample length.

## Decision

1. The decisive harness (`scripts/run_cross_sectional_factor.py`) uses
   `deflated_sharpe_ratio_published` with **per-observation** Sharpe, raw
   kurtosis, and `num_trials = 3` (the frozen variant family).
2. The legacy function is left untouched for backward compatibility and is
   documented as unsafe for gate decisions. Its annualized-Sharpe convention
   is the reason every prior `DSR p = 1.0` audit result should be treated as
   uninformative, not as evidence.
3. Regression tests lock the correct formula and the legacy discrepancy, so
   neither can silently regress before the decisive run.

## References

- Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
  JPM 40(5), 94-107. DOI 10.3905/jpm.2014.40.5.094.
- Mertens, E. (2002). "Comments on the Variance of the IID Estimator in
  'The Statistics of Sharpe Ratios'." Working paper.
- MetricGate (2026), Deflated Sharpe Ratio calculator documentation — used as
  an external worked reference for the standard-error formula.
