"""Regression tests for the published B&LdP 2014 DSR implementation.

See ``docs/dsr_audit.md`` for the full audit write-up. These tests lock the
correct formula and document the legacy implementation's unit bug.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from quantcore.research.deflated_sharpe import deflated_sharpe_ratio as legacy_dsr
from quantcore.research.dsr_published import (
    deflated_sharpe_ratio_published,
    expected_max_sharpe,
    sharpe_se,
)


def _independent_published_formula(
    *,
    sr: float,
    n: int,
    t: int,
    skew: float,
    kurt: float,
) -> float:
    """Independent transcription of the published equations (Eq. 2-4)."""
    var_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2
    se = np.sqrt(var_term / (t - 1.0))
    if n <= 1:
        emax = 0.0
    else:
        # Exact E[max of N normals] via the equivalent integral form:
        # E[max] = N * int x*phi(x)*Phi(x)^(N-1) dx over R.
        from scipy.integrate import quad
        from scipy.stats import norm

        def _kernel(x: float) -> float:
            log_term = (n - 1) * norm.logcdf(x)
            return x * norm.pdf(x) * math.exp(log_term)

        emax = float(n * quad(_kernel, -np.inf, np.inf, limit=400)[0])
    sr_star = se * emax
    dsr = (sr - sr_star) / se
    return float(1.0 - stats.norm.cdf(dsr))


class TestPublishedFormula:
    def test_se_matches_metricgate_reference(self) -> None:
        """Published reference: T=120, SR=0.1760, skew=-0.3649, kurt=6.129."""
        se = sharpe_se(
            observed_sharpe=0.1760,
            sample_length=120,
            skewness=-0.3649,
            kurtosis=6.129,
        )
        assert se == pytest.approx(0.0963, abs=5e-4)

    def test_n_one_is_probabilistic_sharpe(self) -> None:
        """N=1 collapses to PSR: p = 1 - Phi(SR / se)."""
        sr, t = 0.3, 500
        skew, kurt = -0.2, 4.5
        se = sharpe_se(observed_sharpe=sr, sample_length=t, skewness=skew, kurtosis=kurt)
        pv, dsr = deflated_sharpe_ratio_published(
            observed_sharpe=sr, num_trials=1, sample_length=t,
            skewness=skew, kurtosis=kurt,
        )
        assert dsr == pytest.approx(sr / se, rel=1e-10)
        assert pv == pytest.approx(1.0 - stats.norm.cdf(sr / se), rel=1e-10)

    def test_canonical_paper_case(self) -> None:
        """Paper's canonical illustration: SR 1.5 annualized, T=756, N=100.

        Annualized 1.5 = per-observation 1.5/sqrt(252) ~= 0.0945. The published
        DSR is ~0.52 (p ~= 0.48): a coin flip after 100 trials.
        """
        sr_per_obs = 1.5 / np.sqrt(252)
        pv, dsr = deflated_sharpe_ratio_published(
            observed_sharpe=sr_per_obs,
            num_trials=100,
            sample_length=756,
            skewness=-0.15,
            kurtosis=3.0,
        )
        assert 0.40 < (1.0 - pv) < 0.60  # DSR probability ~0.52
        assert 0.40 < pv < 0.55

    def test_legacy_annualized_input_is_wrong(self) -> None:
        """Locks the audit finding: legacy function fed annualized SR -> p~1.0."""
        pv_legacy, _ = legacy_dsr(
            observed_sharpe=1.5,
            num_trials=100,
            sample_length=756,
            skewness=-0.15,
            kurtosis_=0.0,
        )
        assert pv_legacy > 0.99
        pv_published, _ = deflated_sharpe_ratio_published(
            observed_sharpe=1.5 / np.sqrt(252),
            num_trials=100,
            sample_length=756,
            skewness=-0.15,
            kurtosis=3.0,
        )
        assert pv_published < 0.55

    def test_cross_check_independent_transcription(self) -> None:
        rng = np.random.default_rng(7)
        for _ in range(20):
            sr = float(rng.uniform(-0.3, 0.8))
            n = int(rng.integers(1, 300))
            t = int(rng.integers(30, 2000))
            skew = float(rng.uniform(-1.0, 1.0))
            kurt = float(rng.uniform(1.5, 8.0))
            expected = _independent_published_formula(sr=sr, n=n, t=t, skew=skew, kurt=kurt)
            pv, _ = deflated_sharpe_ratio_published(
                observed_sharpe=sr, num_trials=n, sample_length=t,
                skewness=skew, kurtosis=kurt,
            )
            assert pv == pytest.approx(expected, abs=1e-10)

    def test_expected_max_matches_monte_carlo(self) -> None:
        rng = np.random.default_rng(0)
        for n in (2, 5, 10, 50):
            sims = np.max(rng.standard_normal((200_000, n)), axis=1)
            assert expected_max_sharpe(n) == pytest.approx(float(np.mean(sims)), abs=0.01)

    def test_monotonicity(self) -> None:
        base = dict(observed_sharpe=0.2, sample_length=500, skewness=0.0, kurtosis=3.0)
        p1, _ = deflated_sharpe_ratio_published(num_trials=1, **base)
        p100, _ = deflated_sharpe_ratio_published(num_trials=100, **base)
        assert p100 > p1  # more trials => less significant

        p_lo, _ = deflated_sharpe_ratio_published(observed_sharpe=0.1, num_trials=10, sample_length=500, skewness=0.0, kurtosis=3.0)
        p_hi, _ = deflated_sharpe_ratio_published(observed_sharpe=0.5, num_trials=10, sample_length=500, skewness=0.0, kurtosis=3.0)
        assert p_hi < p_lo  # higher Sharpe => more significant

        p_short, _ = deflated_sharpe_ratio_published(observed_sharpe=0.2, num_trials=10, sample_length=100, skewness=0.0, kurtosis=3.0)
        p_long, _ = deflated_sharpe_ratio_published(observed_sharpe=0.2, num_trials=10, sample_length=2000, skewness=0.0, kurtosis=3.0)
        assert p_long < p_short  # more data => more significant

    def test_edge_cases(self) -> None:
        assert deflated_sharpe_ratio_published(
            observed_sharpe=0.5, num_trials=5, sample_length=1
        ) == (1.0, 0.0)
        # Invalid kurtosis falls back to normal (3.0), no crash.
        pv, dsr = deflated_sharpe_ratio_published(
            observed_sharpe=0.5, num_trials=5, sample_length=500,
            skewness=0.0, kurtosis=0.5,
        )
        assert np.isfinite(dsr)
        assert 0.0 <= pv <= 1.0
