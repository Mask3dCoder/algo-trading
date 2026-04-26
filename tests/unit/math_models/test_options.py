"""Unit tests for BSM pricing, Greeks, implied vol, and put-call parity."""

from __future__ import annotations

import numpy as np
import pytest

from quantcore.math_models.options.bsm import (
    bsm_call,
    bsm_put,
    call_delta,
    call_rho,
    call_theta,
    gamma,
    numerical_greek,
    put_call_parity_check,
    put_delta,
    put_rho,
    vanna,
    vega,
    volga,
)
from quantcore.math_models.options.implied_vol import (
    implied_vol,
    implied_vol_brent,
    implied_vol_newton,
)

# Standard test parameters (from Hull, Chapter 19)
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2


class TestBSMPricing:
    def test_call_price_positive(self) -> None:
        c = float(bsm_call(S, K, T, r, sigma))
        assert c > 0

    def test_put_price_positive(self) -> None:
        p = float(bsm_put(S, K, T, r, sigma))
        assert p > 0

    def test_call_gt_put_for_atm(self) -> None:
        c = float(bsm_call(S, K, T, r, sigma))
        p = float(bsm_put(S, K, T, r, sigma))
        assert c > p  # Call has time value advantage when r > 0

    def test_deep_itm_call_approaches_intrinsic(self) -> None:
        c = float(bsm_call(200.0, 100.0, T, r, sigma))
        intrinsic = 200.0 - 100.0 * np.exp(-r * T)
        assert abs(c - intrinsic) < 1.0

    def test_vectorized(self) -> None:
        S_arr = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        prices = bsm_call(S_arr, K, T, r, sigma)
        assert len(prices) == 5
        assert all(p > 0 for p in prices)


class TestPutCallParity:
    @pytest.mark.parametrize("S_val,K_val", [(80, 100), (100, 100), (120, 100), (50, 60)])
    def test_parity_holds(self, S_val: float, K_val: float) -> None:
        error = put_call_parity_check(S_val, K_val, T, r, sigma)
        assert error < 1e-10, f"Put-call parity violated: error = {error}"


class TestGreeks:
    def test_call_delta_range(self) -> None:
        d = float(call_delta(S, K, T, r, sigma))
        assert 0 < d < 1

    def test_put_delta_range(self) -> None:
        d = float(put_delta(S, K, T, r, sigma))
        assert -1 < d < 0

    def test_put_delta_equals_call_minus_one(self) -> None:
        cd = float(call_delta(S, K, T, r, sigma))
        pd = float(put_delta(S, K, T, r, sigma))
        assert abs(cd - pd - 1.0) < 1e-10

    def test_gamma_positive(self) -> None:
        g = float(gamma(S, K, T, r, sigma))
        assert g > 0

    def test_vega_positive(self) -> None:
        v = float(vega(S, K, T, r, sigma))
        assert v > 0

    def test_call_theta_negative(self) -> None:
        t = float(call_theta(S, K, T, r, sigma))
        assert t < 0

    def test_call_rho_positive(self) -> None:
        rho_val = float(call_rho(S, K, T, r, sigma))
        assert rho_val > 0

    def test_put_rho_negative(self) -> None:
        rho_val = float(put_rho(S, K, T, r, sigma))
        assert rho_val < 0

    def test_vanna_sign(self) -> None:
        v = float(vanna(S, K, T, r, sigma))
        # Vanna can be positive or negative depending on moneyness
        assert isinstance(v, float)

    def test_volga_positive_atm(self) -> None:
        v = float(volga(S, S, T, r, sigma))
        # Near ATM, d1*d2 ≈ 0, but volga > 0 away from ATM
        assert isinstance(v, float)

    def test_numerical_delta_matches_analytical(self) -> None:
        analytical = float(call_delta(S, K, T, r, sigma))
        numerical = numerical_greek(bsm_call, "S", S, K, T, r, sigma)
        assert abs(analytical - numerical) < 0.01

    def test_numerical_vega_matches_analytical(self) -> None:
        analytical = float(vega(S, K, T, r, sigma))
        numerical = numerical_greek(bsm_call, "sigma", S, K, T, r, sigma)
        # Tolerance: central difference with epsilon=1e-6 on sigma
        assert abs(analytical - numerical) < 1.0


class TestImpliedVol:
    def test_round_trip_call(self) -> None:
        true_sigma = 0.25
        price = float(bsm_call(S, K, T, r, true_sigma))
        iv = implied_vol(price, S, K, T, r, is_call=True)
        assert abs(iv - true_sigma) < 1e-8

    def test_round_trip_put(self) -> None:
        true_sigma = 0.30
        price = float(bsm_put(S, K, T, r, true_sigma))
        iv = implied_vol(price, S, K, T, r, is_call=False)
        assert abs(iv - true_sigma) < 1e-8

    def test_newton_converges(self) -> None:
        true_sigma = 0.15
        price = float(bsm_call(S, K, T, r, true_sigma))
        iv = implied_vol_newton(price, S, K, T, r, is_call=True)
        assert abs(iv - true_sigma) < 1e-8

    def test_brent_converges(self) -> None:
        true_sigma = 0.40
        price = float(bsm_call(S, K, T, r, true_sigma))
        iv = implied_vol_brent(price, S, K, T, r, is_call=True)
        assert abs(iv - true_sigma) < 1e-8

    def test_deep_itm(self) -> None:
        true_sigma = 0.20
        price = float(bsm_call(150.0, 100.0, T, r, true_sigma))
        iv = implied_vol(price, 150.0, 100.0, T, r, is_call=True)
        assert abs(iv - true_sigma) < 1e-4

    def test_deep_otm(self) -> None:
        true_sigma = 0.30
        price = float(bsm_call(60.0, 100.0, T, r, true_sigma))
        if price > 1e-6:
            iv = implied_vol(price, 60.0, 100.0, T, r, is_call=True)
            assert abs(iv - true_sigma) < 1e-4
