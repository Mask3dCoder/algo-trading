"""Gamma Exposure (GEX) calculation, gamma flip levels, charm/vanna flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from quantcore.math_models.options.bsm import gamma as bsm_gamma


@dataclass
class GEXConfig:
    spot_price: float = 0.0
    risk_free_rate: float = 0.05
    use_net_gex: bool = True


@dataclass
class GEXResult:
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    symbol: str = ""
    total_gex: float = 0.0
    net_gex: float = 0.0
    positive_gex: float = 0.0
    negative_gex: float = 0.0
    gamma_flip_level: float = 0.0
    charm_flow: float = 0.0
    vanna_flow: float = 0.0
    concentration: float = 0.0


class GEXCalculator:
    """Compute Gamma Exposure from an options chain."""

    def __init__(self, config: GEXConfig | None = None) -> None:
        self.config = config or GEXConfig()

    def compute_from_chain(self, spot: float, options: list[dict]) -> GEXResult:
        """Compute GEX from a list of option contracts.

        Each option dict must have: strike, expiration (DTE or date), call_oi, put_oi,
        call_iv, put_iv, option_type, open_interest.
        """
        total_pos = 0.0
        total_neg = 0.0
        gex_by_strike: dict[float, float] = {}
        r = self.config.risk_free_rate

        for opt in options:
            strike = float(opt["strike"])
            dte = float(opt.get("dte", 30))
            tau = max(dte / 365.0, 1.0 / 365.0)
            oi = float(opt.get("open_interest", opt.get("call_oi", opt.get("put_oi", 0))))
            iv = float(opt.get("iv", opt.get("call_iv", opt.get("put_iv", 0.2))))
            opt_type = str(opt.get("option_type", "call")).lower()

            if oi <= 0 or iv <= 0 or strike <= 0 or spot <= 0:
                continue

            try:
                g = bsm_gamma(spot, strike, tau, r, iv)
            except Exception:
                g = 0.0

            gex_contribution = g * spot * 100.0 * oi

            if opt_type == "call":
                total_pos += gex_contribution
            else:
                total_neg -= gex_contribution

            gex_by_strike[strike] = gex_by_strike.get(strike, 0.0) + gex_contribution

        net = total_pos + total_neg
        total = abs(total_pos) + abs(total_neg)
        flip = _find_gamma_flip(gex_by_strike, spot)

        if total > 0:
            concentration = sum(
                (v / total) ** 2 for v in gex_by_strike.values()
            ) if gex_by_strike else 0.0
        else:
            concentration = 0.0

        return GEXResult(
            symbol=options[0].get("symbol", "") if options else "",
            total_gex=total, net_gex=net,
            positive_gex=total_pos, negative_gex=total_neg,
            gamma_flip_level=flip, concentration=float(concentration),
        )

    def find_gamma_flip(
        self, total_gex_by_strike: dict[float, float], spot: float,
    ) -> float:
        """Find the price where cumulative GEX crosses zero."""
        return _find_gamma_flip(total_gex_by_strike, spot)

    def compute_charm(
        self, spot: float, options: list[dict], days_passed: float = 1.0 / 252.0,
    ) -> float:
        """Approximate total charm exposure (dGamma/dTime)."""
        total = 0.0
        for opt in options:
            strike = float(opt["strike"])
            dte = float(opt.get("dte", 30))
            tau = max(dte / 365.0, 1.0 / 365.0)
            oi = float(opt.get("open_interest", opt.get("call_oi", opt.get("put_oi", 0))))
            iv = float(opt.get("iv", 0.2))
            opt_type = str(opt.get("option_type", "call")).lower()
            if oi <= 0 or iv <= 0:
                continue
            g0 = bsm_gamma(spot, strike, tau, self.config.risk_free_rate, iv)
            g1 = bsm_gamma(spot, strike, max(tau - days_passed, 1.0 / 365.0), self.config.risk_free_rate, iv)
            charm = (g1 - g0) * spot * 100.0 * oi
            total += charm if opt_type == "call" else -charm
        return total

    def compute_vanna(self, spot: float, options: list[dict]) -> float:
        """Approximate total vanna exposure (dDelta/dVol or dVega/dSpot)."""
        total = 0.0
        bump = 0.01
        for opt in options:
            strike = float(opt["strike"])
            dte = float(opt.get("dte", 30))
            tau = max(dte / 365.0, 1.0 / 365.0)
            oi = float(opt.get("open_interest", opt.get("call_oi", opt.get("put_oi", 0))))
            iv = float(opt.get("iv", 0.2))
            opt_type = str(opt.get("option_type", "call")).lower()
            if oi <= 0 or iv <= 0:
                continue
            g_up = bsm_gamma(spot * (1 + bump), strike, tau, self.config.risk_free_rate, iv)
            g_dn = bsm_gamma(spot * (1 - bump), strike, tau, self.config.risk_free_rate, iv)
            vanna = (g_up - g_dn) / (2.0 * bump * spot) * spot * 100.0 * oi
            total += vanna if opt_type == "call" else -vanna
        return total


def _find_gamma_flip(gex_by_strike: dict[float, float], spot: float) -> float:
    """Find the strike level where cumulative GEX flips sign."""
    if not gex_by_strike:
        return spot
    sorted_strikes = sorted(gex_by_strike.items())
    cumulative = 0.0
    for strike, gex in sorted_strikes:
        cumulative += gex
        if strike >= spot and cumulative <= 0:
            return strike
        if strike >= spot and gex_by_strike.get(strike, 0) > 0:
            continue
    return spot


def gex_profile(
    spot: float, options: list[dict],
    price_range: tuple[float, float] = (0.9, 1.1), num_points: int = 100,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Compute GEX across a range of spot prices."""
    prices = np.linspace(spot * price_range[0], spot * price_range[1], num_points)
    gex_values = np.zeros(num_points, dtype=float)
    r = 0.05
    for i, p in enumerate(prices):
        total = 0.0
        for opt in options:
            strike = float(opt["strike"])
            dte = float(opt.get("dte", 30))
            tau = max(dte / 365.0, 1.0 / 365.0)
            oi = float(opt.get("open_interest", opt.get("call_oi", opt.get("put_oi", 0))))
            iv = float(opt.get("iv", 0.2))
            opt_type = str(opt.get("option_type", "call")).lower()
            if oi <= 0:
                continue
            try:
                g = bsm_gamma(p, strike, tau, r, iv)
            except Exception:
                g = 0.0
            gex_c = g * p * 100.0 * oi
            total += gex_c if opt_type == "call" else -gex_c
        gex_values[i] = total
    return prices, gex_values


def detect_gamma_squeeze(gex: float, historical_gex: NDArray[np.floating]) -> bool:
    """Detect potential gamma squeeze: very negative GEX vs historical distribution."""
    if len(historical_gex) < 20:
        return False
    mean = float(np.mean(historical_gex))
    std = float(np.std(historical_gex))
    if std == 0:
        return False
    z_score = (gex - mean) / std
    return bool(z_score < -2.0 and gex < 0)
