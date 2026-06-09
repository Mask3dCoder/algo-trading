"""Parametric volatility surface model with skew and term structure.

Replaces the flat ``rv + 0.15`` synthetic IV with a realistic surface that
captures:
  - ATM IV level (from VIX or realized vol + VRP)
  - Strike-dependent skew (OTM puts more expensive, OTM calls cheaper)
  - Term structure (short-dated IV typically higher than long-dated)
  - VRP that varies with the VIX level

This is used for backtesting options strategies when real options chain
data is not available.  It produces more realistic entry/exit timing
and P&L estimates than a flat premium model.

References:
    - Gatheral (2006), The Volatility Surface: A Practitioner's Guide
    - CBOE SKEW methodology
    - Typical SPX implied volatility surface behavior
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skew model
# ---------------------------------------------------------------------------


def _skew_adjustment(vix_level: float) -> float:
    """Compute OTM put skew adjustment as a function of VIX level.

    Based on empirical SPX skew behavior:
      - Low vol (VIX < 15): modest skew (~1-2%)
      - Normal vol (VIX 15-25): moderate skew (~2-4%)
      - Elevated vol (VIX 25-35): pronounced skew (~4-6%)
      - Crisis vol (VIX > 35): extreme skew (~6-10%)

    The relationship is approximately linear in VIX with a kink around 20.
    """
    if vix_level < 12:
        return 0.01
    elif vix_level < 20:
        return 0.01 + 0.002 * (vix_level - 12)
    elif vix_level < 30:
        return 0.026 + 0.003 * (vix_level - 20)
    elif vix_level < 40:
        return 0.056 + 0.004 * (vix_level - 30)
    else:
        return min(0.096 + 0.002 * (vix_level - 40), 0.15)


def _term_structure_adjustment(
    atm_iv: float,
    dte: float,
    long_run_iv: float = 0.20,
) -> float:
    """Adjust IV for term structure (expiration dependence).

    Short-dated options typically have higher IV than long-dated
    (the term structure is usually in "backwardation" for equity options,
    meaning short-term IV > long-term IV).

    Uses a square-root-of-time interpolation with mean reversion:
      IV(T) = long_run + (atm_iv - long_run) * sqrt(T_ref / T)

    where T_ref is a reference expiration (30 days).
    """
    T_ref = 30.0 / 365.0  # 30 days as fraction of year
    T = max(dte / 365.0, 1.0 / 365.0)

    # Square-root scaling relative to reference
    scale = np.sqrt(T_ref / T)

    # Blend: short-dated → more like ATM IV; long-dated → more like long-run
    iv_adjusted = long_run_iv + (atm_iv - long_run_iv) * scale

    # Floor at reasonable minimum
    return max(iv_adjusted, 0.05)


# ---------------------------------------------------------------------------
# Core surface model
# ---------------------------------------------------------------------------


class VolatilitySurface:
    """Parametric implied volatility surface model.

    Builds a realistic IV surface from:
      - ATM IV level (VIX or realized vol + VRP premium)
      - Moneyness-dependent skew (OTM put premium, OTM call discount)
      - DTE-dependent term structure
      - VIX-level-dependent VRP magnitude

    The surface is parameterized by (moneyness, dte) where:
      - moneyness = K / S (strike / spot), so 1.0 = ATM, 0.9 = 10% OTM put
      - dte = days to expiration

    Parameters
    ----------
    atm_iv : float
        Current ATM implied volatility (annualized, e.g. 0.20 = 20%).
    vix_level : float
        Current VIX level.  Used to scale skew and VRP.
    long_run_iv : float
        Long-run average IV level (default 0.20).
    skew_factor : float | None
        Override the automatic skew scaling.  If None, computed from VIX.
    """

    def __init__(
        self,
        atm_iv: float,
        vix_level: float | None = None,
        long_run_iv: float = 0.20,
        skew_factor: float | None = None,
    ) -> None:
        self.atm_iv = atm_iv
        self.vix_level = vix_level if vix_level is not None else atm_iv * 100
        self.long_run_iv = long_run_iv

        # Skew: how much OTM puts are marked up relative to ATM
        if skew_factor is not None:
            self._skew = skew_factor
        else:
            self._skew = _skew_adjustment(self.vix_level)

    def iv_at_strike_dte(
        self,
        moneyness: float,
        dte: float,
    ) -> float:
        """Get IV for a given moneyness and DTE.

        Parameters
        ----------
        moneyness : float
            K / S.  1.0 = ATM, < 1.0 = OTM put, > 1.0 = OTM call.
        dte : float
            Days to expiration.

        Returns
        -------
        float
            Implied volatility (annualized).
        """
        # Start with ATM IV adjusted for term structure
        base_iv = _term_structure_adjustment(self.atm_iv, dte, self.long_run_iv)

        # Apply skew as a function of moneyness
        # log-moneyness: negative for puts, positive for calls
        log_m = np.log(moneyness)

        # Quadratic skew model (similar to SABR backbone):
        #   IV(K) = ATM_IV * (1 + skew * log(K/S) + convexity * log(K/S)^2)
        # OTM puts (log_m < 0): IV increases (positive adjustment)
        # OTM calls (log_m > 0): IV decreases slightly (negative adjustment)
        skew_adj = self._skew * (-log_m)  # negative log_m → positive adj for puts
        convexity = 0.5 * self._skew  # smile curvature
        smile_adj = convexity * log_m ** 2

        iv = base_iv * (1.0 + skew_adj + smile_adj)

        # Bounds
        return float(np.clip(iv, 0.04, 1.50))

    def iv_smile(
        self,
        spot: float,
        dte: float,
        strikes: list[float] | None = None,
        n_strikes: int = 21,
    ) -> tuple[list[float], list[float]]:
        """Generate a smile (IV vs strike) for a given DTE.

        Parameters
        ----------
        spot : float
            Current spot price.
        dte : float
            Days to expiration.
        strikes : list[float] | None
            Specific strikes.  If None, generates from 0.8*S to 1.2*S.
        n_strikes : int
            Number of strikes to generate (if strikes is None).

        Returns
        -------
        strikes : list[float]
            Strike prices.
        ivs : list[float]
            Implied volatilities at each strike.
        """
        if strikes is None:
            strikes = list(np.linspace(spot * 0.80, spot * 1.20, n_strikes))

        ivs = [self.iv_at_strike_dte(k / spot, dte) for k in strikes]
        return strikes, ivs


# ---------------------------------------------------------------------------
# Enhanced synthetic IV generator
# ---------------------------------------------------------------------------


def generate_enhanced_synthetic_iv(
    close: pd.Series,
    lookback: int = 60,
    vix_series: pd.Series | None = None,
    use_skew: bool = True,
    long_run_iv: float = 0.20,
) -> pd.Series:
    """Generate realistic ATM IV with skew-aware VRP premium.

    This replaces the flat ``realized_vol + 0.15`` model with:
      1. Realized vol from log returns
      2. VRP premium that scales with vol level (higher vol → higher premium)
      3. Mean reversion to long-run IV
      4. Optional VIX input for more realistic ATM levels

    Parameters
    ----------
    close : pd.Series
        Daily close prices (index = DatetimeIndex).
    lookback : int
        Window for realized vol computation (default 60).
    vix_series : pd.Series | None
        If available, use VIX as the ATM IV proxy instead of rv + premium.
    use_skew : bool
        Whether to apply the VIX-dependent VRP scaling.
    long_run_iv : float
        Long-run IV level for mean reversion (default 0.20).

    Returns
    -------
    pd.Series
        ATM implied volatility series (same index as close).
    """
    log_returns = np.log(close / close.shift(1)).dropna()
    realized_vol = log_returns.rolling(lookback).std() * np.sqrt(252)

    if vix_series is not None:
        # Use VIX as ATM IV proxy (divide by 100 to get decimal)
        vix_aligned = vix_series.reindex(close.index, method="ffill") / 100.0
        # Fill NaN with realized vol + dynamic premium
        atm_base = vix_aligned.copy()
        nan_mask = atm_base.isna()
        if nan_mask.any():
            atm_base[nan_mask] = realized_vol[nan_mask] + 0.05  # small default premium
        atm_iv = atm_base
    elif use_skew:
        # Dynamic VRP premium: scales with vol level
        # Low vol → smaller premium; high vol → larger premium
        rv_aligned = realized_vol.reindex(close.index, method="ffill")

        # VRP premium model: base + vol-scaled component
        # Based on empirical VRP behavior:
        #   RV ~ 10% → VRP ~ 3-4%
        #   RV ~ 20% → VRP ~ 5-6%
        #   RV ~ 30% → VRP ~ 7-9%
        base_premium = 0.03
        vol_scaled_premium = 0.15 * rv_aligned  # ~15% of realized vol
        dynamic_premium = base_premium + vol_scaled_premium
        dynamic_premium = dynamic_premium.clip(lower=0.02, upper=0.15)

        atm_iv = rv_aligned + dynamic_premium
    else:
        # Fallback: flat premium (original behavior)
        atm_iv = realized_vol + 0.15

    # Mean revert toward long-run IV
    # Use exponential smoothing with mean reversion
    mean_reversion_speed = 0.02
    iv_smooth = atm_iv.copy()
    values = iv_smooth.values.copy()

    for i in range(1, len(values)):
        if np.isnan(values[i]):
            values[i] = values[i - 1] if not np.isnan(values[i - 1]) else long_run_iv
        else:
            # Mean reversion: pull toward long_run_iv
            deviation = values[i] - long_run_iv
            values[i] = values[i - 1] + mean_reversion_speed * (values[i] - values[i - 1])
            # Additional mean reversion when IV is extreme
            if abs(deviation) > 0.10:
                values[i] -= 0.005 * np.sign(deviation)

    iv_smooth = pd.Series(values, index=iv_smooth.index, dtype=float)

    # Clip to reasonable bounds
    iv_smooth = iv_smooth.clip(lower=0.05, upper=0.80)
    iv_smooth = iv_smooth.fillna(long_run_iv)

    return iv_smooth


# ---------------------------------------------------------------------------
# Surface-from-chain fitter (for real data mode)
# ---------------------------------------------------------------------------


def fit_surface_from_chain(
    chain_data: pd.DataFrame,
    spot: float,
    dte: float,
) -> VolatilitySurface:
    """Fit a VolatilitySurface from observed options chain data.

    Takes a DataFrame of options chain data (with columns: strike, iv,
    option_type) and fits the parametric surface to the observed IVs.

    Parameters
    ----------
    chain_data : pd.DataFrame
        Options chain with columns: strike, iv, option_type (call/put).
    spot : float
        Current spot price.
    dte : float
        Days to expiration of this chain.

    Returns
    -------
    VolatilitySurface
        Fitted surface.
    """
    if chain_data.empty:
        return VolatilitySurface(atm_iv=0.20)

    # Find ATM IV (closest strike to spot)
    chain_data = chain_data.copy()
    chain_data["moneyness"] = chain_data["strike"] / spot
    chain_data["dist_from_atm"] = abs(chain_data["moneyness"] - 1.0)

    atm_row = chain_data.loc[chain_data["dist_from_atm"].idxmin()]
    atm_iv = float(atm_row["iv"]) if "iv" in chain_data.columns else 0.20

    # Estimate skew from OTM puts vs ATM
    otm_puts = chain_data[
        (chain_data["option_type"] == "put") &
        (chain_data["moneyness"] < 0.95)
    ]
    if not otm_puts.empty and "iv" in chain_data.columns:
        avg_put_iv = float(otm_puts["iv"].mean())
        estimated_skew = (avg_put_iv - atm_iv) / max(atm_iv, 0.01)
        estimated_skew = np.clip(estimated_skew, 0.0, 0.5)
    else:
        estimated_skew = None

    return VolatilitySurface(
        atm_iv=atm_iv,
        vix_level=atm_iv * 100,
        skew_factor=estimated_skew,
    )
