"""Stress testing framework: historical, hypothetical, factor, reverse."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from quantcore.core.types import PortfolioSnapshot, Position

# ---------------------------------------------------------------------------
# Scenario data models
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """A stress scenario: mapping of asset/risk-factor → shock."""
    name: str
    description: str = ""
    shocks: dict[str, float] = field(default_factory=dict)
    asset_class_shocks: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class StressResult:
    scenario_name: str
    scenario_description: str = ""
    initial_value: float = 0.0
    stressed_value: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    asset_pnl: dict[str, float] = field(default_factory=dict)
    new_greeks: dict[str, float] = field(default_factory=dict)
    is_worst_case: bool = False


# ---------------------------------------------------------------------------
# T-0306: Historical Scenario Library
# ---------------------------------------------------------------------------


def build_scenario_library() -> dict[str, Scenario]:
    """Return the built-in historical scenario library."""
    scenarios = {}

    # GFC 2008 (Sep-Nov)
    scenarios["GFC_2008"] = Scenario(
        name="GFC_2008",
        description="Global Financial Crisis, Sep-Nov 2008",
        asset_class_shocks={
            "equity": {"return": -0.40, "vol": 3.0},
            "fixed_income": {"return": -0.05, "spread": 0.05},
            "credit": {"spread": 5.0, "default_rate": 0.05},
            "volatility": {"level": 3.0, "vix": 60.0},
            "commodity": {"return": -0.30},
            "fx": {"em_currencies": -0.15},
        },
        shocks={"correlation_amplifier": 0.85},
    )

    # COVID 2020
    scenarios["COVID_2020"] = Scenario(
        name="COVID_2020",
        description="COVID-19 crash, Feb-Mar 2020",
        asset_class_shocks={
            "equity": {"return": -0.34, "vol": 4.0},
            "fixed_income": {"return": 0.03, "spread": 0.02},
            "volatility": {"level": 4.0, "vix": 82.0},
            "commodity": {"return": -0.25, "oil": -0.60},
            "fx": {"dollar": 0.05},
        },
        shocks={"correlation_amplifier": 0.70},
    )

    # Dot-com bust
    scenarios["DOTCOM_2000"] = Scenario(
        name="DOTCOM_2000",
        description="Dot-com bust, 2000-2002",
        asset_class_shocks={
            "equity": {"return": -0.50, "tech": -0.80, "vol": 2.0},
            "fixed_income": {"return": 0.10},
            "volatility": {"level": 2.0, "vix": 45.0},
        },
    )

    # Black Monday 1987
    scenarios["BLACK_MONDAY_1987"] = Scenario(
        name="BLACK_MONDAY_1987",
        description="Black Monday, October 19, 1987",
        asset_class_shocks={
            "equity": {"return": -0.22, "vol": 5.0},
            "volatility": {"level": 5.0, "vix": 150.0},
            "fixed_income": {"return": 0.02},
        },
    )

    # LTCM / Russia 1998
    scenarios["LTCM_1998"] = Scenario(
        name="LTCM_1998",
        description="Russian default / LTCM crisis, 1998",
        asset_class_shocks={
            "equity": {"return": -0.20, "em": -0.50, "vol": 2.5},
            "fixed_income": {"return": 0.02, "spread": 0.10},
            "credit": {"spread": 8.0},
            "commodity": {"return": -0.15},
            "fx": {"em_currencies": -0.25},
        },
    )

    # 2022 Rate Shock
    scenarios["RATE_SHOCK_2022"] = Scenario(
        name="RATE_SHOCK_2022",
        description="2022 Fed rate hiking cycle",
        asset_class_shocks={
            "equity": {"return": -0.25, "growth": -0.40, "vol": 1.5},
            "fixed_income": {"return": -0.20, "yield": 2.5},
            "volatility": {"level": 1.5, "vix": 35.0},
            "crypto": {"return": -0.70},
        },
    )

    return scenarios


_SCENARIO_LIBRARY: dict[str, Scenario] | None = None


def get_scenario_library() -> dict[str, Scenario]:
    """Get or create the scenario library (lazy singleton)."""
    global _SCENARIO_LIBRARY
    if _SCENARIO_LIBRARY is None:
        _SCENARIO_LIBRARY = build_scenario_library()
    return _SCENARIO_LIBRARY


# ---------------------------------------------------------------------------
# Scenario application engine
# ---------------------------------------------------------------------------


def _get_position_value(pos: Position) -> float:
    """Get market value of a position in float dollars."""
    return float(pos.market_value)


def _get_position_asset_class(symbol: str) -> str:
    """Map symbol to asset class."""
    symbol_upper = symbol.upper()
    if symbol_upper in ("SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"):
        return "equity"
    if symbol_upper in ("TLT", "IEF", "SHY", "AGG", "BND"):
        return "fixed_income"
    if symbol_upper in ("HYG", "LQD", "JNK"):
        return "credit"
    if symbol_upper in ("USO", "GLD", "SLV", "DBC"):
        return "commodity"
    if symbol_upper in ("VIX", "VXX", "UVXY"):
        return "volatility"
    if symbol_upper in ("EURUSD", "USDJPY", "GBPUSD"):
        return "fx"
    if symbol_upper in ("BTC", "ETH", "BTCUSD", "ETHUSD"):
        return "crypto"
    return "equity"


def apply_scenario(
    portfolio: PortfolioSnapshot,
    scenario: Scenario,
    greeks_fn: Any | None = None,
) -> StressResult:
    """Apply a stress scenario to a portfolio snapshot.

    Args:
        portfolio: Current portfolio state.
        scenario: The stress scenario to apply.
        greeks_fn: Optional function to compute Greeks.

    Returns:
        StressResult with P&L impact.
    """
    initial_value = 0.0
    stressed_value = 0.0
    asset_pnl: dict[str, float] = {}

    for symbol, pos in portfolio.positions.items():
        pos_val = _get_position_value(pos)
        initial_value += pos_val

        asset_class = _get_position_asset_class(symbol)

        # Look up the shock for this asset class
        shock_pct = 0.0
        if asset_class in scenario.asset_class_shocks:
            shock_pct = scenario.asset_class_shocks[asset_class].get("return", 0.0)
        elif symbol.upper() in scenario.shocks:
            shock_pct = scenario.shocks[symbol.upper()]

        # Apply correlation amplification
        corr_amp = scenario.shocks.get("correlation_amplifier", 0.0)
        if corr_amp > 0:
            # Amplify individual shocks under correlation stress
            shock_pct = shock_pct * (1.0 + corr_amp * 0.5)

        stressed_val = pos_val * (1.0 + shock_pct)
        stressed_value += stressed_val
        asset_pnl[symbol] = stressed_val - pos_val

    if initial_value == 0:
        pnl_pct = 0.0
    else:
        pnl_pct = (stressed_value - initial_value) / initial_value

    return StressResult(
        scenario_name=scenario.name,
        scenario_description=scenario.description,
        initial_value=float(initial_value),
        stressed_value=float(stressed_value),
        pnl=float(stressed_value - initial_value),
        pnl_pct=float(pnl_pct),
        asset_pnl=asset_pnl,
    )


def stress_report(
    portfolio: PortfolioSnapshot,
    scenarios: list[str] | None = None,
) -> list[StressResult]:
    """Run all (or selected) scenarios and return results as a list."""
    if scenarios is None:
        scenarios = list(get_scenario_library().keys())

    lib = get_scenario_library()
    results = []
    for name in scenarios:
        sc = lib.get(name)
        if sc is None:
            continue
        result = apply_scenario(portfolio, sc)
        results.append(result)

    results.sort(key=lambda r: r.pnl_pct)
    if results:
        results[0].is_worst_case = True

    return results


def stress_report_dataframe(
    portfolio: PortfolioSnapshot,
    scenarios: list[str] | None = None,
) -> dict[str, list]:
    """Run stress test and return data as dict (easily convertible to DataFrame)."""
    results = stress_report(portfolio, scenarios)
    return {
        "scenario": [r.scenario_name for r in results],
        "pnl": [r.pnl for r in results],
        "pnl_pct": [r.pnl_pct for r in results],
        "initial_value": [r.initial_value for r in results],
        "stressed_value": [r.stressed_value for r in results],
        "is_worst_case": [r.is_worst_case for r in results],
    }


# ---------------------------------------------------------------------------
# T-0307: Hypothetical + Factor Stress Tests
# ---------------------------------------------------------------------------


def custom_scenario(
    name: str,
    shocks: dict[str, float],
    description: str = "",
) -> Scenario:
    """Create a user-defined hypothetical scenario.

    Example:
        custom_scenario("VIX Spike", {"volatility": 0.30, "equity": -0.10})
    """
    return Scenario(
        name=name,
        description=description or f"Custom stress: {shocks}",
        asset_class_shocks={
            asset_class: {"return": shock} for asset_class, shock in shocks.items()
        },
    )


def correlation_stress(
    portfolio: PortfolioSnapshot,
    target_corr: float = 0.80,
) -> StressResult:
    """Stress test where all pairwise correlations rise to target_corr.

    Under correlation → 1 stress, diversification benefit collapses.
    """
    total_value = float(portfolio.total_value)
    if total_value == 0:
        return StressResult(
            scenario_name="Correlation Spike",
            scenario_description=f"All pairwise correlations → {target_corr}",
            initial_value=0.0, stressed_value=0.0, pnl=0.0, pnl_pct=0.0,
        )

    # Correlation stress: amplified shock via position value concentration
    # When correlations → 1, effective positions amplify each other
    n_pos = len(portfolio.positions)
    if n_pos <= 1:
        return StressResult(
            scenario_name="Correlation Spike",
            scenario_description=f"All pairwise correlations → {target_corr}",
            initial_value=total_value, stressed_value=total_value,
            pnl=0.0, pnl_pct=0.0,
        )

    # Under perfect correlation, diversification benefit disappears
    # Stress = -(1 - 1/√n) * correlation_amplifier as loss multiplier
    concentration_factor = 1.0 - 1.0 / np.sqrt(n_pos)
    loss_pct = -concentration_factor * target_corr * 0.10  # 10% base stress

    initial_value = 0.0
    stressed_value = 0.0
    asset_pnl: dict[str, float] = {}
    for symbol, pos in portfolio.positions.items():
        pos_val = float(pos.market_value)
        initial_value += pos_val
        stressed_val = pos_val * (1.0 + loss_pct)
        stressed_value += stressed_val
        asset_pnl[symbol] = stressed_val - pos_val

    return StressResult(
        scenario_name="Correlation Spike",
        scenario_description=f"All pairwise correlations → {target_corr}",
        initial_value=float(initial_value),
        stressed_value=float(stressed_value),
        pnl=float(stressed_value - initial_value),
        pnl_pct=float(loss_pct),
        asset_pnl=asset_pnl,
    )


def factor_stress(
    portfolio: PortfolioSnapshot,
    factor_exposures: dict[str, float],
    factor_shocks: dict[str, float],
) -> StressResult:
    """Apply factor-level shocks via portfolio factor exposures.

    portfolio_pnl = Σ factor_exposure × factor_shock

    Args:
        factor_exposures: {factor_name: portfolio_beta_to_factor}
        factor_shocks: {factor_name: shock_in_std_devs}
    """
    total_shock = 0.0
    for factor, exposure in factor_exposures.items():
        shock = factor_shocks.get(factor, 0.0)
        total_shock += exposure * shock

    initial_value = float(portfolio.total_value)
    stressed_value = initial_value * (1.0 + total_shock)

    return StressResult(
        scenario_name="Factor Stress",
        scenario_description=f"Factor shocks: {factor_shocks}",
        initial_value=initial_value,
        stressed_value=stressed_value,
        pnl=stressed_value - initial_value,
        pnl_pct=float(total_shock),
    )


def liquidity_stress(
    portfolio: PortfolioSnapshot,
    spread_multiplier: float = 5.0,
) -> StressResult:
    """Stress test: widen all bid-ask spreads by spread_multiplier.

    Loss ≈ Σ position_size × (multiplier - 1) × normal_spread / 2
    """
    # Simplified: assume ~10bps normal spread, loss = half-spread × multiplier
    normal_half_spread = 0.0005  # 5bps half-spread is typical
    extra_cost = normal_half_spread * (spread_multiplier - 1.0)

    initial_value = 0.0
    total_extra = 0.0
    for symbol, pos in portfolio.positions.items():
        pos_val = float(pos.market_value)
        initial_value += pos_val
        total_extra += abs(pos_val) * extra_cost

    stressed_value = initial_value - total_extra
    pnl = stressed_value - initial_value

    return StressResult(
        scenario_name="Liquidity Crisis",
        scenario_description=f"Spreads {spread_multiplier}× normal",
        initial_value=float(initial_value),
        stressed_value=float(stressed_value),
        pnl=float(pnl),
        pnl_pct=float(pnl / initial_value) if initial_value > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# T-0308: Reverse Stress Testing
# ---------------------------------------------------------------------------


def reverse_stress(
    portfolio: PortfolioSnapshot,
    loss_threshold: float = -0.20,
    n_results: int = 5,
) -> list[Scenario]:
    """Find minimal scenarios that cause portfolio loss ≥ |loss_threshold|.

    Optimization: minimize Σ|shock_i|² subject to portfolio loss ≤ threshold.
    Uses scipy.optimize.minimize with L-BFGS-B.

    Returns top-n most plausible (minimal shock magnitude) scenarios.
    """
    # Identify the positions
    symbols = list(portfolio.positions.keys())
    n_positions = len(symbols)

    if n_positions == 0:
        return []

    # Position values
    values = np.array([
        float(portfolio.positions[s].market_value) for s in symbols
    ])
    total_val = float(np.sum(np.abs(values)))

    if total_val == 0:
        return []

    # Optimization: find minimal shocks that achieve the loss
    def objective(shocks: NDArray) -> float:
        return float(np.sum(shocks**2))

    def constraint(shocks: NDArray) -> float:
        # Loss = Σ position_value_i × shock_i
        loss = float(np.sum(values * shocks))
        return loss_threshold * total_val - loss  # ≥ 0 means loss ≤ threshold

    constraints = [{"type": "ineq", "fun": constraint}]

    # Bounds: shocks between -1.0 and 0 (loss side)
    bounds = [(-1.0, 0.0)] * n_positions

    # Try multiple starting points and collect results
    n_starts = min(n_results * 2, 10)
    results = []
    rng = np.random.default_rng(42)

    for _ in range(n_starts):
        x0 = rng.uniform(-0.3, 0.0, n_positions)
        try:
            res = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-8},
            )
            if res.success and constraint(res.x) >= -1e-6:
                shock_magnitude = np.sqrt(float(np.sum(res.x**2)))
                results.append((res.x.copy(), shock_magnitude))
        except Exception:
            continue

    # Sort by magnitude (most plausible first), deduplicate
    results.sort(key=lambda x: x[1])
    seen = set()
    scenarios = []
    for shocks, magnitude in results:
        key = tuple(round(float(s), 4) for s in shocks)
        if key in seen:
            continue
        seen.add(key)

        shock_dict = {symbols[i]: float(shocks[i]) for i in range(n_positions) if abs(float(shocks[i])) > 0.001}
        sc = Scenario(
            name=f"Reverse Stress #{len(scenarios)+1}",
            description=f"Shock magnitude={magnitude:.4f}, portfolio loss≥{-loss_threshold:.1%}",
            shocks=shock_dict,
        )
        scenarios.append(sc)
        if len(scenarios) >= n_results:
            break

    return scenarios


def concentration_report(
    reverse_results: list[Scenario],
) -> str:
    """Identify concentration risk from reverse stress results.

    Reports which positions/factors appear most frequently in the most plausible
    adverse scenarios.
    """
    if not reverse_results:
        return "No reverse stress results to analyze."

    # Count symbol appearances in top scenarios
    symbol_counts: dict[str, int] = {}
    symbol_magnitudes: dict[str, list[float]] = {}
    for sc in reverse_results:
        for sym, shock in sc.shocks.items():
            symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
            if sym not in symbol_magnitudes:
                symbol_magnitudes[sym] = []
            symbol_magnitudes[sym].append(abs(shock))

    # Build report
    lines = ["=== Concentration Risk Report ==="]
    lines.append(f"Top {len(reverse_results)} reverse stress scenarios analyzed.")
    lines.append("")
    lines.append("Most vulnerable positions (by frequency in scenarios):")
    for sym, count in sorted(symbol_counts.items(), key=lambda x: -x[1]):
        avg_mag = np.mean(symbol_magnitudes[sym]) if symbol_magnitudes[sym] else 0.0
        lines.append(f"  {sym}: appears in {count}/{len(reverse_results)} scenarios, avg shock magnitude={avg_mag:.2%}")

    return "\n".join(lines)
