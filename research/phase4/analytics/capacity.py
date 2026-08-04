"""Phase 4 — Capacity & Cost Estimation (Stage 7 & 8).

Estimates real-world deployability of each factor:
  1. Capacity: Maximum AUM based on 5% ADV participation rate
  2. Cost: Theoretical round-trip cost (bps) × estimated annual turnover
  3. Minimum IC: The IC needed to survive after costs

Institutional standard: A factor is only deployable if its IC exceeds the
minimum IC required to cover trading costs at the target capacity.

Multi-horizon support:
  - Turnover scales inversely with holding_days: fewer rebalances per year
  - min_IC is recalculated per-horizon, allowing factors that fail at h=1
    to become viable at h=5 or h=21
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Cost Model Constants ───────────────────────────────────────────────────────
DEFAULT_ROUND_TRIP_BPS = 10.0    # Conservative round-trip (commission + spread + slippage)
PARTICIPATION_RATE     = 0.05    # 5% ADV — institutional max without excessive market impact
ANNUAL_TRADING_DAYS     = 252


@dataclass
class CapacityEstimate:
    """Per-factor, per-horizon capacity and cost estimate."""
    factor_name:       str
    holding_days:      int             # Rebalance holding period
    avg_adv_usd:       float           # Average (cross-section) daily dollar volume
    max_capacity_usd:  float           # Max deployable = PARTICIPATION_RATE × ADV
    avg_turnover_annual: float         # Annual portfolio turnover (fraction of AUM)
    round_trip_cost_bps: float        # Round-trip transaction cost (bps)
    annual_cost_bps:   float           # Total annual cost = turnover × round_trip
    min_ic_to_survive: float           # IC needed to cover annual costs
    viable:            bool            # True if factor |IC| > min_ic_to_survive


def estimate_avg_adv(volumes: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    """Compute average daily dollar volume per symbol over the full sample.

    Args:
        volumes: (date × symbol) raw daily volume.
        prices:  (date × symbol) adjusted close prices.

    Returns:
        pd.Series indexed by symbol, values = mean ADV in USD.
    """
    dollar_vol = volumes * prices
    return dollar_vol.mean()


def estimate_turnover(
    signal: pd.DataFrame,
    holding_days: int = 1,
    annualize: bool = True,
) -> float:
    """Estimate annualized portfolio turnover from signal rank changes.

    Turnover = Σ|w_new - w_old| / 2  per rebalance event.
    We measure turnover as the average cross-sectional sum of rank changes
    and convert to a full-portfolio turnover assuming we rebalance every
    `holding_days` trading days.

    Args:
        signal:        (date × symbol) cross-sectionally ranked signal [-0.5, +0.5]
        holding_days:  How many days each portfolio is held before rebalancing
        annualize:    If True, scale to full year; if False, return per-rebalance

    Returns:
        Estimated annual turnover as a fraction (e.g., 5.0 = 500% annual turnover).
    """
    if holding_days < 1:
        holding_days = 1

    diff = signal.diff().dropna()

    if holding_days > 1:
        # Sample the diffs every `holding_days` rows as a proxy for rebalance events
        # e.g., if holding_days=21, we only rebalance 252/21≈12 times/year
        sampled = diff.iloc[::holding_days]
        per_rebalance = sampled.abs().sum(axis=1).mean() / 2.0
        n_rebalances = ANNUAL_TRADING_DAYS / holding_days
        annual_turnover = per_rebalance * n_rebalances
    else:
        # Daily rebalancing: sum of |rank changes| across all dates
        per_rebalance = diff.abs().sum(axis=1).mean() / 2.0
        annual_turnover = per_rebalance * ANNUAL_TRADING_DAYS

    return float(annual_turnover)


def compute_capacity_report(
    factor_name: str,
    signal: pd.DataFrame,
    volumes: pd.DataFrame,
    prices: pd.DataFrame,
    ic_value: float = 0.0,
    holding_days: int = 1,
    round_trip_bps: float = DEFAULT_ROUND_TRIP_BPS,
    target_annual_vol: float = 0.15,
) -> CapacityEstimate:
    """Compute full capacity and cost estimate for a single factor at a given horizon.

    Args:
        factor_name:      Factor identifier.
        signal:            (date × symbol) signal panel.
        volumes:           (date × symbol) daily volume.
        prices:            (date × symbol) adjusted close.
        ic_value:          Factor's full-sample IC (for viability check).
        holding_days:      Number of days each portfolio is held before rebalancing.
                           This reduces the effective annual turnover.
        round_trip_bps:    Round-trip cost per full portfolio rebalance (bps).
        target_annual_vol: Target portfolio annual vol (15% institutional default).

    The holding_days effect:
      - h=1:  rebalance 252 times/year → max turnover
      - h=5:  rebalance 50 times/year  → 5× lower turnover
      - h=21: rebalance 12 times/year   → 21× lower turnover

    min_IC formula:
      min_IC = annual_cost_bps / (target_annual_vol × 10000)

    Since annual_cost_bps scales linearly with holding_days (fewer rebalances
    = fewer costs), min_IC decreases proportionally:

      h=21:  min_IC is ~21× lower than h=1
      h=5:   min_IC is ~5× lower than h=1

    This is the key rescue mechanism: a factor with IC=0.020 can be completely
    viable at h=21 even if it needs IC=0.50 at h=1.
    """
    avg_adv = estimate_avg_adv(volumes, prices)
    mean_adv = float(avg_adv.mean())
    max_cap = mean_adv * PARTICIPATION_RATE

    turnover_annual = estimate_turnover(signal, holding_days=holding_days)

    # Clip unrealistic turnovers (factor must contribute signal, not noise)
    turnover_annual = min(turnover_annual, 1000.0)

    annual_cost = turnover_annual * round_trip_bps

    # Minimum IC to survive: annual_cost_bps / (ann_vol × 10000)
    # annual_cost_bps = turnover × round_trip_bps
    # min_IC = turnover × 10bps / (0.15 × 10000) = turnover × 0.0067
    if target_annual_vol > 0:
        min_ic = annual_cost / (target_annual_vol * 10000)
    else:
        min_ic = float("inf")

    # Viable if |IC| > min_IC (both positive IC as alpha and negative as anti-alpha count)
    viable = not np.isnan(ic_value) and abs(ic_value) > min_ic

    log.debug(
        "  %s (h=%d): ADV=$%.0fM Cap=$%.0fM Turnover=%.1fx/yr Cost=%.1fbps "
        "MinIC=%.4f ActualIC=%.4f → %s",
        factor_name, holding_days, mean_adv / 1e6, max_cap / 1e6,
        turnover_annual, annual_cost, min_ic, ic_value,
        "VIABLE" if viable else "NOT VIABLE",
    )

    return CapacityEstimate(
        factor_name=factor_name,
        holding_days=holding_days,
        avg_adv_usd=mean_adv,
        max_capacity_usd=max_cap,
        avg_turnover_annual=turnover_annual,
        round_trip_cost_bps=round_trip_bps,
        annual_cost_bps=annual_cost,
        min_ic_to_survive=min_ic,
        viable=viable,
    )


def generate_capacity_report(
    all_estimates: dict[str, CapacityEstimate],
) -> str:
    """Generate CAPACITY_COST_REPORT.md as a markdown string."""
    lines: list[str] = []

    lines.append("# Capacity & Cost Report")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("- **Capacity:** 5% of Average Daily Dollar Volume (ADV)")
    lines.append("- **Turnover:** Estimated from signal rank changes, scaled by holding_days")
    lines.append("- **Round-trip cost:** 10 bps (conservative for large-cap equities)")
    lines.append("- **Min IC:** IC required to cover annual trading costs")
    lines.append("- **Target vol:** 15% annual (institutional vol-target standard)")
    lines.append("")
    lines.append(f"- **Holding days adjustment:** Turnover is scaled by 252/holding_days")
    lines.append(f"- **min_IC scaling:** min_IC ∝ 1/holding_days — longer horizons lower the hurdle")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Summary Table (per horizon)")
    lines.append("")

    lines.append("| Factor | h | ADV ($M) | Max Cap ($M) | Turnover (×/yr) | Ann Cost (bps) | Min IC | Viable |")
    lines.append("|--------|--|--------:|-------------:|----------------:|---------------:|-------:|--------|")

    viable_count = 0
    total_count  = len(all_estimates)

    for est in sorted(all_estimates.values(), key=lambda e: (e.factor_name, e.holding_days)):
        adv_m = est.avg_adv_usd / 1e6
        cap_m = est.max_capacity_usd / 1e6
        viable_str = "**VIABLE**" if est.viable else "NOT VIABLE"
        if est.viable:
            viable_count += 1
        lines.append(
            f"| {est.factor_name} | {est.holding_days} | {adv_m:.0f} | {cap_m:.0f} | "
            f"{est.avg_turnover_annual:.1f} | {est.annual_cost_bps:.1f} | "
            f"{est.min_ic_to_survive:.4f} | {viable_str} |"
        )

    lines.append("")
    lines.append(f"**{viable_count}/{total_count} factor-horizon combinations survive cost hurdle.**")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Detailed Analysis")
    lines.append("")

    for name, est in sorted(all_estimates.items(), key=lambda x: x[1].factor_name):
        lines.append(f"### {est.factor_name} (h={est.holding_days})")
        lines.append("")
        lines.append(f"- **Average ADV:** ${est.avg_adv_usd/1e6:.0f}M")
        lines.append(f"- **Max Capacity:** ${est.max_capacity_usd/1e6:.0f}M (5% participation)")
        lines.append(f"- **Annual Turnover:** {est.avg_turnover_annual:.1f}× ({est.avg_turnover_annual/ANNUAL_TRADING_DAYS:.2%} per day)")
        lines.append(f"- **Round-trip Cost:** {est.round_trip_cost_bps:.1f} bps")
        lines.append(f"- **Annual Cost:** {est.annual_cost_bps:.1f} bps")
        lines.append(f"- **Min IC to Survive:** {est.min_ic_to_survive:.4f}")
        lines.append(f"- **Verdict:** {'✓ VIABLE — IC exceeds cost hurdle' if est.viable else '✗ NOT VIABLE'}")
        lines.append("")

    return "\n".join(lines)