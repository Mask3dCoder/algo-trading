"""Optimal execution algorithms for institutional-grade trading.

Implements:
  - Almgren-Chriss optimal execution (minimize market impact + timing risk)
  - VWAP execution (participate proportional to volume)
  - TWAP execution (equal time slices)
  - Transaction cost analysis (TCA)
  - Smart order routing

References:
    - Almgren & Chriss (2000): Optimal Execution of Portfolio Transactions
    - Kissell & Glantz (2003): Optimal Trading Strategies
    - Cartea, Jaimungal, Penalva (2015): Algorithmic and High-Frequency Trading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class UrgencyLevel(Enum):
    """Execution urgency levels."""
    LOW = "low"  # Minimize impact, accept timing risk
    MEDIUM = "medium"  # Balanced
    HIGH = "high"  # Minimize timing risk, accept impact
    AGGRESSIVE = "aggressive"  # Immediate execution


@dataclass
class ExecutionSchedule:
    """Optimal execution schedule."""
    timestamps: list[pd.Timestamp]
    quantities: list[float]
    cumulative_quantities: list[float]
    expected_costs: list[float]
    total_cost: float
    market_impact: float
    timing_risk: float


@dataclass
class TransactionCostBreakdown:
    """Detailed transaction cost breakdown."""
    spread_cost: float  # Bid-ask spread cost
    market_impact: float  # Price impact
    timing_cost: float  # Delay cost
    opportunity_cost: float  # Missed trades
    commission: float  # Explicit costs
    total_cost: float  # Total implementation shortfall
    cost_bps: float  # Cost in basis points


@dataclass
class OrderSlice:
    """Single order slice for execution."""
    timestamp: pd.Timestamp
    quantity: float
    order_type: str  # 'market', 'limit', 'pegged'
    limit_price: float | None = None
    venue: str | None = None


# ---------------------------------------------------------------------------
# Almgren-Chriss Optimal Execution
# ---------------------------------------------------------------------------


class AlmgrenChrissOptimizer:
    """Almgren-Chriss optimal execution trajectory.
    
    Minimizes the trade-off between:
      - Market impact (temporary + permanent)
      - Timing risk (price volatility during execution)
    
    The optimal trajectory follows:
      X_t = X_0 * sinh(κ(T-t)) / sinh(κT)
    
    where κ depends on risk aversion and market impact parameters.
    
    Parameters
    ----------
    sigma : float
        Asset volatility (annualized, e.g., 0.20 = 20%).
    eta : float
        Temporary impact coefficient (default 2.5e-7).
    gamma : float
        Permanent impact coefficient (default 2.5e-7).
    tau : float
        Time horizon in days (default 1.0).
    risk_aversion : float
        Risk aversion parameter (default 1e-6).
        Higher = more urgency, lower = more patience.
    """

    def __init__(
        self,
        sigma: float,
        eta: float = 2.5e-7,
        gamma: float = 2.5e-7,
        tau: float = 1.0,
        risk_aversion: float = 1e-6,
    ) -> None:
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.tau = tau
        self.risk_aversion = risk_aversion

    def compute_optimal_trajectory(
        self,
        total_quantity: float,
        n_periods: int = 10,
        is_buy: bool = True,
    ) -> ExecutionSchedule:
        """Compute optimal execution trajectory.
        
        Parameters
        ----------
        total_quantity : float
            Total shares to execute.
        n_periods : int
            Number of execution periods.
        is_buy : bool
            True for buy, False for sell.
            
        Returns
        -------
        ExecutionSchedule
            Optimal execution schedule.
        """
        # Almgren-Chriss parameters
        kappa = np.sqrt(self.risk_aversion * self.sigma**2 / self.eta)

        # Time grid
        dt = self.tau / n_periods
        times = np.linspace(0, self.tau, n_periods + 1)

        # Optimal trajectory: X_t = X_0 * sinh(κ(T-t)) / sinh(κT)
        if kappa * self.tau > 0.1:
            # Use sinh formula for non-trivial kappa*T
            trajectory = total_quantity * np.sinh(kappa * (self.tau - times)) / np.sinh(kappa * self.tau)
        else:
            # Linear trajectory for very small risk aversion
            trajectory = total_quantity * (1 - times / self.tau)

        # Compute trade quantities (differences)
        quantities = -np.diff(trajectory)  # Negative because we're selling down the position
        if not is_buy:
            quantities = -quantities  # Flip for buys

        # Compute expected costs
        costs = []
        cumulative_qty = np.cumsum(quantities)

        for i in range(n_periods):
            # Temporary impact: η * (q_i / dt)
            temp_impact = self.eta * abs(quantities[i]) / dt

            # Permanent impact: γ * X_t
            perm_impact = self.gamma * abs(cumulative_qty[i])

            # Timing risk: σ * sqrt(t) * remaining
            remaining = total_quantity - cumulative_qty[i]
            timing_risk = self.sigma * np.sqrt(times[i + 1]) * abs(remaining)

            costs.append(temp_impact + perm_impact + timing_risk)

        # Market impact and timing risk decomposition
        total_market_impact = self.eta * np.sum(quantities**2) / dt
        total_timing_risk = self.sigma**2 * np.sum((total_quantity - cumulative_qty)**2) * dt

        timestamps = [pd.Timestamp.now() + pd.Timedelta(days=t) for t in times[1:]]

        return ExecutionSchedule(
            timestamps=timestamps,
            quantities=quantities.tolist(),
            cumulative_quantities=cumulative_qty.tolist(),
            expected_costs=costs,
            total_cost=float(np.sum(costs)),
            market_impact=float(total_market_impact),
            timing_risk=float(total_timing_risk),
        )

    def compute_trajectory_with_urgency(
        self,
        total_quantity: float,
        urgency: UrgencyLevel,
        n_periods: int = 10,
        is_buy: bool = True,
    ) -> ExecutionSchedule:
        """Compute trajectory adjusted for urgency level.
        
        Parameters
        ----------
        total_quantity : float
            Total shares to execute.
        urgency : UrgencyLevel
            Execution urgency.
        n_periods : int
            Number of periods.
        is_buy : bool
            Direction.
            
        Returns
        -------
        ExecutionSchedule
            Adjusted execution schedule.
        """
        # Adjust risk aversion based on urgency
        urgency_multipliers = {
            UrgencyLevel.LOW: 0.1,
            UrgencyLevel.MEDIUM: 1.0,
            UrgencyLevel.HIGH: 10.0,
            UrgencyLevel.AGGRESSIVE: 100.0,
        }

        adjusted_optimizer = AlmgrenChrissOptimizer(
            sigma=self.sigma,
            eta=self.eta,
            gamma=self.gamma,
            tau=self.tau,
            risk_aversion=self.risk_aversion * urgency_multipliers[urgency],
        )

        return adjusted_optimizer.compute_optimal_trajectory(
            total_quantity, n_periods, is_buy,
        )


# ---------------------------------------------------------------------------
# VWAP Execution
# ---------------------------------------------------------------------------


class VWAPExecutor:
    """Volume-Weighted Average Price execution.
    
    Participates proportional to historical volume profile
    to minimize VWAP slippage.
    
    Parameters
    ----------
    volume_profile : NDArray | None
        Historical volume profile (fraction of daily volume per period).
        If None, uses uniform profile.
    participation_rate : float
        Target participation rate (default 0.10 = 10% of volume).
    max_participation : float
        Maximum participation rate (default 0.25).
    """

    def __init__(
        self,
        volume_profile: NDArray | None = None,
        participation_rate: float = 0.10,
        max_participation: float = 0.25,
    ) -> None:
        self.volume_profile = volume_profile
        self.participation_rate = participation_rate
        self.max_participation = max_participation

    def generate_schedule(
        self,
        total_quantity: float,
        total_volume: float,
        n_periods: int = 390,  # Minutes in trading day
        is_buy: bool = True,
    ) -> list[OrderSlice]:
        """Generate VWAP execution schedule.
        
        Parameters
        ----------
        total_quantity : float
            Total shares to execute.
        total_volume : float
            Expected total volume for the day.
        n_periods : int
            Number of execution periods (default 390 = 1 per minute).
        is_buy : bool
            Direction.
            
        Returns
        -------
        list[OrderSlice]
            Execution slices.
        """
        # Get volume profile
        if self.volume_profile is not None:
            profile = self.volume_profile
        else:
            # U-shaped intraday volume profile
            profile = self._generate_default_profile(n_periods)

        # Ensure profile sums to 1
        profile = profile / profile.sum()

        # Generate slices
        slices = []
        base_time = pd.Timestamp.now().normalize() + pd.Timedelta(hours=9, minutes=30)

        for i in range(n_periods):
            # Volume in this period
            period_volume = total_volume * profile[i]

            # Target quantity (participation rate * volume)
            target_qty = period_volume * self.participation_rate

            # Cap at maximum participation
            max_qty = period_volume * self.max_participation
            target_qty = min(target_qty, max_qty)

            # Ensure we don't exceed total
            target_qty = min(target_qty, total_quantity - sum(s.quantity for s in slices))

            if target_qty > 0:
                timestamp = base_time + pd.Timedelta(minutes=i)
                slices.append(OrderSlice(
                    timestamp=timestamp,
                    quantity=target_qty if is_buy else -target_qty,
                    order_type="limit",
                    venue="primary",
                ))

        return slices

    def _generate_default_profile(self, n_periods: int) -> NDArray:
        """Generate default U-shaped intraday volume profile."""
        # Typical intraday volume pattern: high at open, low at lunch, high at close
        times = np.linspace(0, 1, n_periods)

        # U-shaped: high at start and end, low in middle
        profile = 1.0 - 0.5 * np.sin(np.pi * times)

        # Add extra volume at open and close
        profile[:n_periods // 10] *= 1.5
        profile[-n_periods // 10:] *= 1.5

        return profile

    def compute_slippage(
        self,
        executed_prices: NDArray,
        vwap_benchmark: float,
        executed_quantities: NDArray,
    ) -> float:
        """Compute slippage vs VWAP benchmark.
        
        Parameters
        ----------
        executed_prices : NDArray
            Actual execution prices.
        vwap_benchmark : float
            VWAP benchmark price.
        executed_quantities : NDArray
            Quantities executed at each price.
            
        Returns
        -------
        float
            Slippage in basis points.
        """
        # Implementation VWAP
        total_qty = np.sum(executed_quantities)
        if total_qty == 0:
            return 0.0

        implementation_vwap = np.sum(executed_prices * executed_quantities) / total_qty

        # Slippage in bps
        slippage_bps = (implementation_vwap - vwap_benchmark) / vwap_benchmark * 10000

        return float(slippage_bps)


# ---------------------------------------------------------------------------
# TWAP Execution
# ---------------------------------------------------------------------------


class TWAPExecutor:
    """Time-Weighted Average Price execution.
    
    Executes equal quantities over time, regardless of volume.
    Simple but may underperform in high-volume periods.
    
    Parameters
    ----------
    n_slices : int
        Number of execution slices (default 10).
    randomize : bool
        Add random noise to slice times (default True).
    randomization_pct : float
        Randomization percentage (default 0.10).
    """

    def __init__(
        self,
        n_slices: int = 10,
        randomize: bool = True,
        randomization_pct: float = 0.10,
    ) -> None:
        self.n_slices = n_slices
        self.randomize = randomize
        self.randomization_pct = randomization_pct

    def generate_schedule(
        self,
        total_quantity: float,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        is_buy: bool = True,
    ) -> list[OrderSlice]:
        """Generate TWAP execution schedule.
        
        Parameters
        ----------
        total_quantity : float
            Total shares to execute.
        start_time : pd.Timestamp
            Execution start time.
        end_time : pd.Timestamp
            Execution end time.
        is_buy : bool
            Direction.
            
        Returns
        -------
        list[OrderSlice]
            Execution slices.
        """
        # Equal quantity per slice
        qty_per_slice = total_quantity / self.n_slices

        # Time interval
        total_duration = (end_time - start_time).total_seconds()
        interval = total_duration / self.n_slices

        slices = []
        for i in range(self.n_slices):
            # Base timestamp
            timestamp = start_time + pd.Timedelta(seconds=i * interval)

            # Add randomization
            if self.randomize:
                random_offset = np.random.uniform(
                    -self.randomization_pct * interval,
                    self.randomization_pct * interval,
                )
                timestamp += pd.Timedelta(seconds=random_offset)

            slices.append(OrderSlice(
                timestamp=timestamp,
                quantity=qty_per_slice if is_buy else -qty_per_slice,
                order_type="market",
                venue="primary",
            ))

        return slices


# ---------------------------------------------------------------------------
# Transaction Cost Analysis
# ---------------------------------------------------------------------------


class TransactionCostAnalyzer:
    """Pre/post-trade transaction cost analysis.
    
    Estimates and analyzes:
      - Spread cost
      - Market impact (temporary + permanent)
      - Timing cost
      - Opportunity cost
      - Implementation shortfall
    
    Parameters
    ----------
    spread_bps : float
        Average bid-ask spread in bps (default 5.0).
    impact_coefficient : float
        Market impact coefficient (default 0.1).
    volatility : float
        Asset volatility (default 0.20).
    """

    def __init__(
        self,
        spread_bps: float = 5.0,
        impact_coefficient: float = 0.1,
        volatility: float = 0.20,
    ) -> None:
        self.spread_bps = spread_bps
        self.impact_coefficient = impact_coefficient
        self.volatility = volatility

    def pre_trade_cost(
        self,
        quantity: float,
        adv: float,
        price: float,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    ) -> TransactionCostBreakdown:
        """Estimate pre-trade transaction costs.
        
        Parameters
        ----------
        quantity : float
            Shares to trade.
        adv : float
            Average daily volume.
        price : float
            Current price.
        urgency : UrgencyLevel
            Execution urgency.
            
        Returns
        -------
        TransactionCostBreakdown
            Estimated cost breakdown.
        """
        notional = quantity * price

        # Spread cost (half spread for each side)
        spread_cost = notional * self.spread_bps / 10000 / 2

        # Market impact (square-root model)
        participation = quantity / adv
        market_impact = notional * self.impact_coefficient * np.sqrt(participation)

        # Timing cost (volatility risk)
        urgency_multipliers = {
            UrgencyLevel.LOW: 2.0,
            UrgencyLevel.MEDIUM: 1.0,
            UrgencyLevel.HIGH: 0.5,
            UrgencyLevel.AGGRESSIVE: 0.1,
        }
        execution_time = urgency_multipliers[urgency]  # Days
        timing_cost = notional * self.volatility * np.sqrt(execution_time / 252)

        # Opportunity cost (for patient execution)
        opportunity_cost = timing_cost * 0.5 if urgency == UrgencyLevel.LOW else 0.0

        # Commission (assume $0.005 per share)
        commission = quantity * 0.005

        total_cost = spread_cost + market_impact + timing_cost + opportunity_cost + commission
        cost_bps = total_cost / notional * 10000 if notional > 0 else 0.0

        return TransactionCostBreakdown(
            spread_cost=spread_cost,
            market_impact=market_impact,
            timing_cost=timing_cost,
            opportunity_cost=opportunity_cost,
            commission=commission,
            total_cost=total_cost,
            cost_bps=cost_bps,
        )

    def post_trade_cost(
        self,
        decision_price: float,
        execution_prices: NDArray,
        execution_quantities: NDArray,
        side: str = "buy",
    ) -> TransactionCostBreakdown:
        """Compute post-trade implementation shortfall.
        
        Parameters
        ----------
        decision_price : float
            Price at decision time.
        execution_prices : NDArray
            Actual execution prices.
        execution_quantities : NDArray
            Quantities executed at each price.
        side : str
            'buy' or 'sell'.
            
        Returns
        -------
        TransactionCostBreakdown
            Actual cost breakdown.
        """
        total_qty = np.sum(execution_quantities)
        if total_qty == 0:
            return TransactionCostBreakdown(0, 0, 0, 0, 0, 0, 0)

        # Implementation VWAP
        impl_vwap = np.sum(execution_prices * execution_quantities) / total_qty

        # Implementation shortfall
        if side == "buy":
            shortfall = (impl_vwap - decision_price) / decision_price
        else:
            shortfall = (decision_price - impl_vwap) / decision_price

        notional = total_qty * decision_price
        total_cost = shortfall * notional

        # Decompose (simplified)
        spread_cost = total_cost * 0.3  # Assume 30% is spread
        market_impact = total_cost * 0.5  # 50% is impact
        timing_cost = total_cost * 0.2  # 20% is timing

        cost_bps = shortfall * 10000

        return TransactionCostBreakdown(
            spread_cost=spread_cost,
            market_impact=market_impact,
            timing_cost=timing_cost,
            opportunity_cost=0.0,
            commission=0.0,
            total_cost=total_cost,
            cost_bps=cost_bps,
        )


# ---------------------------------------------------------------------------
# Smart Order Router
# ---------------------------------------------------------------------------


class SmartOrderRouter:
    """Intelligent order routing across venues.
    
    Routes orders to minimize costs and maximize fill rates
    by considering:
      - Venue fees/rebates
      - Fill probabilities
      - Adverse selection
      - Speed requirements
    
    Parameters
    ----------
    venues : list[str] | None
        Available venues (default: ['primary', 'dark_pool']).
    fee_schedule : dict[str, float] | None
        Fee per share for each venue.
    fill_probability : dict[str, float] | None
        Estimated fill probability per venue.
    """

    def __init__(
        self,
        venues: list[str] | None = None,
        fee_schedule: dict[str, float] | None = None,
        fill_probability: dict[str, float] | None = None,
    ) -> None:
        self.venues = venues or ["primary", "dark_pool"]
        self.fee_schedule = fee_schedule or {"primary": 0.003, "dark_pool": 0.001}
        self.fill_probability = fill_probability or {"primary": 0.95, "dark_pool": 0.60}

    def route_order(
        self,
        quantity: float,
        price: float,
        urgency: UrgencyLevel = UrgencyLevel.MEDIUM,
    ) -> list[OrderSlice]:
        """Route order across venues.
        
        Parameters
        ----------
        quantity : float
            Total shares to execute.
        price : float
            Limit price.
        urgency : UrgencyLevel
            Execution urgency.
            
        Returns
        -------
        list[OrderSlice]
            Routed order slices.
        """
        slices = []

        # Urgency-based venue allocation
        if urgency in [UrgencyLevel.HIGH, UrgencyLevel.AGGRESSIVE]:
            # Aggressive: mostly lit venues
            allocation = {"primary": 0.8, "dark_pool": 0.2}
        else:
            # Patient: more dark pool
            allocation = {"primary": 0.5, "dark_pool": 0.5}

        base_time = pd.Timestamp.now()

        for venue, fraction in allocation.items():
            venue_qty = quantity * fraction
            if venue_qty > 0:
                # Adjust order type based on venue
                if venue == "dark_pool":
                    order_type = "pegged"
                    limit_price = None  # Pegged to midpoint
                else:
                    order_type = "limit"
                    limit_price = price

                slices.append(OrderSlice(
                    timestamp=base_time,
                    quantity=venue_qty,
                    order_type=order_type,
                    limit_price=limit_price,
                    venue=venue,
                ))

        return slices

    def estimate_routing_cost(
        self,
        quantity: float,
        price: float,
    ) -> dict[str, float]:
        """Estimate routing costs per venue.
        
        Parameters
        ----------
        quantity : float
            Shares to execute.
        price : float
            Execution price.
            
        Returns
        -------
        dict[str, float]
            Cost per venue in dollars.
        """
        costs = {}
        for venue in self.venues:
            fee = self.fee_schedule.get(venue, 0.003)
            fill_prob = self.fill_probability.get(venue, 0.80)

            # Expected cost = fee * quantity * fill_probability
            expected_cost = fee * quantity * fill_prob
            costs[venue] = expected_cost

        return costs


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def create_execution_optimizer(
    method: str = "almgren_chriss",
    **kwargs,
) -> any:
    """Factory function to create execution optimizer.
    
    Parameters
    ----------
    method : str
        'almgren_chriss', 'vwap', 'twap', 'smart_router'.
    **kwargs
        Additional arguments.
        
    Returns
    -------
    Execution optimizer instance.
    """
    optimizers = {
        "almgren_chriss": AlmgrenChrissOptimizer,
        "vwap": VWAPExecutor,
        "twap": TWAPExecutor,
        "smart_router": SmartOrderRouter,
    }

    if method not in optimizers:
        raise ValueError(f"Unknown method: {method}. Choose from {list(optimizers.keys())}")

    return optimizers[method](**kwargs)


def compute_implementation_shortfall(
    decision_price: float,
    execution_prices: NDArray,
    execution_quantities: NDArray,
    side: str = "buy",
    paper_portfolio_return: float = 0.0,
) -> dict[str, float]:
    """Compute implementation shortfall decomposition.
    
    Parameters
    ----------
    decision_price : float
        Price at decision time.
    execution_prices : NDArray
        Actual execution prices.
    execution_quantities : NDArray
        Quantities executed.
    side : str
        'buy' or 'sell'.
    paper_portfolio_return : float
        Return of paper portfolio (for opportunity cost).
        
    Returns
    -------
    dict[str, float]
        Shortfall components in basis points.
    """
    total_qty = np.sum(execution_quantities)
    if total_qty == 0:
        return {"total": 0.0, "delay": 0.0, "trading": 0.0, "opportunity": 0.0}

    # Implementation VWAP
    impl_vwap = np.sum(execution_prices * execution_quantities) / total_qty

    # Total shortfall
    if side == "buy":
        total_shortfall = (impl_vwap - decision_price) / decision_price
    else:
        total_shortfall = (decision_price - impl_vwap) / decision_price

    # Trading cost (vs arrival price)
    arrival_price = execution_prices[0]
    if side == "buy":
        trading_cost = (impl_vwap - arrival_price) / arrival_price
    else:
        trading_cost = (arrival_price - impl_vwap) / arrival_price

    # Delay cost
    delay_cost = total_shortfall - trading_cost

    # Opportunity cost (for unfilled orders)
    opportunity_cost = paper_portfolio_return * 0.1  # Simplified

    return {
        "total": total_shortfall * 10000,  # bps
        "delay": delay_cost * 10000,
        "trading": trading_cost * 10000,
        "opportunity": opportunity_cost * 10000,
    }
