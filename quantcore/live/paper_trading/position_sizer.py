"""Position Sizer — Capital-Allocated Sizing for Delta Neutral VRP.

Replaces the debugging 'single_share' mode with a mathematically scaled
allocation framework that respects:
  - Maximum premium allocation as a percentage of account equity
  - Standard option round-lot sizing (100 shares per contract)
  - Margin requirements (Reg T 50% initial for short options)
  - Portfolio-level position limits

Design Principles
-----------------
  1. Premium-based allocation: the strategy commits at most X% of equity to
     option premium per trade cycle (default 5%).
  2. Round-lot scaling: contract count always rounds DOWN to ensure the
     strategy cannot accidentally over-leverage.
  3. Margin-aware: short options require Reg T margin.  The sizer enforces
     that total margin exposure stays below a configurable limit.
  4. No look-ahead: all inputs are current-state only (equity, IV, Greeks).

Usage
-----
    from quantcore.live.paper_trading.position_sizer import PositionSizer

    sizer = PositionSizer(equity=100_000.0, config=sizing_config)
    contracts = sizer.size_option_entry(premium=3.50, delta=0.30)
    hedge_shares = sizer.size_delta_hedge(option_delta=-0.30, contracts=contracts)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionSizerConfig:
    """Immutable sizing configuration.

    These parameters govern how much capital is allocated per trade cycle.
    Adjustments require Investment Committee approval during paper trading.
    """

    # -- Capital allocation ----------------------------------------------------
    max_premium_allocation_pct: float = 0.05  # 5% of equity per trade cycle
    max_portfolio_premium_pct: float = 0.20   # 20% of equity across all open

    # -- Round-lot constraint --------------------------------------------------
    shares_per_contract: int = 100  # standard US equity option multiplier

    # -- Margin requirements (Reg T) -------------------------------------------
    # Short options: initial margin = premium + max(0.20 * underlying - OTM, 0.10 * underlying)
    # Simplified: we use a conservative estimate of 20% of notional per short leg
    initial_margin_pct: float = 0.20  # 20% of notional per short option
    max_margin_utilization_pct: float = 0.50  # never use > 50% of equity as margin

    # -- Delta hedge sizing ----------------------------------------------------
    delta_hedge_round_lot: int = 1  # hedge in whole shares (no fractional)

    # -- Hard limits -----------------------------------------------------------
    max_contracts_per_trade: int = 10  # cap per single entry
    max_total_contracts: int = 50      # cap across all open positions

    # -- Config source ---------------------------------------------------------
    config_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> PositionSizerConfig:
        """Load sizing parameters from config/paper_trading.yaml."""
        path = Path(path)
        if not path.exists():
            _log.warning("Config not found at %s — using defaults", path)
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        exec_cfg = data.get("execution", {})
        sizing_cfg = data.get("sizing", {})

        return cls(
            max_premium_allocation_pct=sizing_cfg.get(
                "max_premium_allocation_pct", 0.05,
            ),
            max_portfolio_premium_pct=sizing_cfg.get(
                "max_portfolio_premium_pct", 0.20,
            ),
            max_contracts_per_trade=sizing_cfg.get(
                "max_contracts_per_trade", 10,
            ),
            max_total_contracts=sizing_cfg.get(
                "max_total_contracts", 50,
            ),
            config_path=path,
        )


# ---------------------------------------------------------------------------
# Position Sizer
# ---------------------------------------------------------------------------


class PositionSizer:
    """Capital-allocated position sizing for delta-neutral VRP strategy.

    Reads the initial capital from config/paper_trading.yaml and applies a
    premium-based allocation rule with round-lot contract scaling.

    Parameters
    ----------
    equity : float
        Current account equity (updated at each cycle).
    config : PositionSizerConfig
        Immutable sizing configuration.
    open_positions : list[dict]
        Current open positions for portfolio-level limit checks.
    """

    def __init__(
        self,
        equity: float,
        config: PositionSizerConfig | None = None,
        open_positions: list[dict[str, Any]] | None = None,
    ) -> None:
        self._equity = equity
        self._cfg = config or PositionSizerConfig()
        self._open_positions = open_positions or []
        self._peak_equity = equity

        _log.info(
            "PositionSizer initialised: equity=$%.2f, "
            "max_premium_pct=%.1f%%, max_contracts=%d, "
            "shares_per_contract=%d",
            self._equity,
            self._cfg.max_premium_allocation_pct * 100,
            self._cfg.max_contracts_per_trade,
            self._cfg.shares_per_contract,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    def update_equity(self, equity: float) -> None:
        """Update current equity (called after each P&L cycle)."""
        self._equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def update_positions(self, positions: list[dict[str, Any]]) -> None:
        """Update the open position list for portfolio-level checks."""
        self._open_positions = positions

    def size_option_entry(
        self,
        premium: float,
        delta: float = 0.0,
        underlying_price: float = 0.0,
        position_type: str = "iron_condor",
    ) -> dict[str, Any]:
        """Compute contract quantity for an option entry.

        Applies three constraints (most restrictive wins):
          1. Premium allocation: contracts = floor(equity * max_pct / premium / 100)
          2. Portfolio limit: remaining capacity within portfolio_pct
          3. Margin requirement: contracts must not exceed margin utilization

        Parameters
        ----------
        premium : float
            Option premium per contract (mid-price).
        delta : float
            Option delta (for margin estimation). Default 0.0 — not used
            for sizing, only for margin estimation.
        underlying_price : float
            Underlying price for margin computation.
        position_type : str
            Type of position — "iron_condor", "short_strangle", etc.
            Affects margin calculation.

        Returns
        -------
        dict with keys:
            contracts : int       — number of contracts (rounded down)
            premium_cost : float   — total premium outlay
            margin_required : float — estimated Reg T margin
            sizing_basis : str     — which constraint was binding
            capped : bool          — True if a hard cap was hit
        """
        if premium <= 0:
            return self._zero_result("invalid_premium")

        # ── Constraint 1: Premium allocation per trade ──
        max_premium_spend = self._equity * self._cfg.max_premium_allocation_pct
        contracts_by_premium = max_premium_spend / (premium * self._cfg.shares_per_contract)

        # ── Constraint 2: Portfolio-level premium limit ──
        current_premium_exposure = self._compute_current_premium_exposure()
        remaining_portfolio = (
            self._equity * self._cfg.max_portfolio_premium_pct - current_premium_exposure
        )
        contracts_by_portfolio = (
            remaining_portfolio / (premium * self._cfg.shares_per_contract)
            if remaining_portfolio > 0 else 0.0
        )

        # ── Constraint 3: Margin requirement ──
        if underlying_price > 0:
            margin_per_contract = self._estimate_margin_per_contract(
                premium=premium,
                underlying_price=underlying_price,
                position_type=position_type,
            )
            current_margin = self._compute_current_margin_exposure()
            remaining_margin = (
                self._equity * self._cfg.max_margin_utilization_pct - current_margin
            )
            contracts_by_margin = (
                remaining_margin / margin_per_contract
                if margin_per_contract > 0 and remaining_margin > 0 else 0.0
            )
        else:
            # No underlying price → skip margin constraint
            contracts_by_margin = float("inf")
            margin_per_contract = 0.0

        # ── Choose the most restrictive constraint ──
        candidates = {
            "premium": contracts_by_premium,
            "portfolio": contracts_by_portfolio,
            "margin": contracts_by_margin,
        }
        binding_constraint = min(candidates, key=candidates.get)
        raw_contracts = candidates[binding_constraint]

        # ── Round DOWN to nearest whole contract ──
        contracts = max(0, int(math.floor(raw_contracts)))

        # ── Apply hard cap ──
        capped = False
        if contracts > self._cfg.max_contracts_per_trade:
            contracts = self._cfg.max_contracts_per_trade
            capped = True

        # ── Portfolio total contracts cap ──
        current_total = self._count_open_contracts()
        if contracts + current_total > self._cfg.max_total_contracts:
            contracts = max(0, self._cfg.max_total_contracts - current_total)
            capped = True
            if binding_constraint != "portfolio_cap":
                binding_constraint = "portfolio_cap"

        # ── Compute output metrics ──
        premium_cost = contracts * premium * self._cfg.shares_per_contract
        margin_required = contracts * margin_per_contract

        _log.info(
            "Sized option entry: contracts=%d premium_cost=$%.2f "
            "margin=$%.2f basis=%s capped=%s",
            contracts, premium_cost, margin_required,
            binding_constraint, capped,
        )

        return {
            "contracts": contracts,
            "premium_cost": round(premium_cost, 2),
            "margin_required": round(margin_required, 2),
            "sizing_basis": binding_constraint,
            "capped": capped,
            "constraints": {
                k: round(v, 2) for k, v in candidates.items()
            },
        }

    def size_delta_hedge(
        self,
        option_delta: float,
        contracts: int,
        target_net_delta: float = 0.0,
    ) -> dict[str, Any]:
        """Compute the number of hedge shares to achieve target net delta.

        Parameters
        ----------
        option_delta : float
            Aggregate net delta of all option positions (per share).
            For short options: delta is negative of long delta.
        contracts : int
            Number of option contracts.
        target_net_delta : float
            Desired net portfolio delta (default 0.0 for delta-neutral).

        Returns
        -------
        dict with keys:
            hedge_shares : int   — number of shares to trade
            direction : str       — "buy" or "sell"
            net_delta_after : float — expected net delta after hedge
        """
        # Total options delta (per-share delta × contracts × multiplier)
        total_options_delta = option_delta * contracts * self._cfg.shares_per_contract

        # Hedge shares needed to offset options delta
        # If options are net short delta (negative), buy shares to offset
        shares_needed = target_net_delta - total_options_delta

        # Round to whole shares (round toward 0 to avoid over-hedging)
        if shares_needed > 0:
            hedge_shares = int(math.floor(shares_needed))
            direction = "buy"
        elif shares_needed < 0:
            hedge_shares = -int(math.floor(abs(shares_needed)))
            direction = "sell"
        else:
            hedge_shares = 0
            direction = "hold"

        net_delta_after = total_options_delta + hedge_shares

        return {
            "hedge_shares": hedge_shares,
            "direction": direction,
            "net_delta_after": round(net_delta_after, 2),
            "total_options_delta": round(total_options_delta, 2),
            "target_net_delta": target_net_delta,
            "delta_drift": round(net_delta_after - target_net_delta, 2),
        }

    def get_summary(self) -> dict[str, Any]:
        """Return current sizing state for dashboard / reporting."""
        return {
            "equity": round(self._equity, 2),
            "peak_equity": round(self._peak_equity, 2),
            "current_premium_exposure": round(self._compute_current_premium_exposure(), 2),
            "current_margin_exposure": round(self._compute_current_margin_exposure(), 2),
            "open_contracts": self._count_open_contracts(),
            "max_premium_allocation_pct": self._cfg.max_premium_allocation_pct,
            "max_contracts_per_trade": self._cfg.max_contracts_per_trade,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _zero_result(self, reason: str) -> dict[str, Any]:
        """Return a zero-contract result with the reason as sizing_basis."""
        return {
            "contracts": 0,
            "premium_cost": 0.0,
            "margin_required": 0.0,
            "sizing_basis": reason,
            "capped": False,
            "constraints": {},
        }

    def _compute_current_premium_exposure(self) -> float:
        """Sum of premium paid/received across all open positions."""
        return sum(
            pos.get("premium_cost", 0.0)
            for pos in self._open_positions
        )

    def _compute_current_margin_exposure(self) -> float:
        """Sum of margin required across all open positions."""
        return sum(
            pos.get("margin_required", 0.0)
            for pos in self._open_positions
        )

    def _count_open_contracts(self) -> int:
        """Count total contracts across open positions."""
        return sum(
            pos.get("contracts", 0)
            for pos in self._open_positions
        )

    @property
    def margin_utilization_pct(self) -> float:
        """Current margin utilization as a fraction of equity (0.0–1.0)."""
        if self._equity <= 0:
            return 0.0
        return self._compute_current_margin_exposure() / self._equity

    def _estimate_margin_per_contract(
        self,
        premium: float,
        underlying_price: float,
        position_type: str,
    ) -> float:
        """Estimate Reg T initial margin per contract.

        For short options, Reg T requires:
          Short call/put: premium + max(
            20% of underlying - OTM amount,
            10% of underlying
          )

        For defined-risk spreads (iron condor), margin = width of spreads.
        We use a conservative estimate based on position type.
        """
        notional = underlying_price * self._cfg.shares_per_contract

        if position_type in ("iron_condor", "credit_spread", "defined_risk"):
            # Defined risk: margin ≈ max loss = width of the spread
            # Conservative: use 5% of notional as proxy for spread width
            return notional * 0.05
        else:
            # Undefined risk (short strangle, naked short):
            # Reg T: premium + max(20% * notional - OTM, 10% * notional)
            margin = premium * self._cfg.shares_per_contract + max(
                self._cfg.initial_margin_pct * notional,
                0.10 * notional,
            )
            return margin
