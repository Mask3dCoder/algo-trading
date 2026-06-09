"""EconomicEngineCore: single canonical economic kernel for the entire system.

This module contains the ONLY authoritative implementation of position-level
economic logic in QuantCore. Every system that processes fills, updates
positions, or computes PnL MUST delegate to this kernel.

Canonical rules enforced by this kernel:
  Buy  + position qty >= 0  → blended avg cost (adding to long)
  Buy  + position qty < 0   → avg cost unchanged, realize PnL (covering short)
  Sell + position qty <= 0  → blended avg cost (adding to short)
  Sell + position qty > 0   → avg cost unchanged, realize PnL (reducing long)

Design:
  - Zero side effects. All functions return new state.
  - Fully deterministic. Same inputs → same outputs.
  - No I/O, no logging, no system dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


@dataclass(frozen=True)
class FillApplication:
    """Result of applying a single fill to a position.

    Attributes:
        position: New position state, or None if position is fully closed.
        cash_delta: Signed cash change from the fill (negative for buys,
            positive for sells). Does NOT include commission.
        realized_pnl: PnL realized by this fill (positive = gain, negative = loss).
    """
    position: Any | None  # Position-like dict or None
    cash_delta: float
    realized_pnl: float


def _new_position(
    symbol: str,
    quantity: float,
    avg_cost: float,
    market_price: float,
    realized_pnl: float = 0.0,
) -> dict[str, Any]:
    """Create a serializable Position-like dict.

    Using dict for zero-dependency kernel — callers convert to their own
    Position model as needed.
    """
    return {
        "symbol": symbol,
        "quantity": quantity,
        "avg_cost": avg_cost,
        "market_price": market_price,
        "realized_pnl": realized_pnl,
    }


def apply_fill(
    current_position: dict[str, Any] | None,
    *,
    fill_price: float,
    fill_quantity: float,
    is_buy: bool,
    symbol: str,
) -> FillApplication:
    """Apply a fill to a position using the canonical four-case logic.

    This is the SINGLE authoritative implementation of position-level
    economic math. All systems must call this function — none may
    reimplement avg_cost or PnL logic.

    Args:
        current_position: Current position dict with keys 'quantity',
            'avg_cost', 'realized_pnl'. None if no existing position.
        fill_price: Execution price of the fill.
        fill_quantity: Number of shares/contracts filled (always positive).
        is_buy: True for buy, False for sell.
        symbol: Symbol for the position being filled.

    Returns:
        FillApplication with new position state (or None if flat),
        cash delta, and realized PnL from this fill.
    """
    qty = float(fill_quantity)
    price = float(fill_price)

    if current_position is None:
        # New position
        init_qty = qty if is_buy else -qty
        cash_delta = -(qty * price) if is_buy else (qty * price)
        return FillApplication(
            position=_new_position(symbol, init_qty, price, price, 0.0),
            cash_delta=cash_delta,
            realized_pnl=0.0,
        )

    # Existing position
    current_qty = float(current_position["quantity"])
    current_avg = float(current_position["avg_cost"])
    current_realized = float(current_position.get("realized_pnl", 0.0))

    if is_buy:
        new_qty = current_qty + qty
        cash_delta = -(qty * price)
    else:
        new_qty = current_qty - qty
        cash_delta = qty * price

    realized = 0.0

    if is_buy:
        if current_qty >= 0:
            # Case 1: Adding to long — blended avg cost
            total_cost = current_avg * current_qty + price * qty
            new_avg = total_cost / new_qty if abs(new_qty) > 1e-10 else current_avg
        else:
            # Case 2: Covering short — avg cost unchanged, realize PnL
            cover_qty = min(qty, abs(current_qty))
            realized = (current_avg - price) * cover_qty
            new_avg = current_avg
    else:
        if current_qty <= 0:
            # Case 3: Adding to short — blended avg cost
            total_cost = current_avg * abs(current_qty) + price * qty
            new_qty_abs = abs(new_qty)
            new_avg = total_cost / new_qty_abs if new_qty_abs > 1e-10 else current_avg
        else:
            # Case 4: Reducing long — avg cost unchanged, realize PnL
            sell_qty = min(qty, current_qty)
            realized = (price - current_avg) * sell_qty
            new_avg = current_avg

    # Determine new position state
    if abs(new_qty) < 1e-10:
        new_position = None  # Position fully closed
    else:
        new_position = _new_position(
            symbol=symbol,
            quantity=new_qty,
            avg_cost=new_avg,
            market_price=price,
            realized_pnl=current_realized + realized,
        )

    return FillApplication(
        position=new_position,
        cash_delta=cash_delta,
        realized_pnl=realized,
    )


def fill_application_to_position(
    fa: FillApplication,
    position_model: type,
    symbol: str,
) -> Any | None:
    """Convert a FillApplication to a typed Position model, if needed.

    Args:
        fa: The FillApplication result.
        position_model: The Position model class (e.g., from quantcore.core.types).
        symbol: Symbol for the position.

    Returns:
        Position instance, or None if position is None in the result.
    """
    if fa.position is None:
        return None
    return position_model(
        symbol=symbol,
        quantity=Decimal(str(fa.position["quantity"])),
        avg_cost=Decimal(str(fa.position["avg_cost"])),
        market_price=Decimal(str(fa.position["market_price"])),
        realized_pnl=Decimal(str(fa.position["realized_pnl"])),
    )
