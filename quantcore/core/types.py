"""Core Pydantic data models for QuantCore."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class OHLCV(BaseModel):
    """Open-High-Low-Close-Volume bar."""
    symbol: str
    timestamp: datetime
    open: Decimal = Field(gt=0)
    high: Decimal = Field(gt=0)
    low: Decimal = Field(gt=0)
    close: Decimal = Field(gt=0)
    volume: int = Field(ge=0)
    adj_close: Decimal | None = None
    dividend: Decimal = Field(default=Decimal("0"), ge=0)
    split_factor: Decimal = Field(default=Decimal("1"), gt=0)

    @model_validator(mode="after")
    def validate_bar(self) -> Self:
        _TOL = Decimal("1e-10")
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) must be >= low ({self.low})")
        if self.high + _TOL < self.open:
            raise ValueError(f"high ({self.high}) must be >= open ({self.open})")
        if self.high + _TOL < self.close:
            raise ValueError(f"high ({self.high}) must be >= close ({self.close})")
        if self.low > self.open + _TOL:
            raise ValueError(f"low ({self.low}) must be <= open ({self.open})")
        if self.low > self.close + _TOL:
            raise ValueError(f"low ({self.low}) must be <= close ({self.close})")
        # Heal tiny ordering violations by rounding (float-to-Decimal artifacts)
        if self.high < self.close:
            try:
                object.__setattr__(self, "high", self.close)
            except AttributeError:
                pass
        return self

    model_config = {"frozen": True}


class Tick(BaseModel):
    """Trade tick with microsecond timestamp."""
    symbol: str
    timestamp_us: int = Field(ge=0)
    price: Decimal = Field(gt=0)
    size: int = Field(gt=0)
    exchange: str = ""

    model_config = {"frozen": True}


class Quote(BaseModel):
    """Level-1 bid/ask quote."""
    symbol: str
    timestamp: datetime
    bid: Decimal = Field(gt=0)
    ask: Decimal = Field(gt=0)
    bid_size: int = Field(gt=0)
    ask_size: int = Field(gt=0)
    exchange: str = ""

    @field_validator("ask")
    @classmethod
    def ask_gte_bid(cls, v: Decimal, info: Any) -> Decimal:
        if info.data.get("bid") and v < info.data["bid"]:
            raise ValueError("ask must be >= bid")
        return v

    model_config = {"frozen": True}


class Order(BaseModel):
    """Order with full lifecycle tracking."""
    id: str
    symbol: str
    side: Side
    type: OrderType
    quantity: Decimal = Field(gt=0)
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    status: OrderStatus = OrderStatus.PENDING
    strategy_id: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    filled_qty: Decimal = Field(default=Decimal("0"), ge=0)
    avg_fill_price: Decimal | None = None


class Fill(BaseModel):
    """Execution fill record."""
    order_id: str
    price: Decimal = Field(gt=0)
    quantity: Decimal = Field(gt=0)
    commission: Decimal = Field(ge=0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    exchange: str = ""
    side: Side | None = None  # Required for correct position accounting
    slippage_bps: Decimal = Field(default=Decimal("0"), ge=0)
    market_impact_cost: Decimal = Field(default=Decimal("0"), ge=0)
    impact_bps: Decimal = Field(default=Decimal("0"), ge=0)


class Position(BaseModel):
    """Current position in a symbol."""
    symbol: str
    quantity: Decimal = Field(default=Decimal("0"))
    avg_cost: Decimal = Field(gt=0)
    market_price: Decimal = Field(default=Decimal("0"), ge=0)
    unrealized_pnl: Decimal = Field(default=Decimal("0"))
    realized_pnl: Decimal = Field(default=Decimal("0"))

    @property
    def market_value(self) -> Decimal:
        return self.quantity * self.market_price


class Signal(BaseModel):
    """Trading signal from a strategy or indicator."""
    name: str
    symbol: str
    direction: Direction
    strength: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    notional_size: float | None = None


class PortfolioSnapshot(BaseModel):
    """Point-in-time portfolio state."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    positions: dict[str, Position] = Field(default_factory=dict)
    cash: Decimal = Field(default=Decimal("0"), ge=0)
    total_value: Decimal = Field(default=Decimal("0"), ge=0)
    returns: float = Field(default=0.0)
