"""Broker-neutral live execution order types.

These dataclasses are deliberately independent from Alpaca, IBKR, pydantic, and
any broker SDK. They are the only order representation accepted by the live
execution gateway.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any


class InternalTradeSide(str, Enum):
    """Broker-neutral trade side."""

    BUY = "buy"
    SELL = "sell"

    @classmethod
    def from_any(cls, value: Any) -> InternalTradeSide:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"BUY", "B"}:
                return cls.BUY
            if normalized in {"SELL", "S"}:
                return cls.SELL
        if value in (1, True):
            return cls.BUY
        if value in (-1, False):
            return cls.SELL
        raise ValueError(f"Unsupported trade side: {value!r}")


class InternalOrderType(str, Enum):
    """Broker-neutral order type."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

    @classmethod
    def from_any(cls, value: Any) -> InternalOrderType:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            mapping = {
                "mkt": cls.MARKET,
                "market": cls.MARKET,
                "lmt": cls.LIMIT,
                "limit": cls.LIMIT,
                "stp": cls.STOP,
                "stop": cls.STOP,
                "stop_loss": cls.STOP,
                "stop_limit": cls.STOP_LIMIT,
                "stp_lmt": cls.STOP_LIMIT,
            }
            if normalized in mapping:
                return mapping[normalized]
        raise ValueError(f"Unsupported order type: {value!r}")


@dataclass(frozen=True)
class InternalOrder:
    """Broker-neutral executable order.

    ``metadata`` carries broker-specific routing/context fields without making the
    core order broker-aware. Examples:
      - IBKR: exchange, currency, sec_type/security_type/asset_class, primary_exchange
      - Alpaca: fractional, extended_hours, client_order_id, order_class
    """

    id: str
    symbol: str
    side: InternalTradeSide
    order_type: InternalOrderType
    quantity: Decimal
    limit_price: Decimal | None = None
    stop_price: Decimal | None = None
    time_in_force: str = "DAY"
    client_order_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionReport:
    """Unified execution acknowledgement returned by broker adapters."""

    broker_name: str
    internal_order_id: str
    broker_order_id: str | None
    accepted: bool
    status: str = "accepted"
    message: str = ""
    submitted_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def rejected(
        cls,
        broker_name: str,
        internal_order_id: str,
        message: str,
        *,
        status: str = "rejected",
        metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionReport:
        return cls(
            broker_name=broker_name,
            internal_order_id=internal_order_id,
            broker_order_id=None,
            accepted=False,
            status=status,
            message=message,
            metadata=metadata or {},
        )
