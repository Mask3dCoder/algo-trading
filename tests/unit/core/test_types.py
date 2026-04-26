"""Unit tests for core types."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from quantcore.core.types import (
    OHLCV,
    Direction,
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
    Signal,
    Tick,
)


class TestOHLCV:
    def test_valid_ohlcv(self) -> None:
        bar = OHLCV(
            symbol="SPY",
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            open=Decimal("450.00"),
            high=Decimal("452.50"),
            low=Decimal("449.00"),
            close=Decimal("451.00"),
            volume=100_000,
        )
        assert bar.symbol == "SPY"
        assert bar.close == Decimal("451.00")

    def test_negative_volume_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OHLCV(
                symbol="SPY",
                timestamp=datetime(2024, 1, 2, tzinfo=UTC),
                open=Decimal("100"),
                high=Decimal("110"),
                low=Decimal("90"),
                close=Decimal("105"),
                volume=-1,
            )

    def test_high_below_low_rejected(self) -> None:
        with pytest.raises(ValidationError):
            OHLCV(
                symbol="SPY",
                timestamp=datetime(2024, 1, 2, tzinfo=UTC),
                open=Decimal("100"),
                high=Decimal("90"),
                low=Decimal("110"),
                close=Decimal("95"),
                volume=1000,
            )

    def test_frozen(self) -> None:
        bar = OHLCV(
            symbol="SPY",
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=1000,
        )
        with pytest.raises(ValidationError):
            bar.volume = 2000

    def test_serialization_roundtrip(self) -> None:
        bar = OHLCV(
            symbol="AAPL",
            timestamp=datetime(2024, 6, 15, tzinfo=UTC),
            open=Decimal("195.00"),
            high=Decimal("197.50"),
            low=Decimal("194.20"),
            close=Decimal("196.80"),
            volume=50_000_000,
        )
        data = bar.model_dump()
        restored = OHLCV.model_validate(data)
        assert restored == bar


class TestTick:
    def test_valid_tick(self) -> None:
        tick = Tick(
            symbol="SPY",
            timestamp_us=1704067200000000,
            price=Decimal("450.50"),
            size=100,
        )
        assert tick.size == 100

    def test_zero_size_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Tick(
                symbol="SPY",
                timestamp_us=1704067200000000,
                price=Decimal("450.50"),
                size=0,
            )


class TestQuote:
    def test_valid_quote(self) -> None:
        q = Quote(
            symbol="SPY",
            timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            bid=Decimal("450.00"),
            ask=Decimal("450.10"),
            bid_size=500,
            ask_size=300,
        )
        assert q.ask > q.bid

    def test_ask_below_bid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Quote(
                symbol="SPY",
                timestamp=datetime(2024, 1, 2, tzinfo=UTC),
                bid=Decimal("450.10"),
                ask=Decimal("450.00"),
                bid_size=500,
                ask_size=300,
            )


class TestOrder:
    def test_valid_market_order(self) -> None:
        order = Order(
            id="ord-001",
            symbol="SPY",
            side=Side.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("100"),
        )
        assert order.status == OrderStatus.PENDING

    def test_fill_update(self) -> None:
        order = Order(
            id="ord-002",
            symbol="AAPL",
            side=Side.SELL,
            type=OrderType.LIMIT,
            quantity=Decimal("50"),
            limit_price=Decimal("200.00"),
        )
        updated = order.model_copy(update={
            "status": OrderStatus.FILLED,
            "filled_qty": Decimal("50"),
            "avg_fill_price": Decimal("199.95"),
        })
        assert updated.status == OrderStatus.FILLED


class TestFill:
    def test_valid_fill(self) -> None:
        fill = Fill(
            order_id="ord-001",
            price=Decimal("450.25"),
            quantity=Decimal("100"),
            commission=Decimal("0.50"),
        )
        assert fill.commission >= Decimal("0")


class TestSignal:
    def test_valid_signal(self) -> None:
        signal = Signal(
            name="momentum_12_1",
            symbol="SPY",
            direction=Direction.LONG,
            strength=0.8,
            confidence=0.6,
        )
        assert signal.direction == Direction.LONG

    def test_strength_bounds(self) -> None:
        with pytest.raises(ValidationError):
            Signal(
                name="test",
                symbol="SPY",
                direction=Direction.FLAT,
                strength=1.5,
                confidence=0.5,
            )


class TestPosition:
    def test_market_value(self) -> None:
        pos = Position(
            symbol="SPY",
            quantity=Decimal("100"),
            avg_cost=Decimal("450.00"),
            market_price=Decimal("455.00"),
        )
        assert pos.market_value == Decimal("45500")

    def test_unrealized_pnl(self) -> None:
        pos = Position(
            symbol="SPY",
            quantity=Decimal("100"),
            avg_cost=Decimal("450.00"),
            market_price=Decimal("455.00"),
            unrealized_pnl=Decimal("500.00"),
        )
        assert pos.unrealized_pnl == Decimal("500.00")
