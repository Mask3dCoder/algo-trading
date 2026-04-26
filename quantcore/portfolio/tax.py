"""Tax-lot tracking with FIFO/LIFO/specific-ID and tax-loss harvesting."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from decimal import Decimal


class LotMethod(str, Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    SPECIFIC = "specific"


TAX_LONG_TERM_DAYS = 365
WASH_SALE_WINDOW = 30  # Days


@dataclass
class TaxLot:
    """A single tax lot."""
    symbol: str
    purchase_date: datetime
    shares: Decimal
    cost_basis: Decimal  # Per-share cost
    total_cost: Decimal  # Total cost = shares × cost_basis
    lot_id: str = ""


@dataclass
class RealizedGain:
    symbol: str
    lots: list[TaxLot]
    shares: Decimal = Decimal("0")
    proceeds: Decimal = Decimal("0")  # Total sale proceeds
    cost: Decimal = Decimal("0")
    gain: Decimal = Decimal("0")
    holding_period_days: int = 0
    is_long_term: bool = False
    is_wash_sale: bool = False


class TaxLotTracker:
    """Track tax lots and compute gains/losses for tax optimization."""

    def __init__(self) -> None:
        self._lots: dict[str, list[TaxLot]] = {}
        self._realized: list[RealizedGain] = []
        self._wash_sale_disallowed: set[str] = set()
        self._sale_history: list[tuple[datetime, str, Decimal]] = []  # (date, symbol, shares)

    def add_lot(
        self,
        symbol: str,
        shares: Decimal,
        cost_per_share: Decimal,
        purchase_date: datetime | None = None,
        lot_id: str = "",
    ) -> TaxLot:
        """Record a new purchase lot."""
        if symbol not in self._lots:
            self._lots[symbol] = []

        lot = TaxLot(
            symbol=symbol,
            purchase_date=purchase_date or datetime.utcnow(),
            shares=shares,
            cost_basis=cost_per_share,
            total_cost=shares * cost_per_share,
            lot_id=lot_id or f"{symbol}-{len(self._lots[symbol])+1}",
        )

        self._lots[symbol].append(lot)
        return lot

    def realize(
        self,
        symbol: str,
        shares: Decimal,
        sale_price: Decimal,
        method: LotMethod = LotMethod.FIFO,
        sale_date: datetime | None = None,
    ) -> list[RealizedGain]:
        """Realize shares by selling from specific lots.

        Args:
            symbol: Ticker symbol.
            shares: Number of shares to sell.
            sale_price: Price per share.
            method: Lot selection method (FIFO, LIFO, specific).
            sale_date: Date of sale for holding period calculation.

        Returns:
            List of RealizedGain records.
        """
        if symbol not in self._lots or not self._lots[symbol]:
            return []

        remaining = shares
        results = []
        sale_dt = sale_date or datetime.utcnow()

        # Sort lots based on method
        if method == LotMethod.FIFO:
            lots = sorted(self._lots[symbol], key=lambda x: x.purchase_date)
        elif method == LotMethod.LIFO:
            lots = sorted(self._lots[symbol], key=lambda x: x.purchase_date, reverse=True)
        else:
            lots = self._lots[symbol]

        used_lots = []

        for lot in lots:
            if remaining <= 0:
                break

            take = min(remaining, lot.shares)
            if take <= 0:
                continue

            proceeds = take * sale_price
            cost = take * lot.cost_basis
            gain = proceeds - cost
            holding_days = (sale_dt - lot.purchase_date).days
            is_lt = holding_days >= TAX_LONG_TERM_DAYS

            # Wash sale check
            is_wash = self._check_wash_sale(symbol, sale_dt, gain)

            realized = RealizedGain(
                symbol=symbol,
                lots=[lot],
                shares=take,
                proceeds=proceeds,
                cost=cost,
                gain=gain,
                holding_period_days=holding_days,
                is_long_term=is_lt,
                is_wash_sale=is_wash,
            )
            results.append(realized)
            self._realized.append(realized)

            # Deduct from lot
            lot.shares -= take
            lot.total_cost -= cost
            if lot.shares <= 0:
                used_lots.append(lot)

            remaining -= take

        # Remove empty lots
        self._lots[symbol] = [l for l in self._lots[symbol] if l not in used_lots]

        # Record sale for wash-sale tracking
        self._sale_history.append((sale_dt, symbol, shares))

        return results

    def short_vs_long(
        self,
        lots: list[TaxLot],
        current_price: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """Report unrealized short-term vs long-term gains for a list of lots."""
        now = datetime.utcnow()
        st_gain = Decimal("0")
        lt_gain = Decimal("0")

        for lot in lots:
            gain = (current_price - lot.cost_basis) * lot.shares
            days = (now - lot.purchase_date).days
            if days >= TAX_LONG_TERM_DAYS:
                lt_gain += gain
            else:
                st_gain += gain

        return st_gain, lt_gain

    def harvest_candidates(
        self,
        threshold: float = -100.0,
        current_prices: dict[str, float] | None = None,
    ) -> list[TaxLot]:
        """Find lots with unrealized losses exceeding threshold.

        Excludes lots that would trigger wash sales.
        """
        candidates = []
        prices = current_prices or {}

        for symbol, lots in self._lots.items():
            current = Decimal(str(prices.get(symbol, 0)))
            for lot in lots:
                if current <= 0:
                    continue
                unrealized = (current - lot.cost_basis) * lot.shares
                if float(unrealized) < threshold:
                    if not self._check_wash_sale(symbol, datetime.utcnow(), unrealized):
                        candidates.append(lot)

        candidates.sort(key=lambda x: float((Decimal(str(current_prices.get(x.symbol, "0"))) - x.cost_basis) * x.shares))
        return candidates

    def _check_wash_sale(
        self,
        symbol: str,
        sale_date: datetime,
        gain: Decimal,
    ) -> bool:
        """Check if a sale would trigger wash sale rule.

        A wash sale occurs if you buy the same security within 30 days
        before or after selling at a loss.
        """
        if gain >= 0:
            return False  # Wash sale only applies to losses

        window_start = sale_date - timedelta(days=WASH_SALE_WINDOW)
        window_end = sale_date + timedelta(days=WASH_SALE_WINDOW)

        # Check other lots purchased within the wash sale window
        for lot in self._lots.get(symbol, []):
            if lot.shares > 0 and window_start <= lot.purchase_date <= window_end:
                return True

        for date, sym, shares in self._sale_history:
            if sym == symbol and window_start <= date <= window_end:
                return True

        return False

    def unrealized_gains(self, current_prices: dict[str, float]) -> dict[str, Decimal]:
        """Compute unrealized gains for all positions."""
        gains: dict[str, Decimal] = {}
        for symbol, lots in self._lots.items():
            price = Decimal(str(current_prices.get(symbol, 0)))
            total_gain = Decimal("0")
            for lot in lots:
                total_gain += (price - lot.cost_basis) * lot.shares
            gains[symbol] = total_gain
        return gains

    @property
    def total_realized_gains(self) -> Decimal:
        return sum((g.gain for g in self._realized), Decimal("0"))

    @property
    def short_term_realized(self) -> Decimal:
        return sum((g.gain for g in self._realized if not g.is_long_term), Decimal("0"))

    @property
    def long_term_realized(self) -> Decimal:
        return sum((g.gain for g in self._realized if g.is_long_term), Decimal("0"))
