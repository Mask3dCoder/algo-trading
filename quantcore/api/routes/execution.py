"""Execution API routes: OMS, algorithmic orders, market impact, TCA."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from typing import Any
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from quantcore.core.types import Fill, Side
from quantcore.execution.algos.twap import TWAPAlgo, TWAPConfig
from quantcore.execution.algos.vwap import VWAPAlgo, VWAPConfig
from quantcore.execution.impact import (
    predict_cost,
    sqrt_impact_bps,
)
from quantcore.execution.oms import OrderManager, limit_order, market_order
from quantcore.execution.tca import StrategySummary, TCAResult, tca_strategy, tca_trade

router = APIRouter(prefix="/execution", tags=["execution"])

# Shared OMS instance (in production, this would be per-session or persisted)
_oms = OrderManager()


# ---------------------------------------------------------------------------
# OMS
# ---------------------------------------------------------------------------


@router.get("/orders")
def api_list_orders() -> dict[str, Any]:
    """List all orders in the OMS."""
    orders = {}
    for oid, order in _oms.orders.items():
        orders[oid] = {
            "symbol": order.symbol,
            "side": order.side.value,
            "type": order.type.value,
            "quantity": float(order.quantity),
            "filled_qty": float(order.filled_qty),
            "status": order.status.value,
        }
    return {"orders": orders, "count": len(orders)}


@router.get("/orders/{order_id}")
def api_get_order(order_id: str) -> dict[str, Any]:
    """Get a single order by ID."""
    if order_id not in _oms.orders:
        raise HTTPException(404, f"Order {order_id} not found")
    order = _oms.orders[order_id]
    return {
        "id": order.id,
        "symbol": order.symbol,
        "side": order.side.value,
        "type": order.type.value,
        "quantity": float(order.quantity),
        "filled_qty": float(order.filled_qty),
        "limit_price": float(order.limit_price) if order.limit_price else None,
        "status": order.status.value,
    }


@router.post("/orders/market")
def api_place_market_order(
    symbol: str,
    side: str = Query(..., pattern="^(buy|sell)$"),
    quantity: float = Query(..., gt=0),
) -> dict[str, str]:
    """Place a market order."""
    s = Side.BUY if side == "buy" else Side.SELL
    order = market_order(symbol, s, Decimal(str(quantity)))
    _oms.submit(order)
    return {"order_id": order.id, "status": "submitted"}


@router.post("/orders/limit")
def api_place_limit_order(
    symbol: str,
    side: str = Query(..., pattern="^(buy|sell)$"),
    quantity: float = Query(..., gt=0),
    limit_price: float = Query(..., gt=0),
) -> dict[str, str]:
    """Place a limit order."""
    s = Side.BUY if side == "buy" else Side.SELL
    order = limit_order(symbol, s, Decimal(str(quantity)), Decimal(str(limit_price)))
    _oms.submit(order)
    return {"order_id": order.id, "status": "submitted"}


@router.post("/orders/{order_id}/cancel")
def api_cancel_order(order_id: str) -> dict[str, str]:
    """Cancel an open order."""
    try:
        _oms.cancel(order_id)
        return {"order_id": order_id, "status": "cancelled"}
    except Exception as e:
        raise HTTPException(400, str(e)) from e


# ---------------------------------------------------------------------------
# Market Impact
# ---------------------------------------------------------------------------


@router.post("/impact/estimate")
def api_impact_estimate(
    shares: float = Query(..., gt=0),
    adv: float = Query(..., gt=0),
    sigma: float = Query(..., gt=0),
    price: float | None = None,
) -> dict[str, Any]:
    """Estimate market impact cost."""
    cost_bps = sqrt_impact_bps(shares, adv, sigma)
    result = {
        "impact_bps": cost_bps,
        "shares": shares,
        "pct_adv": (shares / adv) * 100 if adv > 0 else 0,
    }
    if price is not None:
        result["impact_per_share"] = price * cost_bps / 10_000
        result["total_impact_dollars"] = shares * price * cost_bps / 10_000
    return result


@router.post("/impact/predict")
def api_predict_cost(
    shares: float,
    adv: float,
    sigma: float,
    spread: float = Query(0.0001, gt=0),
    price: float = Query(..., gt=0),
    urgency: str = Query("medium", pattern="^(low|medium|high)$"),
) -> dict[str, Any]:
    """Predict total trading cost."""
    result = predict_cost(shares, adv, sigma, spread, price, urgency=urgency)
    return {
        "total_bps": result.total_bps,
        "spread_cost_bps": result.spread_cost_bps,
        "impact_cost_bps": result.impact_cost_bps,
        "timing_cost_bps": result.timing_cost_bps,
        "total_dollars": result.total_dollars,
    }


# ---------------------------------------------------------------------------
# TCA
# ---------------------------------------------------------------------------


@router.post("/tca/trade")
def api_tca_trade(
    execution_prices: list[float],
    benchmark_price: float = Query(..., gt=0),
    side: str = Query(..., pattern="^(buy|sell)$"),
    quantities: list[float] | None = None,
) -> dict[str, Any]:
    """Run transaction cost analysis on a trade."""
    s = Side.BUY if side == "buy" else Side.SELL
    if quantities is None:
        quantities = [1.0] * len(execution_prices)
    fills = [
        Fill(
            order_id=str(uuid4()),
            price=Decimal(str(p)),
            quantity=Decimal(str(q)),
            commission=Decimal("0"),
        )
        for p, q in zip(execution_prices, quantities)
    ]
    result = tca_trade(fills, float(benchmark_price), s)
    return {
        "avg_execution_price": result.avg_fill_price,
        "benchmark_price": benchmark_price,
        "slippage_bps": result.total_cost_bps,
        "total_cost_bps": result.total_cost_bps,
    }


@router.post("/tca/strategy")
def api_tca_strategy(
    trades: list[dict[str, Any]],
    benchmark: str = Query("arrival", pattern="^(arrival|vwap|close)$"),
) -> dict[str, Any]:
    """Run TCA across a strategy's trades."""
    tca_results = [TCAResult(**t) for t in trades]
    result = tca_strategy(tca_results, strategy_name=benchmark)
    return asdict(result)


# ---------------------------------------------------------------------------
# Algos
# ---------------------------------------------------------------------------


@router.post("/algos/vwap/schedule")
def api_vwap_schedule(
    total_shares: float = Query(..., gt=0),
    symbol: str = "",
    window_minutes: int = Query(60, gt=0),
    num_slices: int = Query(12, gt=0, le=100),
) -> dict[str, Any]:
    """Compute a VWAP execution schedule."""
    algo = VWAPAlgo(VWAPConfig(window_minutes=window_minutes, num_slices=num_slices))
    slices = algo.compute_schedule(total_shares, symbol)
    return {
        "slices": [
            {
                "slice_id": s.slice_id,
                "target_time": s.target_time.isoformat(),
                "shares": s.shares,
                "volume_pct": s.volume_pct,
            }
            for s in slices
        ],
        "total_shares": total_shares,
    }


@router.post("/algos/twap/schedule")
def api_twap_schedule(
    total_shares: float = Query(..., gt=0),
    window_minutes: int = Query(60, gt=0),
    num_slices: int = Query(10, gt=0, le=100),
) -> dict[str, Any]:
    """Compute a TWAP execution schedule."""
    algo = TWAPAlgo(TWAPConfig(window_minutes=window_minutes, num_slices=num_slices))
    slices = algo.compute_schedule(total_shares)
    return {
        "slices": [
            {
                "slice_id": s.slice_id,
                "target_time": s.target_time.isoformat(),
                "shares": s.shares,
            }
            for s in slices
        ],
        "total_shares": total_shares,
    }
