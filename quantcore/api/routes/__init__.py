"""REST API routes for all QuantCore modules."""

from quantcore.api.routes.execution import router as execution_router
from quantcore.api.routes.portfolio import router as portfolio_router
from quantcore.api.routes.regime import router as regime_router
from quantcore.api.routes.risk import router as risk_router

__all__ = ["execution_router", "portfolio_router", "regime_router", "risk_router"]
