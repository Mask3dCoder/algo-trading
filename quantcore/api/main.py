"""FastAPI application entrypoint: API server with all routes, CORS, and health checks."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quantcore.api.routes import (
    risk_router, portfolio_router, execution_router, regime_router,
)
from quantcore.core.health_state import get_health_state


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    print("[QuantCore API] Starting up...")
    yield
    print("[QuantCore API] Shutting down...")


app = FastAPI(
    title="QuantCore API",
    version="0.1.0",
    description="Institutional-grade algorithmic trading system API",
    lifespan=lifespan,
)

# CORS — permissive in dev, restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(risk_router)
app.include_router(portfolio_router)
app.include_router(execution_router)
app.include_router(regime_router)


@app.get("/")
async def root() -> dict[str, Any]:
    """API root — returns basic info."""
    return {
        "name": "QuantCore API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint — uptime, data freshness, positions, kill switch, memory.

    Returns 200 with operational metrics from the shared health state singleton.
    Does not require authentication. Suitable for Kubernetes liveness/readiness probes.
    """
    return get_health_state()
