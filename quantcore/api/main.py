"""FastAPI application entrypoint: API server with all routes, CORS, and health checks."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from quantcore.api.routes import (
    execution_router,
    portfolio_router,
    regime_router,
    risk_router,
)
from quantcore.core.health_state import get_health_state, set_kill_switch_active


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
async def dashboard():
    """Serve the live monitoring dashboard."""
    dashboard_path = Path(__file__).resolve().parent.parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        return FileResponse(str(dashboard_path), media_type="text/html")
    return {"error": "Dashboard not found", "docs": "/docs"}


@app.get("/api")
async def api_info() -> dict[str, Any]:
    """API root — returns basic info."""
    return {
        "name": "QuantCore API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "status": "/status",
        "dashboard": "/",
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint — uptime, data freshness, positions, kill switch, memory.

    Returns 200 with operational metrics from the shared health state singleton.
    Does not require authentication. Suitable for Kubernetes liveness/readiness probes.
    """
    return get_health_state()


@app.get("/status")
async def status() -> dict[str, Any]:
    """Trading status — positions, equity, recent events, circuit breaker state.

    Reads from the live runner's output files (reports/live_equity.csv,
    reports/live_events.jsonl) and the shared health state.
    """
    health = get_health_state()

    # Read latest equity
    equity_path = Path("reports/live_equity.csv")
    equity_data: dict[str, Any] = {}
    if equity_path.exists():
        try:
            import pandas as pd

            df = pd.read_csv(equity_path)
            if not df.empty:
                last = df.iloc[-1]
                equity_data = {
                    "date": str(last.get("date", "")),
                    "cash": float(last.get("cash", 0)),
                    "positions_value": float(last.get("positions_value", 0)),
                    "total_equity": float(last.get("total_equity", 0)),
                    "daily_return": float(last.get("ret", 0)),
                    "equity_history_length": len(df),
                }
        except Exception:
            pass

    # Read last 10 events
    events_path = Path("reports/live_events.jsonl")
    recent_events: list[dict] = []
    if events_path.exists():
        try:
            lines = events_path.read_text(encoding="utf-8").strip().split("\n")
            for line in lines[-10:]:
                if line.strip():
                    recent_events.append(json.loads(line))
        except Exception:
            pass

    # Kill switch file check
    kill_switch_file = Path("KILL_SWITCH").exists()

    return {
        **health,
        "equity": equity_data,
        "recent_events": recent_events,
        "kill_switch_file": kill_switch_file,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/api/strategies")
async def strategies_status() -> dict[str, Any]:
    """Per-strategy status: name, weight, equity, daily P&L, circuit breaker state.

    Reads from per-strategy equity CSV files written by the production runner.
    Returns a list of strategy objects with current metrics.
    """
    import pandas as pd

    strategies = []
    reports_dir = Path("reports")

    # Read per-strategy equity files
    for csv_file in sorted(reports_dir.glob("live_equity_*.csv")):
        strategy_name = csv_file.stem.replace("live_equity_", "")
        try:
            df = pd.read_csv(csv_file)
            if df.empty:
                continue
            last = df.iloc[-1]
            strategies.append({
                "name": strategy_name,
                "date": str(last.get("date", "")),
                "equity": float(last.get("equity", 0)),
                "daily_pnl": float(last.get("daily_pnl", 0)),
                "weight": float(last.get("weight", 0)),
                "status": str(last.get("status", "unknown")),
                "equity_history_length": len(df),
            })
        except Exception:
            strategies.append({
                "name": strategy_name,
                "status": "error",
                "error": "Could not read equity file",
            })

    # If no per-strategy files, read from config
    if not strategies:
        try:
            import yaml
            config_path = Path("config/live.yaml")
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                for strat_cfg in config.get("strategies", []):
                    strategies.append({
                        "name": strat_cfg["name"],
                        "weight": strat_cfg.get("weight", 0),
                        "status": "enabled" if strat_cfg.get("enabled", True) else "disabled",
                    })
        except Exception:
            pass

    return {
        "strategies": strategies,
        "count": len(strategies),
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/kill-switch/activate")
async def activate_kill_switch() -> dict[str, Any]:
    """Activate the kill switch — halts all trading immediately.

    Sets the in-memory kill switch flag (polled by the runner every 10s)
    and creates the KILL_SWITCH file as a belt-and-suspenders measure.
    """
    set_kill_switch_active(True)
    kill_file = Path("KILL_SWITCH")
    kill_file.write_text(
        f"Kill switch activated via API at {datetime.now(UTC).isoformat()}\n",
        encoding="utf-8",
    )
    return {
        "status": "activated",
        "message": "Kill switch is now ACTIVE. All trading halted.",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.post("/kill-switch/deactivate")
async def deactivate_kill_switch() -> dict[str, Any]:
    """Deactivate the kill switch — allows trading to resume.

    Requires explicit operator action. Removes the KILL_SWITCH file
    and clears the in-memory flag.
    """
    set_kill_switch_active(False)
    kill_file = Path("KILL_SWITCH")
    if kill_file.exists():
        kill_file.unlink()
    return {
        "status": "deactivated",
        "message": "Kill switch deactivated. Trading may resume on next cycle.",
        "timestamp": datetime.now(UTC).isoformat(),
    }
