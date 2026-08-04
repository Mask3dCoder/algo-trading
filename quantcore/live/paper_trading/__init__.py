"""Delta Neutral VRP — Paper Trading Environment.

Three-module architecture for the Investment Committee's paper trading
validation program, with hardcoded risk guardrails.

Modules
-------
risk_controller      : RiskController with mandatory kill switches
execution_environment : PaperExecutionEnvironment with delta tracking
analytics_dashboard  : PaperTradingAnalytics with drift monitoring
runner                : PaperTradingRunner — full daily loop entry point

Active Execution Suite (TCA + Adaptive Hedging + Live Dashboard)
-------------------------------------------------------------
tca_engine            : TCAEngine — slippage tracking, partial fills, CSV logging
adaptive_hedger        : AdaptiveHedger — volatility-adjusted delta thresholds
terminal_dashboard    : TerminalDashboard — real-time ANSI/TUI portfolio monitor

Usage
-----
    python -m quantcore.live.paper_trading.runner    # CLI
    from quantcore.live.paper_trading.runner import run_paper_trading

    # TCA + Dashboard integration:
    from quantcore.live.paper_trading.tca_engine import TCAEngine, TCAConfig
    from quantcore.live.paper_trading.terminal_dashboard import TerminalDashboard
    from quantcore.live.paper_trading.adaptive_hedger import AdaptiveHedger, AdaptiveHedgerConfig
"""

from .risk_controller import ExecutionState, RiskController, RiskControllerConfig
from .execution_environment import PaperExecutionEnvironment, PaperExecutionConfig
from .analytics_dashboard import AnalyticsConfig, PaperTradingAnalytics
from .position_sizer import PositionSizer, PositionSizerConfig
from .delta_telemetry import DeltaTelemetry, DeltaTelemetryConfig, DeltaTelemetryMiddleware
from .live_connector import (
    LiveStreamConnector,
    PassiveExecutionProxy,
    SilentRunOrchestrator,
    Tick,
)
from .connection_monitor import ConnectionMonitor, ConnectionState, FeedHealth
from .runner import PaperTradingConfig, PaperTradingRunner, run_paper_trading

# Active execution suite
from .tca_engine import TCAConfig, TCAEngine, FillMetrics, TCALogger, TCASnapshot
from .adaptive_hedger import (
    AdaptiveHedger,
    AdaptiveHedgerConfig,
    HedgeAction,
    HedgeSignal,
)
from .terminal_dashboard import (
    DashboardSnapshot,
    GreeksState,
    PortfolioState,
    RiskState,
    SystemState,
    TCAState,
    TerminalDashboard,
)

__all__ = [
    # Core modules
    "RiskController",
    "RiskControllerConfig",
    "PaperExecutionEnvironment",
    "PaperExecutionConfig",
    "PaperTradingAnalytics",
    "AnalyticsConfig",
    # Position sizing
    "PositionSizer",
    "PositionSizerConfig",
    # Delta telemetry
    "DeltaTelemetry",
    "DeltaTelemetryConfig",
    "DeltaTelemetryMiddleware",
    # Live streaming
    "LiveStreamConnector",
    "PassiveExecutionProxy",
    "SilentRunOrchestrator",
    "Tick",
    # Connection monitoring
    "ConnectionMonitor",
    "ConnectionState",
    "FeedHealth",
    # Runner
    "PaperTradingRunner",
    "PaperTradingConfig",
    "run_paper_trading",
    # State
    "ExecutionState",
    # TCA Engine
    "TCAEngine",
    "TCAConfig",
    "FillMetrics",
    "TCALogger",
    "TCASnapshot",
    # Adaptive Hedger
    "AdaptiveHedger",
    "AdaptiveHedgerConfig",
    "HedgeAction",
    "HedgeSignal",
    # Terminal Dashboard
    "TerminalDashboard",
    "DashboardSnapshot",
    "GreeksState",
    "PortfolioState",
    "RiskState",
    "SystemState",
    "TCAState",  # The Dashboard TCAState (dashboard snapshot of TCA metrics)
]
