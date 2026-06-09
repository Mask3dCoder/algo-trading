"""Optimal execution algorithms for institutional-grade trading.

Implements:
  - Almgren-Chriss optimal execution
  - VWAP/TWAP execution
  - Transaction cost analysis
  - Smart order routing
"""

from quantcore.execution.optimal_execution import (
    AlmgrenChrissOptimizer,
    OrderSlice,
    SmartOrderRouter,
    TransactionCostAnalyzer,
    TransactionCostBreakdown,
    TWAPExecutor,
    UrgencyLevel,
    VWAPExecutor,
    compute_implementation_shortfall,
    create_execution_optimizer,
)

__all__ = [
    "AlmgrenChrissOptimizer",
    "VWAPExecutor",
    "TWAPExecutor",
    "TransactionCostAnalyzer",
    "SmartOrderRouter",
    "UrgencyLevel",
    "OrderSlice",
    "TransactionCostBreakdown",
    "create_execution_optimizer",
    "compute_implementation_shortfall",
]
