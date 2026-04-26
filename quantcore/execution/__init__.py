"""Order execution: OMS, algorithmic trading, market impact, TCA."""

from quantcore.execution.algos.iceberg import (
    IcebergAlgo,
    IcebergConfig,
    IcebergResult,
    IcebergSlice,
)
from quantcore.execution.algos.implementation_shortfall import (
    ISConfig,
    ISSchedule,
    compute_almgren_chriss,
    compute_trajectory,
    compute_trajectory_with_alpha,
)
from quantcore.execution.algos.ml import (
    MLAlgo,
    MLExecConfig,
    MLResult,
    MLSlice,
)
from quantcore.execution.algos.pov import (
    POVAlgo,
    POVConfig,
    POVResult,
    POVSlice,
)
from quantcore.execution.algos.twap import (
    TWAPAlgo,
    TWAPConfig,
    TWAPResult,
    TWAPSlice,
)
from quantcore.execution.algos.vwap import (
    CRYPTO_VOLUME_PROFILE,
    DEFAULT_VOLUME_PROFILE,
    VWAPAlgo,
    VWAPConfig,
    VWAPResult,
    VWAPSlice,
)
from quantcore.execution.impact import (
    CostEstimate,
    amihud_illiquidity,
    calibrate_impact,
    estimate_kyle_lambda,
    linear_impact,
    predict_cost,
    roll_spread,
    roll_spread_bps,
    sqrt_impact,
    sqrt_impact_bps,
)
from quantcore.execution.oms import (
    TRANSITIONS,
    AuditEntry,
    OrderManager,
    OrderState,
    bracket_order,
    limit_order,
    market_order,
    stop_order,
)
from quantcore.execution.tca import (
    CostCategory,
    StrategySummary,
    TCAResult,
    tca_report,
    tca_report_dataframe,
    tca_strategy,
    tca_trade,
)

__all__ = [
    # oms
    "OrderState", "TRANSITIONS", "AuditEntry", "OrderManager",
    "market_order", "limit_order", "stop_order", "bracket_order",
    # impact
    "CostEstimate", "sqrt_impact", "sqrt_impact_bps", "linear_impact",
    "estimate_kyle_lambda", "amihud_illiquidity", "roll_spread", "roll_spread_bps",
    "predict_cost", "calibrate_impact",
    # tca
    "CostCategory", "TCAResult", "StrategySummary", "tca_trade",
    "tca_strategy", "tca_report", "tca_report_dataframe",
    # vwap
    "VWAPConfig", "VWAPSlice", "VWAPResult", "VWAPAlgo",
    "DEFAULT_VOLUME_PROFILE", "CRYPTO_VOLUME_PROFILE",
    # twap
    "TWAPConfig", "TWAPSlice", "TWAPResult", "TWAPAlgo",
    # pov
    "POVConfig", "POVSlice", "POVResult", "POVAlgo",
    # implementation shortfall
    "ISConfig", "ISSchedule", "compute_almgren_chriss",
    "compute_trajectory", "compute_trajectory_with_alpha",
    # iceberg
    "IcebergConfig", "IcebergSlice", "IcebergResult", "IcebergAlgo",
    # ml
    "MLExecConfig", "MLSlice", "MLResult", "MLAlgo",
]
