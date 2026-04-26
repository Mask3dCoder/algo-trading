"""Order execution: OMS, algorithmic trading, market impact, TCA."""

from quantcore.execution.oms import (
    OrderState,
    TRANSITIONS,
    AuditEntry,
    OrderManager,
    market_order,
    limit_order,
    stop_order,
    bracket_order,
)
from quantcore.execution.impact import (
    CostEstimate,
    sqrt_impact,
    sqrt_impact_bps,
    linear_impact,
    estimate_kyle_lambda,
    amihud_illiquidity,
    roll_spread,
    roll_spread_bps,
    predict_cost,
    calibrate_impact,
)
from quantcore.execution.tca import (
    CostCategory,
    TCAResult,
    StrategySummary,
    tca_trade,
    tca_strategy,
    tca_report,
    tca_report_dataframe,
)
from quantcore.execution.algos.vwap import (
    VWAPConfig,
    VWAPSlice,
    VWAPResult,
    VWAPAlgo,
    DEFAULT_VOLUME_PROFILE,
    CRYPTO_VOLUME_PROFILE,
)
from quantcore.execution.algos.twap import (
    TWAPConfig,
    TWAPSlice,
    TWAPResult,
    TWAPAlgo,
)
from quantcore.execution.algos.pov import (
    POVConfig,
    POVSlice,
    POVResult,
    POVAlgo,
)
from quantcore.execution.algos.implementation_shortfall import (
    ISConfig,
    ISSchedule,
    compute_almgren_chriss,
    compute_trajectory,
    compute_trajectory_with_alpha,
)
from quantcore.execution.algos.iceberg import (
    IcebergConfig,
    IcebergSlice,
    IcebergResult,
    IcebergAlgo,
)
from quantcore.execution.algos.ml import (
    MLExecConfig,
    MLSlice,
    MLResult,
    MLAlgo,
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
