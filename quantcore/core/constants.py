"""Trading constants and market parameters."""

from __future__ import annotations

import math

# Calendar
TRADING_DAYS_PER_YEAR: int = 252
TRADING_DAYS_PER_MONTH: int = 21
TRADING_DAYS_PER_WEEK: int = 5

# Annualization
ANNUALIZATION_SQRT: float = math.sqrt(TRADING_DAYS_PER_YEAR)  # ~15.87
ANNUALIZATION_FACTOR: float = TRADING_DAYS_PER_YEAR  # 252

# Market hours (Eastern Time)
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 30
MARKET_CLOSE_HOUR: int = 16
MARKET_CLOSE_MINUTE: int = 0

# Rates
RISK_FREE_RATE: float = 0.05
BENCHMARK_SYMBOL: str = "SPY"

# Risk
DEFAULT_VAR_CONFIDENCE: float = 0.95
DEFAULT_CVAR_CONFIDENCE: float = 0.975

# Options
MIN_OPTION_DTE: int = 1
DEFAULT_RISK_FREE_RATE: float = 0.05

# Numerical
PRICE_EPSILON: float = 1e-10
VOL_FLOOR: float = 1e-6
VOL_CEILING: float = 5.0
