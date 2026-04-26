"""ML signal pipeline: features, preprocessing, models, evaluation."""

from quantcore.signals.ml.features import (
    TechnicalFeatures, InterMarketFeatures, CalendarFeatures,
    cross_sectional_normalize, build_feature_matrix,
)
from quantcore.signals.ml.preprocessing import (
    winsorize, cross_sectional_zscore, impute_median, impute_knn,
    select_features_mutual_info, TimeSeriesSplit, PreprocessingPipeline,
)
from quantcore.signals.ml.linear_models import (
    LassoSignal, RidgeSignal, ElasticNetSignal, QuantileRegression,
)
from quantcore.signals.ml.tree_models import (
    GradientBoostingSignal, RandomForestSignal,
)
from quantcore.signals.ml.evaluation import (
    information_coefficient, rolling_ic, ic_decay,
    signal_turnover, oos_degradation, quintile_analysis,
)

__all__ = [
    # Features
    "TechnicalFeatures", "InterMarketFeatures", "CalendarFeatures",
    "cross_sectional_normalize", "build_feature_matrix",
    # Preprocessing
    "winsorize", "cross_sectional_zscore", "impute_median", "impute_knn",
    "select_features_mutual_info", "TimeSeriesSplit", "PreprocessingPipeline",
    # Linear models
    "LassoSignal", "RidgeSignal", "ElasticNetSignal", "QuantileRegression",
    # Tree models
    "GradientBoostingSignal", "RandomForestSignal",
    # Evaluation
    "information_coefficient", "rolling_ic", "ic_decay",
    "signal_turnover", "oos_degradation", "quintile_analysis",
]
