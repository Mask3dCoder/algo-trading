"""ML signal pipeline: features, preprocessing, models, evaluation."""

from quantcore.signals.ml.evaluation import (
    ic_decay,
    information_coefficient,
    oos_degradation,
    quintile_analysis,
    rolling_ic,
    signal_turnover,
)
from quantcore.signals.ml.features import (
    CalendarFeatures,
    InterMarketFeatures,
    TechnicalFeatures,
    build_feature_matrix,
    cross_sectional_normalize,
)
from quantcore.signals.ml.linear_models import (
    ElasticNetSignal,
    LassoSignal,
    QuantileRegression,
    RidgeSignal,
)
from quantcore.signals.ml.preprocessing import (
    PreprocessingPipeline,
    TimeSeriesSplit,
    cross_sectional_zscore,
    impute_knn,
    impute_median,
    select_features_mutual_info,
    winsorize,
)
from quantcore.signals.ml.tree_models import (
    GradientBoostingSignal,
    RandomForestSignal,
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
