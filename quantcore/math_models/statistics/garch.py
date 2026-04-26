"""GARCH family models: fitting, forecasting, and comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from arch import arch_model
from numpy.typing import NDArray


@dataclass
class GARCHResult:
    """Fitted GARCH model result."""
    model_type: str
    params: dict[str, float]
    aic: float
    bic: float
    loglikelihood: float
    conditional_volatility: NDArray
    model: Any  # The fitted arch model object


@dataclass
class VolForecast:
    """Volatility forecast."""
    horizon: int
    mean: NDArray
    variance: NDArray


def fit_garch(
    returns: NDArray,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
) -> GARCHResult:
    """Fit GARCH(p,q) model to return series."""
    r = np.asarray(returns, dtype=float) * 100  # arch library expects percentage returns
    am = arch_model(r, vol="Garch", p=p, q=q, dist=dist, mean="Zero")
    res = am.fit(disp="off")

    return GARCHResult(
        model_type=f"GARCH({p},{q})",
        params={k: float(v) for k, v in res.params.items()},
        aic=float(res.aic),
        bic=float(res.bic),
        loglikelihood=float(res.loglikelihood),
        conditional_volatility=res.conditional_volatility / 100.0,
        model=res,
    )


def fit_egarch(returns: NDArray, p: int = 1, q: int = 1, dist: str = "normal") -> GARCHResult:
    """Fit EGARCH model (captures leverage effect — asymmetry)."""
    r = np.asarray(returns, dtype=float) * 100
    am = arch_model(r, vol="EGARCH", p=p, q=q, dist=dist, mean="Zero")
    res = am.fit(disp="off")
    return GARCHResult(
        model_type=f"EGARCH({p},{q})",
        params={k: float(v) for k, v in res.params.items()},
        aic=float(res.aic),
        bic=float(res.bic),
        loglikelihood=float(res.loglikelihood),
        conditional_volatility=res.conditional_volatility / 100.0,
        model=res,
    )


def fit_gjr_garch(returns: NDArray, p: int = 1, q: int = 1, dist: str = "normal") -> GARCHResult:
    """Fit GJR-GARCH (threshold GARCH — asymmetry via indicator)."""
    r = np.asarray(returns, dtype=float) * 100
    am = arch_model(r, vol="GARCH", p=p, q=q, dist=dist, mean="Zero", power=2.0)
    # GJR is GARCH with o=1
    am = arch_model(r, vol="GARCH", p=p, o=1, q=q, dist=dist, mean="Zero")
    res = am.fit(disp="off")
    return GARCHResult(
        model_type=f"GJR-GARCH({p},{q})",
        params={k: float(v) for k, v in res.params.items()},
        aic=float(res.aic),
        bic=float(res.bic),
        loglikelihood=float(res.loglikelihood),
        conditional_volatility=res.conditional_volatility / 100.0,
        model=res,
    )


def fit_figarch(returns: NDArray, p: int = 0, q: int = 1, dist: str = "normal") -> GARCHResult:
    """Fit FIGARCH (fractionally integrated GARCH — long memory)."""
    r = np.asarray(returns, dtype=float) * 100
    am = arch_model(r, vol="FIGARCH", p=p, q=q, dist=dist, mean="Zero")
    res = am.fit(disp="off")
    return GARCHResult(
        model_type=f"FIGARCH({p},{q})",
        params={k: float(v) for k, v in res.params.items()},
        aic=float(res.aic),
        bic=float(res.bic),
        loglikelihood=float(res.loglikelihood),
        conditional_volatility=res.conditional_volatility / 100.0,
        model=res,
    )


@dataclass
class HARResult:
    """HAR-RV model result."""
    params: dict[str, float]
    r_squared: float
    residuals: NDArray
    forecasts: NDArray | None = None


def har_rv(realized_vol: NDArray, lags: tuple[int, ...] = (1, 5, 22)) -> HARResult:
    """HAR-RV model (Heterogeneous Autoregressive Realized Volatility).

    RV_{t+1} = beta_0 + beta_d*RV_t + beta_w*RV_t^5 + beta_m*RV_t^22 + epsilon
    """
    rv = np.asarray(realized_vol, dtype=float)
    max_lag = max(lags)
    n = len(rv)

    # Construct features
    Y = rv[max_lag:]
    X = np.ones((n - max_lag, len(lags) + 1))  # intercept + lag terms

    for i, lag in enumerate(lags):
        if lag == 1:
            X[:, i + 1] = rv[max_lag - 1:-1]
        else:
            # Rolling average of past `lag` periods
            rolled = np.convolve(rv, np.ones(lag) / lag, mode='valid')
            X[:, i + 1] = rolled[max_lag - lag:n - lag]

    # OLS
    coeffs, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    ss_res = np.sum(residuals**2) if len(residuals) > 0 else np.sum((Y - X @ coeffs)**2)
    ss_tot = np.sum((Y - np.mean(Y))**2)
    r_sq = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    param_names = ["intercept"] + [f"beta_{l}d" for l in lags]
    params = dict(zip(param_names, coeffs))

    return HARResult(
        params={k: float(v) for k, v in params.items()},
        r_squared=float(r_sq),
        residuals=np.array(Y - X @ coeffs),
    )


def compare_models(returns: NDArray) -> list[dict[str, Any]]:
    """Fit multiple GARCH variants and rank by AIC."""
    results = []
    for name, fn in [
        ("GARCH(1,1)", lambda r: fit_garch(r)),
        ("EGARCH(1,1)", lambda r: fit_egarch(r)),
        ("GJR-GARCH(1,1)", lambda r: fit_gjr_garch(r)),
    ]:
        try:
            res = fn(returns)
            results.append({
                "model": name,
                "aic": res.aic,
                "bic": res.bic,
                "loglik": res.loglikelihood,
            })
        except Exception:
            continue

    results.sort(key=lambda x: x["aic"])
    return results


def forecast_volatility(fitted_model: Any, horizon: int = 5) -> VolForecast:
    """Forecast volatility from a fitted GARCH model."""
    fcst = fitted_model.forecast(horizon=horizon)
    return VolForecast(
        horizon=horizon,
        mean=fcst.mean.values.flatten() / 100.0,
        variance=fcst.variance.values.flatten() / 10_000.0,  # Convert from %^2 to decimal
    )
