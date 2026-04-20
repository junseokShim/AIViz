"""
Forecasting module.

Provides time-series forecasting via statsmodels.
Supports:
  - Holt-Winters Exponential Smoothing (trend + optional seasonality)
  - ARIMA
  - Simple Exponential Smoothing

All functions return a ForecastResult dataclass.
Prophet is NOT a hard dependency; import is attempted at runtime.

STATUS: Holt-Winters and ARIMA are complete and working.
        Prophet wrapper is stubbed – mark TODO if you want to add it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd


@dataclass
class ForecastResult:
    method: str
    historical: pd.Series
    forecast: pd.Series
    conf_int: Optional[pd.DataFrame]   # columns: lower, upper
    metrics: dict                       # aic, bic, rmse, mae etc.
    model_summary: str
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def forecast_holtwinters(
    series: pd.Series,
    horizon: int = 30,
    trend: str = "add",
    seasonal: Optional[str] = None,
    seasonal_periods: Optional[int] = None,
    damped_trend: bool = False,
) -> ForecastResult:
    """
    Holt-Winters Exponential Smoothing forecast.

    Args:
        series:           Clean, numeric, no-NaN time series.
        horizon:          Number of future steps to forecast.
        trend:            'add' | 'mul' | None
        seasonal:         'add' | 'mul' | None (None disables seasonality)
        seasonal_periods: Number of periods in one season (e.g. 12 for monthly).
        damped_trend:     Apply damping to the trend component.

    Returns:
        ForecastResult with forecast values and confidence bands.
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return _import_error("statsmodels")

    s = series.dropna().astype(float)
    if len(s) < 4:
        return ForecastResult(
            method="Holt-Winters", historical=s, forecast=pd.Series(dtype=float),
            conf_int=None, metrics={}, model_summary="",
            error="Need ≥4 data points for Holt-Winters."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ExponentialSmoothing(
                s,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                damped_trend=damped_trend if trend else False,
            )
            fitted = model.fit(optimized=True, remove_bias=True)

        fcast_values = fitted.forecast(horizon)
        # Build index for forecast
        if isinstance(s.index, pd.DatetimeIndex):
            freq = pd.infer_freq(s.index) or "D"
            fcast_index = pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]
        else:
            fcast_index = pd.RangeIndex(s.index[-1] + 1, s.index[-1] + 1 + horizon)

        fcast = pd.Series(fcast_values, index=fcast_index)

        # Approximate 95% CI using in-sample residual std
        resid_std = fitted.resid.std()
        z = 1.96
        conf_int = pd.DataFrame({
            "lower": fcast - z * resid_std,
            "upper": fcast + z * resid_std,
        })

        metrics = _compute_metrics(s, fitted.fittedvalues, fitted)

        return ForecastResult(
            method="Holt-Winters",
            historical=s,
            forecast=fcast,
            conf_int=conf_int,
            metrics=metrics,
            model_summary=str(fitted.summary()),
        )

    except Exception as exc:
        return ForecastResult(
            method="Holt-Winters", historical=series, forecast=pd.Series(dtype=float),
            conf_int=None, metrics={}, model_summary="",
            error=f"Holt-Winters error: {exc}"
        )


def forecast_arima(
    series: pd.Series,
    order: tuple = (1, 1, 1),
    horizon: int = 30,
) -> ForecastResult:
    """
    ARIMA forecast.

    Args:
        series: Input series.
        order:  (p, d, q) ARIMA order.
        horizon: Steps to forecast.

    Returns:
        ForecastResult.
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return _import_error("statsmodels")

    s = series.dropna().astype(float)
    if len(s) < max(10, sum(order) * 2):
        return ForecastResult(
            method="ARIMA", historical=s, forecast=pd.Series(dtype=float),
            conf_int=None, metrics={}, model_summary="",
            error=f"Too few data points for ARIMA{order}."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(s, order=order)
            fitted = model.fit()

        forecast_res = fitted.get_forecast(steps=horizon)
        fcast_mean = forecast_res.predicted_mean

        if isinstance(s.index, pd.DatetimeIndex):
            freq = pd.infer_freq(s.index) or "D"
            fcast_index = pd.date_range(s.index[-1], periods=horizon + 1, freq=freq)[1:]
            fcast_mean.index = fcast_index

        conf_int_raw = forecast_res.conf_int(alpha=0.05)
        conf_int_raw.columns = ["lower", "upper"]
        try:
            conf_int_raw.index = fcast_mean.index
        except Exception:
            pass

        metrics = _compute_metrics(s, fitted.fittedvalues, fitted)

        return ForecastResult(
            method=f"ARIMA{order}",
            historical=s,
            forecast=fcast_mean,
            conf_int=conf_int_raw,
            metrics=metrics,
            model_summary=str(fitted.summary()),
        )

    except Exception as exc:
        return ForecastResult(
            method=f"ARIMA{order}", historical=series, forecast=pd.Series(dtype=float),
            conf_int=None, metrics={}, model_summary="",
            error=f"ARIMA error: {exc}"
        )


def forecast_simple_es(
    series: pd.Series,
    horizon: int = 30,
    smoothing_level: Optional[float] = None,
) -> ForecastResult:
    """Simple Exponential Smoothing (no trend, no seasonality)."""
    try:
        from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    except ImportError:
        return _import_error("statsmodels")

    s = series.dropna().astype(float)
    if len(s) < 2:
        return ForecastResult(
            method="Simple ES", historical=s, forecast=pd.Series(dtype=float),
            conf_int=None, metrics={}, model_summary="",
            error="Need ≥2 data points."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SimpleExpSmoothing(s)
            fit_kwargs = {}
            if smoothing_level is not None:
                fit_kwargs["smoothing_level"] = smoothing_level
                fit_kwargs["optimized"] = False
            fitted = model.fit(**fit_kwargs)

        fcast_values = fitted.forecast(horizon)
        fcast = pd.Series(
            fcast_values,
            index=pd.RangeIndex(len(s), len(s) + horizon),
        )
        resid_std = fitted.resid.std()
        conf_int = pd.DataFrame({
            "lower": fcast - 1.96 * resid_std,
            "upper": fcast + 1.96 * resid_std,
        })

        return ForecastResult(
            method="Simple ES",
            historical=s,
            forecast=fcast,
            conf_int=conf_int,
            metrics={"alpha": getattr(fitted, "params", {}).get("smoothing_level", None)},
            model_summary=str(fitted.summary()),
        )
    except Exception as exc:
        return ForecastResult(
            method="Simple ES", historical=series, forecast=pd.Series(dtype=float),
            conf_int=None, metrics={}, model_summary="",
            error=f"Simple ES error: {exc}"
        )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_forecast(
    series: pd.Series,
    method: str = "Holt-Winters",
    horizon: int = 30,
    **kwargs,
) -> ForecastResult:
    """
    Unified forecast entry point.

    Args:
        series:  Input time series.
        method:  'Holt-Winters' | 'ARIMA' | 'Simple ES'
        horizon: Number of steps to forecast.
        **kwargs: Passed to the specific forecast function.
    """
    method = method.strip()
    if method == "Holt-Winters":
        return forecast_holtwinters(series, horizon=horizon, **kwargs)
    elif method == "ARIMA":
        order = kwargs.pop("order", (1, 1, 1))
        return forecast_arima(series, order=order, horizon=horizon)
    elif method in ("Simple ES", "Simple"):
        return forecast_simple_es(series, horizon=horizon)
    else:
        return ForecastResult(
            method=method, historical=series, forecast=pd.Series(dtype=float),
            conf_int=None, metrics={}, model_summary="",
            error=f"Unknown method: {method!r}"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    actual: pd.Series, fitted_vals: pd.Series, fitted_model
) -> dict:
    """Compute common accuracy metrics."""
    residuals = actual.values - fitted_vals.reindex(actual.index).ffill().values
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    metrics: dict = {"rmse": rmse, "mae": mae}
    for attr in ("aic", "bic", "aicc"):
        val = getattr(fitted_model, attr, None)
        if val is not None:
            metrics[attr] = float(val)
    return metrics


def _import_error(pkg: str) -> ForecastResult:
    return ForecastResult(
        method="", historical=pd.Series(dtype=float),
        forecast=pd.Series(dtype=float),
        conf_int=None, metrics={}, model_summary="",
        error=f"{pkg} is not installed. Run: pip install {pkg}"
    )
