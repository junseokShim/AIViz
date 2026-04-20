"""Tests for the forecast module."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import pytest

from aiviz.analytics.forecast import (
    forecast_holtwinters, forecast_arima, forecast_simple_es, run_forecast
)


def _sine_series(n=120, freq=0.05, trend=0.02):
    t = np.arange(n)
    return pd.Series(
        5 + trend * t + np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05, n),
        name="signal",
    )


class TestHoltWinters:
    def test_returns_result(self):
        s = _sine_series()
        r = forecast_holtwinters(s, horizon=10)
        assert r.ok, r.error
        assert len(r.forecast) == 10

    def test_metrics_present(self):
        s = _sine_series()
        r = forecast_holtwinters(s, horizon=5)
        assert "rmse" in r.metrics
        assert "mae" in r.metrics

    def test_conf_int_shape(self):
        s = _sine_series()
        r = forecast_holtwinters(s, horizon=10)
        assert r.conf_int is not None
        assert r.conf_int.shape == (10, 2)

    def test_too_few_points(self):
        s = pd.Series([1.0, 2.0])
        r = forecast_holtwinters(s, horizon=5)
        assert not r.ok
        assert r.error is not None

    def test_historical_preserved(self):
        s = _sine_series()
        r = forecast_holtwinters(s, horizon=10)
        assert len(r.historical) == len(s)


class TestARIMA:
    def test_basic(self):
        s = _sine_series(n=80)
        r = forecast_arima(s, order=(1, 1, 1), horizon=10)
        assert r.ok, r.error
        assert len(r.forecast) == 10

    def test_too_few_points(self):
        s = pd.Series([1.0, 2.0, 3.0])
        r = forecast_arima(s, order=(1, 1, 1), horizon=5)
        assert not r.ok


class TestSimpleES:
    def test_basic(self):
        s = _sine_series(n=50)
        r = forecast_simple_es(s, horizon=10)
        assert r.ok, r.error
        assert len(r.forecast) == 10


class TestDispatcher:
    def test_holtwinters_dispatch(self):
        s = _sine_series()
        r = run_forecast(s, method="Holt-Winters", horizon=10)
        assert r.method == "Holt-Winters"

    def test_arima_dispatch(self):
        s = _sine_series(n=80)
        r = run_forecast(s, method="ARIMA", horizon=5, order=(1, 1, 0))
        assert "ARIMA" in r.method

    def test_simple_es_dispatch(self):
        s = _sine_series(n=50)
        r = run_forecast(s, method="Simple ES", horizon=5)
        assert r.method == "Simple ES"

    def test_unknown_method(self):
        s = _sine_series()
        r = run_forecast(s, method="unknown_method", horizon=5)
        assert not r.ok
