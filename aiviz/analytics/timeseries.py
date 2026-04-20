"""
Time-series analysis module.

Provides rolling statistics, trend detection, anomaly hinting,
smoothing, and multi-signal helpers.

All functions accept a plain pandas Series or DataFrame and return
results as pandas objects – no Streamlit dependencies here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class TimeSeriesResult:
    original: pd.Series
    rolling_mean: pd.Series
    rolling_std: pd.Series
    smoothed: Optional[pd.Series]
    trend_slope: float          # linear trend slope per sample
    trend_intercept: float
    anomalies: pd.Series        # boolean mask
    anomaly_threshold: float    # n-sigma used
    stats: dict


def analyze_series(
    series: pd.Series,
    window: int = 10,
    smooth: bool = True,
    anomaly_sigma: float = 3.0,
) -> TimeSeriesResult:
    """
    Full time-series analysis for a single numeric series.

    Args:
        series:         Numeric pandas Series (index may be datetime).
        window:         Rolling window size.
        smooth:         Apply Savitzky-Golay smoothing if True.
        anomaly_sigma:  Z-score threshold for anomaly flagging.

    Returns:
        TimeSeriesResult with all computed signals.
    """
    s = series.dropna().astype(float)

    rolling_mean = s.rolling(window=window, min_periods=1).mean()
    rolling_std = s.rolling(window=window, min_periods=1).std().fillna(0)

    # Linear trend via least-squares
    x = np.arange(len(s), dtype=float)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, s.values, 1)
    else:
        slope, intercept = 0.0, float(s.iloc[0]) if len(s) else 0.0

    # Smoothing (Savitzky-Golay needs window_length >= polyorder+2 and odd)
    smoothed: Optional[pd.Series] = None
    if smooth and len(s) > 10:
        wl = min(window | 1, len(s) - (1 - len(s) % 2))  # ensure odd
        wl = max(wl, 5)
        if wl % 2 == 0:
            wl += 1
        try:
            sg = savgol_filter(s.values, window_length=min(wl, len(s) - 1 if len(s) % 2 == 0 else len(s)), polyorder=2)
            smoothed = pd.Series(sg, index=s.index)
        except Exception:
            smoothed = rolling_mean

    # Anomaly detection: |z-score| > threshold
    z = (s - rolling_mean) / rolling_std.replace(0, np.nan)
    anomalies = z.abs() > anomaly_sigma

    stats = {
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "range": float(s.max() - s.min()),
        "trend_slope": slope,
        "anomaly_count": int(anomalies.sum()),
        "n_points": len(s),
    }

    return TimeSeriesResult(
        original=s,
        rolling_mean=rolling_mean,
        rolling_std=rolling_std,
        smoothed=smoothed,
        trend_slope=slope,
        trend_intercept=intercept,
        anomalies=anomalies,
        anomaly_threshold=anomaly_sigma,
        stats=stats,
    )


def multi_series_stats(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Return a comparison table of time-series statistics for multiple columns.
    """
    rows = []
    for col in columns:
        s = df[col].dropna().astype(float)
        if len(s) < 2:
            continue
        slope, _ = np.polyfit(np.arange(len(s)), s.values, 1)
        rows.append({
            "column": col,
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "max": s.max(),
            "trend_slope": slope,
            "trend_direction": "↑ increasing" if slope > 0 else "↓ decreasing",
        })
    return pd.DataFrame(rows).set_index("column")


def resample_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    freq: str = "1min",
    agg: str = "mean",
) -> pd.DataFrame:
    """
    Resample a time-indexed DataFrame to a regular frequency.

    Args:
        freq: pandas offset string e.g. '1min', '1H', '1D'
        agg:  aggregation method: 'mean', 'sum', 'max', 'min'
    """
    tmp = df[[time_col, value_col]].copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col])
    tmp = tmp.set_index(time_col)
    resampled = getattr(tmp.resample(freq), agg)()
    return resampled.reset_index()
