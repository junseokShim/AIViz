"""
Chart builder module.

All chart functions accept a DataFrame + configuration and return a
plotly Figure. No Streamlit calls here – pure Plotly only.

To add a new chart type:
  1. Add a new function following the _make_* naming convention.
  2. Register it in CHART_REGISTRY at the bottom of this file.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aiviz.analytics.timeseries import TimeSeriesResult
from aiviz.analytics.frequency import FFTResult, SpectrogramResult


# ---------------------------------------------------------------------------
# Tabular / general charts
# ---------------------------------------------------------------------------

def make_line(df: pd.DataFrame, x: str, y: list[str], title: str = "") -> go.Figure:
    fig = px.line(df, x=x, y=y, title=title or f"Line Chart: {', '.join(y)}")
    fig.update_layout(_base_layout())
    return fig


def make_scatter(
    df: pd.DataFrame, x: str, y: str,
    color: Optional[str] = None,
    title: str = "",
) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color,
                     title=title or f"Scatter: {x} vs {y}")
    fig.update_layout(_base_layout())
    return fig


def make_bar(
    df: pd.DataFrame, x: str, y: str,
    agg: str = "sum",
    title: str = "",
) -> go.Figure:
    tmp = df.groupby(x)[y].agg(agg).reset_index()
    fig = px.bar(tmp, x=x, y=y,
                 title=title or f"Bar Chart: {y} by {x} ({agg})")
    fig.update_layout(_base_layout())
    return fig


def make_histogram(df: pd.DataFrame, col: str, bins: int = 40, title: str = "") -> go.Figure:
    fig = px.histogram(df, x=col, nbins=bins,
                       title=title or f"Histogram: {col}")
    fig.update_layout(_base_layout())
    return fig


def make_box(df: pd.DataFrame, columns: list[str], title: str = "") -> go.Figure:
    fig = px.box(df[columns], title=title or "Box Plot")
    fig.update_layout(_base_layout())
    return fig


def make_heatmap_correlation(corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title,
    )
    fig.update_layout(_base_layout())
    return fig


def make_heatmap_missing(missing_mask: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        missing_mask.T,
        color_continuous_scale=[[0, "#e8f4ea"], [1, "#d32f2f"]],
        title="Missing Value Map (red = missing)",
        aspect="auto",
    )
    fig.update_layout(_base_layout())
    return fig


# ---------------------------------------------------------------------------
# Time-series charts
# ---------------------------------------------------------------------------

def make_timeseries_analysis(result: TimeSeriesResult, col_name: str = "signal") -> go.Figure:
    """Composite chart: original + rolling mean ± std + smoothed + anomalies."""
    fig = go.Figure()

    s = result.original
    x = s.index if not isinstance(s.index, pd.RangeIndex) else list(range(len(s)))

    # Rolling std band
    upper = result.rolling_mean + result.rolling_std
    lower = result.rolling_mean - result.rolling_std
    fig.add_trace(go.Scatter(
        x=list(x) + list(x)[::-1],
        y=list(upper) + list(lower)[::-1],
        fill="toself",
        fillcolor="rgba(99,110,250,0.15)",
        line=dict(width=0),
        name="Rolling ±1σ",
        showlegend=True,
    ))

    # Original signal
    fig.add_trace(go.Scatter(x=x, y=s.values, mode="lines",
                             name=col_name, line=dict(color="#636EFA", width=1)))

    # Rolling mean
    fig.add_trace(go.Scatter(x=x, y=result.rolling_mean.values, mode="lines",
                             name="Rolling Mean", line=dict(color="#EF553B", width=2, dash="dash")))

    # Smoothed
    if result.smoothed is not None:
        fig.add_trace(go.Scatter(x=x, y=result.smoothed.values, mode="lines",
                                 name="Smoothed", line=dict(color="#00CC96", width=2)))

    # Anomalies
    anom_x = [xi for xi, flag in zip(x, result.anomalies) if flag]
    anom_y = [yi for yi, flag in zip(s.values, result.anomalies) if flag]
    if anom_x:
        fig.add_trace(go.Scatter(
            x=anom_x, y=anom_y,
            mode="markers",
            name=f"Anomalies (>{result.anomaly_threshold}σ)",
            marker=dict(color="#FF0000", size=8, symbol="x"),
        ))

    # Trend line
    x_num = np.arange(len(s))
    trend = result.trend_slope * x_num + result.trend_intercept
    fig.add_trace(go.Scatter(x=x, y=trend, mode="lines",
                             name="Linear Trend",
                             line=dict(color="#AB63FA", width=1, dash="dot")))

    direction = "↑" if result.trend_slope > 0 else "↓"
    fig.update_layout(
        title=f"Time-Series Analysis: {col_name}  |  trend {direction}  |  "
              f"{int(result.anomalies.sum())} anomalies",
        xaxis_title="Index / Time",
        yaxis_title=col_name,
        **_base_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Frequency-domain charts
# ---------------------------------------------------------------------------

def make_fft_amplitude(result: FFTResult, log_scale: bool = False, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.freqs, y=result.amplitude,
        mode="lines", name="Amplitude",
        line=dict(color="#636EFA"),
    ))
    # Mark dominant peaks
    if not result.peaks.empty:
        fig.add_trace(go.Scatter(
            x=result.peaks["frequency"],
            y=result.peaks["amplitude"],
            mode="markers+text",
            marker=dict(color="#EF553B", size=10, symbol="triangle-up"),
            text=[f"{f:.3g} Hz" for f in result.peaks["frequency"]],
            textposition="top center",
            name="Peaks",
        ))
    fig.update_layout(
        title=title or f"Amplitude Spectrum  |  dominant: {result.dominant_freq:.4g} Hz",
        xaxis_title="Frequency",
        yaxis_title="Amplitude",
        yaxis_type="log" if log_scale else "linear",
        **_base_layout(),
    )
    return fig


def make_fft_power(result: FFTResult, log_scale: bool = True, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.freqs, y=result.power,
        mode="lines", name="Power",
        line=dict(color="#00CC96"),
        fill="tozeroy",
        fillcolor="rgba(0,204,150,0.15)",
    ))
    fig.update_layout(
        title=title or "Power Spectrum",
        xaxis_title="Frequency",
        yaxis_title="Power",
        yaxis_type="log" if log_scale else "linear",
        **_base_layout(),
    )
    return fig


def make_spectrogram(result: SpectrogramResult, title: str = "Spectrogram") -> go.Figure:
    fig = go.Figure(go.Heatmap(
        x=result.times,
        y=result.freqs,
        z=10 * np.log10(result.Sxx + 1e-12),  # dB scale
        colorscale="Viridis",
        colorbar=dict(title="dB"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Frequency",
        **_base_layout(),
    )
    return fig


def make_band_energy_bar(band_stats: list) -> go.Figure:
    names = [b.band_name for b in band_stats]
    fractions = [b.energy_fraction * 100 for b in band_stats]
    fig = go.Figure(go.Bar(
        x=names, y=fractions,
        marker_color="#636EFA",
        text=[f"{f:.1f}%" for f in fractions],
        textposition="outside",
    ))
    fig.update_layout(
        title="Band Energy Distribution (%)",
        yaxis_title="% of Total Energy",
        **_base_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Image charts
# ---------------------------------------------------------------------------

def make_image_histogram(histograms: dict, title: str = "Pixel Intensity Histogram") -> go.Figure:
    colors = {"R": "red", "G": "green", "B": "blue", "L": "gray"}
    fig = go.Figure()
    for channel, (counts, edges) in histograms.items():
        centers = (edges[:-1] + edges[1:]) / 2
        fig.add_trace(go.Bar(
            x=centers, y=counts,
            name=channel,
            marker_color=colors.get(channel, "steelblue"),
            opacity=0.6,
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Pixel Value (0–255)",
        yaxis_title="Count",
        barmode="overlay",
        **_base_layout(),
    )
    return fig


def make_dominant_colors_bar(dominant_colors: pd.DataFrame) -> go.Figure:
    colors_hex = [
        f"rgb({r},{g},{b})"
        for r, g, b in zip(
            dominant_colors["r"],
            dominant_colors["g"],
            dominant_colors["b"],
        )
    ]
    fig = go.Figure(go.Bar(
        x=dominant_colors["color_rgb"],
        y=dominant_colors["fraction"] * 100,
        marker_color=colors_hex,
        text=[f"{f*100:.1f}%" for f in dominant_colors["fraction"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Dominant Colors",
        yaxis_title="% of Pixels",
        xaxis_title="Color (quantized)",
        **_base_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Layout helper
# ---------------------------------------------------------------------------

def _base_layout() -> dict:
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
    )


# ---------------------------------------------------------------------------
# Chart registry – used by UI chart builder
# ---------------------------------------------------------------------------

CHART_TYPES = {
    "Line Chart": "line",
    "Scatter Plot": "scatter",
    "Bar Chart": "bar",
    "Histogram": "histogram",
    "Box Plot": "box",
    "Correlation Heatmap": "correlation",
}
