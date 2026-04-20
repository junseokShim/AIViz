"""
Matplotlib chart factories for the PyQt desktop panels.

All functions accept an Axes object and render onto it.
They return nothing – callers are responsible for calling canvas.draw().

To add a new chart:
  1. Write a plot_<name>(ax, ...) function here.
  2. Call it from the relevant panel after getting ax = plot_widget.get_ax().
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from aiviz.app.style import SERIES_COLORS, C_RED, C_GREEN, C_ORANGE, C_MAUVE, C_BLUE


# ---------------------------------------------------------------------------
# Tabular / general charts
# ---------------------------------------------------------------------------

def plot_line(ax: Axes, df: pd.DataFrame, x: str, y_cols: list[str]) -> None:
    for i, col in enumerate(y_cols):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(df[x], df[col], label=col, color=color, linewidth=1.5)
    ax.set_xlabel(x)
    ax.legend(fontsize=9)
    ax.set_title(f"Line Chart: {', '.join(y_cols)}")


def plot_scatter(
    ax: Axes, df: pd.DataFrame, x: str, y: str,
    color_col: Optional[str] = None,
) -> None:
    if color_col:
        cats = df[color_col].unique()
        for i, cat in enumerate(cats):
            mask = df[color_col] == cat
            ax.scatter(
                df.loc[mask, x], df.loc[mask, y],
                label=str(cat), color=SERIES_COLORS[i % len(SERIES_COLORS)],
                alpha=0.7, s=25,
            )
        ax.legend(fontsize=9, title=color_col)
    else:
        ax.scatter(df[x], df[y], color=C_BLUE, alpha=0.7, s=25)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"Scatter: {x} vs {y}")


def plot_bar(ax: Axes, df: pd.DataFrame, x: str, y: str, agg: str = "sum") -> None:
    tmp = df.groupby(x)[y].agg(agg).sort_values(ascending=False).head(30)
    bars = ax.bar(
        [str(v) for v in tmp.index], tmp.values,
        color=SERIES_COLORS[:len(tmp)],
    )
    ax.set_xlabel(x)
    ax.set_ylabel(f"{agg}({y})")
    ax.set_title(f"Bar: {y} by {x} ({agg})")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    # Add value labels
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h * 1.01,
            f"{h:.3g}", ha="center", va="bottom", fontsize=8, color="#cdd6f4",
        )


def plot_histogram(ax: Axes, series: pd.Series, bins: int = 40) -> None:
    ax.hist(series.dropna(), bins=bins, color=C_BLUE, edgecolor="#1e1e2e",
            alpha=0.85, linewidth=0.4)
    mean = series.mean()
    ax.axvline(mean, color=C_ORANGE, linestyle="--", linewidth=1.5,
               label=f"mean={mean:.4g}")
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("count")
    ax.set_title(f"Histogram: {series.name or 'values'}")
    ax.legend(fontsize=9)


def plot_box(ax: Axes, df: pd.DataFrame, columns: list[str]) -> None:
    data = [df[c].dropna().values for c in columns]
    bp = ax.boxplot(
        data, labels=columns, patch_artist=True,
        medianprops=dict(color=C_ORANGE, linewidth=2),
        whiskerprops=dict(color="#cdd6f4"),
        capprops=dict(color="#cdd6f4"),
        flierprops=dict(marker="x", color=C_RED, markersize=4),
    )
    for patch, color in zip(bp["boxes"], SERIES_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Box Plot")
    ax.tick_params(axis="x", rotation=25, labelsize=9)


def plot_heatmap_correlation(ax: Axes, corr: pd.DataFrame) -> None:
    import matplotlib.colors as mcolors
    cmap = _diverging_cmap()
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    ax.set_title("Correlation Matrix")
    # Annotate cells
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.5 else "#cdd6f4")
    ax.figure.colorbar(im, ax=ax, shrink=0.8)


def plot_missing_heatmap(ax: Axes, df_missing: pd.DataFrame) -> None:
    """Binary null mask heatmap."""
    if df_missing.empty:
        ax.text(0.5, 0.5, "No missing values!", ha="center", va="center",
                color=C_GREEN, fontsize=13)
        return
    ax.imshow(df_missing.T.values, cmap=_missing_cmap(), aspect="auto")
    ax.set_yticks(range(len(df_missing.columns)))
    ax.set_yticklabels(df_missing.columns, fontsize=8)
    ax.set_xlabel("row index (sampled)")
    ax.set_title("Missing Values  (red = NaN)")


# ---------------------------------------------------------------------------
# Time-series charts
# ---------------------------------------------------------------------------

def plot_timeseries_analysis(
    ax: Axes,
    result,       # TimeSeriesResult from analytics.timeseries
    col_name: str = "signal",
) -> None:
    s = result.original
    x = np.arange(len(s))
    try:
        x_vals = s.index.to_numpy() if hasattr(s.index, "to_numpy") else x
    except Exception:
        x_vals = x

    # Rolling std band
    upper = result.rolling_mean + result.rolling_std
    lower = result.rolling_mean - result.rolling_std
    ax.fill_between(x_vals, lower.values, upper.values,
                    alpha=0.15, color=C_BLUE, label="Rolling ±1σ")

    ax.plot(x_vals, s.values, color=C_BLUE, linewidth=1.0,
            alpha=0.9, label=col_name)
    ax.plot(x_vals, result.rolling_mean.values, color=C_ORANGE,
            linewidth=1.8, linestyle="--", label="Rolling mean")

    if result.smoothed is not None:
        ax.plot(x_vals, result.smoothed.values, color=C_GREEN,
                linewidth=1.5, label="Smoothed")

    # Anomalies
    anom_mask = result.anomalies.values
    ax.scatter(
        x_vals[anom_mask], s.values[anom_mask],
        color=C_RED, zorder=5, marker="x", s=60,
        label=f"Anomalies (>{result.anomaly_threshold}σ)",
    )

    # Trend line
    trend = result.trend_slope * x + result.trend_intercept
    ax.plot(x_vals, trend, color=C_MAUVE, linewidth=1.0, linestyle=":",
            label="Linear trend")

    n_anom = int(anom_mask.sum())
    direction = "↑" if result.trend_slope > 0 else "↓"
    ax.set_title(
        f"Time-Series: {col_name}  |  trend {direction}  |  {n_anom} anomalies"
    )
    ax.set_xlabel("time / index")
    ax.set_ylabel(col_name)
    ax.legend(fontsize=8, ncol=2)


# ---------------------------------------------------------------------------
# Frequency-domain charts
# ---------------------------------------------------------------------------

def plot_fft_amplitude(
    ax: Axes, result, log_scale: bool = False
) -> None:
    ax.plot(result.freqs, result.amplitude, color=C_BLUE, linewidth=1.2)
    if not result.peaks.empty:
        ax.scatter(
            result.peaks["frequency"], result.peaks["amplitude"],
            color=C_RED, zorder=5, marker="^", s=50, label="Peaks",
        )
        for _, row in result.peaks.iterrows():
            ax.annotate(
                f"{row['frequency']:.3g}",
                (row["frequency"], row["amplitude"]),
                textcoords="offset points", xytext=(0, 6),
                fontsize=7, color="#cdd6f4", ha="center",
            )
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Amplitude Spectrum  |  dominant: {result.dominant_freq:.4g} Hz")
    if log_scale:
        ax.set_yscale("log")
    ax.legend(fontsize=8)


def plot_fft_power(ax: Axes, result, log_scale: bool = True) -> None:
    ax.fill_between(result.freqs, result.power, alpha=0.4, color=C_GREEN)
    ax.plot(result.freqs, result.power, color=C_GREEN, linewidth=1.2)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectrum")
    if log_scale:
        ax.set_yscale("log")


def plot_spectrogram(ax: Axes, result) -> None:
    pcm = ax.pcolormesh(
        result.times, result.freqs,
        10 * np.log10(result.Sxx + 1e-12),
        cmap="inferno", shading="gouraud",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (STFT)")
    ax.figure.colorbar(pcm, ax=ax, label="dB")


def plot_band_energy(ax: Axes, band_stats: list) -> None:
    names = [b.band_name for b in band_stats]
    fractions = [b.energy_fraction * 100 for b in band_stats]
    colors = SERIES_COLORS[:len(names)]
    bars = ax.bar(names, fractions, color=colors)
    for bar, val in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=9, color="#cdd6f4",
        )
    ax.set_ylabel("% of Total Energy")
    ax.set_title("Band Energy Distribution")
    ax.tick_params(axis="x", rotation=15, labelsize=8)


# ---------------------------------------------------------------------------
# Image charts
# ---------------------------------------------------------------------------

def plot_pixel_histogram(ax: Axes, histograms: dict) -> None:
    channel_colors = {"R": "#f38ba8", "G": "#a6e3a1", "B": "#89b4fa", "L": "#a6adc8"}
    for ch, (counts, edges) in histograms.items():
        centers = (edges[:-1] + edges[1:]) / 2
        color = channel_colors.get(ch, C_BLUE)
        ax.fill_between(centers, counts, alpha=0.5, color=color, label=ch)
        ax.plot(centers, counts, color=color, linewidth=0.8)
    ax.set_xlabel("Pixel value (0–255)")
    ax.set_ylabel("Count")
    ax.set_title("Pixel Intensity Histogram")
    ax.legend(fontsize=9)


def plot_dominant_colors(ax: Axes, dom_colors) -> None:
    colors_hex = [
        f"#{int(r):02x}{int(g):02x}{int(b):02x}"
        for r, g, b in zip(dom_colors["r"], dom_colors["g"], dom_colors["b"])
    ]
    fractions = dom_colors["fraction"].values * 100
    bars = ax.bar(range(len(dom_colors)), fractions, color=colors_hex,
                  edgecolor="#1e1e2e", linewidth=0.5)
    ax.set_xticks(range(len(dom_colors)))
    ax.set_xticklabels([f"#{c[1:]}" for c in colors_hex], rotation=40,
                       ha="right", fontsize=7)
    ax.set_ylabel("% pixels")
    ax.set_title("Dominant Colors")


# ---------------------------------------------------------------------------
# Forecast chart
# ---------------------------------------------------------------------------

def plot_forecast(
    ax: Axes,
    historical: pd.Series,
    forecast: pd.Series,
    conf_int=None,
    col_name: str = "value",
) -> None:
    x_hist = np.arange(len(historical))
    x_fcast = np.arange(len(historical), len(historical) + len(forecast))

    ax.plot(x_hist, historical.values, color=C_BLUE, linewidth=1.5, label="Historical")
    ax.plot(x_fcast, forecast.values, color=C_ORANGE, linewidth=1.8,
            linestyle="--", label="Forecast")
    ax.axvline(len(historical) - 1, color=C_MAUVE, linewidth=1, linestyle=":")

    if conf_int is not None:
        try:
            lo = conf_int.iloc[:, 0].values
            hi = conf_int.iloc[:, 1].values
            ax.fill_between(x_fcast, lo, hi, alpha=0.2, color=C_ORANGE,
                            label="95% CI")
        except Exception:
            pass

    ax.set_xlabel("time step")
    ax.set_ylabel(col_name)
    ax.set_title(f"Forecast: {col_name}")
    ax.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "aiviz_div",
        ["#f38ba8", "#1e1e2e", "#89b4fa"],
    )


def _missing_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "aiviz_missing",
        ["#313244", "#f38ba8"],
    )
