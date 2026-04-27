"""
Matplotlib chart factories for the PyQt desktop panels.

All functions accept an Axes object and render onto it.
They return nothing – callers are responsible for calling canvas.draw().

Large-data safety:
  All time-series and frequency functions pass data through _safe_downsample()
  before rendering.  Full-resolution data is preserved in analytics objects;
  only the *display* is downsampled.  A warning annotation is added to the plot
  when downsampling is applied so the user knows what they are seeing.

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
# Safe-rendering constants
# ---------------------------------------------------------------------------

MAX_LINE_POINTS   = 10_000    # max points for line / frequency plots
MAX_SCATTER_POINTS = 5_000    # max points for scatter plots
MAX_ANNOTATE_PEAKS = 20       # max peak annotations to avoid clutter
_SAFE_FONT_SIZE    = 8        # minimum font size used in annotations


def _safe_downsample(
    x: np.ndarray,
    y: np.ndarray,
    max_pts: int = MAX_LINE_POINTS,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Uniformly subsample (x, y) to at most *max_pts* points.

    Returns (x_ds, y_ds, was_downsampled).
    """
    n = len(x)
    if n <= max_pts:
        return x, y, False
    idx = np.round(np.linspace(0, n - 1, max_pts)).astype(int)
    return x[idx], y[idx], True


def _add_downsample_note(ax: Axes, original_n: int, shown_n: int) -> None:
    """Annotate the plot with a sampling notice."""
    ax.annotate(
        f"⚠ 표시: {shown_n:,} / {original_n:,} 포인트 (다운샘플링)",
        xy=(0.02, 0.97), xycoords="axes fraction",
        fontsize=_SAFE_FONT_SIZE, color=C_ORANGE, va="top",
    )


# ---------------------------------------------------------------------------
# Tabular / general charts
# ---------------------------------------------------------------------------

def plot_line(ax: Axes, df: pd.DataFrame, x: str, y_cols: list[str]) -> None:
    x_arr = df[x].to_numpy()
    for i, col in enumerate(y_cols):
        y_arr = df[col].to_numpy()
        x_ds, y_ds, ds = _safe_downsample(x_arr, y_arr)
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(x_ds, y_ds, label=col, color=color, linewidth=1.5)
    if ds:
        _add_downsample_note(ax, len(x_arr), len(x_ds))
    ax.set_xlabel(x)
    ax.legend(fontsize=_SAFE_FONT_SIZE + 1, loc="upper right")
    ax.set_title(f"Line Chart: {', '.join(y_cols)}")


def plot_scatter(
    ax: Axes, df: pd.DataFrame, x: str, y: str,
    color_col: Optional[str] = None,
) -> None:
    x_arr = df[x].to_numpy()
    y_arr = df[y].to_numpy()
    n_original = len(x_arr)
    was_ds = False

    if color_col:
        cats = df[color_col].unique()
        for i, cat in enumerate(cats):
            mask = (df[color_col] == cat).to_numpy()
            xc, yc = x_arr[mask], y_arr[mask]
            if len(xc) > MAX_SCATTER_POINTS:
                idx = np.round(np.linspace(0, len(xc) - 1, MAX_SCATTER_POINTS)).astype(int)
                xc, yc = xc[idx], yc[idx]
                was_ds = True
            ax.scatter(
                xc, yc, label=str(cat),
                color=SERIES_COLORS[i % len(SERIES_COLORS)],
                alpha=0.7, s=20, rasterized=True,
            )
        ax.legend(fontsize=_SAFE_FONT_SIZE + 1, loc="upper right", title=color_col)
    else:
        if n_original > MAX_SCATTER_POINTS:
            idx = np.round(np.linspace(0, n_original - 1, MAX_SCATTER_POINTS)).astype(int)
            x_arr, y_arr = x_arr[idx], y_arr[idx]
            was_ds = True
        ax.scatter(x_arr, y_arr, color=C_BLUE, alpha=0.7, s=20, rasterized=True)

    if was_ds:
        _add_downsample_note(ax, n_original, MAX_SCATTER_POINTS)

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
    ax.tick_params(axis="x", rotation=35, labelsize=_SAFE_FONT_SIZE)
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h * 1.01,
            f"{h:.3g}", ha="center", va="bottom", fontsize=_SAFE_FONT_SIZE, color="#cdd6f4",
        )


def plot_histogram(ax: Axes, series: pd.Series, bins: int = 40) -> None:
    data = series.dropna().to_numpy()
    # Cap bins for very large datasets
    effective_bins = min(bins, max(10, len(data) // 100))
    ax.hist(data, bins=effective_bins, color=C_BLUE, edgecolor="#1e1e2e",
            alpha=0.85, linewidth=0.4)
    mean = float(np.mean(data))
    ax.axvline(mean, color=C_ORANGE, linestyle="--", linewidth=1.5,
               label=f"mean={mean:.4g}")
    ax.set_xlabel(series.name or "value")
    ax.set_ylabel("count")
    ax.set_title(f"Histogram: {series.name or 'values'}")
    ax.legend(fontsize=_SAFE_FONT_SIZE + 1, loc="upper right")


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
    ax.tick_params(axis="x", rotation=25, labelsize=_SAFE_FONT_SIZE + 1)


def plot_heatmap_correlation(ax: Axes, corr: pd.DataFrame) -> None:
    cmap = _diverging_cmap()
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=_SAFE_FONT_SIZE)
    ax.set_yticklabels(corr.columns, fontsize=_SAFE_FONT_SIZE)
    ax.set_title("Correlation Matrix")
    # Only annotate if matrix is not too large (avoid cell block limit)
    if len(corr) <= 30:
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha="center", va="center", fontsize=max(6, _SAFE_FONT_SIZE - 1),
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
    ax.set_yticklabels(df_missing.columns, fontsize=_SAFE_FONT_SIZE)
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
    n_original = len(s)
    x = np.arange(n_original)
    try:
        x_vals = s.index.to_numpy() if hasattr(s.index, "to_numpy") else x
    except Exception:
        x_vals = x

    # Downsample for display
    s_arr = s.values
    rm_arr = result.rolling_mean.values
    rs_arr = result.rolling_std.values
    x_ds, s_ds, was_ds = _safe_downsample(x_vals, s_arr)
    _, rm_ds, _ = _safe_downsample(x_vals, rm_arr)
    _, rs_ds, _ = _safe_downsample(x_vals, rs_arr)

    upper = rm_ds + rs_ds
    lower = rm_ds - rs_ds
    ax.fill_between(x_ds, lower, upper, alpha=0.15, color=C_BLUE, label="Rolling ±1σ")
    ax.plot(x_ds, s_ds, color=C_BLUE, linewidth=1.0, alpha=0.9, label=col_name)
    ax.plot(x_ds, rm_ds, color=C_ORANGE, linewidth=1.8, linestyle="--", label="Rolling mean")

    if result.smoothed is not None:
        _, sm_ds, _ = _safe_downsample(x_vals, result.smoothed.values)
        ax.plot(x_ds, sm_ds, color=C_GREEN, linewidth=1.5, label="Smoothed")

    # Anomalies – always show all (typically few points)
    anom_mask = result.anomalies.values
    anom_x = x_vals[anom_mask]
    anom_y = s_arr[anom_mask]
    if len(anom_x) > 0:
        ax.scatter(anom_x, anom_y, color=C_RED, zorder=5, marker="x", s=60,
                   label=f"Anomalies (>{result.anomaly_threshold}σ)")

    # Trend line (compute on downsampled x for display)
    x_numeric = np.arange(n_original)
    trend_full = result.trend_slope * x_numeric + result.trend_intercept
    _, tr_ds, _ = _safe_downsample(x_vals, trend_full)
    ax.plot(x_ds, tr_ds, color=C_MAUVE, linewidth=1.0, linestyle=":",
            label="Linear trend")

    if was_ds:
        _add_downsample_note(ax, n_original, len(x_ds))

    n_anom = int(anom_mask.sum())
    direction = "↑" if result.trend_slope > 0 else "↓"
    ax.set_title(
        f"Time-Series: {col_name}  |  trend {direction}  |  {n_anom} anomalies"
    )
    ax.set_xlabel("time / index")
    ax.set_ylabel(col_name)
    ax.legend(fontsize=max(_SAFE_FONT_SIZE, 8), ncol=2, loc="upper right")


# ---------------------------------------------------------------------------
# Frequency-domain charts
# ---------------------------------------------------------------------------

def plot_fft_amplitude(
    ax: Axes, result, log_scale: bool = False
) -> None:
    freqs = np.asarray(result.freqs)
    amp   = np.asarray(result.amplitude)
    n_original = len(freqs)

    freqs_ds, amp_ds, was_ds = _safe_downsample(freqs, amp)

    ax.plot(freqs_ds, amp_ds, color=C_BLUE, linewidth=1.2)

    if not result.peaks.empty:
        # Cap annotations to avoid cluttering the plot
        peaks = result.peaks.head(MAX_ANNOTATE_PEAKS)
        ax.scatter(
            peaks["frequency"], peaks["amplitude"],
            color=C_RED, zorder=5, marker="^", s=50, label="Peaks",
        )
        for _, row in peaks.iterrows():
            ax.annotate(
                f"{row['frequency']:.3g}",
                (row["frequency"], row["amplitude"]),
                textcoords="offset points", xytext=(0, 6),
                fontsize=max(6, _SAFE_FONT_SIZE - 1), color="#cdd6f4", ha="center",
            )

    if was_ds:
        _add_downsample_note(ax, n_original, len(freqs_ds))

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Amplitude Spectrum  |  dominant: {result.dominant_freq:.4g} Hz")
    if log_scale:
        # Only set log scale if values are positive
        if np.all(amp_ds > 0):
            ax.set_yscale("log")
    ax.legend(fontsize=max(_SAFE_FONT_SIZE, 8), loc="upper right")


def plot_fft_power(ax: Axes, result, log_scale: bool = True) -> None:
    freqs = np.asarray(result.freqs)
    power = np.asarray(result.power)
    n_original = len(freqs)

    freqs_ds, power_ds, was_ds = _safe_downsample(freqs, power)

    ax.fill_between(freqs_ds, power_ds, alpha=0.4, color=C_GREEN)
    ax.plot(freqs_ds, power_ds, color=C_GREEN, linewidth=1.2)

    if was_ds:
        _add_downsample_note(ax, n_original, len(freqs_ds))

    ax.set_xlabel("Frequency")
    ax.set_ylabel("Power")
    ax.set_title("Power Spectrum")
    if log_scale:
        if np.all(power_ds > 0):
            ax.set_yscale("log")


def plot_spectrogram(ax: Axes, result) -> None:
    # Downsample spectrogram columns if too many time bins
    times = result.times
    freqs = result.freqs
    Sxx = result.Sxx
    MAX_TIME_BINS = 500
    if Sxx.shape[1] > MAX_TIME_BINS:
        idx = np.round(np.linspace(0, Sxx.shape[1] - 1, MAX_TIME_BINS)).astype(int)
        times = times[idx]
        Sxx = Sxx[:, idx]

    pcm = ax.pcolormesh(
        times, freqs,
        10 * np.log10(np.maximum(Sxx, 1e-12)),
        cmap="inferno", shading="gouraud",
        rasterized=True,
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
            f"{val:.1f}%", ha="center", fontsize=_SAFE_FONT_SIZE + 1, color="#cdd6f4",
        )
    ax.set_ylabel("% of Total Energy")
    ax.set_title("Band Energy Distribution")
    ax.tick_params(axis="x", rotation=15, labelsize=_SAFE_FONT_SIZE + 1)


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
    ax.legend(fontsize=_SAFE_FONT_SIZE + 1, loc="upper right")


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
                       ha="right", fontsize=_SAFE_FONT_SIZE)
    ax.set_ylabel("% pixels")
    ax.set_title("Dominant Colors")


def plot_edge_result(ax: Axes, edge_arr: np.ndarray, method: str = "edge") -> None:
    """Display a 2-D uint8 edge map."""
    ax.imshow(edge_arr, cmap="gray", aspect="auto", interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"Edge Detection: {method}")


def plot_segmentation_result(
    ax: Axes,
    base_img: np.ndarray,
    mask: np.ndarray,
    n_regions: int,
    method: str = "segmentation",
) -> None:
    """Display image with coloured segmentation overlay."""
    ax.imshow(base_img, aspect="auto", interpolation="nearest")
    # Create a colormap overlay
    import matplotlib.cm as cm
    cmap = cm.get_cmap("tab20", max(n_regions, 1))
    overlay = cmap(mask.astype(float) / max(n_regions, 1))
    overlay[mask == 0, 3] = 0.0     # background transparent
    overlay[mask > 0, 3] = 0.45    # regions semi-transparent
    ax.imshow(overlay, aspect="auto", interpolation="nearest")
    ax.axis("off")
    ax.set_title(f"Segmentation: {method}  ({n_regions} regions)")


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
    hist_arr = historical.values
    fcast_arr = forecast.values
    x_hist = np.arange(len(hist_arr))
    x_fcast = np.arange(len(hist_arr), len(hist_arr) + len(fcast_arr))

    # Downsample historical if needed
    x_h_ds, h_ds, _ = _safe_downsample(x_hist, hist_arr)

    ax.plot(x_h_ds, h_ds, color=C_BLUE, linewidth=1.5, label="Historical")
    ax.plot(x_fcast, fcast_arr, color=C_ORANGE, linewidth=1.8,
            linestyle="--", label="Forecast")
    ax.axvline(len(hist_arr) - 1, color=C_MAUVE, linewidth=1, linestyle=":")

    if conf_int is not None:
        try:
            lo = conf_int.iloc[:, 0].values
            hi = conf_int.iloc[:, 1].values
            ax.fill_between(x_fcast, lo, hi, alpha=0.2, color=C_ORANGE, label="95% CI")
        except Exception:
            pass

    ax.set_xlabel("time step")
    ax.set_ylabel(col_name)
    ax.set_title(f"Forecast: {col_name}")
    ax.legend(fontsize=_SAFE_FONT_SIZE + 1, loc="upper right")


# ---------------------------------------------------------------------------
# Advanced signal analysis charts
# ---------------------------------------------------------------------------

def plot_psd(ax: Axes, result, log_scale: bool = True) -> None:
    """Plot Welch PSD."""
    freqs = np.asarray(result.freqs)
    psd   = np.asarray(result.psd)
    n_orig = len(freqs)

    freqs_ds, psd_ds, was_ds = _safe_downsample(freqs, psd)

    ax.fill_between(freqs_ds, psd_ds, alpha=0.35, color=C_MAUVE)
    ax.plot(freqs_ds, psd_ds, color=C_MAUVE, linewidth=1.3)

    if was_ds:
        _add_downsample_note(ax, n_orig, len(freqs_ds))

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (V²/Hz)")
    ax.set_title(
        f"Welch PSD  |  dominant: {result.dominant_freq:.4g} Hz  |  "
        f"total power: {result.total_power:.4g}"
    )
    if log_scale and np.all(psd_ds > 0):
        ax.set_yscale("log")


def plot_cwt_scalogram(ax: Axes, result) -> None:
    """Plot CWT scalogram (time × frequency heatmap)."""
    if not result.available:
        ax.text(
            0.5, 0.5, result.error or "CWT 사용 불가",
            ha="center", va="center", color=C_RED, fontsize=11,
            wrap=True, transform=ax.transAxes,
        )
        ax.set_title("CWT 스칼로그램 (사용 불가)")
        return

    times = result.times
    freqs = result.freqs
    coefs = result.coefs   # (n_scales, n_times)

    # Downsample time axis if too many columns
    MAX_T = 500
    if coefs.shape[1] > MAX_T:
        idx = np.round(np.linspace(0, coefs.shape[1] - 1, MAX_T)).astype(int)
        times = times[idx]
        coefs = coefs[:, idx]

    # Log-scale power for better dynamic range
    log_coefs = np.log10(coefs + 1e-12)

    pcm = ax.pcolormesh(
        times, freqs, log_coefs,
        cmap="magma", shading="gouraud", rasterized=True,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"CWT 스칼로그램  |  wavelet: {result.wavelet}")
    ax.figure.colorbar(pcm, ax=ax, label="log₁₀ |CWT|")


def plot_s_transform(ax: Axes, result) -> None:
    """Plot S-Transform time-frequency heatmap."""
    if not result.available:
        ax.text(
            0.5, 0.5, result.error or "S-Transform 사용 불가",
            ha="center", va="center", color=C_RED, fontsize=11,
            wrap=True, transform=ax.transAxes,
        )
        ax.set_title("S-Transform (사용 불가)")
        return

    times = result.times
    freqs = result.freqs
    ST = result.ST   # (n_freqs, n_times)

    if ST.size == 0 or len(freqs) == 0:
        ax.text(0.5, 0.5, "결과 없음", ha="center", va="center", transform=ax.transAxes)
        return

    # Downsample time axis
    MAX_T = 500
    if ST.shape[1] > MAX_T:
        idx = np.round(np.linspace(0, ST.shape[1] - 1, MAX_T)).astype(int)
        times = times[idx]
        ST = ST[:, idx]

    pcm = ax.pcolormesh(
        times, freqs, ST,
        cmap="plasma", shading="gouraud", rasterized=True,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    title = "S-Transform 시간-주파수"
    if result.decimated:
        title += f"  [다운샘플링: {result.original_n} → {len(result.times)}]"
    ax.set_title(title)
    ax.figure.colorbar(pcm, ax=ax, label="|S(t,f)|")


def plot_envelope_spectrum(ax: Axes, result, log_scale: bool = False) -> None:
    """Plot envelope spectrum (Hilbert-based)."""
    freqs = np.asarray(result.freqs)
    spec  = np.asarray(result.envelope_spectrum)
    n_orig = len(freqs)

    freqs_ds, spec_ds, was_ds = _safe_downsample(freqs, spec)

    ax.plot(freqs_ds, spec_ds, color=C_GREEN, linewidth=1.2)
    ax.fill_between(freqs_ds, spec_ds, alpha=0.3, color=C_GREEN)

    # Mark dominant frequency
    ax.axvline(result.dominant_freq, color=C_RED, linewidth=1.0,
               linestyle="--", label=f"dominant: {result.dominant_freq:.4g} Hz")

    if was_ds:
        _add_downsample_note(ax, n_orig, len(freqs_ds))

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Envelope Spectrum  |  dominant: {result.dominant_freq:.4g} Hz")
    ax.legend(fontsize=_SAFE_FONT_SIZE + 1, loc="upper right")
    if log_scale and np.all(spec_ds > 0):
        ax.set_yscale("log")


def plot_cepstrum(ax: Axes, result) -> None:
    """Plot real cepstrum."""
    q   = np.asarray(result.quefrency)
    cep = np.asarray(result.cepstrum)
    n_orig = len(q)

    q_ds, cep_ds, was_ds = _safe_downsample(q, cep)

    ax.plot(q_ds, cep_ds, color=C_ORANGE, linewidth=1.0)

    if result.dominant_quefrency > 0:
        ax.axvline(
            result.dominant_quefrency, color=C_RED, linewidth=1.2,
            linestyle="--",
            label=(
                f"dominant: {result.dominant_quefrency*1000:.2f} ms  "
                f"({result.dominant_pitch:.2f} Hz)"
            ),
        )
        ax.legend(fontsize=_SAFE_FONT_SIZE + 1, loc="upper right")

    if was_ds:
        _add_downsample_note(ax, n_orig, len(q_ds))

    ax.set_xlabel("Quefrency (s)")
    ax.set_ylabel("Cepstrum")
    ax.set_title(
        f"Real Cepstrum  |  dominant quefrency: {result.dominant_quefrency*1000:.2f} ms"
    )


def plot_new_band_power(ax: Axes, result) -> None:
    """Bar chart for BandPowerResult (from signal.band_power)."""
    bands = result.bands
    if not bands:
        ax.text(0.5, 0.5, "밴드 데이터 없음", ha="center", va="center",
                transform=ax.transAxes)
        return

    names = [b["band"] for b in bands]
    powers = [b["relative_power"] for b in bands]
    colors = SERIES_COLORS[:len(names)]

    bars = ax.bar(names, powers, color=colors)
    for bar, val in zip(bars, powers):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", fontsize=_SAFE_FONT_SIZE + 1, color="#cdd6f4",
        )

    ax.set_ylabel("상대 전력 (%)")
    ax.set_title(f"Band Power 분포  ({result.method})")
    ax.set_ylim(0, max(powers) * 1.15 if powers else 1)
    ax.tick_params(axis="x", rotation=20, labelsize=_SAFE_FONT_SIZE)


def plot_dominant_freq_over_time(
    ax: Axes,
    times: np.ndarray,
    dominant_freqs: np.ndarray,
    title: str = "Dominant Frequency over Time",
) -> None:
    """Line plot of dominant frequency at each time step."""
    t_ds, f_ds, was_ds = _safe_downsample(times, dominant_freqs)
    ax.plot(t_ds, f_ds, color=C_ORANGE, linewidth=1.2)
    if was_ds:
        _add_downsample_note(ax, len(times), len(t_ds))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Dominant Frequency (Hz)")
    ax.set_title(title)


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
