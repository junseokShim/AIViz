"""
Frequency-domain analysis module.

Provides FFT, power spectral density, dominant frequency extraction,
band energy statistics, and a STFT spectrogram helper.

All functions are pure analytics – no UI/Streamlit dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal as sci_signal


@dataclass
class FFTResult:
    freqs: np.ndarray          # frequency axis (Hz if sample_rate given, else bins)
    amplitude: np.ndarray      # single-sided amplitude spectrum
    power: np.ndarray          # power (amplitude^2)
    dominant_freq: float       # frequency of highest amplitude peak
    dominant_amplitude: float
    peaks: pd.DataFrame        # top N peaks with freq & amplitude
    stats: dict


@dataclass
class BandStats:
    band_name: str
    freq_low: float
    freq_high: float
    mean_power: float
    rms: float
    energy_fraction: float     # fraction of total energy in this band


@dataclass
class SpectrogramResult:
    times: np.ndarray
    freqs: np.ndarray
    Sxx: np.ndarray            # spectrogram matrix (power)


# ---------------------------------------------------------------------------
# Core FFT
# ---------------------------------------------------------------------------

def compute_fft(
    series: pd.Series,
    sample_rate: float = 1.0,
    window: str = "hann",
    n_peaks: int = 5,
) -> FFTResult:
    """
    Compute FFT of a numeric time series.

    Args:
        series:      Input signal (numeric, no NaN).
        sample_rate: Samples per second. Use 1.0 for normalised freq (0–0.5).
        window:      Window function name: 'hann', 'hamming', 'blackman', 'none'.
        n_peaks:     Number of dominant peaks to report.

    Returns:
        FFTResult with amplitude/power spectra and peak info.
    """
    x = series.dropna().astype(float).values
    n = len(x)
    if n < 4:
        raise ValueError("Signal must have at least 4 samples for FFT.")

    # Apply window
    win = _get_window(window, n)
    x_win = x * win

    # FFT – keep single-sided spectrum
    fft_vals = np.fft.rfft(x_win)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    amplitude = np.abs(fft_vals) * 2 / n   # normalise & single-sided
    amplitude[0] /= 2                        # DC component not doubled
    power = amplitude ** 2

    # Peak detection
    peak_indices, properties = sci_signal.find_peaks(
        amplitude,
        height=amplitude.max() * 0.05,  # ignore < 5% of max
        distance=max(1, n // 50),
    )
    peak_indices = peak_indices[np.argsort(amplitude[peak_indices])[::-1]][:n_peaks]

    peaks_df = pd.DataFrame({
        "frequency": freqs[peak_indices],
        "amplitude": amplitude[peak_indices],
        "power": power[peak_indices],
    })

    dominant_idx = int(np.argmax(amplitude[1:]) + 1)  # skip DC
    dominant_freq = float(freqs[dominant_idx])
    dominant_amp = float(amplitude[dominant_idx])

    stats = {
        "n_samples": n,
        "sample_rate": sample_rate,
        "freq_resolution": float(freqs[1] - freqs[0]) if len(freqs) > 1 else 0,
        "nyquist": sample_rate / 2,
        "dominant_freq": dominant_freq,
        "dominant_amplitude": dominant_amp,
        "total_power": float(power.sum()),
        "rms": float(np.sqrt(np.mean(x ** 2))),
        "window": window,
    }

    return FFTResult(
        freqs=freqs,
        amplitude=amplitude,
        power=power,
        dominant_freq=dominant_freq,
        dominant_amplitude=dominant_amp,
        peaks=peaks_df,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Band statistics
# ---------------------------------------------------------------------------

def compute_band_stats(
    fft_result: FFTResult,
    bands: Optional[dict[str, tuple[float, float]]] = None,
) -> list[BandStats]:
    """
    Compute energy statistics within user-defined frequency bands.

    Args:
        fft_result: Output from compute_fft().
        bands:      Dict mapping band name to (low_hz, high_hz) tuples.
                    If None, auto-divides spectrum into 4 equal bands.

    Returns:
        List of BandStats, one per band.
    """
    freqs = fft_result.freqs
    power = fft_result.power
    total_power = power.sum() or 1.0

    if bands is None:
        nyq = freqs[-1]
        q = nyq / 4
        bands = {
            "Band 1 (DC–25%)": (0, q),
            "Band 2 (25–50%)": (q, 2 * q),
            "Band 3 (50–75%)": (2 * q, 3 * q),
            "Band 4 (75–Nyq)": (3 * q, nyq),
        }

    results = []
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs <= hi)
        if mask.sum() == 0:
            continue
        band_power = power[mask]
        results.append(BandStats(
            band_name=name,
            freq_low=lo,
            freq_high=hi,
            mean_power=float(band_power.mean()),
            rms=float(np.sqrt(band_power.mean())),
            energy_fraction=float(band_power.sum() / total_power),
        ))
    return results


# ---------------------------------------------------------------------------
# STFT / Spectrogram
# ---------------------------------------------------------------------------

def compute_spectrogram(
    series: pd.Series,
    sample_rate: float = 1.0,
    nperseg: int = 64,
    noverlap: Optional[int] = None,
) -> SpectrogramResult:
    """
    Compute Short-Time Fourier Transform spectrogram.

    Args:
        series:     Input signal.
        sample_rate: Samples per second.
        nperseg:    Samples per STFT segment.
        noverlap:   Overlap between segments. Defaults to nperseg // 2.

    Returns:
        SpectrogramResult with times, freqs, and power matrix.
    """
    x = series.dropna().astype(float).values
    if noverlap is None:
        noverlap = nperseg // 2

    freqs, times, Sxx = sci_signal.spectrogram(
        x,
        fs=sample_rate,
        nperseg=min(nperseg, len(x)),
        noverlap=noverlap,
        scaling="density",
    )
    return SpectrogramResult(times=times, freqs=freqs, Sxx=Sxx)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_window(name: str, n: int) -> np.ndarray:
    name = name.lower()
    if name == "hann":
        return np.hanning(n)
    elif name == "hamming":
        return np.hamming(n)
    elif name == "blackman":
        return np.blackman(n)
    elif name in ("none", "rectangular", "rect"):
        return np.ones(n)
    else:
        return np.hanning(n)  # safe default
